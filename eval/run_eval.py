"""Live, oracle-scoped retrieval experiments; this is not an end-to-end QA test.

The annotated item_names are supplied directly, and query rewriting / product
resolution / answer generation are skipped. Runtime dependencies are imported
inside the functions that actually need them. Importing this module or asking
for --help never loads .env, models or database clients.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import evaluate_dataset, is_relevant, validate_questions
from eval.offline_eval import configure_console, read_json, render_markdown, write_reports


EVAL_DIR = Path(__file__).resolve().parent
CONFIG_LABELS = {
    "C0": "C0 dense+sparse hybrid",
    "C1": "C1 hybrid+HyDE+RRF",
    "C2": "C2 C1+reranker",
    "C3": "C3 C2+web candidates",
}
OUTPUT_FIELDS = ["chunk_id", "content", "file_title", "title", "parent_title", "item_name"]


def load_questions(path):
    return validate_questions(read_json(path))


def _store():
    from app.clients.milvus_utils import get_milvus_client
    from app.conf.milvus_config import milvus_config

    client = get_milvus_client()
    if client is None or not milvus_config.chunks_collection:
        raise RuntimeError("Milvus client or CHUNKS_COLLECTION is unavailable")
    return client, milvus_config.chunks_collection


def iter_corpus_chunks():
    client, collection = _store()
    if not hasattr(client, "query_iterator"):
        raise RuntimeError("Corpus validation requires query_iterator; refusing a silently truncated export")
    iterator = client.query_iterator(
        collection_name=collection, batch_size=200,
        filter="chunk_id >= 0", output_fields=OUTPUT_FIELDS,
    )
    try:
        while True:
            batch = iterator.next()
            if not batch:
                return
            yield from batch
    finally:
        close = getattr(iterator, "close", None)
        if close:
            close()


def dump_corpus(output_dir):
    rows = list(iter_corpus_chunks())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "corpus_dump.json"
    # This is an explicit, local export for annotation; it may contain corpus text.
    path.write_text(json.dumps({"total_chunks": len(rows), "chunks": rows}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Exported {len(rows)} chunks for annotation: {path}")


def check_questions(questions):
    from app.query_process.agent.nodes.retrieval_common import local_candidate

    candidates = [local_candidate(row) for row in iter_corpus_chunks()]
    errors = []
    item_names = {row.get("item_name") for row in candidates}
    for q in questions:
        for name in q.get("item_names", []):
            if name not in item_names:
                errors.append(f"{q['question_id']}: item_name not found: {name}")
        for ref in q["relevant_chunks"]:
            single = {**q, "relevant_chunks": [ref]}
            if not any(is_relevant(single, candidate) for candidate in candidates):
                errors.append(f"{q['question_id']}: evidence not found: {ref}")
    if errors:
        raise ValueError("\n".join(errors))
    print("All answerable evidence identities exist in the corpus. No-answer correctness still requires human review.")


def hybrid_search_chunks(query_text, item_names, topk):
    from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search
    from app.lm.embedding_utils import generate_embeddings

    client, collection = _store()
    embeddings = generate_embeddings([query_text])
    requests = create_hybrid_search_requests(
        dense_vector=embeddings["dense"][0], sparse_vector=embeddings["sparse"][0],
        expr="item_name in " + json.dumps(item_names, ensure_ascii=False), limit=topk,
    )
    response = hybrid_search(
        client=client, collection_name=collection, reqs=requests,
        ranker_weights=(0.9, 0.1), norm_score=True, limit=topk,
        output_fields=OUTPUT_FIELDS,
    )
    return response[0] if response else []


def fetch_web_docs(question):
    from app.query_process.agent.nodes.node_web_search_mcp import mcp_call_streamable

    result = asyncio.run(mcp_call_streamable(question))
    return json.loads(result.content[0].text).get("pages", [])


def run_config(config, question, topk):
    from app.query_process.agent.nodes.retrieval_common import local_candidate, reciprocal_rank_fusion

    timings = {}

    def stage(name, function):
        started = time.perf_counter()
        try:
            return function()
        finally:
            timings[name] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    original_query = question["question"]
    names = question["item_names"]
    degraded = False
    hits = stage("vector_ms", lambda: hybrid_search_chunks(original_query, names, topk))
    if config == "C0":
        docs = [local_candidate(hit) for hit in hits]
    else:
        from app.query_process.agent.nodes.node_search_embedding_hyde import step_1_create_hyde_doc

        def hyde():
            hypothetical = step_1_create_hyde_doc(original_query)
            return hybrid_search_chunks(original_query + hypothetical, names, topk)

        hyde_hits = stage("hyde_ms", hyde)
        fused = stage("fusion_ms", lambda: reciprocal_rank_fusion([(hits, 1.0), (hyde_hits, 1.0)], top_k=topk))
        if config == "C1":
            docs = [local_candidate(hit) for hit in fused]
        else:
            from app.query_process.agent.nodes.node_rerank import step_1_merge_rrf_mcp, rerank_with_fallback

            web_docs = stage("web_ms", lambda: fetch_web_docs(original_query)) if config == "C3" else []
            combined = step_1_merge_rrf_mcp({"rrf_chunks": fused, "web_search_docs": web_docs})
            docs, degraded = stage("rerank_ms", lambda: rerank_with_fallback(combined, {"rewritten_query": original_query}))
    timings["total_ms"] = (time.perf_counter() - started) * 1000
    # Normalize third-party scalar types before entering the pure JSON protocol.
    docs = json.loads(json.dumps(docs, ensure_ascii=False, default=str))
    return docs, timings, degraded


def evaluate(questions, configs, topk):
    predictions, diagnostics = {}, {}
    for config in configs:
        rankings, rows = {}, []
        for q in questions:
            start = time.perf_counter()
            try:
                docs, timing, degraded = run_config(config, q, topk)
                error = None
            except Exception as exc:
                docs, degraded = [], False
                timing = {"total_ms": (time.perf_counter() - start) * 1000}
                error = f"{type(exc).__name__}: {exc}"[:500]
            rankings[q["question_id"]] = docs
            rows.append({"question_id": q["question_id"], "timing_ms": timing, "rerank_degraded": degraded, "error": error})
            print(f"{config} {q['question_id']}: candidates={len(docs)} degraded={degraded} error={error}")
        predictions[CONFIG_LABELS[config]] = rankings
        diagnostics[CONFIG_LABELS[config]] = rows
    report = evaluate_dataset(questions, predictions, (1, 3, 5, topk))
    for label, rows in diagnostics.items():
        report["configs"][label]["runtime"] = {
            "per_question": rows,
            "failed_questions": sum(row["error"] is not None for row in rows),
            "rerank_degraded_questions": sum(row["rerank_degraded"] for row in rows),
            "mean_total_ms": mean(row["timing_ms"]["total_ms"] for row in rows),
        }
    report.update(
        synthetic=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        execution_scope="LIVE oracle item_names / retrieval only; NOT end-to-end QA",
        scope_notes=[
            "item_names supplied by annotations: oracle document scope; entity resolution and query rewriting are skipped.",
            "No answer generation, citation correctness, refusal behavior or ingestion quality is evaluated.",
            "Retrieval/RRF use the supplied topk; reranking follows the runtime max-topk and score-gap policy, so returned list lengths can differ.",
            "Reranker fallback counts are recorded; fallback rankings are not presented as successful neural reranking.",
            "Runtime failures stay in the denominator with empty candidates and are separately recorded, never silently dropped.",
            "Timing is local wall-clock time for these calls, not production latency or throughput evidence.",
        ],
    )
    return report


def main(argv=None):
    configure_console()
    parser = argparse.ArgumentParser(description="Live oracle-scoped retrieval experiments. Requires real services; not end-to-end QA.")
    parser.add_argument("--questions", default=str(EVAL_DIR / "questions.example.json"))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--configs", default="C0,C1,C2,C3")
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--dump-corpus", action="store_true")
    parser.add_argument("--check-questions", action="store_true")
    parser.add_argument("--output-dir", default=str(EVAL_DIR / "reports" / "live"))
    args = parser.parse_args(argv)
    if args.topk <= 0:
        parser.error("--topk must be positive")
    try:
        if args.dump_corpus:
            dump_corpus(args.output_dir)
            return 0
        questions = load_questions(args.questions)
        if any(q.get("synthetic", False) for q in questions):
            raise ValueError("The bundled questions are SYNTHETIC. Use offline_eval for the demo, or supply your own real annotated question set.")
        if args.check_questions:
            check_questions(questions)
            return 0
        if any(not q.get("item_names") for q in questions):
            raise ValueError("Live oracle evaluation requires explicit item_names for every question")
        configs = list(dict.fromkeys(value.strip().upper() for value in args.configs.split(",") if value.strip()))
        if not configs or any(config not in CONFIG_LABELS for config in configs):
            raise ValueError("--configs must select C0,C1,C2,C3")
        if args.skip_web:
            configs = [config for config in configs if config != "C3"]
        if not configs:
            raise ValueError("No configurations remain after --skip-web")
        report = evaluate(questions, configs, args.topk)
        report["questions_path"] = str(args.questions)
        paths = write_reports(report, args.output_dir)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(render_markdown(report))
    for path in paths:
        print(path)
    return int(any(block["runtime"]["failed_questions"] for block in report["configs"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
