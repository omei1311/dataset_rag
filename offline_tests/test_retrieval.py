"""Standard-library regression tests; no .env, model downloads or services."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.query_process.agent.nodes import node_rerank as rerank
from app.query_process.agent.nodes.node_rrf import step_3_reciprocal_rank_fusion
from app.query_process.agent.nodes.retrieval_common import (
    build_evidence_context,
    deduplicate_candidates,
    local_candidate,
    merge_candidates,
)


def hit(chunk_id=0, file_title="doc-a", text="evidence", **metadata):
    return {"id": chunk_id, "entity": {
        "chunk_id": chunk_id, "file_title": file_title, "item_name": "topic",
        "title": "section", "parent_title": "chapter", "content": text, **metadata,
    }}


class RetrievalTests(unittest.TestCase):
    def test_id_zero_and_document_identity_survive(self):
        items = step_3_reciprocal_rank_fusion([([hit(0, "a"), hit(0, "b")], 1)], 10)
        self.assertEqual(len(items), 2)
        self.assertEqual([x["entity"]["file_title"] for x in items], ["a", "b"])
        self.assertEqual(local_candidate(items[0])["chunk_id"], 0)

    def test_route_duplicates_do_not_inflate_rrf(self):
        items = step_3_reciprocal_rank_fusion([([hit(1), hit(1), hit(2)], 1)])
        self.assertEqual(len(items), 2)
        self.assertAlmostEqual(items[0]["rrf_score"], 1 / 61)
        self.assertAlmostEqual(items[1]["rrf_score"], 1 / 62)

    def test_independent_routes_vote_and_do_not_mutate(self):
        sources = [([hit(1)], 1), ([hit(1)], 0.5)]
        before = copy.deepcopy(sources)
        items = step_3_reciprocal_rank_fusion(sources)
        self.assertAlmostEqual(items[0]["rrf_score"], 1.5 / 61)
        items[0]["entity"]["title"] = "changed"
        self.assertEqual(sources, before)

    def test_empty_routes_and_zero_topk(self):
        self.assertEqual(step_3_reciprocal_rank_fusion([(None, 1), ([], 1)]), [])
        self.assertEqual(step_3_reciprocal_rank_fusion([([hit()], 1)], 0), [])

    def test_invalid_weights_and_topk_are_rejected(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                step_3_reciprocal_rank_fusion([([hit()], weight)])
        for topk in (-1, True, 1.5):
            with self.subTest(topk=topk), self.assertRaises(ValueError):
                step_3_reciprocal_rank_fusion([], topk)

    def test_metadata_preserved_through_merge(self):
        merged = merge_candidates([hit(page_number=3, part=1)], [])
        self.assertEqual(merged[0]["document_id"], "doc-a")
        for field in ("file_title", "parent_title", "title", "item_name", "page_number", "part"):
            self.assertIn(field, merged[0])

    def test_missing_metadata_is_not_invented(self):
        candidate = local_candidate({"id": 0, "entity": {"content": "a"}})
        self.assertEqual(candidate["chunk_id"], 0)
        self.assertEqual(candidate["document_id"], "")

    def test_anonymous_different_text_is_not_collapsed(self):
        docs = [{"chunk_id": 0, "text": "a"}, {"chunk_id": 0, "text": "b"}]
        self.assertEqual(len(deduplicate_candidates(docs)), 2)

    def test_merge_deduplicates_web_urls_and_keeps_sources_separate(self):
        merged = merge_candidates([hit()], [
            {"url": "https://example.test/a", "snippet": "one"},
            {"url": "https://example.test/a", "snippet": "duplicate"},
        ])
        self.assertEqual([d["source"] for d in merged], ["local", "web"])
        self.assertEqual(merged[1]["document_id"], "https://example.test/a")

    def test_empty_text_candidates_are_dropped(self):
        self.assertEqual(merge_candidates([hit(text="   ")], [{"snippet": ""}]), [])

    def test_rerank_preserves_identity_and_input(self):
        docs = merge_candidates([hit(0), hit(1, "other")], [])
        before = copy.deepcopy(docs)
        model = SimpleNamespace(compute_score=lambda pairs, normalize: [0.2, 0.9])
        with patch.object(rerank, "get_reranker_model", return_value=model):
            ranked = rerank.step_2_rerank_doc_list(docs, {"original_query": "q"})
        self.assertEqual(ranked[0]["file_title"], "other")
        self.assertEqual(ranked[1]["chunk_id"], 0)
        self.assertEqual(docs, before)

    def test_single_scalar_score_supported(self):
        model = SimpleNamespace(compute_score=lambda pairs, normalize: 0.8)
        with patch.object(rerank, "get_reranker_model", return_value=model):
            ranked = rerank.step_2_rerank_doc_list(merge_candidates([hit()], []), {"original_query": "q"})
        self.assertEqual(ranked[0]["score"], 0.8)

    def test_empty_candidates_never_load_model(self):
        with patch.object(rerank, "get_reranker_model", side_effect=AssertionError):
            self.assertEqual(rerank.rerank_with_fallback([], {}), ([], False))

    def test_model_failure_keeps_rrf_order_and_input(self):
        docs = merge_candidates([hit(2), hit(1)], [{"url": "u", "snippet": "web"}])
        before = copy.deepcopy(docs)
        with patch.object(rerank, "get_reranker_model", side_effect=RuntimeError("unavailable")):
            result, degraded = rerank.rerank_with_fallback(docs, {"original_query": "q"})
        self.assertTrue(degraded)
        self.assertEqual([d["chunk_id"] for d in result], [2, 1, None])
        self.assertTrue(all(d["rerank_status"] == "fallback" for d in result))
        self.assertEqual(docs, before)

    def test_invalid_score_responses_fall_back_without_partial_mutation(self):
        docs = merge_candidates([hit(1), hit(2)], [])
        before = copy.deepcopy(docs)
        for scores in ([0.8], [0.8, float("nan")], [0.8, float("inf")], [0.8, "bad"],
                       [0.8, 1.5], [True, 0.5], "12", {"a": 0.5, "b": 0.4}):
            model = SimpleNamespace(compute_score=lambda pairs, normalize: scores)
            with self.subTest(scores=scores), patch.object(rerank, "get_reranker_model", return_value=model):
                result, degraded = rerank.rerank_with_fallback(docs, {"original_query": "q"})
                self.assertTrue(degraded)
                self.assertEqual([d["chunk_id"] for d in result], [1, 2])
                self.assertEqual(docs, before)

    def test_score_gap_truncation_keeps_complete_metadata(self):
        docs = [{**local_candidate(hit(i)), "score": score} for i, score in enumerate((0.9, 0.88, 0.1))]
        self.assertEqual(len(rerank.step_3_topk_and_gap(docs)), 2)
        self.assertIn("file_title", rerank.step_3_topk_and_gap(docs)[0])

    def test_context_only_reports_included_evidence_and_is_bounded(self):
        docs = merge_candidates([hit(1, text="x" * 2000), hit(2)], [])
        context, references = build_evidence_context(docs, 350)
        self.assertEqual(len(context), 350)
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0]["chunk_id"], 1)
        self.assertTrue(references[0]["truncated"])

    def test_context_small_budget_has_no_false_sources(self):
        self.assertEqual(build_evidence_context(merge_candidates([hit()], []), 1), ("", []))

    def test_context_numbering_and_distinct_document_sources(self):
        context, references = build_evidence_context(merge_candidates([hit(0, "a"), hit(0, "b")], []))
        self.assertEqual([r["citation"] for r in references], [1, 2])
        self.assertIn('"document_id": "a"', context)
        self.assertIn('"document_id": "b"', context)

    def test_pure_imports_do_not_load_runtime_dependencies(self):
        root = str(Path(__file__).resolve().parents[1])
        code = (
            "import sys; sys.path.insert(0, " + repr(root) + "); "
            "import app.query_process.agent.nodes.node_rrf; "
            "import app.query_process.agent.nodes.node_rerank; "
            "assert not any(m in sys.modules for m in "
            "['torch','FlagEmbedding','fastapi','dotenv','app.conf.lm_config','app.core.logger'])"
        )
        completed = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class AnswerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.results, self.events, self.prompts = {}, [], []
        self.llm = Mock()
        self.llm.invoke.return_value = SimpleNamespace(content="answer [1]")
        task = ModuleType("app.utils.task_utils")
        task.add_running_task = lambda *a, **k: None
        task.add_done_task = lambda *a, **k: None
        task.set_task_result = lambda session, key, value: self.results.update({key: value})
        sse = ModuleType("app.utils.sse_utils")
        sse.push_to_session = lambda session, event, payload: self.events.append((event, payload))
        sse.SSEEvent = SimpleNamespace(DELTA="delta", FINAL="final")
        log = ModuleType("app.core.logger")
        log.logger = Mock()
        prompt = ModuleType("app.core.load_prompt")
        def load_prompt(name, **kwargs):
            self.prompts.append(kwargs)
            return json.dumps(kwargs)
        prompt.load_prompt = load_prompt
        llm = ModuleType("app.lm.lm_utils")
        llm.get_llm_client = lambda: self.llm
        mongo = ModuleType("app.clients.mongo_history_utils")
        mongo.save_chat_message = lambda **kwargs: None
        self.stubs = patch.dict(sys.modules, {
            task.__name__: task, sse.__name__: sse, log.__name__: log,
            prompt.__name__: prompt, llm.__name__: llm, mongo.__name__: mongo,
        })
        self.stubs.start()
        module_name = "app.query_process.agent.nodes.node_answer_output"
        self.module = importlib.import_module(module_name)
        self.addCleanup(sys.modules.pop, module_name, None)
        self.addCleanup(self.stubs.stop)

    def test_sources_are_in_prompt_and_synchronous_result(self):
        state = {"session_id": "s", "original_query": "q", "reranked_docs": merge_candidates([hit()], [])}
        result = self.module.node_answer_output(state)
        self.assertEqual(self.results["sources"][0]["chunk_id"], 0)
        self.assertEqual(result["sources"], self.results["sources"])
        self.assertIn('"document_id": "doc-a"', self.prompts[0]["context"])

    def test_empty_evidence_does_not_invoke_model(self):
        state = {"session_id": "s", "original_query": "q", "reranked_docs": []}
        self.module.node_answer_output(state)
        self.llm.invoke.assert_not_called()
        self.assertEqual(self.results["sources"], [])
        self.assertIn("没有可用", self.results["answer"])
        self.assertEqual(state["image_urls"], [])

    def test_images_in_truncated_tail_and_omitted_document_are_not_returned(self):
        docs = merge_candidates([
            hit(1, text="x" * 2000 + "![tail](https://example.test/tail.jpg)"),
            hit(2, text="![omitted](https://example.test/omitted.jpg)"),
        ], [])
        state = {"session_id": "s", "original_query": "q", "reranked_docs": docs}
        with patch.object(self.module, "MAX_CONTEXT_CHARS", 350):
            self.module.node_answer_output(state)
        self.assertEqual([s["chunk_id"] for s in state["sources"]], [1])
        self.assertEqual(state["image_urls"], [])

    def test_images_in_retained_context_and_web_source_are_preserved(self):
        docs = merge_candidates([
            hit(1, text="![included](https://example.test/included.jpg)"),
        ], [{"url": "https://example.test/web.PNG?token=demo", "snippet": "image description"}])
        state = {"session_id": "s", "original_query": "q", "reranked_docs": docs}
        self.module.node_answer_output(state)
        self.assertEqual(state["image_urls"], [
            "https://example.test/included.jpg", "https://example.test/web.PNG?token=demo",
        ])

    def test_clarification_answer_does_not_reuse_prior_sources_or_images(self):
        state = {"session_id": "s", "original_query": "q", "answer": "请明确文档",
                 "sources": [{"url": "https://example.test/stale.jpg"}],
                 "evidence_context": "![old](https://example.test/old.jpg)",
                 "image_urls": ["https://example.test/old.jpg"]}
        self.module.node_answer_output(state)
        self.llm.invoke.assert_not_called()
        self.assertEqual(state["sources"], [])
        self.assertEqual(state["image_urls"], [])

    def test_streaming_final_event_exists_even_without_images(self):
        self.llm.stream.return_value = [SimpleNamespace(content="answer [1]")]
        state = {"session_id": "s", "original_query": "q", "is_stream": True,
                 "reranked_docs": merge_candidates([hit()], []), "rerank_degraded": True}
        self.module.node_answer_output(state)
        final = [payload for event, payload in self.events if event == "final"]
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["sources"][0]["file_title"], "doc-a")
        self.assertTrue(final[0]["rerank_degraded"])
        self.assertEqual(final[0]["image_urls"], [])


if __name__ == "__main__":
    unittest.main()
