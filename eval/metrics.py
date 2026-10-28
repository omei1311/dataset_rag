"""Pure standard-library retrieval metrics with explicit evidence identities.

No application configuration, model, client or environment file is imported.
Local evidence is relevant only when both its document scope and chunk ID match.
The ranked list is deduplicated before applying K; missing metadata never widens
the matching rule. No-answer questions have separate diagnostics, not free hits.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from statistics import mean
from typing import Any


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _chunk_id(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} cannot be a boolean")
    if isinstance(value, int) and value >= 0:
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"{label} must be a non-negative integer or a non-empty string")


def _reference(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if value.get("source", "local") != "local":
        raise ValueError(f"{label}: only local chunk ground truth is supported")
    field = "document_id" if value.get("document_id") else "file_title"
    document = _text(value.get(field), f"{label}.{field}")
    chunk = _chunk_id(value.get("chunk_id"), f"{label}.chunk_id")
    return {field: document, "chunk_id": chunk, "source": "local"}


def validate_questions(data: Any) -> list[dict[str, Any]]:
    """Validate and copy a canonical or explicitly document-scoped legacy set.

    Canonical: question_id, question, answerable, relevant_chunks. A legacy
    expected_chunk_ids list needs expected_document_id or expected_file_title.
    Keyword-only labels are intentionally rejected: they are not exact evidence.
    """
    if not isinstance(data, list) or not data:
        raise ValueError("questions must be a non-empty JSON array")
    result, seen_ids = [], set()
    for index, original in enumerate(data, 1):
        if not isinstance(original, Mapping):
            raise ValueError(f"question {index} must be an object")
        q = deepcopy(dict(original))
        qid = _text(q.get("question_id"), f"question {index}.question_id")
        if qid in seen_ids:
            raise ValueError(f"duplicate question_id: {qid}")
        seen_ids.add(qid)
        question = _text(q.get("question"), f"{qid}.question")
        answerable = q.get("answerable", True)
        if not isinstance(answerable, bool):
            raise ValueError(f"{qid}.answerable must be a boolean")
        if "synthetic" in q and not isinstance(q["synthetic"], bool):
            raise ValueError(f"{qid}.synthetic must be a boolean")
        if "relevant_chunks" in q:
            refs = q["relevant_chunks"]
        elif "expected_chunk_ids" in q:
            ids = q["expected_chunk_ids"]
            if not isinstance(ids, list):
                raise ValueError(f"{qid}.expected_chunk_ids must be an array")
            field = "document_id" if q.get("expected_document_id") else "file_title"
            document = q.get("expected_document_id") or q.get("expected_file_title")
            refs = [{field: document, "chunk_id": chunk} for chunk in ids]
        else:
            refs = []
        if not isinstance(refs, list):
            raise ValueError(f"{qid}.relevant_chunks must be an array")
        refs = [_reference(ref, f"{qid}.relevant_chunks[{i}]") for i, ref in enumerate(refs)]
        ref_keys = [json.dumps(ref, sort_keys=True) for ref in refs]
        if len(ref_keys) != len(set(ref_keys)):
            raise ValueError(f"{qid}: duplicate relevant chunk labels")
        if answerable and not refs:
            raise ValueError(f"{qid}: answerable questions need document+chunk labels; keywords alone are insufficient")
        if not answerable and refs:
            raise ValueError(f"{qid}: no-answer questions must have no relevant chunks")
        if "item_names" in q:
            names = q["item_names"]
            if not isinstance(names, list) or not names:
                raise ValueError(f"{qid}.item_names must be a non-empty string array")
            q["item_names"] = [_text(name, f"{qid}.item_names") for name in names]
            if len(q["item_names"]) != len(set(q["item_names"])):
                raise ValueError(f"{qid}: duplicate item_names")
        q.update(question_id=qid, question=question, answerable=answerable, relevant_chunks=refs)
        result.append(q)
    return result


def validate_ks(ks) -> tuple[int, ...]:
    values = tuple(ks)
    if not values or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 for k in values):
        raise ValueError("K values must be positive integers")
    return tuple(sorted(set(values)))


def _optional_text(candidate, field):
    value = candidate.get(field)
    if value is None or value == "":
        return ""
    return _text(value, f"candidate.{field}")


def _candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError("each candidate must be an object")
    result = deepcopy(dict(candidate))
    result["source"] = result.get("source", "local")
    if result["source"] not in ("local", "web"):
        raise ValueError("candidate.source must be local or web")
    for field in ("document_id", "file_title", "url"):
        result[field] = _optional_text(result, field)
    value = result.get("chunk_id")
    result["chunk_id"] = None if value is None or value == "" else _chunk_id(value, "candidate.chunk_id")
    return result


def _key(candidate):
    if candidate["source"] == "web" and candidate["url"]:
        return ("web", candidate["url"])
    document = candidate["document_id"] or candidate["file_title"]
    if document and candidate["chunk_id"] is not None:
        return (candidate["source"], document, candidate["chunk_id"])
    # Missing identity is never invented from item_name or a bare chunk ID.
    return ("anonymous", json.dumps(candidate, sort_keys=True, ensure_ascii=False))


def deduplicate_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be an array")
    seen, result = set(), []
    for raw in candidates:
        candidate = _candidate(raw)
        key = _key(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _matches(reference, candidate):
    if candidate["source"] != "local" or candidate["chunk_id"] != reference["chunk_id"]:
        return False
    if "document_id" in reference:
        # Runtime normalizes file_title into document_id when no ID is stored.
        document = candidate["document_id"] or candidate["file_title"]
        return bool(document) and document == reference["document_id"]
    return bool(candidate["file_title"]) and candidate["file_title"] == reference["file_title"]


def is_relevant(question, candidate) -> bool:
    """Compatibility helper for live tooling, using the same strict protocol."""
    q = validate_questions([question])[0]
    return q["answerable"] and any(_matches(ref, _candidate(candidate)) for ref in q["relevant_chunks"])


def evaluate_question(question, candidates, ks=(1, 3, 5)) -> dict[str, Any]:
    ks = validate_ks(ks)
    q = validate_questions([question])[0]
    unique = deduplicate_candidates(candidates)
    matches = [
        {index for index, ref in enumerate(q["relevant_chunks"]) if _matches(ref, candidate)}
        for candidate in unique
    ]
    first_rank = next((rank for rank, found in enumerate(matches, 1) if found), None)
    result = {
        "question_id": q["question_id"],
        "question": q["question"],
        "answerable": q["answerable"],
        "raw_candidate_count": len(candidates),
        "unique_candidate_count": len(unique),
        "duplicate_candidate_count": len(candidates) - len(unique),
        "missing_local_identity_count": sum(
            c["source"] == "local" and (not (c["document_id"] or c["file_title"]) or c["chunk_id"] is None)
            for c in unique
        ),
        "relevant_chunk_count": len(q["relevant_chunks"]),
        "first_relevant_rank": first_rank,
        "reciprocal_rank": (1.0 / first_rank if first_rank else 0.0) if q["answerable"] else None,
        "hit_at": {},
        "recall_at": {},
        "candidate_identities": [
            {field: c[field] for field in ("source", "document_id", "file_title", "chunk_id", "url")}
            for c in unique
        ],
    }
    for k in ks:
        found = set().union(*matches[:k]) if matches[:k] else set()
        result["hit_at"][str(k)] = int(bool(found)) if q["answerable"] else None
        result["recall_at"][str(k)] = len(found) / len(q["relevant_chunks"]) if q["answerable"] else None
    return result


def evaluate_dataset(questions, configs, ks=(1, 3, 5)) -> dict[str, Any]:
    """Evaluate complete config -> question_id -> ranked candidates mappings."""
    questions = validate_questions(questions)
    ks = validate_ks(ks)
    if not isinstance(configs, Mapping) or not configs:
        raise ValueError("configs must be a non-empty object")
    expected = {q["question_id"] for q in questions}
    blocks = {}
    for config, predictions in configs.items():
        _text(config, "config name")
        if not isinstance(predictions, Mapping):
            raise ValueError(f"{config}: predictions must be an object")
        actual = set(predictions)
        if actual != expected:
            raise ValueError(f"{config}: question IDs differ; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        details = [evaluate_question(q, predictions[q["question_id"]], ks) for q in questions]
        answerable = [row for row in details if row["answerable"]]
        no_answer = [row for row in details if not row["answerable"]]
        metrics = {"answerable_questions": len(answerable), "no_answer_questions": len(no_answer)}
        for k in ks:
            metrics[f"hit_at_{k}"] = mean(row["hit_at"][str(k)] for row in answerable) if answerable else None
            metrics[f"recall_at_{k}"] = mean(row["recall_at"][str(k)] for row in answerable) if answerable else None
        metrics["mrr"] = mean(row["reciprocal_rank"] for row in answerable) if answerable else None
        empty = sum(row["unique_candidate_count"] == 0 for row in no_answer)
        metrics["no_answer_empty_candidate_count"] = empty
        metrics["no_answer_empty_candidate_rate"] = empty / len(no_answer) if no_answer else None
        metrics["duplicate_candidates_removed"] = sum(row["duplicate_candidate_count"] for row in details)
        metrics["missing_local_identity_count"] = sum(row["missing_local_identity_count"] for row in details)
        blocks[config] = {"metrics": metrics, "per_question": details}
    return {
        "schema_version": 1,
        "question_count": len(questions),
        "k_values": list(ks),
        "protocol": {
            "relevance": "exact local document scope + chunk_id; no keyword fallback",
            "ranking": "deduplicate evidence before applying K; keep first occurrence",
            "mrr": "mean reciprocal first-hit rank across answerable questions, using all submitted unique candidates",
            "recall": "macro mean of unique labeled relevant chunks found / labeled relevant chunks",
            "no_answer": "excluded from Hit/Recall/MRR; empty-candidate rate is not answer refusal accuracy",
            "unlabeled_candidates": "non-relevant under the supplied labels; incomplete labels can bias relevance and recall estimates",
        },
        "configs": blocks,
    }
