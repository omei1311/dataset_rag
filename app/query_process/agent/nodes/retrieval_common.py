"""Pure retrieval transformations, shared by runtime nodes and offline tests.

No model, database, logging configuration or environment variables are loaded here.
Evidence identity includes the source document: chunk 0 in two files is not one
piece of evidence. Missing identities are handled conservatively by content hash.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any


def _present(value: Any) -> bool:
    return value is not None and value != ""


def local_candidate(hit: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a Milvus hit or a flat local candidate without losing evidence."""
    entity = hit.get("entity")
    entity = entity if isinstance(entity, Mapping) else hit
    chunk_id = entity.get("chunk_id")
    if not _present(chunk_id):
        chunk_id = hit.get("id")
    doc = {
        "chunk_id": chunk_id,
        "document_id": entity.get("document_id") or entity.get("file_title") or "",
        "file_title": entity.get("file_title") or "",
        "item_name": entity.get("item_name") or "",
        "title": entity.get("title") or "",
        "parent_title": entity.get("parent_title") or "",
        "text": entity.get("text") or entity.get("content") or "",
        "source": "local",
        "url": entity.get("url") or "",
    }
    for field in ("page", "page_number", "page_start", "page_end", "section", "part"):
        if field in entity:
            doc[field] = deepcopy(entity[field])
    for field in ("rrf_score", "score"):
        if field in hit:
            doc[field] = hit[field]
    return doc


def evidence_key(doc: Mapping[str, Any]) -> tuple[str, ...]:
    """Stable identity for deduplication, preserving ID 0 and document scope.

    Anonymous candidates may be deduplicated by content but never acquire an
    invented document identity. Evaluators must still reject missing metadata.
    """
    source = doc.get("source") or "local"
    if source == "web" and doc.get("url"):
        return ("web", str(doc["url"]))
    document = doc.get("document_id") or doc.get("file_title")
    chunk_id = doc.get("chunk_id")
    if _present(document) and _present(chunk_id):
        return (str(source), str(document), str(chunk_id))
    text = str(doc.get("text") or doc.get("content") or "")
    digest = sha256(text.encode("utf-8")).hexdigest()
    return (str(source), str(document or ""), str(chunk_id), digest)


def deduplicate_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first occurrence, copying records so downstream scoring is safe."""
    seen = set()
    result = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("candidate must be a mapping")
        if not str(candidate.get("text") or "").strip():
            continue
        key = evidence_key(candidate)
        if key not in seen:
            seen.add(key)
            result.append(deepcopy(dict(candidate)))
    return result


def merge_candidates(rrf_chunks, web_docs) -> list[dict[str, Any]]:
    """Keep local evidence first and web sources explicit, including all metadata."""
    candidates = [local_candidate(hit) for hit in (rrf_chunks or [])]
    for raw in web_docs or []:
        candidates.append({
            "chunk_id": raw.get("chunk_id"),
            "document_id": raw.get("document_id") or raw.get("url") or "",
            "file_title": raw.get("file_title") or "",
            "item_name": raw.get("item_name") or "",
            "text": raw.get("snippet") or raw.get("text") or raw.get("content") or "",
            "title": raw.get("title") or "",
            "parent_title": raw.get("parent_title") or "",
            "source": "web",
            "url": raw.get("url") or "",
        })
    return deduplicate_candidates(candidates)


def reciprocal_rank_fusion(source_with_weight, top_k: int = 5, *, k: float = 60.0):
    """Weighted RRF over distinct evidence per route; stable ties, no mutation.

    Repeating a chunk inside one route cannot inflate its score. The same piece
    of evidence retrieved by independent routes still receives both votes.
    Return the original hit shape, with a separate ``rrf_score`` field.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not math.isfinite(k) or k < 0:
        raise ValueError("RRF k must be finite and non-negative")
    scores: dict[tuple[str, ...], float] = {}
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    for source, weight in source_with_weight:
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and non-negative")
        if weight == 0:
            continue
        seen = set()
        rank = 0
        for hit in source or []:
            candidate = local_candidate(hit)
            if not str(candidate["text"]).strip():
                continue
            key = evidence_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            rank += 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            if key not in records:
                records[key] = deepcopy(dict(hit))
            else:
                # A route may omit metadata returned by another route.
                prior = records[key].get("entity")
                incoming = hit.get("entity")
                if isinstance(prior, dict) and isinstance(incoming, Mapping):
                    for name, value in incoming.items():
                        if not _present(prior.get(name)):
                            prior[name] = deepcopy(value)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{**records[key], "rrf_score": scores[key]} for key in ordered]


def build_evidence_context(candidates, max_chars: int = 12000):
    """Return bounded context and the exact evidence records included in it.

    References describe retrieved evidence, not verified model citations. Text
    that was omitted by the budget must not appear in the returned references.
    A first oversized chunk is partially retained instead of yielding no context.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 0:
        raise ValueError("max_chars must be a non-negative integer")
    blocks, references = [], []
    used = 0
    for doc in deduplicate_candidates(candidates):
        index = len(references) + 1
        metadata = {name: deepcopy(doc.get(name)) for name in (
            "source", "document_id", "file_title", "chunk_id", "title", "parent_title", "url"
        )}
        for name in ("page", "page_number", "page_start", "page_end"):
            if name in doc:
                metadata[name] = doc[name]
        header = f"[{index}] " + json.dumps(metadata, ensure_ascii=False) + "\n"
        separator = "\n\n" if blocks else ""
        remaining = max_chars - used - len(separator) - len(header)
        if remaining <= 0:
            continue
        text = str(doc["text"])
        retained = text[:remaining]
        block = separator + header + retained
        blocks.append(block)
        references.append({
            "citation": index, **metadata,
            "truncated": len(retained) < len(text),
        })
        used += len(block)
        if len(retained) < len(text):
            break
    return "".join(blocks), references
