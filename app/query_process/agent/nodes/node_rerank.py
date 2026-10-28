"""Evidence-preserving reranking with a deterministic, non-mutating fallback.

Importing this module does not load models, read .env or connect to services.
The model is loaded only when scoring a nonempty candidate list.
"""
from copy import deepcopy
from collections.abc import Mapping
import logging
import math

from app.query_process.agent.nodes.retrieval_common import (
    deduplicate_candidates,
    merge_candidates,
)

logger = logging.getLogger(__name__)
RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 1
RERANK_GAP_RATIO = 0.25
RERANK_GAP_ABS = 0.5


def get_reranker_model():
    from app.lm.reranker_utils import get_reranker_model as load_model

    return load_model()


def step_1_merge_rrf_mcp(state):
    return merge_candidates(state.get("rrf_chunks"), state.get("web_search_docs"))


def step_2_rerank_doc_list(doc_list, state):
    docs = deduplicate_candidates(doc_list)
    if not docs:
        return []
    query = state.get("rewritten_query") or state.get("original_query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("reranking requires a nonempty query")
    pairs = [[query, doc["text"]] for doc in docs]
    scores = get_reranker_model().compute_score(pairs, normalize=True)
    if isinstance(scores, (str, bytes, Mapping)):
        raise ValueError("reranker returned an invalid score container")
    # FlagReranker may return a scalar for a single pair.
    if not hasattr(scores, "__iter__"):
        scores = [scores]
    scores = list(scores)
    if len(scores) != len(docs):
        raise ValueError("reranker score count does not match candidate count")
    ranked = []
    for doc, raw_score in zip(docs, scores):
        score = float(raw_score)
        if isinstance(raw_score, bool) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("reranker returned an invalid normalized score")
        ranked.append({**doc, "score": score})
    return sorted(ranked, key=lambda doc: doc["score"], reverse=True)


def step_3_topk_and_gap(rerank_score_list):
    topk = min(RERANK_MAX_TOPK, len(rerank_score_list))
    for index in range(RERANK_MIN_TOPK - 1, topk - 1):
        first = float(rerank_score_list[index].get("score", 0.0))
        second = float(rerank_score_list[index + 1].get("score", 0.0))
        gap = first - second
        relative = gap / (abs(first) + 1e-6)
        if gap >= RERANK_GAP_ABS or relative >= RERANK_GAP_RATIO:
            topk = index + 1
            break
    return deepcopy(rerank_score_list[:topk])


def rerank_with_fallback(doc_list, state):
    """Return (candidates, degraded); fallback keeps RRF then web input order.

    A failed/invalid model response must not leak partially scored records or
    invoke score-gap truncation on synthetic fallback scores.
    """
    docs = deduplicate_candidates(doc_list)
    if not docs:
        return [], False
    try:
        ranked = step_2_rerank_doc_list(docs, state)
        return step_3_topk_and_gap(ranked), False
    except Exception as exc:
        logger.warning("Reranker unavailable; keeping retrieval order (%s)", type(exc).__name__)
        return [
            {**deepcopy(doc), "score": 0.0, "rerank_status": "fallback"}
            for doc in docs[:RERANK_MAX_TOPK]
        ], True


def node_rerank(state):
    from app.utils.task_utils import add_done_task, add_running_task

    name = "node_rerank"
    add_running_task(state["session_id"], name, state.get("is_stream", False))
    try:
        docs = step_1_merge_rrf_mcp(state)
        candidates, degraded = rerank_with_fallback(docs, state)
        state["reranked_docs"] = candidates
        state["rerank_degraded"] = degraded
        return state
    finally:
        add_done_task(state["session_id"], name, state.get("is_stream", False))
