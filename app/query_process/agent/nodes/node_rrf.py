"""RRF runtime adapter; the ranking transformation itself is dependency-free."""
from app.query_process.agent.nodes.retrieval_common import reciprocal_rank_fusion


def step_3_reciprocal_rank_fusion(source_with_weight, top_k: int = 5):
    return reciprocal_rank_fusion(source_with_weight, top_k)


def node_rrf(state):
    from app.utils.task_utils import add_done_task, add_running_task

    name = "node_rrf"
    add_running_task(state["session_id"], name, state.get("is_stream", False))
    try:
        state["rrf_chunks"] = step_3_reciprocal_rank_fusion([
            (state.get("embedding_chunks") or [], 1.0),
            (state.get("hyde_embedding_chunks") or [], 1.0),
        ])
        return state
    finally:
        add_done_task(state["session_id"], name, state.get("is_stream", False))
