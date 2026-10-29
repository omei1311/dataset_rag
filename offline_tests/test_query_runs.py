"""Exercise the real query entry functions without importing service clients."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid

from app.query_process.agent.state import create_query_default_state
from app.utils.query_runs import QueryInProgressError, SessionRunRegistry


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class QueryRunTests(unittest.TestCase):
    def setUp(self):
        self.results = {"s": {"answer": "old", "sources": ["old"], "rerank_degraded": True}}
        self.statuses = {}
        self.events = []
        self.history = ["existing MongoDB history"]
        self.registry = SessionRunRegistry()
        self.graph = Mock()
        def clear_task(session):
            self.results.pop(session, None)
            self.statuses.pop(session, None)
        def update_status(session, status, stream=False):
            self.statuses[session] = status
        def get_result(session, key, default=""):
            self.assertTrue(self.registry.is_active(session), "result must be read while owned")
            return self.results.get(session, {}).get(key, default)
        self.namespace = {
            "_query_runs": self.registry, "QueryInProgressError": QueryInProgressError,
            "clear_task": clear_task, "update_task_status": update_status,
            "get_task_result": get_result, "create_query_default_state": create_query_default_state,
            "query_app": self.graph, "logger": Mock(), "uuid": uuid,
            "push_to_session": lambda s, event, payload: self.events.append((event, payload)),
            "create_sse_queue": Mock(), "SSEEvent": SimpleNamespace(ERROR="error"),
            "QueryRequest": SimpleNamespace, "BackgroundTasks": Mock,
            "HTTPException": FakeHTTPException,
        }
        # Compile the unchanged function bodies from the service module. The API
        # imports itself initialize external clients, so tests isolate only those
        # dependencies rather than loading .env / models / databases.
        source = Path(__file__).resolve().parents[1] / "app/query_process/api/query_server.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and node.name in ("run_query_graph", "query")]
        for node in definitions:
            node.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
        exec(compile(module, str(source), "exec"), self.namespace)
        self.run = self.namespace["run_query_graph"]
        self.query = self.namespace["query"]

    def request(self, stream=False):
        return SimpleNamespace(query="new question", session_id="s", is_stream=stream)

    def test_new_run_clears_previous_results_without_deleting_history(self):
        def invoke(state):
            self.assertNotIn("s", self.results)
            self.results["s"] = {"answer": "new", "sources": ["new"]}
        self.graph.invoke.side_effect = invoke
        self.assertTrue(self.run("q", "s", False))
        self.assertEqual(self.statuses["s"], "completed")
        self.assertEqual(self.history, ["existing MongoDB history"])
        self.assertFalse(self.registry.is_active("s"))

    def test_graph_failure_discards_old_and_partial_results(self):
        def invoke(state):
            self.results["s"] = {"answer": "partial", "sources": ["partial"]}
            raise RuntimeError("upstream failed")
        self.graph.invoke.side_effect = invoke
        self.assertFalse(self.run("q", "s", False))
        self.assertNotIn("s", self.results)
        self.assertEqual(self.statuses["s"], "failed")
        self.assertFalse(self.registry.is_active("s"))

    def test_synchronous_failure_raises_instead_of_returning_stale_success(self):
        self.graph.invoke.side_effect = RuntimeError("failed")
        with self.assertRaises(FakeHTTPException) as caught:
            asyncio.run(self.query(self.request(), Mock()))
        self.assertEqual(caught.exception.status_code, 500)
        self.assertNotIn("s", self.results)
        self.assertFalse(self.registry.is_active("s"))

    def test_running_session_reentry_rejected_without_clearing_active_results(self):
        self.registry.acquire("s")
        with self.assertRaises(FakeHTTPException) as caught:
            asyncio.run(self.query(self.request(), Mock()))
        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(self.results["s"]["answer"], "old")
        self.assertTrue(self.registry.is_active("s"))
        with self.assertRaises(QueryInProgressError):
            self.run("q", "s", False)
        self.assertTrue(self.registry.is_active("s"))

    def test_synchronous_success_reads_only_new_results_while_owned(self):
        self.graph.invoke.side_effect = lambda state: self.results.update({
            "s": {"answer": "new", "sources": [{"chunk_id": 0}], "rerank_degraded": False}
        })
        response = asyncio.run(self.query(self.request(), Mock()))
        self.assertEqual(response["answer"], "new")
        self.assertEqual(response["sources"], [{"chunk_id": 0}])
        self.assertFalse(response["rerank_degraded"])
        self.assertFalse(self.registry.is_active("s"))

    def test_streaming_run_claims_before_scheduling_and_releases_after_failure(self):
        background = Mock()
        response = asyncio.run(self.query(self.request(True), background))
        self.assertEqual(response["session_id"], "s")
        self.assertNotIn("s", self.results)
        self.assertTrue(self.registry.is_active("s"))
        self.assertEqual(self.statuses["s"], "pending")
        callback, *args = background.add_task.call_args.args
        kwargs = background.add_task.call_args.kwargs
        self.graph.invoke.side_effect = RuntimeError("contains-sensitive-provider-details")
        self.assertFalse(callback(*args, **kwargs))
        self.assertFalse(self.registry.is_active("s"))
        self.assertEqual(self.events[-1][0], "error")
        self.assertNotIn("sensitive", self.events[-1][1]["error"])

    def test_threaded_ownership_has_exactly_one_winner(self):
        with ThreadPoolExecutor(max_workers=16) as executor:
            claims = list(executor.map(lambda _: self.registry.acquire("s"), range(64)))
        self.assertEqual(sum(claims), 1)
        self.registry.release("s")
        self.assertTrue(self.registry.acquire("s"))


if __name__ == "__main__":
    unittest.main()
