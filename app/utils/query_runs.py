"""Single-process ownership of query runs; independent of the service clients.

History remains keyed by session_id. This registry only prevents concurrent
requests from sharing the in-memory result and SSE slot for that session.
Multi-process deployments need a shared run store/queue instead.
"""
from threading import Lock


class QueryInProgressError(RuntimeError):
    pass


class SessionRunRegistry:
    def __init__(self):
        self._lock = Lock()
        self._active = set()

    def acquire(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._active:
                return False
            self._active.add(session_id)
            return True

    def release(self, session_id: str) -> None:
        with self._lock:
            self._active.discard(session_id)

    def is_active(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._active
