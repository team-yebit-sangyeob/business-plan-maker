"""인메모리 세션 저장소 (dev용). 영속화는 비범위."""
from __future__ import annotations

import uuid
from threading import Lock

from common.schema import PlanState, initial_state


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, PlanState] = {}
        self._pdfs: dict[str, dict] = {}  # plan_id → 메타 (PDF 카드용)
        self._lock = Lock()

    def create(self) -> tuple[str, PlanState]:
        session_id = uuid.uuid4().hex[:12]
        state = initial_state()
        state["session_id"] = session_id
        with self._lock:
            self._sessions[session_id] = state
        return session_id, state

    def get(self, session_id: str) -> PlanState | None:
        return self._sessions.get(session_id)

    def update(self, session_id: str, state: PlanState) -> None:
        with self._lock:
            self._sessions[session_id] = state

    def save_pdf(self, plan_id: str, meta: dict) -> None:
        with self._lock:
            self._pdfs[plan_id] = meta

    def get_pdf(self, plan_id: str) -> dict | None:
        return self._pdfs.get(plan_id)


_STORE = SessionStore()


def get_store() -> SessionStore:
    return _STORE
