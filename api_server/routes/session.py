from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api_server.session_store import get_store
from common.schema.labels import SourceLabel

router = APIRouter()


def _serialize_state(state) -> dict:
    slots = {}
    for name, slot in (state.get("slots") or {}).items():
        slots[name] = {
            "value": slot.get("value"),
            "source_label": SourceLabel(slot.get("source_label", SourceLabel.EMPTY)).value,
            "status": slot.get("status", "empty"),
        }
    return {
        "session_id": state.get("session_id"),
        "turn": state.get("turn", 0),
        "slots": slots,
        "pending_question": state.get("pending_question", ""),
        "output_request": state.get("output_request"),
        "correction_count": len(state.get("correction_log") or []),
    }


@router.post("/session")
def create_session() -> dict:
    sid, state = get_store().create()
    return {"session_id": sid, "state": _serialize_state(state)}


@router.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    state = get_store().get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    return _serialize_state(state)
