"""POST /chat — SSE 스트리밍. Orchestrator 그래프 한 턴 실행 후 이벤트들을 차례로 emit."""
from __future__ import annotations

import asyncio
import json
from typing import Annotated, AsyncIterator, TypedDict

from fastapi import APIRouter, Body, HTTPException
from sse_starlette.sse import EventSourceResponse

from agents.orchestrator import run_turn
from api_server.routes.session import _serialize_state
from api_server.session_store import get_store
from common.schema.labels import SourceLabel

router = APIRouter()


class ChatRequest(TypedDict):
    session_id: str
    text: str


def _slot_update_event(slot_name: str, slot: dict) -> dict:
    return {
        "type": "slot_update",
        "slot": slot_name,
        "value": slot.get("value"),
        "source_label": SourceLabel(slot.get("source_label", SourceLabel.EMPTY)).value,
        "status": slot.get("status", "empty"),
    }


async def _stream(session_id: str, text: str) -> AsyncIterator[dict]:
    store = get_store()
    state = store.get(session_id)
    if state is None:
        yield {"event": "error", "data": json.dumps({"detail": "session not found"})}
        return

    prev_slots = {k: dict(v) for k, v in (state.get("slots") or {}).items()}

    # 그래프 실행 (LLM 호출은 stub 단계에서 없음 — 빠르게 끝남)
    new_state = await run_turn(state, text)
    store.update(session_id, new_state)

    # 1) 응답 토큰을 잘게 흘려서 SSE 느낌 살리기 (pending_question을 타이핑처럼)
    question = new_state.get("pending_question") or ""
    for chunk in _chunk_text(question, size=12):
        yield {"event": "message", "data": json.dumps({"type": "token", "text": chunk})}
        await asyncio.sleep(0.03)

    # 2) 검증 리포트들
    new_reports = new_state.get("validation_reports") or []
    prev_count = len(state.get("validation_reports") or [])
    for report in new_reports[prev_count:]:
        yield {
            "event": "message",
            "data": json.dumps({"type": "validation_report", **report}),
        }

    # 3) 슬롯 변경분
    new_slots = new_state.get("slots") or {}
    for name, slot in new_slots.items():
        if prev_slots.get(name, {}).get("value") != slot.get("value"):
            yield {
                "event": "message",
                "data": json.dumps(_slot_update_event(name, slot)),
            }

    # 4) 끝
    yield {
        "event": "message",
        "data": json.dumps(
            {
                "type": "done",
                "next_question": question,
                "output_request": new_state.get("output_request"),
            }
        ),
    }


def _chunk_text(text: str, size: int = 16):
    if not text:
        return
    for i in range(0, len(text), size):
        yield text[i : i + size]


@router.post("/chat")
async def chat(req: Annotated[ChatRequest, Body()]):
    if get_store().get(req["session_id"]) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return EventSourceResponse(_stream(req["session_id"], req["text"]))
