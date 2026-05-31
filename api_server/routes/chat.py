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

    # 1) 에이전트 활동 — 실행 중 → 결과 (클로드 도구 사용처럼). 답변 텍스트보다 먼저.
    #    이번 턴 디스패치 결과만(turn_validation_reports). 리서치·RAG → 비평 순서로 보여
    #    실제 2단계 디스패치를 반영한다.
    turn_reports = new_state.get("turn_validation_reports") or []
    _stage = {"research": 0, "rag": 1, "critic": 2}
    ordered = sorted(turn_reports, key=lambda r: _stage.get(r.get("cluster", ""), 3))
    for r in ordered:  # 먼저 전부 '실행 중'으로 띄움
        yield {
            "event": "message",
            "data": json.dumps(
                {"type": "agent_start", "cluster": r.get("cluster"), "subject": r.get("subject", "")}
            ),
        }
        await asyncio.sleep(0.12)
    for r in ordered:  # 결과 카드로 하나씩 해소
        yield {"event": "message", "data": json.dumps({"type": "validation_report", **r})}
        await asyncio.sleep(0.18)

    # 2) 응답 토큰을 잘게 흘려서 SSE 느낌 살리기 (pending_question을 타이핑처럼)
    question = new_state.get("pending_question") or ""
    for chunk in _chunk_text(question, size=12):
        yield {"event": "message", "data": json.dumps({"type": "token", "text": chunk})}
        await asyncio.sleep(0.03)

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
