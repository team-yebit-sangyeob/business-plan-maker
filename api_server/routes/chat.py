"""POST /chat — SSE 스트리밍. Orchestrator 그래프 한 턴 실행 후 이벤트들을 차례로 emit."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated, AsyncIterator, TypedDict

from fastapi import APIRouter, Body, HTTPException
from sse_starlette.sse import EventSourceResponse

from common.schema.labels import SourceLabel
from agents.orchestrator import run_turn
from agents.orchestrator.progress import set_emitter
from api_server.session_store import get_store

router = APIRouter()

# 흐름을 폴백/예외로 강등하는 지점은 서버 로그에 warning을 남긴다(결정론 변환은 로그 없음).
logger = logging.getLogger(__name__)


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

    # 실시간 진행 이벤트 파이프라인.
    # run_turn을 별도 태스크로 돌리고, dispatch가 emit한 에이전트 활동(agent_start →
    # validation_report)을 그래프 실행 '도중에' 큐로 받아 곧바로 흘린다.
    # (과거엔 run_turn이 끝난 뒤 turn_validation_reports를 회고적으로 재생했다.)
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()
    result_box: dict = {}

    async def _driver() -> None:
        # 이 코루틴 컨텍스트에서 emitter 설정 → create_task의 copy_context로
        # run_turn(→dispatch) 태스크에 전파된다.
        set_emitter(queue.put_nowait)
        try:
            result_box["state"] = await run_turn(state, text)
        except Exception as exc:  # SSE error 이벤트로 변환 + 서버 로그에 남김
            logger.warning("run_turn 실패 → SSE error 변환: %s", exc)
            result_box["error"] = exc
        finally:
            queue.put_nowait(sentinel)  # 소비 루프 종료 보장(성공/실패 공통)

    task = asyncio.create_task(_driver())

    # 1) run_turn이 도는 동안 에이전트 활동을 실시간으로 흘린다.
    #    센티넬은 _driver의 finally에서(모든 emit·상태 확정 후) 들어오므로
    #    앞선 이벤트는 FIFO로 모두 소비된 뒤 루프가 끝난다(트레일링 유실 없음).
    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield {"event": "message", "data": json.dumps(item)}

    await task  # 깔끔히 회수 (_driver가 예외를 삼켰으므로 여기선 안 던짐)

    if "error" in result_box:
        yield {"event": "error", "data": json.dumps({"detail": str(result_box["error"])})}
        return

    new_state = result_box["state"]
    store.update(session_id, new_state)

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
    """한 턴을 실행하고 진행·응답·슬롯 변경을 SSE로 스트리밍한다(세션 없으면 404)."""
    if get_store().get(req["session_id"]) is None:
        raise HTTPException(status_code=404, detail="session not found")
    return EventSourceResponse(_stream(req["session_id"], req["text"]))
