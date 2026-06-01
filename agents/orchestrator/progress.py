"""턴 진행 이벤트(에이전트 활동)를 깊은 호출 스택(dispatch 노드)에서
SSE 스트림(_stream)으로 실시간 전달하기 위한 ContextVar 기반 emitter.

원칙:
- emitter는 ContextVar에만 둔다. PlanState에는 절대 넣지 않는다(상태 직렬화 보존).
- emit은 동기(put_nowait) — 워커 코드가 await할 필요 없고 발행 순서가 보존된다.
- emitter 미설정(단위 테스트·직접 호출) 시 emit은 무해한 no-op.

_stream(api_server/routes/chat.py)이 요청마다 set_emitter(queue.put_nowait)를 호출하고
run_turn을 asyncio.create_task로 띄우면, create_task의 copy_context 덕분에
같은 컨텍스트에서 돈 dispatch의 emit이 그 큐로 들어간다.
"""
from __future__ import annotations

import contextvars
from typing import Callable, Optional

# 설정되면 SSE 큐의 put_nowait. 미설정이면 None → emit은 no-op.
_emitter: contextvars.ContextVar[Optional[Callable[[dict], None]]] = (
    contextvars.ContextVar("bpm_progress_emitter", default=None)
)


def set_emitter(fn: Optional[Callable[[dict], None]]) -> contextvars.Token:
    """현재 컨텍스트에 emitter 설정. 반환 Token으로 reset 가능(보통 불필요)."""
    return _emitter.set(fn)


def reset_emitter(token: contextvars.Token) -> None:
    _emitter.reset(token)


def emit(event: dict) -> None:
    """진행 이벤트 1건 발행. emitter 없으면 조용히 무시."""
    fn = _emitter.get()
    if fn is not None:
        fn(event)
