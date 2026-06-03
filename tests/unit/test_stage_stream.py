"""graph._staged — 노드 시작 직전 stage 이벤트 emit(라벨 있을 때만), emitter 없으면 no-op."""
import asyncio

from agents.orchestrator import graph as g
from agents.orchestrator.progress import set_emitter


async def _noop(state):
    return {"ok": True}


def test_labeled_node_emits_stage():
    events = []
    set_emitter(events.append)
    try:
        result = asyncio.run(g._staged("segment", _noop)({}))
    finally:
        set_emitter(None)
    assert result == {"ok": True}  # 원래 반환 보존
    assert events == [{"type": "stage", "node": "segment", "label": "발화 뜯어보기"}]


def test_unlabeled_node_stays_silent():
    events = []
    set_emitter(events.append)
    try:
        asyncio.run(g._staged("dispatch", _noop)({}))  # dispatch는 라벨 없음
    finally:
        set_emitter(None)
    assert events == []


def test_no_emitter_is_noop():
    set_emitter(None)
    result = asyncio.run(g._staged("conversation", _noop)({}))  # emitter 미설정 — 던지지 않는다
    assert result == {"ok": True}
