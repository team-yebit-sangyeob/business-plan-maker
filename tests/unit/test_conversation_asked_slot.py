"""conversation_node가 ask_slot을 실제로 물은 턴에만 last_asked_slot을 기록하는지.

이 값은 fill이 "직전 질문에 직접 답"(kind=decision (b))을 결정론으로 잡는 근거다.
"""
import asyncio

import agents.orchestrator.graph as _graph  # 패키지 선초기화 — 순환 import 회피(앱 진입 순서)  # noqa: F401
import agents.conversation.agent as conv
from agents.conversation.agent import ConversationOut, conversation_node
from common.schema.state import initial_state


def _fake_llm(monkeypatch, message="응답"):
    async def fake(system, user, schema):
        return ConversationOut(message=message)

    monkeypatch.setattr(conv, "call_json", fake)


def test_records_asked_slot_when_asking(monkeypatch):
    _fake_llm(monkeypatch)
    state = initial_state()  # 전부 빈 슬롯 → 첫 빈칸 problem을 ask_slot
    out = asyncio.run(conversation_node(state))
    assert out["last_asked_slot"] == "problem"


def test_omits_asked_slot_when_suppressed(monkeypatch):
    _fake_llm(monkeypatch)
    state = initial_state()
    # 확인 대기(pending)가 있으면 confirm_slot이 ask_slot을 누른다 → 키 미설정(이전 값 유지)
    state["pending_confirmations"] = [
        {"value": "일본 시장", "proposed_slot": "market",
         "candidate_slots": ["market"], "confirm_kind": "commit"}
    ]
    out = asyncio.run(conversation_node(state))
    assert "last_asked_slot" not in out
