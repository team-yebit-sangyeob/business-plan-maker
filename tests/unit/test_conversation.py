"""conversation_node(사용자 최종 응답) 테스트.

회귀 배경: 전역 reasoning_effort=low가 이 노드의 intent·findings 렌더를 메타 자리표시자
("ACK"·"응답 준비됐습니다" 등)로 뭉갰다. 픽스는 ① 이 노드만 medium 추론으로 분리(call_json에
reasoning_effort 전달), ② 빈/메타 출력이 새면 intent 내용으로 결정론 대체.

call_json은 실제 키가 있으면 OpenAI를 부르므로 항상 스텁한다 — 패치 타깃은 importing 모듈
네임스페이스(agents.conversation.agent.call_json), 스텁 시그니처는 새 reasoning_effort 키워드까지 받는다.
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)

from common.config import conversation_reasoning_effort
from common.schema.state import ALL_SLOTS, initial_state
from agents.conversation.agent import ConversationOut, conversation_node


def _claim_seg(text):
    return {
        "text": text,
        "canonical_text": text,
        "utterance_types": ["claim"],
        "in_scope": True,
        "target_slot": None,
        "routes": ["research", "rag", "logic_validator"],
    }


def _report(subject, findings, *, cluster="research", agreement="confirms"):
    return {
        "subject": subject,
        "cluster": cluster,
        "findings": findings,
        "agreement": agreement,
        "citations": [],
    }


# ---- reasoning_effort 분리: 이 노드는 medium으로 렌더한다 -------------------

def test_conversation_passes_configured_reasoning(monkeypatch):
    captured: dict = {}

    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        captured["reasoning_effort"] = reasoning_effort
        return ConversationOut(message="문제 슬롯부터 채워볼까, 어떤 문제를 풀려는 거야?")

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    out = asyncio.run(conversation_node(initial_state()))

    # 전역 low가 아니라 conversation 전용 노브(기본 medium)가 call_json에 도달해야 한다.
    assert captured["reasoning_effort"] == conversation_reasoning_effort()
    assert out["pending_question"]  # 비어있지 않음


def test_conversation_returns_message_and_last_asked_slot(monkeypatch):
    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(message="어떤 문제를 풀려는 거야?")

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    out = asyncio.run(conversation_node(initial_state()))

    assert out["pending_question"] == "어떤 문제를 풀려는 거야?"
    # 빈 슬롯이 전부면 ask_slot은 ALL_SLOTS 첫 칸(problem)을 묻고 기록한다.
    assert out["last_asked_slot"] == ALL_SLOTS[0]


# ---- 퇴화 출력 가드: 빈/메타가 새면 intent 내용으로 대체 -------------------

def test_conversation_guard_replaces_degenerate_with_findings(monkeypatch):
    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(message="ACK")  # 회귀 증상 그대로

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    state = initial_state()
    state["turn"] = 1
    state["turn_segments"] = [_claim_seg("웹툰 시장 규모")]
    state["turn_validation_reports"] = [
        _report("웹툰 시장 규모", ["국내 웹툰 시장은 2023년 약 1.8조 원 규모로 추정된다"])
    ]
    out = asyncio.run(conversation_node(state))

    # 자리표시자 대신 실제 findings가 본문에 담겨야 한다.
    assert out["pending_question"] != "ACK"
    assert "1.8조" in out["pending_question"]


def test_conversation_reason_fallback_renders_context(monkeypatch):
    # reason_over_context도 퇴화 출력이 새면 누적 근거(context)로 결정론 대체된다.
    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(message="ACK")  # 회귀 증상 그대로

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    state = initial_state()
    state["turn"] = 1
    state["turn_segments"] = [{
        "text": "여기서 문제점 추론해봐",
        "canonical_text": "여기서 문제점 추론해봐",
        "utterance_types": ["reason"],
        "in_scope": True,
        "target_slot": None,
        "routes": ["none"],
    }]
    state["session_evidence"] = [{
        "subject": "국내 커피 시장 수익성",
        "cluster": "research",
        "findings": ["점포 포화로 가맹점 매출 감소"],
        "agreement": "confirms",
        "citations": [],
        "target_slot": None,
        "turn": 1,
    }]
    out = asyncio.run(conversation_node(state))

    assert out["pending_question"] != "ACK"
    assert "가맹점 매출 감소" in out["pending_question"]


def test_conversation_guard_passes_normal_short_reply(monkeypatch):
    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(message="좋아, 그렇게 둘게.")

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    out = asyncio.run(conversation_node(initial_state()))
    # 정상적인 짧은 답은 가드에 걸리지 않는다.
    assert out["pending_question"] == "좋아, 그렇게 둘게."


# ---- 설정 노브 -------------------------------------------------------------

def test_conversation_reasoning_effort_default_and_override(monkeypatch):
    monkeypatch.delenv("BPM_CONVERSATION_REASONING", raising=False)
    assert conversation_reasoning_effort() == "medium"
    monkeypatch.setenv("BPM_CONVERSATION_REASONING", "high")
    assert conversation_reasoning_effort() == "high"
