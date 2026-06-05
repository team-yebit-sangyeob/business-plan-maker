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
from common.schema.labels import SourceLabel
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


def test_reason_request_proposes_not_asks(monkeypatch):
    # "네가 생각하는 문제는 뭐야?"처럼 어시스턴트에게 제안을 청하면(=reason),
    # 모은 근거로 제안하는 reason_over_context가 떠야 하고 ask_slot으로 되묻지 않는다.
    captured: dict = {}

    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        captured["payload"] = user
        return ConversationOut(
            message="모은 내용을 보면 저가 커피 포화로 신규 점포 수익성이 떨어지는 게 문제 같아 — 이렇게 잡아볼까?"
        )

    monkeypatch.setattr("agents.conversation.agent.call_json", fake_call_json)
    state = initial_state()
    state["turn"] = 1
    state["turn_segments"] = [{
        "text": "네가 생각하는 문제는 뭐야?",
        "canonical_text": "네가 생각하는 문제는 뭐야?",
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

    # reason 발화는 누적 근거를 실은 reason_over_context로 처리되고, 그 근거가 LLM payload에 닿는다.
    assert "가맹점 매출 감소" in captured["payload"]
    assert "reason_over_context" in captured["payload"]
    # 되묻기(ask_slot) 차단 — last_asked_slot 키를 내보내지 않는다(슬롯 질문을 안 했으므로).
    assert "last_asked_slot" not in out
    # LLM이 돌려준 제안 문구가 그대로 사용자에게 간다(퇴화 대체 안 탐).
    assert out["pending_question"].endswith("이렇게 잡아볼까?")


# ---- reason 제안 → pending_confirmations 등록(채택→슬롯반영 닫힌 루프의 앞단) -----

def _reason_seg(text="네가 생각하는 문제는 뭐야?"):
    return {
        "text": text, "canonical_text": text, "utterance_types": ["reason"],
        "in_scope": True, "target_slot": None, "routes": ["none"],
    }


def test_reason_proposal_registers_pending_commit(monkeypatch):
    # 빈 슬롯에 어시스턴트가 구체적 값을 제안(모드②) → commit 확인 큐로 등록(다음 턴 수락 대상).
    async def fake(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(
            message="문제는 공급 불안정 같아 — 이렇게 잡아볼까?",
            proposal={"slot": "problem", "value": "납품 품질·공급 불안정으로 운영 손실"},
        )

    monkeypatch.setattr("agents.conversation.agent.call_json", fake)
    state = initial_state()
    state["turn_segments"] = [_reason_seg()]
    out = asyncio.run(conversation_node(state))

    pend = out["pending_confirmations"]
    assert len(pend) == 1
    assert pend[0]["proposed_slot"] == "problem"
    assert pend[0]["confirm_kind"] == "commit"   # 빈 슬롯
    assert pend[0]["value"] == "납품 품질·공급 불안정으로 운영 손실"
    assert pend[0]["candidate_values"] == []      # 단일 안
    assert pend[0]["adequate"] is True


def test_reason_proposal_multi_candidate_replace(monkeypatch):
    # 이미 찬 슬롯에 여러 안 제시 → replace + candidate_values(추천 + 대안들).
    async def fake(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(
            message="세 모델 제안 — 1순위는 D2C. 어느 걸로?",
            proposal={"slot": "solution", "value": "로스팅 D2C+구독",
                      "alternatives": ["로스터리 카페+로컬", "B2B 집중형"]},
        )

    monkeypatch.setattr("agents.conversation.agent.call_json", fake)
    state = initial_state()
    state["slots"]["solution"] = {"value": "기존 카페 모델", "source_label": SourceLabel.USER,
                                  "status": "filled"}
    state["turn_segments"] = [_reason_seg("솔루션 추천해줘")]
    out = asyncio.run(conversation_node(state))

    pend = out["pending_confirmations"]
    assert pend[0]["proposed_slot"] == "solution"
    assert pend[0]["confirm_kind"] == "replace"        # 이미 찬 슬롯
    assert pend[0]["previous_value"] == "기존 카페 모델"
    assert pend[0]["candidate_values"] == ["로스팅 D2C+구독", "로스터리 카페+로컬", "B2B 집중형"]


def test_proposal_ignored_without_reason_intent(monkeypatch):
    # reason 세그먼트가 없는 턴(claim 등)엔 LLM이 proposal을 내도 코드가 등록하지 않는다(가드).
    async def fake(system, user, schema, *, reasoning_effort=None):
        return ConversationOut(message="음, 그건 좀 더 좁혀보자.",
                               proposal={"slot": "problem", "value": "아무거나"})

    monkeypatch.setattr("agents.conversation.agent.call_json", fake)
    state = initial_state()
    state["turn_segments"] = [_claim_seg("커피 시장이 크다")]
    out = asyncio.run(conversation_node(state))
    assert "pending_confirmations" not in out


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
