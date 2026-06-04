"""제안 요청 처리 — 실 LLM 통합 테스트.

단위테스트(test_conversation.py)는 call_json을 스텁해 _build_intents 라우팅만 고정한다.
정작 이번 픽스의 알맹이는 프롬프트다 — classify가 "네가 생각하는 문제는 뭐야?"를 실제로
reason으로 찍는지, conversation이 되묻지 않고 모은 근거로 답하는지는 실 LLM으로만 검증된다.

call_json은 mock 모드가 없어 OPENAI_API_KEY가 없으면 RuntimeError다(llm.py). 그래서 키가
있을 때만 돈다 — 기본 `pytest tests/unit`에는 안 잡히고, `pytest tests/integration`로 돌린다.
LLM 응답은 비결정적이라 정확 문자열이 아니라 '유형 라벨'과 '근거 참조 여부'로 느슨하게 단언한다.
"""
import asyncio
import os

import pytest

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)

from common.schema.state import initial_state
from agents.orchestrator.nodes.classify import classify_node
from agents.conversation.agent import conversation_node

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="실 LLM 호출 — OPENAI_API_KEY 필요(call_json은 mock 모드 없음)",
)


def _coffee_context_state():
    """커피로 시작해 problem을 막 물어본 직후 상태 — '네가 생각하는 문제는?'의 현실적 맥락."""
    state = initial_state()
    state["turn"] = 2
    state["last_asked_slot"] = "problem"  # 직전에 problem을 물었다(짧은 답 유혹 ↔ reason 경계)
    state["messages"] = [
        {"role": "user", "content": "커피로 사업 계획 짜고 싶어", "turn": 1},
        {
            "role": "assistant",
            "content": "국내 저가 커피 시장은 점포 포화 상태야. 어떤 문제를 풀려는 거야?",
            "turn": 1,
        },
    ]
    state["session_evidence"] = [
        {
            "subject": "국내 저가 커피 시장",
            "cluster": "research",
            "findings": [
                "저가 커피 프랜차이즈 점포가 5년 새 3배로 급증해 포화 상태",
                "업계는 푸드 강화·해외 진출·프리미엄 전환으로 돌파구를 찾는다",
            ],
            "agreement": "confirms",
            "citations": [],
            "target_slot": None,
            "turn": 1,
        }
    ]
    return state


def test_classify_labels_proposal_request_as_reason():
    """'네가 생각하는 문제는 뭐야?'는 question(새 리서치)이 아니라 reason으로 분류돼야 한다."""
    state = _coffee_context_state()
    state["turn_segments"] = [
        {
            "text": "네가 생각하는 문제는 뭐야?",
            "canonical_text": "네가 생각하는 문제는 뭐야?",
            "utterance_types": [],
            "in_scope": True,
            "target_slot": None,
            "routes": [],
        }
    ]

    out = asyncio.run(classify_node(state))
    seg = out["turn_segments"][0]

    # 핵심: 어시스턴트에게 제안을 청한 발화는 reason으로 잡혀 워커(리서치/RAG)를 안 부른다.
    assert "reason" in seg["utterance_types"], seg
    assert "question" not in seg["utterance_types"], seg
    assert seg["routes"] == ["none"], seg  # reason은 interaction → 디스패치 없음


def test_conversation_proposes_from_evidence_not_bounce_back():
    """reason 발화 + 누적 커피 근거면, 되묻기/이론설명이 아니라 근거에 기반한 답을 낸다."""
    state = _coffee_context_state()
    state["turn_segments"] = [
        {
            "text": "네가 생각하는 문제는 뭐야?",
            "canonical_text": "네가 생각하는 문제는 뭐야?",
            "utterance_types": ["reason"],  # classify가 매긴 결과를 입력으로 고정
            "in_scope": True,
            "target_slot": None,
            "routes": ["none"],
        }
    ]

    out = asyncio.run(conversation_node(state))
    msg = out["pending_question"]

    # 슬롯 질문을 새로 던진 게 아니다(되묻기 차단).
    assert "last_asked_slot" not in out, out
    assert msg and len(msg) > 10, msg
    # 모은 커피 근거를 실제로 끌어와 추론했다 — 도메인 토큰이 응답에 드러난다(패러프레이즈 대비 다중 허용).
    assert any(tok in msg for tok in ("포화", "수익", "커피", "점포", "프랜차이즈")), msg
    # "누가·어떤 상황에서…"식으로 같은 질문을 그대로 되묻고 끝내지 않는다.
    assert msg.strip() not in {"어떤 문제를 풀려는 거야?", "네가 생각하는 문제는 뭐야?"}, msg
