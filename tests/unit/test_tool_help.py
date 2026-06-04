"""tool_help(도구/슬롯 메타질문) 처리 + interaction-precedence 가드 테스트.

- classify: tool_help는 워커 라우트 0(["none"]); content와 섞여도 가드가 ["none"]로 덮는다.
- conversation: tool_help 세그먼트 → explain_tool intent(SLOT_SPECS·APP_OVERVIEW), ask_slot 보류.
- state: tool_help_text가 SLOT_SPECS 단일 원천에서 렌더된다.
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정

from common.schema.state import SLOT_SPECS, initial_state, tool_help_text
from agents.orchestrator.nodes.classify import (
    ClassifyItem,
    ClassifyOut,
    classify_node,
    derive_routes,
)
from agents.conversation.agent import _build_intents


# ---- 순수: derive_routes + tool_help_text ----------------------------------

def test_tool_help_routes_to_none():
    assert derive_routes(["tool_help"]) == ["none"]


def test_tool_help_text_bundles_overview_and_every_slot():
    from common.schema.state import ALL_SLOTS, APP_OVERVIEW

    ref = tool_help_text()
    # 참고 자료 한 덩이: 도구 개요 + 슬롯 정의 전부(스코프 판정은 LLM 몫이라 항상 전체를 준다)
    assert APP_OVERVIEW in ref
    for name in ALL_SLOTS:
        assert SLOT_SPECS[name]["title"] in ref
        assert SLOT_SPECS[name]["definition"] in ref
        assert SLOT_SPECS[name]["boundary"] in ref


# ---- conversation: _build_intents ------------------------------------------

def _seg(text, utypes, in_scope=True):
    return {
        "text": text,
        "canonical_text": text,
        "utterance_types": utypes,
        "in_scope": in_scope,
        "target_slot": None,
        "routes": ["none"],
    }


def _state(seg):
    st = initial_state()
    st["turn"] = 1
    st["turn_segments"] = [seg]
    return st


def test_explain_tool_carries_question_and_full_reference():
    # 코드는 질문(subject)과 슬롯 정의 전체(body)만 싣는다 — 어느 슬롯을 얼마나 답할지는 LLM 몫.
    q = "솔루션 슬롯이 뭐하는 칸이야?"
    intents = _build_intents(_state(_seg(q, ["tool_help"])))
    types = [i["type"] for i in intents]
    et = next(i for i in intents if i["type"] == "explain_tool")
    assert et["subject"] == q
    assert "slot" not in et and "scope" not in et  # 코드가 스코프를 정하지 않는다
    assert SLOT_SPECS["solution"]["definition"] in et["body"]
    assert SLOT_SPECS["risks"]["definition"] in et["body"]  # body는 항상 전체 참고 자료
    assert "ask_slot" not in types  # 도구 답이 곧 응답 — 다음 슬롯 질문 보류


def test_explain_tool_all_question_gets_full_reference():
    # "각 슬롯" 질문도 같은 경로 — body에 모든 슬롯의 역할(정의)이 들어가 LLM이 전부 펼친다.
    intents = _build_intents(_state(_seg("각 슬롯의 역할을 설명해줄래?", ["tool_help"])))
    et = next(i for i in intents if i["type"] == "explain_tool")
    assert SLOT_SPECS["problem"]["definition"] in et["body"]
    assert SLOT_SPECS["risks"]["definition"] in et["body"]
    assert "ask_slot" not in [i["type"] for i in intents]


def test_explain_tool_beats_redirect_when_offscope():
    # in_scope=false로 잘못 매겨져도 redirect 대신 explain_tool로 답한다
    intents = _build_intents(_state(_seg("솔루션 슬롯이 뭐야?", ["tool_help"], in_scope=False)))
    types = [i["type"] for i in intents]
    assert "explain_tool" in types
    assert "redirect" not in types


# ---- classify_node: precedence 가드(call_json 스텁) ------------------------

def _run_classify(utterance_types, monkeypatch):
    async def fake_call_json(system, user, schema):
        return ClassifyOut(
            items=[ClassifyItem(canonical_text="x", utterance_types=utterance_types, in_scope=True)]
        )

    monkeypatch.setattr("agents.orchestrator.nodes.classify.call_json", fake_call_json)
    state = {
        "turn_segments": [
            {"text": "x", "canonical_text": "x", "utterance_types": [], "target_slot": None, "routes": []}
        ],
        "messages": [],
    }
    out = asyncio.run(classify_node(state))
    return out["turn_segments"][0]


def test_classify_tool_help_no_workers(monkeypatch):
    seg = _run_classify(["tool_help"], monkeypatch)
    assert seg["routes"] == ["none"]
    assert seg["utterance_types"] == ["tool_help"]


def test_classify_tool_help_plus_claim_guard(monkeypatch):
    # 원래 misfire 회귀: tool_help+claim이면 content를 떨궈 워커가 안 새야 한다
    seg = _run_classify(["tool_help", "claim"], monkeypatch)
    assert seg["routes"] == ["none"]
    assert seg["utterance_types"] == ["tool_help"]


# ---- reason(추론·도출·종합) — 워커 없이 conversation이 누적 근거로 답 ---------

def _evidence(subject, findings, *, cluster="research", agreement="confirms"):
    return {
        "subject": subject,
        "cluster": cluster,
        "findings": findings,
        "agreement": agreement,
        "citations": [],
        "target_slot": None,
        "turn": 1,
    }


def test_classify_reason_no_workers(monkeypatch):
    seg = _run_classify(["reason"], monkeypatch)
    assert seg["routes"] == ["none"]  # 추론 요청은 워커를 안 부른다
    assert seg["utterance_types"] == ["reason"]


def test_classify_reason_plus_claim_guard(monkeypatch):
    # 추론 요청이 claim과 섞여도 interaction-precedence가 content를 떨궈 워커가 안 새야 한다
    # (원래 misfire 재발점: reason을 그대로 리서치 쿼리로 디스패치하던 회귀)
    seg = _run_classify(["reason", "claim"], monkeypatch)
    assert seg["routes"] == ["none"]
    assert seg["utterance_types"] == ["reason"]


def test_reason_over_context_intent_built():
    st = _state(_seg("여기서 문제점 추론해봐", ["reason"]))
    st["session_evidence"] = [
        _evidence("국내 커피 시장 수익성", ["점포 포화로 가맹점 매출 감소", "원두·인건비 상승"])
    ]
    intents = _build_intents(st)
    types = [i["type"] for i in intents]
    ro = next(i for i in intents if i["type"] == "reason_over_context")
    assert ro["subject"] == "여기서 문제점 추론해봐"
    # 누적 근거가 context에 추론 재료로 실려야 한다
    findings = [f for c in ro["context"] for f in c["findings"]]
    assert "원두·인건비 상승" in findings
    assert "ask_slot" not in types  # 추론 답이 곧 응답 — 다음 슬롯 질문 보류


def test_reason_over_context_empty_evidence():
    # 모아둔 근거가 없어도 intent는 뜬다(빈 context) — conversation이 솔직히 답하게 한다.
    st = _state(_seg("여기서 문제점 추론해봐", ["reason"]))
    st["session_evidence"] = []
    intents = _build_intents(st)
    types = [i["type"] for i in intents]
    ro = next(i for i in intents if i["type"] == "reason_over_context")
    assert ro["context"] == []
    assert "ask_slot" not in types


def test_reason_beats_redirect_when_offscope():
    # in_scope=false로 잘못 매겨져도 redirect 대신 추론으로 답한다(explain_tool과 같은 방어)
    intents = _build_intents(_state(_seg("여기서 문제점 추론해봐", ["reason"], in_scope=False)))
    types = [i["type"] for i in intents]
    assert "reason_over_context" in types
    assert "redirect" not in types
