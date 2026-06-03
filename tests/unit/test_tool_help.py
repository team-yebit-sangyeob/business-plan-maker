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
from agents.conversation.agent import _build_intents, _match_help_slot


# ---- 순수: derive_routes + tool_help_text + 슬롯 매칭 -----------------------

def test_tool_help_routes_to_none():
    assert derive_routes(["tool_help"]) == ["none"]


def test_tool_help_text_uses_slot_specs():
    sol = tool_help_text("solution")
    assert SLOT_SPECS["solution"]["definition"] in sol
    assert SLOT_SPECS["solution"]["boundary"] in sol
    # 슬롯 없으면 도구 전체 설명 + 슬롯 목록, 잘못된 슬롯도 같은 overview로 폴백
    overview = tool_help_text(None)
    assert SLOT_SPECS["solution"]["title"] in overview
    assert tool_help_text("nope") == overview


def test_match_help_slot():
    assert _match_help_slot("솔루션 슬롯이 뭐하는 칸이야?") == "solution"
    assert _match_help_slot("넌 뭐 할 수 있어?") is None
    assert _match_help_slot("문제랑 타겟 슬롯 차이?") is None  # 둘 이상 매칭 → None(overview)


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


def test_explain_tool_for_slot_question():
    intents = _build_intents(_state(_seg("솔루션 슬롯이 뭐하는 칸이야?", ["tool_help"])))
    types = [i["type"] for i in intents]
    et = next(i for i in intents if i["type"] == "explain_tool")
    assert et["slot"] == "solution"
    assert SLOT_SPECS["solution"]["definition"] in et["body"]
    assert "ask_slot" not in types  # 도구 답이 곧 응답 — 다음 슬롯 질문 보류


def test_explain_tool_generic_overview():
    intents = _build_intents(_state(_seg("넌 뭐 할 수 있어?", ["tool_help"])))
    et = next(i for i in intents if i["type"] == "explain_tool")
    assert et["slot"] is None
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
