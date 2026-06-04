"""대화 상태 주입 메커니즘 테스트.

- conversation_state_text: 열린 제안(commit/slot/replace)·직전 질문 슬롯을 [대화 상태]로 렌더.
- slot_snapshot_text: 현재 슬롯 값(빈 칸 포함).
- segment·classify가 그 블록(+classify는 [현재 슬롯])을 프롬프트에 실어 보낸다.
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정

from common.schema.labels import SourceLabel
from common.schema.state import (
    conversation_state_text,
    initial_state,
    slot_snapshot_text,
)
from agents.orchestrator.nodes.segment import SegmentItem, SegmentOut, segment_node
from agents.orchestrator.nodes.classify import ClassifyItem, ClassifyOut, classify_node


def _op(value="웹툰 선정성이 문제다", proposed="problem", candidates=None,
        confirm_kind="commit", previous_value=""):
    return {
        "value": value,
        "proposed_slot": proposed,
        "candidate_slots": candidates or [proposed],
        "source_text": value,
        "reason": "",
        "attempts": 0,
        "confirm_kind": confirm_kind,
        "previous_value": previous_value,
    }


# ---- conversation_state_text (순수) ----------------------------------------

def test_state_text_empty_when_nothing_open():
    assert conversation_state_text(initial_state()) == ""


def test_state_text_commit_proposal():
    st = initial_state()
    st["open_proposal"] = _op(confirm_kind="commit")
    block = conversation_state_text(st)
    assert "[대화 상태]" in block
    assert "넣기 확인 대기" in block
    assert "웹툰 선정성이 문제다" in block
    assert "problem" in block


def test_state_text_slot_proposal_lists_candidates():
    st = initial_state()
    st["open_proposal"] = _op(value="AI 검수", proposed="solution",
                              candidates=["solution", "advantage"], confirm_kind="slot")
    block = conversation_state_text(st)
    assert "슬롯 선택 확인 대기" in block
    assert "solution" in block and "advantage" in block


def test_state_text_replace_shows_previous():
    st = initial_state()
    st["open_proposal"] = _op(value="웹툰 작가", proposed="target",
                              confirm_kind="replace", previous_value="플랫폼 사업자")
    block = conversation_state_text(st)
    assert "교체 확인 대기" in block
    assert "웹툰 작가" in block and "플랫폼 사업자" in block


def test_state_text_includes_last_asked_slot():
    st = initial_state()
    st["last_asked_slot"] = "target"
    block = conversation_state_text(st)
    assert "직전 질문 슬롯" in block and "target" in block


def test_slot_snapshot_shows_values_and_empties():
    st = initial_state()
    st["slots"]["problem"] = {"value": "웹툰 선정성", "source_label": SourceLabel.USER, "status": "filled"}
    snap = slot_snapshot_text(st)
    assert "웹툰 선정성" in snap
    assert "[비어있음]" in snap  # 안 찬 슬롯들


# ---- segment 프롬프트 주입 (call_json 스텁으로 user payload 캡처) ----------

def _capture_segment(state, monkeypatch):
    captured = {}

    async def fake(system, user, schema, **kw):
        captured["user"] = user
        return SegmentOut(segments=[SegmentItem(text="이대로 넣어",
                                                canonical_text="방금 제안 값을 그대로 넣으라고 한다")])

    monkeypatch.setattr("agents.orchestrator.nodes.segment.call_json", fake)
    asyncio.run(segment_node(state))
    return captured["user"]


def test_segment_prompt_carries_state_block(monkeypatch):
    st = initial_state()
    st["user_input"] = "이대로 넣어"
    st["open_proposal"] = _op(confirm_kind="commit")
    user = _capture_segment(st, monkeypatch)
    assert "[대화 상태]" in user
    assert "넣기 확인 대기" in user


def test_segment_no_state_block_on_plain_turn(monkeypatch):
    st = initial_state()
    st["user_input"] = "웹툰 시장 규모 어때?"
    user = _capture_segment(st, monkeypatch)
    assert "[대화 상태]" not in user  # 열린 제안·직전 슬롯 없으면 블록 없음


def test_segment_output_has_no_target_slot(monkeypatch):
    # segment는 더는 슬롯을 고르지 않는다 — target_slot은 None.
    st = initial_state()
    st["user_input"] = "타깃은 웹툰 작가"

    async def fake(system, user, schema, **kw):
        return SegmentOut(segments=[SegmentItem(text="타깃은 웹툰 작가", canonical_text="타깃은 웹툰 작가다")])

    monkeypatch.setattr("agents.orchestrator.nodes.segment.call_json", fake)
    out = asyncio.run(segment_node(st))
    assert out["turn_segments"][0]["target_slot"] is None


# ---- classify 프롬프트 주입 ([대화 상태] + [현재 슬롯]) ---------------------

def test_classify_prompt_carries_state_and_slots(monkeypatch):
    captured = {}

    async def fake(system, user, schema, **kw):
        captured["user"] = user
        return ClassifyOut(items=[ClassifyItem(canonical_text="x", utterance_types=["meta"], in_scope=True)])

    monkeypatch.setattr("agents.orchestrator.nodes.classify.call_json", fake)
    st = initial_state()
    st["open_proposal"] = _op(confirm_kind="commit")
    st["slots"]["target"] = {"value": "플랫폼 사업자", "source_label": SourceLabel.USER, "status": "filled"}
    st["turn_segments"] = [
        {"text": "이대로 넣어", "canonical_text": "이대로 넣어", "utterance_types": [],
         "target_slot": None, "routes": []}
    ]
    asyncio.run(classify_node(st))
    assert "[대화 상태]" in captured["user"]
    assert "[현재 슬롯]" in captured["user"]
    assert "플랫폼 사업자" in captured["user"]  # 충돌 판정용 슬롯 값
