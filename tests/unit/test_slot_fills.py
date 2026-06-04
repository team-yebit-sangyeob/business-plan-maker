"""extract_slot_fills 충분성·충돌 게이트 + 근거 태그 테스트.

- 충분(adequate)한 결정은 빈 슬롯에 주입, 공허(inadequate)한 결정은 안 채우고 비워둠(다음 턴 되묻기).
- 이미 찬 슬롯에 다른 값(정정 마커 없음) → replace 확인 큐(덮어쓰기는 확인 후).
- dispatch된 claim 세그먼트에 근거→슬롯 연결용 target_slot 태그.
- conversation: replace 확인 intent가 기존 값(previous)을 싣는다.
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정

from common.schema.labels import SourceLabel
from common.schema.state import initial_state
from agents.orchestrator.nodes.correction import (
    FillItem,
    FillOut,
    extract_slot_fills_node,
)
from agents.conversation.agent import _build_intents


def _claim_seg(text):
    return {
        "text": text,
        "canonical_text": text,
        "utterance_types": ["claim"],
        "in_scope": True,
        "target_slot": None,
        "routes": ["research", "rag", "logic_validator"],
    }


def _state(segs, filled=None, last_asked=None):
    st = initial_state()
    st["turn"] = 3
    st["turn_segments"] = segs
    st["last_asked_slot"] = last_asked
    for slot, val in (filled or {}).items():
        st["slots"][slot] = {"value": val, "source_label": SourceLabel.USER, "status": "filled"}
    return st


def _run_fill(state, fills, monkeypatch):
    async def fake(system, user, schema, **kw):
        return FillOut(fills=fills)

    monkeypatch.setattr("agents.orchestrator.nodes.correction.call_json", fake)
    return asyncio.run(extract_slot_fills_node(state))


# ---- 충분성 게이트 ----------------------------------------------------------

def test_adequate_decision_fills_empty_slot(monkeypatch):
    out = _run_fill(
        _state([_claim_seg("6개월 내 월 1500만, 미달 시 재검토")]),
        [FillItem(slot="goal", value="6개월 내 월 1500만, 미달 시 재검토",
                  kind="decision", confidence="clear", adequate=True)],
        monkeypatch,
    )
    assert out["slots"]["goal"]["value"] == "6개월 내 월 1500만, 미달 시 재검토"
    assert out["slots"]["goal"]["status"] == "filled"


def test_inadequate_decision_is_not_filled(monkeypatch):
    # "골은 결과물" — goal 기준 미달이라 박지 않고 비워둔다(다음 턴 되묻기).
    out = _run_fill(
        _state([_claim_seg("목표는 결과물이다")]),
        [FillItem(slot="goal", value="결과물", kind="decision", confidence="clear", adequate=False)],
        monkeypatch,
    )
    assert out["slots"]["goal"]["value"] is None      # 공허값 안 박음
    assert out["pending_confirmations"] == []          # 확인 큐로도 안 보냄
    assert out["turn_segments"][0]["target_slot"] == "goal"  # 근거 태그는 남김


def test_inadequate_value_is_not_queued_to_commit(monkeypatch):
    # "명시되지 않음 —..." 같은 비-답: exploration이어도 commit 큐로 안 보낸다(헛질문 방지).
    # 리포트 회귀: 이 값이 commit 확인을 거쳐 problem에 filled로 박히던 버그.
    out = _run_fill(
        _state([_claim_seg("커피 사업 하고 싶어")]),
        [FillItem(slot="problem", value="명시되지 않음 — 고객 고통이 제공되지 않음",
                  kind="exploration", confidence="clear", adequate=False)],
        monkeypatch,
    )
    assert out["slots"]["problem"]["value"] is None       # 비워둠
    assert out["pending_confirmations"] == []             # commit 큐로도 안 감
    assert out["turn_segments"][0]["target_slot"] == "problem"  # 근거 태그만


def test_exploration_queues_commit(monkeypatch):
    out = _run_fill(
        _state([_claim_seg("일본 시장도 괜찮으려나")]),
        [FillItem(slot="market", value="일본 시장", kind="exploration", confidence="clear")],
        monkeypatch,
    )
    pend = out["pending_confirmations"]
    assert pend and pend[0]["confirm_kind"] == "commit" and pend[0]["proposed_slot"] == "market"


# ---- 충돌/교체 게이트 -------------------------------------------------------

def test_filled_slot_conflict_queues_replace(monkeypatch):
    # target이 이미 찬 상태에서 다른 값(정정 마커 없음) → replace 확인 큐, 덮어쓰지 않음.
    out = _run_fill(
        _state([_claim_seg("타깃은 웹툰 작가다")], filled={"target": "플랫폼 사업자"}),
        [FillItem(slot="target", value="웹툰 작가", kind="decision", confidence="clear", adequate=True)],
        monkeypatch,
    )
    assert out["slots"]["target"]["value"] == "플랫폼 사업자"  # 아직 안 바꿈
    pend = out["pending_confirmations"]
    assert pend and pend[0]["confirm_kind"] == "replace"
    assert pend[0]["proposed_slot"] == "target"
    assert pend[0]["value"] == "웹툰 작가"
    assert pend[0]["previous_value"] == "플랫폼 사업자"


def test_filled_slot_same_value_no_queue(monkeypatch):
    out = _run_fill(
        _state([_claim_seg("타깃은 플랫폼 사업자")], filled={"target": "플랫폼 사업자"}),
        [FillItem(slot="target", value="플랫폼 사업자", kind="decision", confidence="clear", adequate=True)],
        monkeypatch,
    )
    assert out["pending_confirmations"] == []  # 같은 값 → 교체 확인 안 함


# ---- 근거 태그 (Part 1) -----------------------------------------------------

def test_evidence_tag_on_dispatched_claim(monkeypatch):
    segs = [_claim_seg("국내 웹툰 시장은 1.8조 규모다")]
    out = _run_fill(
        _state(segs),
        [FillItem(slot="market", value="국내 웹툰 시장 1.8조", kind="exploration", confidence="clear")],
        monkeypatch,
    )
    assert out["turn_segments"][0]["target_slot"] == "market"  # segment 힌트 대체 — 근거 백필용


# ---- conversation: replace 확인 intent --------------------------------------

def test_confirm_slot_intent_carries_previous_for_replace():
    st = initial_state()
    st["turn"] = 4
    st["pending_confirmations"] = [{
        "value": "웹툰 작가",
        "proposed_slot": "target",
        "candidate_slots": ["target"],
        "source_text": "웹툰 작가",
        "reason": "",
        "attempts": 0,
        "confirm_kind": "replace",
        "previous_value": "플랫폼 사업자",
    }]
    cs = next(i for i in _build_intents(st) if i["type"] == "confirm_slot")
    assert cs["confirm_kind"] == "replace"
    assert cs["previous"] == "플랫폼 사업자"
    assert cs["value"] == "웹툰 작가"
