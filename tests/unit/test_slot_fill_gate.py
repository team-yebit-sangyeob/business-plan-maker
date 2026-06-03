"""extract_slot_fills_node 쓰기 게이트 — kind(결정/탐색) × confidence(명확/애매) 세 갈래.

LLM(fill 추출)은 가짜로 바꿔 키 없이 결정론적으로 돈다 — 검증 대상은 게이트 분기다:
  결정+명확 → 즉시 주입 / 결정+애매 → confirm_kind=slot 큐 / 탐색 → confirm_kind=commit 큐.
"""
import asyncio

import agents.orchestrator.nodes.correction as correction
from agents.orchestrator.nodes.correction import FillItem, FillOut, extract_slot_fills_node


def _state(slots=None, last_asked=None):
    return {
        "turn_segments": [{"canonical_text": "테스트 발화", "utterance_types": ["claim"]}],
        "slots": slots or {},
        "messages": [],
        "pending_confirmations": [],
        "last_asked_slot": last_asked,
    }


def _run(monkeypatch, fills, slots=None, last_asked=None):
    async def fake(system, user, schema):
        return FillOut(fills=fills)

    monkeypatch.setattr(correction, "call_json", fake)
    return asyncio.run(extract_slot_fills_node(_state(slots, last_asked)))


def test_decision_clear_fills_immediately(monkeypatch):
    out = _run(monkeypatch, [FillItem(slot="target", value="네이버팀", kind="decision", confidence="clear")])
    assert out["slots"]["target"]["value"] == "네이버팀"
    assert out["slots"]["target"]["status"] == "filled"
    assert not out["pending_confirmations"]


def test_decision_ambiguous_queues_slot_confirm(monkeypatch):
    out = _run(
        monkeypatch,
        [FillItem(slot="solution", value="AI 검수 툴", kind="decision",
                  confidence="ambiguous", alt_slots=["advantage"])],
    )
    assert not (out["slots"].get("solution") or {}).get("value")  # 주입 안 됨
    pend = out["pending_confirmations"]
    assert len(pend) == 1
    assert pend[0]["confirm_kind"] == "slot"
    assert set(pend[0]["candidate_slots"]) == {"solution", "advantage"}


def test_exploration_queues_commit_confirm(monkeypatch):
    out = _run(monkeypatch, [FillItem(slot="market", value="일본 시장", kind="exploration", confidence="clear")])
    assert not (out["slots"].get("market") or {}).get("value")  # 주입 안 됨
    pend = out["pending_confirmations"]
    assert len(pend) == 1
    assert pend[0]["confirm_kind"] == "commit"
    assert pend[0]["candidate_slots"] == ["market"]
    assert pend[0]["value"] == "일본 시장"


def test_exploration_on_filled_slot_skips(monkeypatch):
    filled = {"market": {"value": "이미 있음", "source_label": "user", "status": "filled"}}
    out = _run(monkeypatch, [FillItem(slot="market", value="일본 시장", kind="exploration")], slots=filled)
    assert out["slots"]["market"]["value"] == "이미 있음"  # 안 건드림
    assert not out["pending_confirmations"]  # 빈 슬롯 없으니 확인도 안 함


def test_answer_to_asked_slot_promoted_to_decision(monkeypatch):
    # LLM이 보수적으로 exploration을 줘도, 직전에 물은 슬롯(target)에 대한 답이면 채운다.
    out = _run(
        monkeypatch,
        [FillItem(slot="target", value="네이버 운영팀", kind="exploration", confidence="clear")],
        last_asked="target",
    )
    assert out["slots"]["target"]["value"] == "네이버 운영팀"
    assert not out["pending_confirmations"]  # 확인 질문 없이 바로 채움


def test_commit_confirm_capped_to_one_per_turn(monkeypatch):
    out = _run(
        monkeypatch,
        [
            FillItem(slot="market", value="일본 시장", kind="exploration"),
            FillItem(slot="solution", value="구독 모델", kind="exploration"),
        ],
    )
    pend = [p for p in out["pending_confirmations"] if p.get("confirm_kind") == "commit"]
    assert len(pend) == 1  # 턴당 1건만
