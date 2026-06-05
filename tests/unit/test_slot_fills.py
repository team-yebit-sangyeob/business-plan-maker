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


# ---- 알맹이 모자란 빈 슬롯 → incomplete_fills로 되묻기(A) ----------------------

def test_inadequate_empty_slot_records_incomplete_fill(monkeypatch):
    # "goal에 10% 해소"처럼 수치는 있지만 기한·실패선이 빠진 부분값 → 안 채우되 조용히 버리지 않고
    # incomplete_fills에 남긴다(다음 응답에서 그 슬롯을 우선 되묻고 받은 부분은 인정하게).
    out = _run_fill(
        _state([_claim_seg("Goal 슬롯에 선정성 불일치 10% 해소를 목표로")]),
        [FillItem(slot="goal", value="선정성 불일치 10% 해소",
                  kind="decision", confidence="clear", adequate=False)],
        monkeypatch,
    )
    assert out["slots"]["goal"]["value"] is None             # 여전히 안 채움
    assert out["pending_confirmations"] == []                # 확인 큐도 아님
    inc = out["incomplete_fills"]
    assert inc == [{"slot": "goal", "partial_value": "선정성 불일치 10% 해소"}]


def test_inadequate_filled_slot_does_not_record_incomplete(monkeypatch):
    # 이미 찬 슬롯에 알맹이 모자란 값 → 다시 캐묻지 않는다(빈 슬롯만 incomplete 대상).
    out = _run_fill(
        _state([_claim_seg("목표는 잘 됐으면")], filled={"goal": "6개월 내 월 1500만, 미달 시 재검토"}),
        [FillItem(slot="goal", value="잘 됐으면", kind="exploration", confidence="clear", adequate=False)],
        monkeypatch,
    )
    assert out["incomplete_fills"] == []
    assert out["slots"]["goal"]["value"] == "6개월 내 월 1500만, 미달 시 재검토"  # 기존 값 보존


def test_non_answer_placeholder_not_recorded_as_incomplete(monkeypatch):
    # "명시되지 않음 —..." 류 비-답은 부분값이 아니라 부재 서술 → 되묻기 신호에서도 거른다
    # (안 그러면 "'명시되지 않음'은 받았어 —"라는 헛 인정이 나간다).
    out = _run_fill(
        _state([_claim_seg("커피 사업 하고 싶어")]),
        [FillItem(slot="problem", value="명시되지 않음 — 고객 고통이 제공되지 않음",
                  kind="exploration", confidence="clear", adequate=False)],
        monkeypatch,
    )
    assert out["incomplete_fills"] == []
    assert out["slots"]["problem"]["value"] is None


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


def test_exploration_fills_empty_slot(monkeypatch):
    # 적극 채움: 떠보는 말(exploration)이어도 빈 슬롯이면 즉시 채운다(가역 — 틀리면 정정).
    # (이전: commit 확인 큐로 보냈다가 미응답이면 드롭 → "너무 안 채워줌" 마찰의 원인.)
    out = _run_fill(
        _state([_claim_seg("일본 시장도 괜찮으려나")]),
        [FillItem(slot="market", value="일본 시장", kind="exploration", confidence="clear")],
        monkeypatch,
    )
    assert out["slots"]["market"]["value"] == "일본 시장"
    assert out["pending_confirmations"] == []   # 확인 큐로 안 보냄 — 바로 채움


def test_multiple_empty_slots_all_fill_no_cap(monkeypatch):
    # 한 턴에 사실 여럿 → 빈 슬롯을 다 채운다(턴당 1건 캡 제거). 탐색이어도.
    out = _run_fill(
        _state([_claim_seg("타겟은 20대"), _claim_seg("강남에서"), _claim_seg("월 1000만 목표")]),
        [
            FillItem(slot="target", value="20대", kind="exploration", confidence="clear"),
            FillItem(slot="market", value="강남", kind="exploration", confidence="clear"),
            FillItem(slot="goal", value="월 1000만", kind="exploration", confidence="clear"),
        ],
        monkeypatch,
    )
    assert out["slots"]["target"]["value"] == "20대"
    assert out["slots"]["market"]["value"] == "강남"
    assert out["slots"]["goal"]["value"] == "월 1000만"
    assert out["pending_confirmations"] == []


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


# ---- conversation: incomplete 슬롯 우선 ask + partial 인정(A+C) ----------------

def _state_with_incomplete(incomplete, filled):
    st = initial_state()
    st["turn"] = 4
    for slot, val in filled.items():
        st["slots"][slot] = {"value": val, "source_label": SourceLabel.USER, "status": "filled"}
    st["incomplete_fills"] = incomplete
    return st


def test_ask_slot_prefers_incomplete_over_canonical_first_empty():
    # problem·target만 차 있으면 캐논 첫 빈칸은 solution. 하지만 사용자가 이번 턴 goal을 채우려다
    # 알맹이가 모자랐으면(incomplete) solution이 아니라 goal을 먼저 물어야 한다(C).
    st = _state_with_incomplete(
        incomplete=[{"slot": "goal", "partial_value": "선정성 불일치 10% 해소"}],
        filled={"problem": "웹툰 작가 인식 격차", "target": "웹툰 작가"},
    )
    ask = next(i for i in _build_intents(st) if i["type"] == "ask_slot")
    assert ask["slot"] == "goal"                              # solution 아님
    assert ask["partial"] == "선정성 불일치 10% 해소"          # 받은 부분값을 함께 실음(A)


def test_ask_slot_falls_back_to_canonical_when_no_incomplete():
    # incomplete가 없으면 평소대로 캐논 첫 빈칸(solution)을 묻는다.
    st = _state_with_incomplete(incomplete=[], filled={"problem": "x", "target": "y"})
    ask = next(i for i in _build_intents(st) if i["type"] == "ask_slot")
    assert ask["slot"] == "solution"
    assert "partial" not in ask


def test_ask_slot_ignores_incomplete_for_slot_filled_meanwhile():
    # incomplete에 goal이 있어도 그새 goal이 다른 경로로 찼으면 무시하고 캐논 첫 빈칸을 묻는다.
    st = _state_with_incomplete(
        incomplete=[{"slot": "goal", "partial_value": "10% 해소"}],
        filled={"problem": "x", "target": "y", "goal": "6개월 내 10% 해소, 미달 시 재검토"},
    )
    ask = next(i for i in _build_intents(st) if i["type"] == "ask_slot")
    assert ask["slot"] == "solution"


# ---- 통합 회귀: "Goal에 써줘"인데 솔루션을 묻던 버그(A+C 끝까지) --------------

def test_named_goal_inadequate_then_asks_goal_not_solution(monkeypatch):
    # 신고 시나리오 재현: problem·target만 차고 goal·solution은 빈 상태에서 사용자가 goal에
    # 기한·실패선 없는 값을 직접 넣어달라고 한다. 예전엔 goal이 조용히 드롭되고 캐논 첫 빈칸인
    # solution을 물었다. 이제는 extract_fills가 incomplete로 남기고 → conversation이 goal을
    # 우선 묻는다(받은 부분 인정).
    st = _state(
        [_claim_seg("Goal 슬롯에 선정성 불일치 10% 해소를 목표로 작성해줘")],
        filled={"problem": "웹툰 작가 인식 격차", "target": "웹툰 작가"},
    )
    out = _run_fill(
        st,
        [FillItem(slot="goal", value="선정성 불일치 10% 해소",
                  kind="decision", confidence="clear", adequate=False)],
        monkeypatch,
    )
    # extract_fills 출력을 상태에 합쳐 conversation에 넘긴다.
    merged = {**st, **out}
    ask = next(i for i in _build_intents(merged) if i["type"] == "ask_slot")
    assert ask["slot"] == "goal"                              # 솔루션이 아니라 goal
    assert ask["partial"] == "선정성 불일치 10% 해소"
