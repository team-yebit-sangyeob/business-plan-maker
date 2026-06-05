"""confirm_resolve 단일권위 + graph 분기 테스트.

- 명령형 긍정("이대로 넣어")이 accept로 주입되고 confirmation_consumed=True를 낸다(리포트 버그 회귀).
- commit kind에서 accept가 값을 버리지 않는다(기존 unclear→drop 버그 수정).
- pick/reject/revise/unrelated 태도별 거동, replace 교체 확인, 추가내용 fall-through.
- _post_confirm_branch: consumed면 conversation, 아니면 segment.
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정

from common.schema.labels import SourceLabel
from common.schema.state import initial_state
from agents.orchestrator.nodes.confirm import ConfirmOut, confirm_resolve_node
from agents.orchestrator.graph import _post_confirm_branch


def _pending(value="V", proposed="problem", candidates=None,
             confirm_kind="commit", attempts=0, previous_value="", adequate=True,
             candidate_values=None):
    return {
        "value": value,
        "proposed_slot": proposed,
        "candidate_slots": candidates or [proposed],
        "candidate_values": candidate_values or [],
        "source_text": value,
        "reason": "",
        "attempts": attempts,
        "confirm_kind": confirm_kind,
        "previous_value": previous_value,
        "adequate": adequate,
    }


def _state(pending, user_input="이대로 넣어", filled=None):
    st = initial_state()
    st["turn"] = 2
    st["user_input"] = user_input
    st["pending_confirmations"] = list(pending)
    for slot, val in (filled or {}).items():
        st["slots"][slot] = {"value": val, "source_label": SourceLabel.USER, "status": "filled"}
    return st


def _resolve(state, decision, monkeypatch, *, slot=None, additional=False, chosen_index=None):
    async def fake(system, user, schema, **kw):
        return ConfirmOut(decision=decision, slot=slot, chosen_index=chosen_index,
                          has_additional_content=additional)

    monkeypatch.setattr("agents.orchestrator.nodes.confirm.call_json", fake)
    return asyncio.run(confirm_resolve_node(state))


# ---- accept(명령형 긍정) — 리포트 버그 직접 회귀 ----------------------------

def test_accept_imperative_fills_and_consumes(monkeypatch):
    out = _resolve(_state([_pending(value="웹툰 선정성이 문제다", proposed="problem")]),
                   "accept", monkeypatch)
    assert out["slots"]["problem"]["value"] == "웹툰 선정성이 문제다"
    assert out["slots"]["problem"]["status"] == "filled"
    assert out["slots"]["problem"]["source_label"] == SourceLabel.USER
    assert out["pending_confirmations"] == []
    assert out["confirmation_consumed"] is True  # 순수 확인 답 → 파이프라인 우회


def test_accept_commit_kind_does_not_drop_value(monkeypatch):
    # 기존 unclear→commit→drop 버그: 명령형 긍정이 commit kind에서 값을 잃지 않아야 한다.
    out = _resolve(_state([_pending(value="V", proposed="problem", confirm_kind="commit")]),
                   "accept", monkeypatch)
    assert out["slots"]["problem"]["value"] == "V"


def test_accept_inadequate_value_does_not_fill(monkeypatch):
    # 코드 백스톱: 공허값(adequate=False)은 사용자가 수락해도 슬롯에 안 박힌다.
    # 큐는 해소(소비)하되 슬롯은 비워둔다 — 다음 턴 ask_slot이 되묻는다.
    out = _resolve(
        _state([_pending(value="명시되지 않음 — 고객 고통 미제공",
                         proposed="problem", confirm_kind="commit", adequate=False)]),
        "accept", monkeypatch,
    )
    assert out["slots"]["problem"]["value"] is None
    assert out["slots"]["problem"]["status"] == "empty"
    assert out["pending_confirmations"] == []
    assert out["confirmation_consumed"] is True


def test_accept_with_additional_content_not_consumed(monkeypatch):
    # "응 넣고, 타겟은 20대야" — 채우되 소비 안 함(잔여 내용은 파이프라인이 처리).
    out = _resolve(_state([_pending(value="V", proposed="problem")]),
                   "accept", monkeypatch, additional=True)
    assert out["slots"]["problem"]["value"] == "V"
    assert out["confirmation_consumed"] is False


# ---- pick / reject -----------------------------------------------------------

def test_pick_named_candidate(monkeypatch):
    out = _resolve(
        _state([_pending(value="AI 자동검수", proposed="solution",
                         candidates=["solution", "advantage"], confirm_kind="slot")]),
        "pick", monkeypatch, slot="advantage",
    )
    assert out["slots"]["advantage"]["value"] == "AI 자동검수"
    assert out["slots"]["solution"]["value"] is None
    assert out["confirmation_consumed"] is True


def test_reject_clears_queue_without_write(monkeypatch):
    out = _resolve(_state([_pending(value="V", proposed="problem")]), "reject", monkeypatch)
    assert "slots" not in out  # 슬롯 미변경(부분 dict 반환)
    assert out["pending_confirmations"] == []
    assert out["confirmation_consumed"] is True


# ---- 다중 후보 값 pick (reason 제안 모드② — 한 슬롯에 여러 안 제시) ----------

def _multi(confirm_kind="commit", proposed="solution", previous_value=""):
    # 추천안(1번) + 두 대안. chosen_index로 사용자가 고른 안을 정한다.
    return _pending(
        value="로스팅 D2C+구독", proposed=proposed, confirm_kind=confirm_kind,
        previous_value=previous_value,
        candidate_values=["로스팅 D2C+구독", "로스터리 카페+로컬", "B2B 집중형"],
    )


def test_pick_chosen_value_fills_that_option(monkeypatch):
    # 빈 슬롯에 세 안 제시 → 사용자가 3번(B2B 집중형) 고름 → 그 값으로 채운다(추천안 아님).
    out = _resolve(_state([_multi(confirm_kind="commit")]),
                   "pick", monkeypatch, chosen_index=3)
    assert out["slots"]["solution"]["value"] == "B2B 집중형"
    assert out["confirmation_consumed"] is True


def test_accept_without_index_uses_recommended(monkeypatch):
    # 특정 안을 콕 집지 않고 추천안 그대로 수락 → 1번(추천/value)으로 채운다.
    out = _resolve(_state([_multi(confirm_kind="commit")]), "accept", monkeypatch)
    assert out["slots"]["solution"]["value"] == "로스팅 D2C+구독"


def test_pick_chosen_value_replace_overwrites_and_logs(monkeypatch):
    # 이미 찬 solution에 세 안 제시 → 2번 고름 → 덮어쓰기 + correction_log 기록.
    out = _resolve(
        _state([_multi(confirm_kind="replace", previous_value="기존 카페 모델")],
               filled={"solution": "기존 카페 모델"}),
        "pick", monkeypatch, chosen_index=2,
    )
    assert out["slots"]["solution"]["value"] == "로스터리 카페+로컬"
    assert any(c["slot"] == "solution" and c["previous"] == "기존 카페 모델"
               and c["new"] == "로스터리 카페+로컬" for c in out["correction_log"])


def test_pick_out_of_range_index_falls_back_to_recommended(monkeypatch):
    # 범위 밖 인덱스(방어) → 추천안으로 안전 폴백.
    out = _resolve(_state([_multi(confirm_kind="commit")]),
                   "pick", monkeypatch, chosen_index=9)
    assert out["slots"]["solution"]["value"] == "로스팅 D2C+구독"


# ---- revise / unrelated (fall-through, 소비 안 함) ---------------------------

def test_revise_drops_stale_and_falls_through(monkeypatch):
    out = _resolve(_state([_pending(value="V", proposed="problem")]), "revise", monkeypatch)
    assert "slots" not in out                              # 스테일 값 안 박음
    assert out["pending_confirmations"] == []              # 큐에서 뺌
    assert not out.get("confirmation_consumed")            # fall through → segment 이하 처리


def test_unrelated_commit_drops_and_falls_through(monkeypatch):
    out = _resolve(_state([_pending(value="V", proposed="problem", confirm_kind="commit")]),
                   "unrelated", monkeypatch)
    assert out["pending_confirmations"] == []
    assert not out.get("confirmation_consumed")


def test_unrelated_slot_increments_attempts(monkeypatch):
    out = _resolve(
        _state([_pending(value="V", proposed="solution",
                         candidates=["solution", "advantage"], confirm_kind="slot", attempts=0)]),
        "unrelated", monkeypatch,
    )
    assert out["pending_confirmations"][0]["attempts"] == 1  # 유지·재질문
    assert "slots" not in out                                # 아직 자동 확정 안 함
    assert not out.get("confirmation_consumed")


def test_unrelated_slot_autocommit_at_limit(monkeypatch):
    out = _resolve(
        _state([_pending(value="V", proposed="solution",
                         candidates=["solution", "advantage"], confirm_kind="slot", attempts=1)]),
        "unrelated", monkeypatch,
    )
    assert out["slots"]["solution"]["value"] == "V"  # 한도(2) 도달 → 제안 슬롯 자동 확정
    assert out["pending_confirmations"] == []


# ---- replace(교체 확인) -----------------------------------------------------

def test_replace_accept_overwrites_and_logs(monkeypatch):
    st = _state(
        [_pending(value="웹툰 작가", proposed="target", confirm_kind="replace",
                  previous_value="플랫폼 사업자")],
        filled={"target": "플랫폼 사업자"},
    )
    out = _resolve(st, "accept", monkeypatch)
    assert out["slots"]["target"]["value"] == "웹툰 작가"          # 덮어씀
    log = out["correction_log"]
    assert log and log[-1]["slot"] == "target"
    assert log[-1]["previous"] == "플랫폼 사업자" and log[-1]["new"] == "웹툰 작가"
    assert out["confirmation_consumed"] is True


def test_replace_reject_keeps_existing(monkeypatch):
    st = _state(
        [_pending(value="웹툰 작가", proposed="target", confirm_kind="replace",
                  previous_value="플랫폼 사업자")],
        filled={"target": "플랫폼 사업자"},
    )
    out = _resolve(st, "reject", monkeypatch)
    assert "slots" not in out                              # 기존 값 그대로(미변경)
    assert out["pending_confirmations"] == []


def test_replace_unrelated_drops_keeps_existing(monkeypatch):
    st = _state(
        [_pending(value="웹툰 작가", proposed="target", confirm_kind="replace",
                  previous_value="플랫폼 사업자")],
        filled={"target": "플랫폼 사업자"},
    )
    out = _resolve(st, "unrelated", monkeypatch)
    assert "slots" not in out                  # 기존 값 그대로(미변경)
    assert out["pending_confirmations"] == []  # replace는 미응답이면 드롭(교체 안 함)


# ---- no-op / 다중 pending ----------------------------------------------------

def test_no_pending_is_noop(monkeypatch):
    st = initial_state()
    st["user_input"] = "아무거나"
    # call_json은 호출되면 안 된다(pending 없음 → 즉시 반환).
    out = asyncio.run(confirm_resolve_node(st))
    assert out == {}
    assert "confirmation_consumed" not in out  # 일반 턴 무영향


def test_multiple_pending_resolves_first_only(monkeypatch):
    second = _pending(value="W", proposed="goal", confirm_kind="commit")
    out = _resolve(_state([_pending(value="V", proposed="problem"), second]),
                   "accept", monkeypatch)
    assert out["slots"]["problem"]["value"] == "V"
    assert out["pending_confirmations"] == [second]  # 둘째는 그대로 남는다


# ---- graph 분기 (_post_confirm_branch) --------------------------------------

def test_branch_consumed_goes_conversation():
    assert _post_confirm_branch({"confirmation_consumed": True}) == "conversation"


def test_branch_default_goes_segment():
    assert _post_confirm_branch({}) == "segment"
    assert _post_confirm_branch({"confirmation_consumed": False}) == "segment"
