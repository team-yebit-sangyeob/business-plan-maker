"""정정 처리 + 슬롯 채움 — LLM 기반 (기획서 Case E).

correction_node: utterance_types에 correction 포함 세그먼트만 모아 LLM에 넘김.
  현재 슬롯·correction_log·최근 messages를 보고 "어느 슬롯의 어떤 값을 어떻게"
  바꿀지 판단. action은 replace(교체)/clear(비움)/ignore(모호하면 건너뜀).

  예: 슬롯 solution="B2B 감수 서비스" 상태에서
      "아 그냥 감수 말고 AI 자동 검수 툴로 바꾸자"
      → action: replace solution = "AI 자동 검수 툴"
      → correction_log += {slot:"solution", previous:"B2B 감수 서비스",
                           new:"AI 자동 검수 툴", turn:5}
      (target·goal 등 다른 슬롯은 그대로 — 부분 롤백)

  예: 슬롯 target="네이버·카카오" 상태에서 "카카오는 빼자"
      → action: replace target = "네이버" (또는 맥락상 clear)

extract_slot_fills_node: 정정 이후 단계에서, claim 세그먼트의 값을 슬롯에 처리한다.
  빈 슬롯 채움은 가역(빼/바꿔/correction이 되돌린다)이라 적극적으로 채운다 — 확인을 받느라
  안 채우고 드롭하던 마찰을 없앤다. 확인 큐는 '비가역에 가까운' 경우에만 남긴다:
  - 빈 슬롯 + 슬롯 명확 → 즉시 주입(결정/탐색 무관 — 떠보는 말이어도 빈 칸이면 박는다).
    한 턴에 빈 슬롯이 여럿이면 다 채운다(턴당 캡 없음).
  - 빈 슬롯 + 슬롯 애매(어느 칸인지 불명) → pending_confirmations(confirm_kind="slot")에 쌓아
    다음 턴 confirm_resolve가 "이거 솔루션이에요 차별점이에요?" 답으로 확정(잘못된 칸 방지).
  - 이미 찬 슬롯 + 다른 값(결정, 정정 마커 없음) → confirm_kind="replace"로 "바꿀까?" 확인
    (덮어쓰기는 비가역적 손실이라 확인 후에만). 탐색이면 기존 값은 안 건드린다.
  - 값이 공허(adequate=false) → 어느 경로로도 안 채운다(다음 턴 ask_slot이 되묻는다).
  (confirm_kind="commit"은 이제 conversation의 reason 제안 경로에서만 쓰인다 — agent.py.)

  예: 직전에 "타겟이 누구예요?" 물음 + 세그먼트 "(claim) 네이버 콘텐츠 운영팀"
      → target = "네이버 콘텐츠 운영팀" 즉시 주입.
  예: 세그먼트 "(claim) AI로 자동 검수해주는 거" → solution/advantage 경계 → ambiguous
      → 보류, pending += {value, proposed=solution, candidates=[solution,advantage], confirm_kind=slot}
  예: 세그먼트 "(claim) 일본 시장도 괜찮으려나?" → market 빈 칸이면 즉시 주입(탐색이어도).
      틀리면 "빼"로 되돌리면 된다 — 안 채우고 드롭하던 것보다 마찰이 적다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from common.schema import PlanState, Correction
from common.schema.labels import SourceLabel
from common.schema.state import (
    ALL_SLOTS,
    PendingConfirmation,
    recent_history,
    slot_guide_text,
)
from agents.orchestrator.llm import call_json


# --- correction ------------------------------------------------------------

_CORR_SYSTEM = (
    """오케스트레이터 정정 해소
사용자가 정정 신호를 낸 세그먼트 목록과 현재 슬롯 상태를 보고, 어떤 슬롯을 어떻게 갱신할지 결정한다.

- action="clear": 슬롯을 비운다(특정 값을 빼는 경우).
- action="replace": 슬롯 값을 new_value로 교체한다.
- action="ignore": 슬롯이 어디인지 모호하면 건너뛴다.

slot은 아래 정의와 경계를 따른다:
"""
    + slot_guide_text()
    + """

JSON만 출력."""
)


class CorrectionAction(BaseModel):
    slot: str
    action: str  # "clear" | "replace" | "ignore"
    new_value: Optional[str] = None
    previous_hint: Optional[str] = None


class CorrectionOut(BaseModel):
    actions: list[CorrectionAction] = Field(default_factory=list)


def _slot_snapshot_lines(state: PlanState) -> str:
    slots = state.get("slots") or {}
    return "\n".join(
        f"- {name}: {(slots.get(name) or {}).get('value') or '[비어있음]'}"
        for name in ALL_SLOTS
    )


async def correction_node(state: PlanState) -> dict:
    """정정 세그먼트를 슬롯에 반영한다 → {"slots","correction_log","turn_segments"}(없으면 {})."""
    segments = state.get("turn_segments") or []
    targets = [s for s in segments if "correction" in (s.get("utterance_types") or [])]
    if not targets:
        return {}

    slots = dict(state.get("slots") or {})
    log = list(state.get("correction_log") or [])
    turn = state.get("turn", 0)

    user_payload = (
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        "[정정 세그먼트]\n"
        + "\n".join(
            f"{i+1}. {s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(targets)
        )
    )
    out = await call_json(_CORR_SYSTEM, user_payload, CorrectionOut)

    valid = set(ALL_SLOTS)
    for action in out.actions:
        if action.slot not in valid:
            continue
        existing = slots.get(action.slot) or {}
        previous = existing.get("value")
        if action.action == "clear":
            slots[action.slot] = {
                "value": None,
                "source_label": SourceLabel.EMPTY,
                "status": "empty",
            }
            log.append(Correction(slot=action.slot, previous=previous, new=None, turn=turn))
        elif action.action == "replace":
            new_val = action.new_value or ""
            if not new_val.strip():
                continue
            slots[action.slot] = {
                "value": new_val,
                "source_label": SourceLabel.USER,
                "status": "filled",
            }
            log.append(
                Correction(slot=action.slot, previous=previous, new=new_val, turn=turn)
            )
        # ignore는 패스

    # TODO(재검증 보류): 기획서 5장 매트릭스는 정정 시 리서치·RAG '재발동'을 요구하나,
    # 현재는 슬롯 덮어쓰기만 하고 교체된 새 값에 대한 재검증은 하지 않는다(워커 stub 단계).
    # 실 워커 연결 시 replace된 슬롯의 새 value를 dispatch subject로 재투입할 것.

    # 정정으로 처리한 세그먼트에 target_slot 표시
    for seg in targets:
        if not seg.get("target_slot") and out.actions:
            seg["target_slot"] = out.actions[0].slot if out.actions[0].slot in valid else None

    return {"slots": slots, "correction_log": log, "turn_segments": segments}


# --- slot fills ------------------------------------------------------------

_FILL_SYSTEM = (
    """오케스트레이터 슬롯 채움
사업 계획 슬롯과 비어있는 항목을 보고, 사용자 세그먼트에서 채울 수 있는 값을 추출한다.

**근거 있는 슬롯만 — 추론·창작 금지.** 사용자가 이번 발화에서 '직접 말한' 슬롯만 fill로 낸다. 한 마디에서 말하지 않은 슬롯까지 그럴듯하게 지어내 채우지 마라 — 발화에 근거가 없으면 fill을 내지 않는다(빈 칸으로 둬 나중에 사용자가 직접 말하거나 ask_slot이 묻게 한다). 값은 사용자가 말한 내용을 슬롯에 맞게 다듬는 정도지, 없는 세부를 보태 부풀리는 게 아니다.
예: "강남에서 로스터리 카페 하려고" → solution(로스터리 카페 운영)·market(강남 상권) 정도만. risks·milestones·resources·revenue·advantage·goal은 사용자가 말 안 했으니 **내지 마라**(고임대료 리스크·준비 일정·초기자본 등을 추론해 채우면 계획이 사용자가 안 한 말로 오염된다).
예: "타겟은 20대고 강남에서 할 거고 목표는 월 1000만" → target(20대)·market(강남)·goal(월 매출 1000만) 셋 다 낸다(셋 다 직접 말함).

슬롯 정의와 경계 (질문 순서):
"""
    + slot_guide_text()
    + """

각 채움(fill)마다:
- slot: 위 정의와 경계에 비춰 값이 깔끔하게 들어맞는 슬롯.
- value: 채울 값.
- kind: 사용자가 그 값을 "확정"(decision)했는지 아직 "떠보는"(exploration) 말인지 가린다. 빈 슬롯은
  둘 다 채우므로(가역 — 틀리면 정정하면 된다) 이 구분은 주로 '이미 찬 슬롯을 덮어쓸지'를 가른다:
  확정이면 "바꿀까?" 확인을 거쳐 교체하고, 떠보는 말이면 기존 값은 안 건드린다.
  - "decision": (a) 사용자가 값을 명시적으로 확정한다 — "X로 하자/가자/확정/정했어/그걸로" 같은 약속, 또는
    (b) 직전에 어시스턴트가 물은 슬롯 질문([직전 대화] 참고)에 사용자가 그 슬롯의 값으로 직접 답한다.
  - "exploration": 단순 탐색·가설·비교·생각 말하기 — "X가 좋을 것 같은데", "X는 어때?", "아마 X일 수도".
  - 애매하면 "exploration"으로 둔다(빈 칸이면 어차피 채워진다 — 확정 아닌 걸 확정으로 부풀리진 마라).
- confidence: 경계 규칙으로 한 슬롯에 명확히 들어맞으면 "clear", 두 슬롯 이상에 그럴듯해 단정하기 어려우면 "ambiguous".
- alt_slots: "ambiguous"일 때 함께 후보가 되는 다른 슬롯들(명확하면 빈 배열).
- reason: "ambiguous"라면 왜 헷갈리는지 한 구절(예: "과금 방식이자 솔루션 형태로 모두 읽힘").
- adequate: 값이 그 슬롯 정의가 요구하는 알맹이를 갖췄으면 true. 갖추지 못해 공허하면 false — 예: goal에 "결과물"·"잘 됐으면"(수치·기한 없음), target에 "사람들"·"누구나"(구체 대상 없음). 값이 슬롯 내용이 아니라 "그 슬롯이 아직 비었다"는 사실을 서술하는 비-답(예: "명시되지 않음", "제공되지 않음", "해당 없음", "알 수 없음")이면 값이 없는 것이니 false다 — 그런 비-답을 value로 지어내지 말고 fill을 내지 마라. false면 코드가 채우지 않고 되묻는다.

억지로 하나로 밀어넣지 말고, 진짜 경계선이면 "ambiguous"로 둔다.
이미 채워진 슬롯은 보통 건드리지 마라(정정은 correction이 처리). 단, 사용자가 정정 마커("빼/말고/취소") 없이 이미 찬 슬롯을 분명히 '다른 값으로' 다시 정하면, 그 슬롯과 새 값을 kind=decision으로 뽑아라 — 코드가 "바꿀까?"를 확인한다(여기서 직접 덮어쓰지 않는다).

kind 예:
- [직전 대화] [assistant] "타겟이 누구예요?" / [user] "네이버 콘텐츠 운영팀" → target, kind=decision (물은 슬롯에 직접 답)
- [user] "일본 시장도 괜찮으려나?" → market, kind=exploration (아직 정한 게 아님)

JSON만 출력."""
)


class FillItem(BaseModel):
    slot: str
    value: str
    kind: Literal["decision", "exploration"] = "exploration"  # 결정/탐색 — exploration은 확인 후에만 채운다
    confidence: Literal["clear", "ambiguous"] = "clear"  # 기본 clear (하위호환)
    alt_slots: list[str] = Field(default_factory=list)   # ambiguous일 때 다른 후보
    reason: str = ""                                     # 왜 애매한지(확인 질문 문구용)
    adequate: bool = True                                # 값이 슬롯 정의의 알맹이를 갖췄나 — False면 안 채우고 되묻는다


class FillOut(BaseModel):
    fills: list[FillItem] = Field(default_factory=list)


async def extract_slot_fills_node(state: PlanState) -> dict:
    """claim 세그먼트에서 슬롯 값을 추출해 즉시 주입하거나 확인 큐로 보낸다 →
    {"slots","turn_segments","pending_confirmations"}(없으면 {}).

    - 빈 슬롯: 결정+명확 → 즉시 주입(단 값이 공허하면 안 채우고 비워둔다 — 다음 턴 되묻기),
      결정+애매 → slot 확인 큐, 탐색 → commit 확인 큐(턴당 1건).
    - 이미 찬 슬롯: 정정 마커 없이 '다른 값'으로 다시 정하면 replace 확인 큐(교체는 확인 후에만).
    - dispatch된 claim 세그먼트엔 근거→슬롯 연결용 target_slot을 태그한다(segment가 더는 슬롯
      힌트를 주지 않으므로). [직전 대화]는 kind 판정 (b)("방금 물은 슬롯에 직접 답했나") 근거.
    """
    segments = state.get("turn_segments") or []
    candidates = [s for s in segments if "claim" in (s.get("utterance_types") or [])]
    if not candidates:
        return {}

    slots = dict(state.get("slots") or {})
    empty_slots = [name for name in ALL_SLOTS if not (slots.get(name) or {}).get("value")]

    user_payload = (
        "[직전 대화]\n" + recent_history(state, n=4) + "\n\n"
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        f"[비어있는 슬롯]\n{', '.join(empty_slots) or '없음'}\n\n"
        "[세그먼트]\n"
        + "\n".join(
            f"{i+1}. ({','.join(s.get('utterance_types') or [])}) "
            f"{s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(candidates)
        )
    )
    out = await call_json(_FILL_SYSTEM, user_payload, FillOut)

    valid = set(ALL_SLOTS)
    empty_set = set(empty_slots)
    last_asked = state.get("last_asked_slot")            # 직전 어시스턴트가 ask_slot으로 물은 슬롯
    pending = list(state.get("pending_confirmations") or [])
    queued = {p.get("proposed_slot") for p in pending}  # 이미 확인 대기 중인 슬롯
    decided: set[str] = set()                            # 이번 턴에 쓰거나 큐에 넣은 슬롯

    def _tag_segment(slot: str) -> None:
        # 근거→슬롯 연결용 — fill이 본 슬롯을 아직 태그 없는 claim 세그먼트에 순서대로 단다.
        for seg in candidates:
            if seg.get("target_slot") is None:
                seg["target_slot"] = slot
                return

    def _queue(slot: str, candidate_slots: list[str], confirm_kind: str, value: str,
               reason: str, previous_value: str = "", adequate: bool = True) -> None:
        # 주입하지 말고 사용자 확인 큐로(다음 턴 confirm_resolve가 해소).
        pending.append(
            {
                "value": value,
                "proposed_slot": slot,
                "candidate_slots": candidate_slots or [slot],
                "source_text": value,
                "reason": reason,
                "attempts": 0,
                "confirm_kind": confirm_kind,
                "previous_value": previous_value,
                "adequate": adequate,
            }
        )
        queued.add(slot)
        decided.add(slot)

    for fill in out.fills:
        value = (fill.value or "").strip()
        if not value or fill.slot not in valid:
            continue
        slot = fill.slot

        # 직전에 물은 슬롯에 대한 답이면 결정으로 본다(짧은 명사구라도). kind는 이제 '이미 찬 슬롯을
        # 덮어쓸지'(replace)만 가른다 — 빈 슬롯은 결정/탐색 무관하게 채우므로(아래), 직전 질문에 답한
        # 값이 그 슬롯에 이미 다른 값이 있을 때 교체 확인으로 가도록 결정으로 승격해 둔다.
        kind = "decision" if slot == last_asked else fill.kind

        # 값이 공허(슬롯 알맹이 미달, 또는 "명시되지 않음" 류 비-답) → 어느 경로로도 채우지 않는다.
        # 확인 큐(commit/slot/replace)에도 안 올린다 — "이 비-값을 넣을까?"는 헛질문이라서다.
        # 근거 태그만 남기고 비워둔다(다음 턴 ask_slot이 되묻는다).
        if not fill.adequate:
            _tag_segment(slot)
            continue

        # 이미 찬 슬롯 — 정정 마커 없이 '다른 값'으로 다시 정함 → 교체 확인(덮어쓰기는 확인 후에만).
        if slot not in empty_set:
            existing = ((slots.get(slot) or {}).get("value") or "").strip()
            if (
                kind == "decision"
                and value != existing
                and slot not in decided
                and slot not in queued
            ):
                _tag_segment(slot)
                _queue(slot, [slot], "replace", value, (fill.reason or "").strip(), existing)
            continue

        # 빈 슬롯 채움은 가역이다(빼/바꿔/correction이 되돌린다) — 결정이든 탐색이든 일단 채운다.
        # 떠보는 말(exploration)이어도 빈 칸이면 박고, 틀리면 싸게 정정한다(확인 받느라 안 채우고
        # 드롭하던 마찰 제거). 확인 큐는 '비가역에 가까운' 경우에만: 이미 찬 슬롯 교체(위 replace)와
        # 어느 칸인지 애매(아래 slot). 한 턴에 빈 슬롯이 여럿이면 다 채운다(턴당 1건 캡 없음).
        ambiguous = fill.confidence == "ambiguous" or bool(fill.alt_slots)
        if not ambiguous:
            # 슬롯 명확 — 즉시 주입.
            if slot in decided:
                continue
            slots[slot] = {
                "value": value,
                "source_label": SourceLabel.USER,
                "status": "filled",
            }
            empty_set.discard(slot)
            decided.add(slot)
            _tag_segment(slot)
            continue

        # 슬롯 애매 — 어느 슬롯인지 확인 큐로(잘못된 칸 방지).
        candidate_slots = [s for s in [slot, *fill.alt_slots] if s in valid]
        # 후보 중 채울 수 있는(빈) 슬롯이 하나도 없으면 물어봐야 의미 없음 → 스킵
        if not any(s in empty_set for s in candidate_slots):
            continue
        if slot in decided or slot in queued:
            continue
        _tag_segment(slot)
        _queue(slot, candidate_slots, "slot", value, (fill.reason or "").strip())

    return {
        "slots": slots,
        "turn_segments": segments,
        "pending_confirmations": pending,
    }
