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

extract_slot_fills_node: 정정 이후 단계에서, claim/opinion 세그먼트 중
  '비어 있는' 슬롯에 들어맞는 값을 골라 채움(이미 찬 슬롯은 안 건드림).
  명확(confidence=clear)하면 즉시 주입하고, 어느 슬롯인지 애매(ambiguous)하면
  주입하지 않고 pending_confirmations에 쌓아 다음 턴 confirm_resolve가 사용자
  답으로 확정한다("이거 솔루션이에요 차별점이에요?").

  예: 빈 슬롯 [target, goal] + 세그먼트 "(claim) 타겟은 네이버 콘텐츠 운영팀"
      → fills: target = "네이버 콘텐츠 운영팀" (source=user, clear). goal은 근거 없으면 그대로 빔.
  예: 세그먼트 "(claim) AI로 자동 검수해주는 거" → solution/advantage 경계 → ambiguous
      → 주입 보류, pending_confirmations += {value, proposed=solution, candidates=[solution,advantage]}
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from common.schema import PlanState, Correction
from common.schema.labels import SourceLabel
from common.schema.state import ALL_SLOTS, PendingConfirmation, slot_guide_text
from agents.orchestrator.llm import call_json


# --- correction ------------------------------------------------------------

_CORR_SYSTEM = (
    """오케스트레이터 정정 해소
사용자가 정정 신호를 낸 세그먼트 목록과 현재 슬롯 상태를 보고, 어떤 슬롯을 어떻게 갱신할지 결정한다.

- action="clear": 슬롯을 비움 (특정 값을 빼는 경우).
- action="replace": 슬롯 값을 new_value로 교체.
- action="ignore": 슬롯 매칭이 모호하면 건너뜀.

slot은 아래 정의·경계를 따른다:
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

슬롯 정의·경계 (질문 순서):
"""
    + slot_guide_text()
    + """

각 채움(fill)마다:
- slot: 위 정의·경계로 값이 깔끔히 떨어지는 슬롯.
- value: 채울 값.
- confidence: 경계 규칙으로 한 슬롯에 명확히 들어맞으면 "clear", 두 슬롯 이상에 그럴듯해 단정하기 어려우면 "ambiguous".
- alt_slots: "ambiguous"일 때 함께 후보가 되는 다른 슬롯들(명확하면 빈 배열).
- reason: "ambiguous"라면 왜 헷갈리는지 한 구절(예: "과금 방식이자 솔루션 형태로 모두 읽힘").

세그먼트에 붙은 [힌트:슬롯]은 참고만 — 정의·경계가 우선이다.
이미 채워진 슬롯은 건드리지 마라(정정 노드가 처리함). 억지로 하나로 밀어넣지 말고, 진짜 경계선이면 "ambiguous"로 둔다.

JSON만 출력."""
)


class FillItem(BaseModel):
    slot: str
    value: str
    confidence: Literal["clear", "ambiguous"] = "clear"  # 기본 clear (하위호환)
    alt_slots: list[str] = Field(default_factory=list)   # ambiguous일 때 다른 후보
    reason: str = ""                                     # 왜 애매한지(확인 질문 문구용)


class FillOut(BaseModel):
    fills: list[FillItem] = Field(default_factory=list)


async def extract_slot_fills_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    candidates = [
        s
        for s in segments
        if any(
            t in (s.get("utterance_types") or [])
            for t in ("claim", "opinion")
        )
    ]
    if not candidates:
        return {}

    slots = dict(state.get("slots") or {})
    empty_slots = [
        name for name in ALL_SLOTS if not (slots.get(name) or {}).get("value")
    ]
    if not empty_slots:
        return {}

    # 세그먼트 힌트(segment의 target_slot)를 prior로 같이 넘긴다 — 슬롯 결정은 fill이
    # 단일 권위로 하되, segment가 본 슬롯을 참고하게 해 두 판단이 갈리는 걸 줄인다.
    user_payload = (
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        f"[비어있는 슬롯]\n{', '.join(empty_slots)}\n\n"
        "[세그먼트]\n"
        + "\n".join(
            f"{i+1}. ({','.join(s.get('utterance_types') or [])}) "
            f"[힌트:{s.get('target_slot') or '없음'}] "
            f"{s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(candidates)
        )
    )
    out = await call_json(_FILL_SYSTEM, user_payload, FillOut)

    valid = set(ALL_SLOTS)
    empty_set = set(empty_slots)
    pending = list(state.get("pending_confirmations") or [])
    queued = {p.get("proposed_slot") for p in pending}  # 이미 확인 대기 중인 슬롯
    decided: set[str] = set()                            # 이번 턴에 쓰거나 큐에 넣은 슬롯

    for fill in out.fills:
        value = (fill.value or "").strip()
        if not value or fill.slot not in valid:
            continue

        ambiguous = fill.confidence == "ambiguous" or bool(fill.alt_slots)
        if not ambiguous:
            # 명확 — 빈 슬롯이면 즉시 주입
            if fill.slot not in empty_set or fill.slot in decided:
                continue
            slots[fill.slot] = {
                "value": value,
                "source_label": SourceLabel.USER,
                "status": "filled",
            }
            empty_set.discard(fill.slot)
            decided.add(fill.slot)
            # 세그먼트에 target_slot 표시
            for seg in candidates:
                if seg.get("target_slot") is None:
                    seg["target_slot"] = fill.slot
                    break
            continue

        # 애매 — 주입하지 말고 사용자 확인 큐로(다음 턴 confirm_resolve가 해소)
        candidate_slots = [s for s in [fill.slot, *fill.alt_slots] if s in valid]
        # 후보 중 채울 수 있는(빈) 슬롯이 하나도 없으면 물어봐야 의미 없음 → 스킵
        if not any(s in empty_set for s in candidate_slots):
            continue
        if fill.slot in decided or fill.slot in queued:
            continue
        item: PendingConfirmation = {
            "value": value,
            "proposed_slot": fill.slot,
            "candidate_slots": candidate_slots or [fill.slot],
            "source_text": value,
            "reason": (fill.reason or "").strip(),
            "attempts": 0,
        }
        pending.append(item)
        queued.add(fill.slot)
        decided.add(fill.slot)

    return {
        "slots": slots,
        "turn_segments": segments,
        "pending_confirmations": pending,
    }
