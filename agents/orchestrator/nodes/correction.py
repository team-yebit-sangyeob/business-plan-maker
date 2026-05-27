"""정정 처리 + 슬롯 채움 — LLM 기반 (Case E).

correction_node: utterance_types에 correction 포함 세그먼트만 모아 LLM에 넘김.
  현재 슬롯·correction_log·최근 messages를 보고 "어느 슬롯의 어떤 값을 어떻게"
  바꿀지 판단.

extract_slot_fills_node: 정정 이후 단계에서, 검증·결정·제약 세그먼트 중
  비어 있는 슬롯에 들어갈 값만 골라 채움.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from common.schema import PlanState, Correction, Slot
from common.schema.labels import SourceLabel
from common.schema.state import ALL_SLOTS
from agents.orchestrator.llm import call_json


# --- correction ------------------------------------------------------------

_CORR_SYSTEM = """오케스트레이터 정정 해소
사용자가 정정 신호를 낸 세그먼트 목록과 현재 슬롯 상태를 보고, 어떤 슬롯을 어떻게 갱신할지 결정한다.

- action="clear": 슬롯을 비움 (특정 값을 빼는 경우).
- action="replace": 슬롯 값을 new_value로 교체.
- action="ignore": 슬롯 매칭이 모호하면 건너뜀.

slot은 다음 중 하나: problem, target, goal, solution, market, revenue, milestones, risks, resources.

JSON만 출력."""


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

    # 정정으로 처리한 세그먼트에 target_slot 표시
    for seg in targets:
        if not seg.get("target_slot") and out.actions:
            seg["target_slot"] = out.actions[0].slot if out.actions[0].slot in valid else None

    return {"slots": slots, "correction_log": log, "turn_segments": segments}


# --- slot fills ------------------------------------------------------------

_FILL_SYSTEM = """오케스트레이터 슬롯 채움
사업 계획 슬롯 9개와 비어있는 항목을 보고, 사용자 세그먼트에서 채울 수 있는 값을 추출한다.

슬롯 의미:
- problem: 해결하려는 문제 — 누가/어떤 상황에서/무엇 때문에/어떤 손실
- target: 타겟 고객 — 회사/부서/직책/규모
- goal: 목표 수치 — 언제까지 얼마, 실패 임계값
- solution: 솔루션 형태 (서비스/제품/플랫폼/툴)
- market: 시장 규모·경쟁 데이터
- revenue: 수익 모델 (구독/건당/라이선싱 등)
- milestones: 일정 단계
- risks: 리스크
- resources: 인력·예산 규모

이미 채워진 슬롯은 건드리지 마라(정정 노드가 처리함). 세그먼트 내용이 슬롯에 명백히 들어맞을 때만 추출.

JSON만 출력."""


class FillItem(BaseModel):
    slot: str
    value: str


class FillOut(BaseModel):
    fills: list[FillItem] = Field(default_factory=list)


async def extract_slot_fills_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    candidates = [
        s
        for s in segments
        if any(
            t in (s.get("utterance_types") or [])
            for t in ("fact_claim", "decision", "constraint", "hypothesis", "opinion")
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

    user_payload = (
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        f"[비어있는 슬롯]\n{', '.join(empty_slots)}\n\n"
        "[세그먼트]\n"
        + "\n".join(
            f"{i+1}. ({','.join(s.get('utterance_types') or [])}) "
            f"{s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(candidates)
        )
    )
    out = await call_json(_FILL_SYSTEM, user_payload, FillOut)

    empty_set = set(empty_slots)
    for fill in out.fills:
        if fill.slot not in empty_set:
            continue
        value = (fill.value or "").strip()
        if not value:
            continue
        slots[fill.slot] = {
            "value": value,
            "source_label": SourceLabel.USER,
            "status": "filled",
        }
        empty_set.discard(fill.slot)
        # 세그먼트에 target_slot 표시
        for seg in candidates:
            if seg.get("target_slot") is None:
                seg["target_slot"] = fill.slot
                break

    return {"slots": slots, "turn_segments": segments}


# 하위 호환 — 기존 라우터가 import할 수 있는 동기 함수 자리(deprecated)
def collect_slot_fills(state: PlanState) -> dict:
    """deprecated — extract_slot_fills_node로 대체됨."""
    return {"slots": dict(state.get("slots") or {})}
