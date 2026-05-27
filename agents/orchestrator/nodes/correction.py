"""정정 신호 감지 → correction_log + 슬롯 롤백 (6장)."""
from __future__ import annotations

from common.schema import PlanState, Correction
from common.schema.labels import SourceLabel
from common.schema.state import ALL_SLOTS


def correction_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    slots = dict(state.get("slots") or {})
    log = list(state.get("correction_log") or [])
    turn = state.get("turn", 0)

    for seg in segments:
        if seg["utterance_type"] != "correction":
            continue
        # 정정이 어떤 슬롯에 걸리는지: 텍스트에 슬롯 키워드가 있으면 그 슬롯 비움
        target = _guess_slot(seg["text"])
        if target and target in slots:
            previous = slots[target].get("value")
            log.append(Correction(slot=target, previous=previous, new=None, turn=turn))
            slots[target] = {
                "value": None,
                "source_label": SourceLabel.EMPTY,
                "status": "empty",
            }
            seg["target_slot"] = target

    return {"slots": slots, "correction_log": log, "turn_segments": segments}


_SLOT_HINTS = {
    "target": ("타겟", "고객", "네이버", "카카오"),
    "goal": ("목표", "매출", "kpi", "지표"),
    "solution": ("솔루션", "제품", "서비스", "툴"),
    "revenue": ("수익", "가격", "구독", "요금"),
    "market": ("시장", "경쟁"),
    "milestones": ("일정", "마일스톤", "스케줄"),
    "risks": ("리스크", "위험"),
    "resources": ("인력", "예산", "리소스"),
    "problem": ("문제", "감수성"),
}


def _guess_slot(text: str) -> str | None:
    t = text.lower()
    for slot, hints in _SLOT_HINTS.items():
        if any(h in t for h in hints):
            return slot
    # 슬롯이 명시 안 되면 정정 자체로 표시 (실제론 LLM call로 바뀜)
    return None


def collect_slot_fills(state: PlanState) -> dict:
    """결정·제약·사실 유형 세그먼트를 슬롯에 채움 (1차 휴리스틱)."""
    segments = state.get("turn_segments") or []
    slots = dict(state.get("slots") or {})

    for seg in segments:
        utype = seg["utterance_type"]
        if utype in ("correction", "meta", "clarification_needed"):
            continue
        slot = _guess_slot(seg["text"])
        if not slot:
            continue
        existing = slots.get(slot) or {}
        if existing.get("value"):
            continue
        slots[slot] = {
            "value": seg["text"],
            "source_label": SourceLabel.USER,
            "status": "filled",
        }
        seg["target_slot"] = slot

    return {"slots": slots, "turn_segments": segments}
