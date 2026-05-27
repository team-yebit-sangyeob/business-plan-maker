"""필수 슬롯(P·T·G) 게이트 + 출력 요청 Type 0/1/2/3 (8장)."""
from __future__ import annotations

from typing import Literal

from common.schema import PlanState
from common.schema.state import REQUIRED_SLOTS, OPTIONAL_SLOTS


OutputType = Literal["type0", "type1", "type2", "type3"]


def required_missing(state: PlanState) -> list[str]:
    slots = state.get("slots") or {}
    return [s for s in REQUIRED_SLOTS if not (slots.get(s) or {}).get("value")]


def optional_missing(state: PlanState) -> list[str]:
    slots = state.get("slots") or {}
    return [s for s in OPTIONAL_SLOTS if not (slots.get(s) or {}).get("value")]


def detect_output_request(state: PlanState) -> tuple[bool, bool]:
    """(출력 요청 있나, 자동 채움 요청 있나)."""
    segments = state.get("turn_segments") or []
    output_keywords = ("뽑아", "출력", "여기까지", "계획서 생성", "그만")
    autofill_keywords = ("알아서 채워", "나머지 채워", "자동 채움")
    text = " ".join(s["text"] for s in segments) or state.get("user_input", "")
    return (
        any(k in text for k in output_keywords),
        any(k in text for k in autofill_keywords),
    )


def gate_node(state: PlanState) -> dict:
    wants_output, wants_autofill = detect_output_request(state)
    if not wants_output and not wants_autofill:
        return {"output_request": None}

    missing_req = required_missing(state)
    missing_opt = optional_missing(state)

    if missing_req:
        decision: OutputType = "type0"
    elif wants_autofill:
        decision = "type3"
    elif not missing_opt:
        decision = "type1"
    else:
        decision = "type2"

    return {"output_request": decision}
