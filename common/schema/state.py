"""슬롯 9개 + 턴 처리 상태 (LangGraph PlanState)."""
from __future__ import annotations

from typing import Literal, TypedDict

from common.schema.labels import SourceLabel


REQUIRED_SLOTS: tuple[str, ...] = ("problem", "target", "goal")
OPTIONAL_SLOTS: tuple[str, ...] = (
    "solution",
    "market",
    "revenue",
    "milestones",
    "risks",
    "resources",
)
ALL_SLOTS: tuple[str, ...] = REQUIRED_SLOTS + OPTIONAL_SLOTS


# 5장 발화 유형 8종
UtteranceType = Literal[
    "clarification_needed",
    "fact_claim",
    "opinion",
    "hypothesis",
    "decision",
    "constraint",
    "correction",
    "meta",
]


class Slot(TypedDict, total=False):
    value: str | None
    source_label: SourceLabel
    status: Literal["empty", "needs_clarification", "filled"]


class Segment(TypedDict):
    text: str
    utterance_type: UtteranceType
    target_slot: str | None


class Correction(TypedDict):
    slot: str
    previous: str | None
    new: str | None
    turn: int


class ValidationReport(TypedDict, total=False):
    subject: str
    findings: list[str]
    sources: list[str]
    agreement: Literal["confirms", "contradicts", "partial", "unknown"]


def _empty_slot() -> Slot:
    return {"value": None, "source_label": SourceLabel.EMPTY, "status": "empty"}


def initial_state() -> "PlanState":
    return {
        "session_id": "",
        "turn": 0,
        "user_input": "",
        "turn_segments": [],
        "slots": {name: _empty_slot() for name in ALL_SLOTS},
        "correction_log": [],
        "validation_reports": [],
        "pending_question": "",
        "output_request": None,
    }


class PlanState(TypedDict, total=False):
    session_id: str
    turn: int
    user_input: str

    turn_segments: list[Segment]
    slots: dict[str, Slot]
    correction_log: list[Correction]
    validation_reports: list[ValidationReport]

    pending_question: str
    # 출력 요청 분기 결과 (8장 Type 0/1/2/3)
    output_request: Literal["type0", "type1", "type2", "type3"] | None
