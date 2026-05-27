"""긴 발화를 의미 단위로 분해 (6장 Fig.0). 1차 버전 — 휴리스틱."""
from __future__ import annotations

import re

from common.schema import PlanState, Segment


_SPLIT_PATTERN = re.compile(r"(?<=[.!?。])\s+|(?<=다)\.|(?<=어요)\.|\n+|,\s+(?=그리고|근데|하지만|또)")


def split_segments(text: str) -> list[str]:
    raw = [s.strip() for s in _SPLIT_PATTERN.split(text) if s and s.strip()]
    # 너무 짧은 조각은 앞과 합침
    merged: list[str] = []
    for chunk in raw:
        if merged and len(chunk) < 6:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged or ([text.strip()] if text.strip() else [])


def segment_node(state: PlanState) -> dict:
    user_input = state.get("user_input", "")
    pieces = split_segments(user_input)
    segments: list[Segment] = [
        {"text": p, "utterance_type": "fact_claim", "target_slot": None}
        for p in pieces
    ]
    return {"turn_segments": segments}
