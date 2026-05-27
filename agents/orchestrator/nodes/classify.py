"""8개 발화 유형 분류 (5장). 1차 — 키워드 기반 휴리스틱."""
from __future__ import annotations

import re

from common.schema import PlanState
from common.schema.state import UtteranceType


_CORRECTION = re.compile(r"(아니[, ]|아\s*잠깐|말고|빼자|다시 생각|사실은|취소)")
_META = re.compile(r"(다음|넘어가|일단 출력|뽑아|그만|여기까지)")
_HYPOTHESIS = re.compile(r"(같아|같다|것 같|통할|먹힐|할지도|아닐까)")
_OPINION = re.compile(r"(생각해|봐$|난 |개인적으로|색깔|선호)")
_DECISION = re.compile(r"(정했어|결정|가자|할 거야|확정|가기로)")
_CONSTRAINT_NUM = re.compile(r"(\d+\s*(?:억|만\s*원|명|개월|개사|일|주|월|%|퍼센트))")
_QUESTION_TAIL = re.compile(r"(\?$|좀 해보고|싶은데|어때|어떨까)")


def classify_segment(text: str) -> UtteranceType:
    if _CORRECTION.search(text):
        return "correction"
    if _META.search(text):
        return "meta"
    if _CONSTRAINT_NUM.search(text):
        return "constraint"
    if _DECISION.search(text):
        return "decision"
    if _HYPOTHESIS.search(text):
        return "hypothesis"
    if _QUESTION_TAIL.search(text) or len(text) < 18:
        return "clarification_needed"
    if _OPINION.search(text):
        return "opinion"
    return "fact_claim"


def classify_node(state: PlanState) -> dict:
    segments = list(state.get("turn_segments") or [])
    for seg in segments:
        seg["utterance_type"] = classify_segment(seg["text"])
    return {"turn_segments": segments}
