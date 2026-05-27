"""필수 슬롯(P·T·G) 게이트 + 출력 요청 Type 0/1/2/3 (8장).

출력/자동채움 의도는 LLM 1회 호출로 판정 (키워드 false positive 줄임).
Type 분기는 결정론.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.state import REQUIRED_SLOTS, OPTIONAL_SLOTS
from agents.orchestrator.llm import call_json


OutputType = Literal["type0", "type1", "type2", "type3"]


def required_missing(state: PlanState) -> list[str]:
    slots = state.get("slots") or {}
    return [s for s in REQUIRED_SLOTS if not (slots.get(s) or {}).get("value")]


def optional_missing(state: PlanState) -> list[str]:
    slots = state.get("slots") or {}
    return [s for s in OPTIONAL_SLOTS if not (slots.get(s) or {}).get("value")]


_INTENT_SYSTEM = """오케스트레이터 출력 의도 판정
사용자 발화에서 다음 두 의도를 판정한다.

- wants_output: 지금까지 채워진 슬롯으로 계획서를 뽑아달라는 요청.
  예: "여기까지 뽑아줘", "출력해", "그만 정리해줘", "PDF 만들어"
- wants_autofill: 비어있는 선택 슬롯을 에이전트가 알아서 채워달라는 요청.
  예: "나머지 알아서 채워줘", "자동으로 채워", "네가 채워서 마무리"

일반 대화·정보 입력·정정·질문은 둘 다 false.
JSON만 출력."""


class IntentOut(BaseModel):
    wants_output: bool
    wants_autofill: bool


async def detect_output_request(state: PlanState) -> tuple[bool, bool]:
    text = state.get("user_input", "")
    if not text.strip():
        return False, False
    out = await call_json(_INTENT_SYSTEM, text, IntentOut)
    return out.wants_output, out.wants_autofill


async def gate_node(state: PlanState) -> dict:
    wants_output, wants_autofill = await detect_output_request(state)
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
