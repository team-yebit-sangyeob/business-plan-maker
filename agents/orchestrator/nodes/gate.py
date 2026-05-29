"""필수 슬롯(P·T·G) 게이트 + 출력 요청 Type 0/1/2/3 (기획서 8장).

출력 가능 여부의 단일 기준은 필수 슬롯 3개(problem·target·goal). 비평 severity나
선택 슬롯 충족도는 게이트를 막지 않는다. "뽑아줘/채워줘" 같은 출력·자동채움 의도는
키워드가 아니라 LLM 1회로 판정(예: "여기까지 그만"=출력, "그만해 시끄러"=출력 아님).
의도 판정 후 Type 분기는 슬롯 상태만 보는 결정론.

Type 분기 예시 (P=problem, T=target, G=goal, opt=선택 7개):
    wants_output=F, wants_autofill=F                     → None   (그냥 대화/정보입력)
    wants_output=T, P·T·G 중 하나라도 빔                  → type0  (거절, 빈 필수슬롯 안내)
    wants_autofill=T, P·T·G ✓                            → type3  (선택슬롯 에이전트가 채움)
    wants_output=T, P·T·G ✓, opt 전부 ✓                  → type1  (완전 출력)
    wants_output=T, P·T·G ✓, opt 일부 빔                 → type2  (조기 출력, 빈칸 [미정])
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.state import REQUIRED_SLOTS, OPTIONAL_SLOTS
from agents.orchestrator.llm import call_json


OutputType = Literal["type0", "type1", "type2", "type3"]


def required_missing(state: PlanState) -> list[str]:
    """필수 슬롯 중 값이 빈 것들. 예: target만 비면 ["target"]. 빈 리스트면 게이트 통과 가능."""
    slots = state.get("slots") or {}
    return [s for s in REQUIRED_SLOTS if not (slots.get(s) or {}).get("value")]


def optional_missing(state: PlanState) -> list[str]:
    """선택 슬롯 중 값이 빈 것들. 예: ["market","revenue"]. type1/type2 구분에 쓰임."""
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

    # 분기 순서가 곧 우선순위: 필수 미달이면 무엇을 요청했든 무조건 거절(type0)이 먼저.
    # 그 다음 자동채움 요청(type3)이 완전/조기 판정보다 앞선다 — 사용자가 "채워줘"라고 한
    # 이상 빈 선택슬롯 유무로 type1/2를 따질 게 아니라 채움 분기로 보내야 하므로.
    if missing_req:
        decision: OutputType = "type0"   # 예: goal 미달인데 "뽑아줘" → 거절 + "goal부터 정하자"
    elif wants_autofill:
        decision = "type3"               # 예: P·T·G ✓ + "나머지 알아서" → 데이터/추론/후보로 채움
    elif not missing_opt:
        decision = "type1"               # 예: 10개 전부 user 입력 → 완전 출력
    else:
        decision = "type2"               # 예: P·T·G만 ✓ + "여기까지" → 나머지 [미정]로 조기 출력

    return {"output_request": decision}
