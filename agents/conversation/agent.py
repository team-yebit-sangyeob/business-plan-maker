"""대화 에이전트 — 오케가 결정한 의도(intent)들을 자연어 한 응답으로 변환.

판단은 안 한다 — 표현만 한다(conversation_spec). 무엇을 물을지·무엇을 보고할지는
state에서 결정론으로 뽑고(_build_intents), 그 intent 목록을 LLM 1회로 한 메시지로 엮는다.

지원 intent(conversation_spec TRIGGER MATRIX 전체):
  ask_slot         — 비어있는 슬롯 질문(기본 질문 순서 = ALL_SLOTS 첫 빈칸)
  confirm_slot     — 애매해서 보류된 주입을 어느 슬롯에 넣을지 확인(있으면 ask_slot 보류)
  clarify          — 모호한 발화 좁히기(있으면 다음 슬롯 질문은 보류)
  report_findings  — 리서치(외부)·RAG(내부)·논리검증 결과 전달 + 전제 교정
  answer_question  — 사용자 질문에 리서치·RAG가 찾은 답 전달(질문은 논리검증 미경유)
  recall           — 되묻기: 직전 대화 내용을 대화 이력에서 찾아 답(워커·검색 없이)
  redirect         — 스코프 밖 발화를 부드럽게 되돌림
  reject_output    — 필수 슬롯 미달 상태의 출력 요청 거절(type0)
  acknowledge      — 정정 반영 확인
  deliver_plan     — 출력 가능 안내(type1/type2)

intent 묶음 규칙: 한 턴에 여러 개가 잡히면(예: report_findings + ask_slot) 한 응답으로
자연스럽게 잇는다. 보고할 결과는 '이번 턴' 리포트(turn_validation_reports)만 본다.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.state import ALL_SLOTS, SLOT_SPECS, recent_history, slot_title
from agents.orchestrator.llm import call_json
from agents.orchestrator.nodes.gate import required_missing, optional_missing


# 슬롯별 질문 톤은 SLOT_SPECS[...]["question"](단일 원천)에서 가져온다.


_SYSTEM = """대화 에이전트
오케스트레이터가 결정한 의도(intent) 목록을 받아, 사용자에게 보낼 자연어 응답 하나로 엮는다.
판단은 하지 않는다 — 주어진 intent만 자연스럽게 한 메시지로 엮어 표현한다.

- 문체: 친근한 반말~부드러운 존댓말 혼용, 사업 파트너 톤. 한두 문장 위주로 간결하게.
- 여러 intent가 오면 매끄럽게 연결한다(예: 정정 확인 → 찾은 근거 → 다음 질문).
- 입력의 recent_messages는 최근 대화 이력이다 — recall intent를 답할 때만 근거로 쓰고, 다른 intent엔 끌어들이지 않는다.
- intent별 표현 규칙:
  - acknowledge: 사용자의 정정이나 확인을 짧게 받아준다.
  - report_findings: research는 외부 사실, rag는 회사 내부 자료, logic_validator는 claim과 근거 사이의 논리 검증 결과다. 1~2문장으로 전달하고, 사용자 전제와 어긋나면 부드럽게 교정을 제안한다.
  - answer_question: 사용자가 물은 것에 research와 rag가 찾은 답을 전달한다.
  - recall: 사용자가 직전 대화에 나온 내용을 되묻거나 확인하는 발화. recent_messages(최근 대화 이력)에서 찾아 간결하고 직접적으로 답한다(새 검색·워커 없이). 이력에 없으면 솔직히 모른다고 하고 부드럽게 잇는다.
  - clarify: 모호한 발화를 좁히는 질문을 한다(이게 있으면 보통 ask_slot은 보류된다).
  - redirect: 스코프 밖 발화를 부드럽게 넘기고 본론으로 잇는다.
  - reject_output: 필수 슬롯이 미달이라 지금은 출력이 이르다고 알리고, 무엇을 채우면 되는지 안내한다.
  - deliver_plan: 계획서를 뽑을 수 있다고 안내한다(type2면 빈 항목은 [미정]으로 들어간다고 덧붙인다).
  - confirm_slot: 방금 사용자가 말한 값이 어느 슬롯인지 애매할 때, 그 값과 후보 슬롯들을 제시하고 "어디에 넣을까요?"를 한 문장으로 묻는다. 사용자가 답하기 전엔 다음 슬롯 질문(ask_slot)은 하지 않는다.
  - ask_slot: 다음 채울 슬롯을 맥락 있게 한 문장으로 묻는다(참고 예시의 톤을 살려서).

반드시 {"message": "..."} JSON만 출력."""


class ConversationOut(BaseModel):
    message: str


_AGREEMENT_PRIORITY = {"contradicts": 0, "partial": 1, "confirms": 2, "unknown": 3}


def _next_empty_slot(slots: dict) -> str | None:
    return next((s for s in ALL_SLOTS if not (slots.get(s) or {}).get("value")), None)


def _build_intents(state: PlanState) -> list[dict]:
    """state → 결정론으로 뽑은 의도 목록(conversation_spec 매트릭스)."""
    slots = state.get("slots") or {}
    segments = state.get("turn_segments") or []
    reports = state.get("turn_validation_reports") or []
    output_request = state.get("output_request")
    turn = state.get("turn", 0)
    correction_log = state.get("correction_log") or []

    intents: list[dict] = []
    suppress_ask = False
    next_empty = _next_empty_slot(slots)

    # 1) acknowledge — 이번 턴 정정
    for c in correction_log:
        if c.get("turn") == turn:
            intents.append(
                {
                    "type": "acknowledge",
                    "slot": c.get("slot"),
                    "previous": c.get("previous"),
                    "new": c.get("new"),
                }
            )

    # 1.2) recall — 되묻기: 워커 없이 대화 이력에서 답(routes=["none"]이라 리포트가 없다)
    recalls = [
        (seg.get("canonical_text") or seg.get("text", "")).strip()
        for seg in segments
        if "recall" in (seg.get("utterance_types") or []) and seg.get("in_scope", True)
    ]
    for subj in [r for r in recalls if r]:
        intents.append({"type": "recall", "subject": subj})
        suppress_ask = True  # 되묻기 응답이 곧 답 — 다음 슬롯 질문은 보류

    # 1.5) confirm_slot — 애매해서 보류된 주입(있으면 다음 슬롯 질문은 보류)
    pending = state.get("pending_confirmations") or []
    pending_item = pending[0] if pending else None
    if pending_item:
        intents.append(
            {
                "type": "confirm_slot",
                "value": pending_item.get("value", ""),
                "candidates": [
                    {"slot": c, "title": slot_title(c)}
                    for c in (pending_item.get("candidate_slots") or [])
                ],
                "reason": pending_item.get("reason", ""),
            }
        )

    # 2) redirect — 스코프 밖 발화
    off_topic = next(
        (
            (seg.get("canonical_text") or seg.get("text", "")).strip()
            for seg in segments
            if seg.get("in_scope") is False
        ),
        None,
    )
    if off_topic:
        intents.append({"type": "redirect", "off_topic": off_topic, "next_slot": next_empty})

    # 3) report_findings / answer_question — 이번 턴 워커 결과(주제별 그룹)
    seg_types = {
        ((seg.get("canonical_text") or seg.get("text", "")).strip())[:80]: (
            seg.get("utterance_types") or []
        )
        for seg in segments
    }
    by_subject: dict[str, dict] = {}
    order: list[str] = []
    for r in reports:
        subj = r.get("subject", "")
        if subj not in by_subject:
            by_subject[subj] = {}
            order.append(subj)
        by_subject[subj][r.get("cluster", "")] = r

    for subj in order:
        clusters = by_subject[subj]
        is_question = "question" in seg_types.get(subj, [])
        research = clusters.get("research")
        rag = clusters.get("rag")
        logic_validator = clusters.get("logic_validator")
        if is_question:
            intents.append(
                {
                    "type": "answer_question",
                    "subject": subj,
                    "research": (research or {}).get("findings", []),
                    "rag": (rag or {}).get("findings", []),
                }
            )
        else:
            agreement = min(
                (
                    (rep or {}).get("agreement", "unknown")
                    for rep in (research, logic_validator, rag)
                    if rep is not None
                ),
                key=lambda a: _AGREEMENT_PRIORITY.get(a, 3),
                default="unknown",
            )
            intents.append(
                {
                    "type": "report_findings",
                    "subject": subj,
                    "research": (research or {}).get("findings", []),
                    "rag": (rag or {}).get("findings", []),
                    "logic_validator": (logic_validator or {}).get("findings", []),
                    "agreement": agreement,
                }
            )

    # 4) 출력 게이트
    if output_request == "type0":
        intents.append({"type": "reject_output", "missing_required": required_missing(state)})
        suppress_ask = True
    elif output_request in ("type1", "type2"):
        intents.append(
            {
                "type": "deliver_plan",
                "output_type": output_request,
                "empty_slots": optional_missing(state),
            }
        )
        suppress_ask = True

    # 5) clarify — 있으면 다음 슬롯 질문 보류(사용자 답 받고 다음 턴)
    clarifications = [
        (seg.get("canonical_text") or seg.get("text", "")).strip()
        for seg in segments
        if "clarify" in (seg.get("routes") or [])
    ]
    clarifications = [c for c in clarifications if c]
    if clarifications:
        for text in clarifications[:2]:
            intents.append({"type": "clarify", "text": text})
        suppress_ask = True

    # 6) ask_slot — 위에서 막지 않았고 확인 대기도 없으면 다음 빈칸 1개
    if not suppress_ask and not pending_item:
        if next_empty is None:
            intents.append({"type": "deliver_plan", "output_type": "ready", "empty_slots": []})
        else:
            intents.append(
                {
                    "type": "ask_slot",
                    "slot": next_empty,
                    "example": SLOT_SPECS.get(next_empty, {}).get("question", ""),
                }
            )

    return intents


def _slot_values(state: PlanState) -> dict:
    slots = state.get("slots") or {}
    return {name: (slots.get(name) or {}).get("value") for name in ALL_SLOTS}


async def conversation_node(state: PlanState) -> dict:
    intents = _build_intents(state)
    payload = json.dumps(
        {
            "tone": "casual_business",
            "slots": _slot_values(state),
            "recent_messages": recent_history(state),
            "intents": intents,
        },
        ensure_ascii=False,
    )
    out = await call_json(_SYSTEM, payload, ConversationOut)
    message = out.message.strip()
    return {"pending_question": message}
