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
  explain_tool     — 도구/슬롯/사용법 메타질문에 SLOT_SPECS·APP_OVERVIEW로 답(워커·검색 없이)
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
from common.schema.state import ALL_SLOTS, SLOT_SPECS, recent_history, slot_title, tool_help_text
from agents.orchestrator.llm import call_json
from agents.orchestrator.nodes.gate import required_missing, optional_missing


# 슬롯별 질문 톤은 SLOT_SPECS[...]["question"](단일 원천)에서 가져온다.


_SYSTEM = """대화 에이전트
오케스트레이터가 결정한 의도(intent) 목록을 받아, 사용자에게 보낼 자연어 응답 하나로 엮는다.
판단은 하지 않는다 — 주어진 intent만 자연스럽게 한 메시지로 엮어 표현한다.

- 문체: 친근한 반말~부드러운 존댓말 혼용, 사업 파트너 톤. 짧은 대화나 한 가지만 말할 때는 한두 문장 평문으로 둔다.
- 포맷: 한 응답에 여러 부분이 섞이거나(예: 찾은 근거 보고 + 다음 질문) 나열할 항목이 둘 이상이면, 줄바꿈으로 부분을 나누고 항목은 "- " 불릿으로 정리한다. 강조는 **굵게**를 쓸 수 있다(마크다운으로 렌더된다). 항목이 하나거나 짧은 답이면 불릿 없이 평문으로 둔다.
- 여러 intent가 오면 매끄럽게 연결한다(예: 정정 확인 → 찾은 근거 → 다음 질문).
- 입력의 recent_messages는 최근 대화 이력이다 — recall intent를 답할 때만 근거로 쓰고, 다른 intent엔 끌어들이지 않는다.
- intent별 표현 규칙:
  - acknowledge: 사용자의 정정이나 확인을 짧게 받아준다.
  - report_findings: research는 외부 사실, rag는 회사 내부 자료, logic_validator는 claim과 근거 사이의 논리 검증 결과다. 1~2문장으로 전달하고, 사용자 전제와 어긋나면 부드럽게 교정을 제안한다.
  - answer_question: 사용자가 물은 것에 research와 rag가 찾은 답을 전달한다.
  - recall: 사용자가 직전 대화에 나온 내용을 되묻거나 확인하는 발화. recent_messages(최근 대화 이력)에서 찾아 간결하고 직접적으로 답한다(새 검색·워커 없이). 이력에 없으면 솔직히 모른다고 하고 부드럽게 잇는다.
  - explain_tool: 사용자가 이 도구·슬롯·사용법을 물었다. 주어진 body(도구/슬롯 설명)만 근거로 친근하게 답한다(새 검색·워커 없이, recent_messages도 안 씀). scope가 "all"이면 body의 각 슬롯을 "- 제목: 정의" 불릿 그대로 빠짐없이 보여준다(한두 문장으로 압축하지 않는다). scope가 "slot"이나 "general"이면 한두 문장으로 답한다. 답한 뒤 한 문장으로 본론(계획 채우기)으로 가볍게 잇는다.
  - clarify: 모호한 발화를 좁히는 질문을 한다(이게 있으면 보통 ask_slot은 보류된다).
  - redirect: 스코프 밖 발화를 부드럽게 넘기고 본론으로 잇는다.
  - reject_output: 필수 슬롯이 미달이라 지금은 출력이 이르다고 알리고, 무엇을 채우면 되는지 안내한다.
  - deliver_plan: 계획서를 뽑을 수 있다고 안내한다(type2면 빈 항목은 [미정]으로 들어간다고 덧붙인다).
  - confirm_slot: 방금 사용자가 말한 값을 슬롯에 넣기 전에 확인한다. 후보 슬롯이 둘 이상이면 그 값과 후보들을 제시하고 "어디에 넣을까요?"를 한 문장으로 묻고, 후보가 하나면 "이거 [그 슬롯]에 넣어둘까요?"처럼 넣을지 말지를 한 문장으로 묻는다(사용자가 아직 정하지 않고 떠본 값이다). 사용자가 답하기 전엔 다음 슬롯 질문(ask_slot)은 하지 않는다.
  - ask_slot: 다음 채울 슬롯을 맥락 있게 한 문장으로 묻는다(참고 예시의 톤을 살려서).

반드시 {"message": "..."} JSON만 출력."""


class ConversationOut(BaseModel):
    message: str


_AGREEMENT_PRIORITY = {"contradicts": 0, "partial": 1, "confirms": 2, "unknown": 3}


def _next_empty_slot(slots: dict) -> str | None:
    return next((s for s in ALL_SLOTS if not (slots.get(s) or {}).get("value")), None)


# 슬롯 식별용 별칭 — 사용자가 그 칸을 부르는 흔한 말. 슬롯 '정의'는 SLOT_SPECS가 단일 원천이고,
# 이건 '어느 칸을 가리키나'만 잡는 표면 매칭(tool_help 응답 라우팅용)이라 여기 둔다.
_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "problem": ("문제",),
    "target": ("타겟", "고객"),
    "solution": ("솔루션",),
    "market": ("시장",),
    "advantage": ("차별점", "경쟁우위"),
    "revenue": ("수익",),
    "goal": ("목표",),
    "resources": ("리소스", "자원"),
    "milestones": ("마일스톤", "일정"),
    "risks": ("리스크",),
}


def _match_help_slot(text: str) -> str | None:
    """tool_help 발화에서 슬롯 1개를 결정론으로 식별 — 영문 key 또는 한국어 별칭 표면 매칭.

    정확히 1개만 잡히면 그 슬롯, 0개나 2개 이상이면 None(도구 전체 설명으로 답한다).
    segment의 target_slot 대신 여기서 잡는다 — 그 필드는 fill·evidence가 읽어 오염되기 때문.
    """
    t = text or ""
    tl = t.lower()
    hits = [
        name
        for name, aliases in _SLOT_ALIASES.items()
        if name in tl or any(a in t for a in aliases)
    ]
    return hits[0] if len(hits) == 1 else None


# 슬롯을 통칭하는 말(특정 슬롯 별칭이 아님)과 '여럿/전부'를 가리키는 말 — "all" 스코프 판정용.
# 부분문자열 매칭이라 짧은 음절(각·다·어떤·무슨)은 흔한 단어(생각·있다)에 걸려 과발동하므로
# 뺀다. 핵심 케이스 "각 슬롯의 역할"은 통칭어 '슬롯'으로 이미 잡혀 enumerator가 필요 없다.
_GENERIC_SLOT_WORDS: tuple[str, ...] = ("슬롯", "항목", "칸")
_ENUMERATOR_WORDS: tuple[str, ...] = (
    "각각", "전부", "모든", "모두", "전체", "뭐뭐", "리스트", "종류",
)


def _help_scope(text: str) -> tuple[str | None, str]:
    """tool_help 발화 → (slot, scope). scope ∈ {"slot","all","general"}.

    특정 슬롯 1개를 먼저 본다("솔루션 슬롯이 뭐야?" → ("solution","slot")). 아니면 슬롯을
    통칭하거나 '각/전부' 식으로 묻는지로 all/general을 가른다("각 슬롯의 역할?" → (None,"all"),
    "넌 뭐 할 수 있어?" → (None,"general")). 두 슬롯 이상이 섞여도(예: "문제랑 타겟 슬롯 차이?")
    _match_help_slot이 None을 줘 "all"로 떨어진다 — 전체를 보여줘 사용자가 찾게 한다.
    """
    slot = _match_help_slot(text)
    if slot:
        return slot, "slot"
    t = text or ""
    generic = any(w in t for w in _GENERIC_SLOT_WORDS)
    enumerator = any(w in t for w in _ENUMERATOR_WORDS)
    if generic or enumerator:
        return None, "all"
    return None, "general"


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

    # 1.3) explain_tool — 도구/슬롯/사용법 메타질문: 워커 없이 SLOT_SPECS·APP_OVERVIEW에서 답.
    # recall과 달리 in_scope 필터를 두지 않는다 — 도구 질문이 in_scope=false로 잘못 매겨져도 답한다
    # (아래 redirect는 tool_help 세그먼트를 건너뛰어 explain_tool이 우선한다).
    tool_helps = [
        seg for seg in segments if "tool_help" in (seg.get("utterance_types") or [])
    ]
    for seg in tool_helps:
        subj = (seg.get("canonical_text") or seg.get("text", "")).strip()
        slot, scope = _help_scope(subj)
        intents.append(
            {
                "type": "explain_tool",
                "slot": slot,
                "scope": scope,
                "body": tool_help_text(slot, scope),
            }
        )
        suppress_ask = True  # 도구 설명이 곧 응답 — 다음 슬롯 질문은 보류

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
                "confirm_kind": pending_item.get("confirm_kind", "slot"),
                "reason": pending_item.get("reason", ""),
            }
        )

    # 2) redirect — 스코프 밖 발화
    off_topic = next(
        (
            (seg.get("canonical_text") or seg.get("text", "")).strip()
            for seg in segments
            if seg.get("in_scope") is False
            and "tool_help" not in (seg.get("utterance_types") or [])
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
    """결정론 intent 목록을 자연어 한 응답으로 엮는다 → {"pending_question"(+ ask_slot을 실제로 물은 턴엔 "last_asked_slot": 물은 슬롯)}."""
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
    result: dict = {"pending_question": message}
    # ask_slot을 실제로 물었으면 그 슬롯을 기록 — 다음 턴 fill이 "직전 질문에 직접 답"을
    # 결정론으로 잡는다(kind=decision 기준 (b)). 안 물은 턴엔 이 키를 안 내보내 이전 값 유지.
    asked = next((i["slot"] for i in intents if i.get("type") == "ask_slot"), None)
    if asked is not None:
        result["last_asked_slot"] = asked
    return result
