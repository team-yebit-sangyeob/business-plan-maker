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
  reason_over_context — 추론·도출·종합: 이미 모은 근거(session_evidence)와 대화 이력에서 직접 추론해 답(워커·검색 없이)
  redirect         — 스코프 밖 발화를 부드럽게 되돌림
  acknowledge      — 정정 반영 확인
  deliver_plan     — 슬롯이 전부 차서 계획서 준비 완료 안내(ready). 생성은 '계획서 생성' 버튼(POST /plan)

intent 묶음 규칙: 한 턴에 여러 개가 잡히면(예: report_findings + ask_slot) 한 응답으로
자연스럽게 잇는다. 보고할 결과는 '이번 턴' 리포트(turn_validation_reports)만 본다.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.state import ALL_SLOTS, SLOT_SPECS, recent_history, slot_title, tool_help_text
from common.config import conversation_reasoning_effort
from agents.orchestrator.llm import call_json


# 슬롯별 질문 톤은 SLOT_SPECS[...]["question"](단일 원천)에서 가져온다.


_SYSTEM = """대화 에이전트
오케스트레이터가 결정한 의도(intent) 목록을 받아, 사용자에게 보낼 자연어 응답 하나로 엮는다.
판단은 하지 않는다 — 주어진 intent만 자연스럽게 한 메시지로 엮어 표현한다.

[절대 금지]
- 빈 응답·자리표시자·메타 응답을 내지 않는다. "ACK", "확인했습니다", "응답 준비됐습니다",
  "알겠습니다"만 단독으로 두거나, "답변을 준비했다"·"방금 말한 게 맞다"처럼 응답이 준비됐다거나
  내용을 되풀이했다는 식의 빈 문장으로 끝내면 안 된다.
- intent에 담긴 실제 내용(찾은 근거·질문·확인·되묻기 답 등)을 반드시 메시지 본문으로 펼쳐 쓴다.
  findings가 있으면 그 사실을 사용자에게 직접 전한다 — 있다는 사실만 알리고 마는 건 응답이 아니다.

[문체·포맷]
- 문체: 친근한 반말~부드러운 존댓말 혼용, 사업 파트너 톤. 짧은 대화나 한 가지만 말할 때는 한두 문장 평문으로 둔다.
- 포맷: 한 응답에 여러 부분이 섞이거나(예: 찾은 근거 보고 + 다음 질문) 나열할 항목이 둘 이상이면, 줄바꿈으로 부분을 나누고 항목은 "- " 불릿으로 정리한다. 강조는 **굵게**를 쓸 수 있다(마크다운으로 렌더된다). 항목이 하나거나 짧은 답이면 불릿 없이 평문으로 둔다.
- 여러 intent가 오면 매끄럽게 연결한다(예: 정정 확인 → 찾은 근거 → 다음 질문).
- 입력의 recent_messages는 최근 대화 이력이다 — recall·reason_over_context intent를 답할 때만 근거로 쓰고, 다른 intent엔 끌어들이지 않는다.

[intent별 표현 규칙]
  - acknowledge: 사용자의 정정이나 확인을 짧게 받아준다.
  - report_findings: research는 외부 사실, rag는 회사 내부 자료, logic_validator는 claim과 근거 사이의 논리 검증 결과다. findings의 핵심을 1~3문장으로 자연스럽게 종합해 전한다(상세 근거는 화면 카드가 보여주니 모든 항목을 재나열하지 않는다). 있다는 사실만 알리지 말고 핵심 내용을 담는다. agreement가 contradicts·partial이면 사용자 전제와 어긋나는 지점을 짚어 부드럽게 교정을 제안한다.
  - answer_question: 사용자가 물은 것에 research·rag가 찾은 답(findings)을 1~3문장으로 직접 전한다 — 질문에 대한 답이 본문이 되게 핵심을 풀어 쓴다. findings가 비어 있으면 솔직히 못 찾았다고 하고 본론으로 잇는다.
  - recall: 사용자가 직전 대화에 나온 내용을 되묻거나 확인하는 발화. recent_messages(최근 대화 이력)에서 찾아 간결하고 직접적으로 답한다(새 검색·워커 없이). 이력에 없으면 솔직히 모른다고 하고 부드럽게 잇는다.
  - explain_tool: 사용자가 이 도구·슬롯·사용법을 물었다(subject). 주어진 body(도구 설명 + 슬롯 정의 전부)를 참고해 subject가 묻는 만큼만 친근하게 답한다(새 검색·워커 없이, recent_messages도 안 씀). 특정 슬롯 하나를 물으면 그 슬롯만 한두 문장으로, 여러·모든 슬롯을 물으면 해당 슬롯들을 "- 제목: 역할" 불릿으로 빠짐없이, 도구 전반을 물으면 개요로 답한다. body에 없는 내용은 지어내지 않는다. 답한 뒤 한 문장으로 본론(계획 채우기)으로 가볍게 잇는다.
  - reason_over_context: 사용자가 이미 모은 근거·대화 내용에서 결론·문제점·시사점을 추론·도출·종합해달라고 했다(subject). 주어진 context(누적 근거: subject·cluster·agreement·findings)와 recent_messages를 근거로 직접 추론해 핵심을 1~5문장으로 정리해 전한다(새 검색·워커 없이). 가진 근거 범위에서만 추론하고 없는 사실은 지어내지 않는다. 항목이 여럿이면 "- " 불릿으로 정리한다. context가 비어 있으면 아직 모아둔 근거가 없다고 솔직히 말하고, 무엇을 먼저 찾아보면 좋을지 한 문장으로 제안한다.
  - clarify: 모호한 발화를 좁히는 질문을 한다(이게 있으면 보통 ask_slot은 보류된다).
  - redirect: 스코프 밖 발화를 부드럽게 넘기고 본론으로 잇는다.
  - deliver_plan: 슬롯이 모두 채워져 계획서를 만들 준비가 됐다고 알린다(생성은 화면의 '계획서 생성' 버튼).
  - confirm_slot: 방금 사용자가 말한 값을 슬롯에 넣기 전에 확인한다. 후보 슬롯이 둘 이상이면 그 값과 후보들을 제시하고 "어디에 넣을까요?"를 한 문장으로 묻고, 후보가 하나면 "이거 [그 슬롯]에 넣어둘까요?"처럼 넣을지 말지를 한 문장으로 묻는다(사용자가 아직 정하지 않고 떠본 값이다). 사용자가 답하기 전엔 다음 슬롯 질문(ask_slot)은 하지 않는다.
  - ask_slot: 다음 채울 슬롯을 맥락 있게 한 문장으로 묻는다(참고 예시의 톤을 살려서).

[예시]
입력 intents: [{"type":"report_findings","subject":"웹툰 시장 규모","research":["국내 웹툰 시장은 2023년 약 1.8조 원 규모로 추정된다","네이버·카카오가 거래액의 다수를 차지한다"],"rag":[],"logic_validator":[],"agreement":"confirms"}, {"type":"ask_slot","slot":"target","example":"주로 어떤 독자층을 노리고 있어?"}]
좋은 응답:
찾아봤어 — 국내 웹툰 시장은 2023년 기준 약 **1.8조 원** 규모로 추정되고, 네이버·카카오가 거래액 대부분을 가져가고 있어.
그럼 타겟으로 넘어가 볼까, 주로 어떤 독자층을 노리고 있어?

입력 intents: [{"type":"recall","subject":"내가 방금 뭐라고 했지"}]
좋은 응답: 방금 타겟을 20대 직장인으로 잡는다고 했어.

입력 intents: [{"type":"reason_over_context","subject":"여기서 도출할 문제점을 추론해줘","context":[{"subject":"국내 커피 시장 수익성","cluster":"research","agreement":"confirms","findings":["점포 포화로 가맹점 매출 감소","원두·인건비 상승"]}]}]
좋은 응답: 모은 내용을 종합하면 — 국내 커피 시장은 점포 포화와 원두·인건비 상승이 겹쳐 가맹점 수익성이 떨어지고 있어서, 단순 출점 확대보다 운영 효율화·차별화가 핵심 과제로 보여.

반드시 {"message": "..."} JSON만 출력하고, message는 위 규칙대로 intent 내용을 실제로 담은 비어있지 않은 문자열이어야 한다."""


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
    # 코드는 '재료'(도구 설명+슬롯 정의 전부)와 '질문'(subject)만 넘기고, 어느 슬롯을 얼마나
    # 답할지(특정 1개/여럿/전체/개요)는 conversation LLM이 정한다 — 표현 결정은 코드가 안 한다.
    # recall과 달리 in_scope 필터를 두지 않는다 — 도구 질문이 in_scope=false로 잘못 매겨져도 답한다
    # (아래 redirect는 tool_help 세그먼트를 건너뛰어 explain_tool이 우선한다).
    tool_helps = [
        seg for seg in segments if "tool_help" in (seg.get("utterance_types") or [])
    ]
    for seg in tool_helps:
        subj = (seg.get("canonical_text") or seg.get("text", "")).strip()
        intents.append({"type": "explain_tool", "subject": subj, "body": tool_help_text()})
        suppress_ask = True  # 도구 설명이 곧 응답 — 다음 슬롯 질문은 보류

    # 1.4) reason_over_context — 추론·도출·종합: 워커 없이 conversation이 누적 근거에서 직접 추론.
    # recall(대화 이력)·explain_tool(슬롯 정의)과 같은 '재료로 답하는' interaction이지만, 재료가
    # session_evidence(턴을 넘어 모은 research·rag·논리검증 findings)다. explain_tool이 body를 싣듯
    # 재료를 intent에 실어 넘긴다(payload 계약 유지). in_scope 필터는 두지 않는다 — off-scope로
    # 잘못 매겨져도 답한다(아래 redirect가 reason 세그먼트를 건너뛴다).
    reasons = [
        (seg.get("canonical_text") or seg.get("text", "")).strip()
        for seg in segments
        if "reason" in (seg.get("utterance_types") or [])
    ]
    reasons = [r for r in reasons if r]
    if reasons:
        evidence = state.get("session_evidence") or []
        context = [
            {
                "subject": r.get("subject", ""),
                "cluster": r.get("cluster", ""),
                "agreement": r.get("agreement", "unknown"),
                "findings": r.get("findings") or [],
            }
            for r in evidence[-15:]  # 누적 근거는 (subject,cluster) 중복제거라 보통 작다 — 최근분만 상한
            if r.get("findings")
        ]
        for subj in reasons:
            intents.append({"type": "reason_over_context", "subject": subj, "context": context})
            suppress_ask = True  # 추론 답이 곧 응답 — 다음 슬롯 질문은 보류

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
            and "reason" not in (seg.get("utterance_types") or [])
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

    # 4) clarify — 있으면 다음 슬롯 질문 보류(사용자 답 받고 다음 턴)
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

    # 5) ask_slot — 위에서 막지 않았고 확인 대기도 없으면 다음 빈칸 1개
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


# LLM이 드물게 빈/메타 자리표시자로 뭉개면(저추론 회귀의 잔상) 결정론으로 대체한다 —
# intents에 실린 내용을 직접 렌더해 빈 턴을 막는다(LLM 재호출은 안 한다 — 지연·루프 회피).
_DEGENERATE_EXACT = {"ack", "확인했습니다", "확인했어", "응답 준비됐습니다", "알겠습니다", "알겠어"}


def _is_degenerate(message: str) -> bool:
    """렌더 실패로 자리표시자/메타로 뭉개졌는지 판정 — 정상 짧은 답은 통과시킨다."""
    norm = message.strip().rstrip(" .!~").lower()
    return norm in _DEGENERATE_EXACT


def _fallback_message(intents: list[dict]) -> str:
    """LLM 렌더가 빈/메타로 실패했을 때 intent 내용을 결정론으로 한 메시지로 엮는다."""
    parts: list[str] = []
    for it in intents:
        t = it.get("type")
        if t in ("report_findings", "answer_question"):
            findings = [
                f
                for f in (
                    *(it.get("research") or []),
                    *(it.get("rag") or []),
                    *(it.get("logic_validator") or []),
                )
                if (f or "").strip()
            ]
            if findings:
                parts.append("\n".join(f"- {f}" for f in findings))
        elif t == "ask_slot":
            example = (it.get("example") or "").strip()
            if example:
                parts.append(example)
        elif t == "clarify":
            text = (it.get("text") or "").strip()
            if text:
                parts.append(text)
        elif t == "reason_over_context":
            ctx = it.get("context") or []
            findings = [
                f for r in ctx for f in (r.get("findings") or []) if (f or "").strip()
            ]
            if findings:
                parts.append("모은 내용을 정리하면:\n" + "\n".join(f"- {f}" for f in findings))
            else:
                parts.append("아직 추론에 쓸 만큼 모아둔 근거가 없어 — 먼저 어떤 점이 궁금한지 알려주면 같이 찾아볼게.")
        elif t == "deliver_plan":
            parts.append("슬롯이 다 찼어 — 화면의 '계획서 생성' 버튼으로 계획서를 만들면 돼.")
    if parts:
        return "\n\n".join(parts)
    return "방금 건 잘 못 알아들었어 — 한 번만 더 말해줄래?"


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
    out = await call_json(
        _SYSTEM, payload, ConversationOut,
        reasoning_effort=conversation_reasoning_effort(),
    )
    message = out.message.strip()
    # 저추론 회귀로 LLM이 빈/메타 자리표시자를 내면 intent 내용으로 결정론 대체.
    if not message or _is_degenerate(message):
        message = _fallback_message(intents)
    result: dict = {"pending_question": message}
    # ask_slot을 실제로 물었으면 그 슬롯을 기록 — 다음 턴 fill이 "직전 질문에 직접 답"을
    # 결정론으로 잡는다(kind=decision 기준 (b)). 안 물은 턴엔 이 키를 안 내보내 이전 값 유지.
    asked = next((i["slot"] for i in intents if i.get("type") == "ask_slot"), None)
    if asked is not None:
        result["last_asked_slot"] = asked
    return result
