"""대화 에이전트 — 오케가 결정한 의도(intent)들을 자연어 한 응답으로 변환.

판단은 안 한다 — 표현만 한다(conversation_spec). 무엇을 물을지·무엇을 보고할지는
state에서 결정론으로 뽑고(_build_intents), 그 intent 목록을 LLM 1회로 한 메시지로 엮는다.

지원 intent(conversation_spec TRIGGER MATRIX 전체):
  ask_slot         — 비어있는 슬롯 질문(기본 질문 순서 = ALL_SLOTS 첫 빈칸)
  clarify          — 모호한 발화 좁히기(있으면 다음 슬롯 질문은 보류)
  report_findings  — 리서치(외부)·RAG(내부)·비평(추론·정합성) 결과 전달 + 전제 교정
  answer_question  — 사용자 질문에 리서치·RAG가 찾은 답 전달(질문은 비평 미경유)
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
from common.schema.state import ALL_SLOTS
from agents.orchestrator.llm import call_json
from agents.orchestrator.nodes.gate import required_missing, optional_missing


# 질문 순서(ALL_SLOTS)대로 — 슬롯별 톤 예시
_FEW_SHOT = {
    "problem": "어떤 문제예요? — 누가 · 어떤 상황에서 · 무엇 때문에 · 어떤 손실을 보는지까지 얘기해주면 좋아요.",
    "target": "타겟이 누구예요? — '어느 회사'가 아니라 그 안에서 계약서에 도장 찍는 사람·부서·규모·접촉 경로까지.",
    "solution": "솔루션 형태는 어떻게 가져갈 거예요? (서비스 / 제품 / 플랫폼 중에)",
    "market": "시장 규모나 경쟁사 쪽은 짚어둔 데이터 있어요? 없으면 제가 찾아볼게요.",
    "advantage": "기존 대안이나 경쟁사 대비 우리만의 차별점·이기는 이유는 뭐예요?",
    "revenue": "수익 모델 — 구독, 건당, 라이선싱 중 어떤 쪽 그림이에요?",
    "goal": "목표 수치는요? — 언제까지 얼마, 그리고 어디까지 안 되면 접거나 방향을 트는지 실패 임계값도 같이.",
    "resources": "필요한 인력·예산 규모는 어떻게 보세요?",
    "milestones": "마일스톤 — 언제까지 어느 단계까지 가야 한다고 보세요?",
    "risks": "걱정되는 리스크부터 하나 짚어주실래요?",
}


_SYSTEM = """대화 에이전트
오케스트레이터가 결정한 의도(intent) 목록을 받아 사용자에게 보낼 자연어 응답 한 덩어리로 변환한다.
판단은 하지 않는다 — 주어진 intent만 자연스럽게 한 메시지로 엮어 표현한다.

- 문체: 친근한 반말~부드러운 존댓말 혼용, 사업 파트너 톤. 한두 문장 위주로 간결하게.
- 여러 intent가 오면 매끄럽게 연결한다(예: 정정 확인 → 찾은 근거 → 다음 질문).
- intent별 표현 규칙:
  - acknowledge: 사용자의 정정/확인을 짧게 받아준다.
  - report_findings: research=외부 사실, rag=회사 내부 자료, critic=추론·정합성 점검. 1~2문장으로 전달하고, 사용자 전제와 어긋나면 부드럽게 교정 제안.
  - answer_question: 사용자가 물은 것에 research·rag가 찾은 답을 전달.
  - clarify: 모호한 발화를 좁히는 질문. (이게 있으면 ask_slot은 보통 보류된다)
  - redirect: 스코프 밖 발화를 부드럽게 넘기고 본론으로 잇는다.
  - reject_output: 필수 슬롯 미달이라 지금은 출력이 이르다고 알리고, 무엇을 채우면 되는지 안내.
  - deliver_plan: 계획서를 뽑을 수 있음을 안내(type2면 빈 항목은 [미정]으로 들어간다고).
  - ask_slot: 다음 채울 슬롯을 맥락 있게 한 문장으로 묻는다(참고 예시 톤 활용).

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
        critic = clusters.get("critic")
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
                    for rep in (research, critic, rag)
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
                    "critic": (critic or {}).get("findings", []),
                    "agreement": agreement,
                }
            )

    # 4) 출력 게이트
    suppress_ask = False
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

    # 6) ask_slot — 위에서 막지 않았으면 다음 빈칸 1개
    if not suppress_ask:
        if next_empty is None:
            intents.append({"type": "deliver_plan", "output_type": "ready", "empty_slots": []})
        else:
            intents.append(
                {"type": "ask_slot", "slot": next_empty, "example": _FEW_SHOT.get(next_empty, "")}
            )

    return intents


def _slot_values(state: PlanState) -> dict:
    slots = state.get("slots") or {}
    return {name: (slots.get(name) or {}).get("value") for name in ALL_SLOTS}


async def conversation_node(state: PlanState) -> dict:
    intents = _build_intents(state)
    payload = json.dumps(
        {"tone": "casual_business", "slots": _slot_values(state), "intents": intents},
        ensure_ascii=False,
    )
    out = await call_json(_SYSTEM, payload, ConversationOut)
    message = out.message.strip()
    return {"pending_question": message}
