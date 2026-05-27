"""응답 통합기 (Fig.0 마지막 박스) — 결정론.

명확화 우선 + 검증 백그라운드 통지 + 다음 질문 1개를 한 문단으로 합친다.
LLM 호출 없이 conversation_node 결과와 segment 분류 결과만 보고 조합.
"""
from __future__ import annotations

from common.schema import PlanState


def response_integrator_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    base_question = (state.get("pending_question") or "").strip()

    clarifications: list[str] = []
    for seg in segments:
        if seg.get("priority") == 1:
            text = (seg.get("canonical_text") or seg.get("text", "")).strip()
            if text:
                clarifications.append(text)

    verify_subjects: list[str] = []
    for seg in segments:
        if seg.get("priority") == 2 and seg.get("routes"):
            text = (seg.get("canonical_text") or seg.get("text", "")).strip()
            if text:
                verify_subjects.append(text)

    parts: list[str] = []
    if clarifications:
        parts.append(
            "먼저 명확히 — " + " / ".join(clarifications[:2])
            + " 이 부분 조금만 풀어주실래요?"
        )
        if verify_subjects:
            parts.append(
                f"({', '.join(verify_subjects[:2])} 쪽은 백그라운드에서 같이 찾아볼게요.)"
            )
        # 명확화가 있으면 다음 질문은 보류 (사용자 답변 받고 다음 턴)
    else:
        if verify_subjects:
            parts.append(
                f"{', '.join(verify_subjects[:2])} 쪽은 백그라운드에서 검증 중이에요."
            )
        if base_question:
            parts.append(base_question)

    final = "\n".join(p for p in parts if p).strip()
    if not final:
        final = base_question

    return {
        "pending_question": final,
        "pending_clarifications": clarifications,
    }
