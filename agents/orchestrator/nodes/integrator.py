"""응답 통합기 (Fig.0 마지막 박스) — 결정론, LLM 호출 없음.

한 턴에 여러 처리가 동시에 났어도 응답은 한꺼번에 다 던지지 않는다(기획서 6장
'응답 통합 방식'): 명확화가 있으면 그것만 묻고 검증은 백그라운드로 미루고,
없으면 검증 진행 통지 + 다음 질문 1개. conversation_node가 만든 pending_question과
세그먼트의 routes(clarify=명확화, 워커 라우트=백그라운드 검증)만 보고 조합.

worked example
--------------
세그먼트: [clarification "신사업이 추상적"(routes=clarify),
          claim "게임 시장 포화"(routes=research/rag/critic),
          claim "일본서 통할 것"(routes=research/rag/critic)]
→ pending_question:
    "먼저 명확히 — 신사업이 추상적 이 부분 조금만 풀어주실래요?
     (한국 게임 시장 포화, 웹툰 IP 일본 통함 쪽은 백그라운드에서 같이 찾아볼게요.)"
  (명확화가 있으니 base_question은 보류 — 사용자 답을 받고 다음 턴에 다음 질문)

명확화 없이 검증만 있을 때:
→ "한국 게임 시장 포화 쪽은 백그라운드에서 검증 중이에요.\n<다음 질문>"
"""
from __future__ import annotations

from common.schema import PlanState

_WORKER_ROUTES = frozenset({"research", "rag", "critic"})


def response_integrator_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    base_question = (state.get("pending_question") or "").strip()

    clarifications: list[str] = []
    for seg in segments:
        if "clarify" in (seg.get("routes") or []):
            text = (seg.get("canonical_text") or seg.get("text", "")).strip()
            if text:
                clarifications.append(text)

    dispatch_subjects: list[str] = []
    for seg in segments:
        if _WORKER_ROUTES & set(seg.get("routes") or []):
            text = (seg.get("canonical_text") or seg.get("text", "")).strip()
            if text:
                dispatch_subjects.append(text)

    # 스코프 밖(무맥락·잡담) 세그먼트가 하나라도 있으면 부드럽게 되돌린다.
    # classify가 이미 routes=["none"]로 워커를 막았으므로 여기선 표현만 — 거절이 아니라
    # 다음 질문으로 자연스럽게 흐름을 잇는다(순수 off-topic 턴이면 base_question만 남음).
    has_off_topic = any(seg.get("in_scope") is False for seg in segments)

    parts: list[str] = []
    if has_off_topic:
        parts.append("그건 지금 짜는 사업 계획과는 좀 떨어진 얘기 같아 그쪽은 넘어갈게요.")
    if clarifications:
        parts.append(
            "먼저 명확히 — " + " / ".join(clarifications[:2])
            + " 이 부분 조금만 풀어주실래요?"
        )
        if dispatch_subjects:
            parts.append(
                f"({', '.join(dispatch_subjects[:2])} 쪽은 백그라운드에서 같이 찾아볼게요.)"
            )
        # 명확화가 있으면 다음 질문은 보류 (사용자 답변 받고 다음 턴)
    else:
        if dispatch_subjects:
            parts.append(
                f"{', '.join(dispatch_subjects[:2])} 쪽은 백그라운드에서 검증 중이에요."
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
