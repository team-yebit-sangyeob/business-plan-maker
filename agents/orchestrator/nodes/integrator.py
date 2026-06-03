"""응답 통합기 (Fig.0 마지막 박스) — 결정론, LLM 호출 없음.

spec v0.7.5: 한 턴에 여러 처리(명확화·검증·정정)가 동시에 나도 응답은 한꺼번에
다 던지지 않는다. 그 '무엇을 어떻게 묶을지'는 이제 conversation_node가 intent 목록
(_build_intents)으로 결정하고 한 메시지로 렌더한다 — 그래서 통합기는 pending_question을
다시 만들지 않는다(대화 에이전트의 결과를 덮어쓰지 않음).

여기서는 디버깅·세션 표시에 쓰는 pending_clarifications(이번 턴 명확화 대상 목록)만
세그먼트에서 추려 기록한다.
"""
from __future__ import annotations

from common.schema import PlanState


def response_integrator_node(state: PlanState) -> dict:
    """이번 턴 명확화(clarify) 대상만 추려 기록한다 → {"pending_clarifications"}(LLM 없음)."""
    segments = state.get("turn_segments") or []
    clarifications = [
        text
        for seg in segments
        if "clarify" in (seg.get("routes") or [])
        and (text := (seg.get("canonical_text") or seg.get("text", "")).strip())
    ]
    return {"pending_clarifications": clarifications}
