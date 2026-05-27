"""다음 질문 한 줄 생성 — 3장 필수 슬롯 통과 조건 기반."""
from __future__ import annotations

from common.schema import PlanState
from agents.orchestrator.nodes.gate import required_missing, optional_missing


_REQUIRED_PROMPTS = {
    "problem": "어떤 문제예요? — 누가 · 어떤 상황에서 · 무엇 때문에 · 어떤 손실을 보는지까지 얘기해주면 좋아요.",
    "target": "타겟이 누구예요? — '어느 회사'가 아니라 그 안에서 계약서에 도장 찍는 사람·부서·규모·접촉 경로까지.",
    "goal": "목표 수치는요? — 언제까지 얼마, 그리고 어디까지 안 되면 접거나 방향을 트는지 실패 임계값도 같이.",
}

_OPTIONAL_PROMPTS = {
    "solution": "솔루션 형태는 어떻게 가져갈 거예요? (서비스 / 제품 / 플랫폼 중에)",
    "market": "시장 규모나 경쟁사 쪽은 짚어둔 데이터 있어요? 없으면 제가 찾아볼게요.",
    "revenue": "수익 모델 — 구독, 건당, 라이선싱 중 어떤 쪽 그림이에요?",
    "milestones": "마일스톤 — 언제까지 어느 단계까지 가야 한다고 보세요?",
    "risks": "걱정되는 리스크부터 하나 짚어주실래요?",
    "resources": "필요한 인력·예산 규모는 어떻게 보세요?",
}


def next_question(state: PlanState) -> str:
    missing_req = required_missing(state)
    if missing_req:
        return _REQUIRED_PROMPTS[missing_req[0]]

    missing_opt = optional_missing(state)
    if missing_opt:
        return _OPTIONAL_PROMPTS[missing_opt[0]]

    return "9개 항목 다 채워졌어요. '계획서 생성' 눌러서 뽑아볼까요?"


def conversation_node(state: PlanState) -> dict:
    output_request = state.get("output_request")
    if output_request == "type0":
        missing = required_missing(state)
        slots_kor = {
            "problem": "Problem(문제)",
            "target": "Target(타겟)",
            "goal": "Goal(목표)",
        }
        names = " · ".join(slots_kor[m] for m in missing)
        return {
            "pending_question": f"지금 뽑으면 메모지 수준이에요. {names} 채우면 바로 출력할게요.",
        }
    return {"pending_question": next_question(state)}
