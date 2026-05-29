"""대화 에이전트 — 오케가 결정한 "어느 슬롯·어떤 의도"를 자연어 한 문장으로.

판단은 안 한다(무엇을 물을지는 오케 결정) — 표현만 한다. 채울 슬롯을 고르는 규칙은
결정론: 기본 질문 순서 = ALL_SLOTS 자연 순서의 첫 빈칸(필수/선택 안 가림). 그래서
goal(필수)도 solution·market·advantage·revenue 뒤에 물어진다 — 솔루션·수익모델을
모르면 현실적 목표 수치가 안 나오므로. 다만 출력 요청인데 필수 미달이면(type0) 막은
필수 슬롯을 콕 집어 거절. 문장 생성만 LLM.

NOTE(정합성): conversation_spec은 8+종 intent(ASK_SLOT·CLARIFY·REPORT_CRITIQUE·
REPORT_RESEARCH·PRESENT_CANDIDATES·REJECT_OUTPUT·DELIVER_PLAN·ACKNOWLEDGE)를 정의하지만,
현 stub은 슬롯 질문 + type0 거절만 처리(mode: required/optional/type0_reject).
나머지 intent는 워커(비평/리서치/후보)·Planner 연결 시 확장.

worked example
--------------
output_request=None, 필수 [target,goal] 빔 → mode="required", target_slot="target"
  → "타겟이 누구예요? — 어느 회사가 아니라 그 안에서 도장 찍는 사람·부서·규모까지."
output_request="type0", 필수 [goal] 빔 → mode="type0_reject"
  → "지금 뽑으면 메모지 수준이에요. goal만 정하면 바로 출력 가능 — 언제까지 얼마부터 정해볼까요?"
"""
from __future__ import annotations

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.state import ALL_SLOTS, REQUIRED_SLOTS
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
오케스트레이터가 결정한 "다음 채울 슬롯"을 받아 사용자에게 자연어 한 문장으로 묻는다.
- 문체는 친근한 반말~ 부드러운 존댓말 혼용. 사업 파트너처럼.
- 한 문장. 너무 길게 늘이지 말 것. 사용자 부담 최소.
- 이미 채워진 슬롯 정보를 활용해 맥락 잡힌 질문.
- output_request="type0"이면 부족한 필수 슬롯을 콕 짚어 거절 + 어떤 항목 더 필요한지 안내.

JSON만 출력."""


class ConversationOut(BaseModel):
    question: str


def _slot_snapshot(state: PlanState) -> str:
    slots = state.get("slots") or {}
    return "\n".join(
        f"- {name}: {(slots.get(name) or {}).get('value') or '[비어있음]'}"
        for name in ALL_SLOTS
    )


async def conversation_node(state: PlanState) -> dict:
    output_request = state.get("output_request")
    missing_req = required_missing(state)
    missing_opt = optional_missing(state)

    if output_request == "type0":
        # 출력 요청인데 필수 미달 → 출력을 막은 필수 슬롯을 콕 집어 거절
        target_slot = missing_req[0]
        mode = "type0_reject"
    else:
        # 기본 질문 순서 = ALL_SLOTS 자연 순서의 첫 빈칸 (required/optional 안 가림).
        # 그래서 goal(필수)도 solution·market·advantage·revenue 뒤에 물어진다.
        slots = state.get("slots") or {}
        target_slot = next(
            (s for s in ALL_SLOTS if not (slots.get(s) or {}).get("value")), None
        )
        if target_slot is None:
            # 다 찼고 출력 안 했으면 출력 권유
            return {
                "pending_question": "10개 항목 다 채워졌어요. '계획서 생성' 눌러서 뽑아볼까요?"
            }
        mode = "required" if target_slot in REQUIRED_SLOTS else "optional"

    example = _FEW_SHOT.get(target_slot, "")
    user_payload = (
        f"[모드] {mode}\n"
        f"[채울 슬롯] {target_slot}\n"
        f"[필수 부족] {', '.join(missing_req) or '없음'}\n"
        f"[선택 부족] {', '.join(missing_opt) or '없음'}\n"
        f"[현재 슬롯]\n{_slot_snapshot(state)}\n\n"
        f"[참고 예시 — 이 톤으로]\n{example}"
    )

    out = await call_json(_SYSTEM, user_payload, ConversationOut)
    return {"pending_question": out.question.strip()}
