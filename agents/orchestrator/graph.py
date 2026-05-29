"""LangGraph 구성 (Fig.0 우선순위 토폴로지, spec v0.7.5):

  segment → classify
          → correction              (priority=0 처리)
          → clarify_gate            (priority=1만 있으면 dispatch 우회)
            ├ dispatch+fills 경로   (priority=2 발견 → 리서치/RAG/비평 호출)
            └ skip 경로             (명확화 우선)
          → gate → conversation → integrator → END

end-to-end trace 예시 (turn 5, "카카오는 빼자. 예산은 1억으로 가자."):
  segment      → [seg1 "카카오는 빼자"(hints=correction), seg2 "예산 1억으로 가자"]
  classify     → seg1=["correction"](p0), seg2=["claim"](p2, routes=research/rag/critic)
  correction   → target "네이버·카카오" → "네이버" (correction_log에 기록)
  clarify_gate → p1 없고 p2 있음 → "dispatch"
  dispatch     → seg2 canonical을 research·rag·critic 병렬 호출 → validation_reports 누적
  extract_fills→ 빈 슬롯에 "예산 1억" 채울 수 있으면 resources 등에 반영
  gate         → "가자"는 출력요청 아님 → output_request=None
  conversation → 다음 빈 필수/선택 슬롯 1개 질문 생성
  integrator   → 검증 백그라운드 통지 + 그 질문을 한 문단으로 → pending_question
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import StateGraph, START, END

from common.schema import PlanState, Message
from agents.orchestrator.nodes.segment import segment_node
from agents.orchestrator.nodes.classify import classify_node
from agents.orchestrator.nodes.correction import (
    correction_node,
    extract_slot_fills_node,
)
from agents.orchestrator.nodes.dispatch import parallel_dispatch_workers_node
from agents.orchestrator.nodes.gate import gate_node
from agents.orchestrator.nodes.integrator import response_integrator_node
from agents.conversation.agent import conversation_node


def _clarify_branch(state: PlanState) -> Literal["dispatch", "gate"]:
    """우선순위 1(명확화)만 있고 우선순위 2(워커 호출)는 없으면 디스패치 우회."""
    segments = state.get("turn_segments") or []
    has_clarify = any(s.get("priority") == 1 for s in segments)
    has_dispatch = any(s.get("priority") == 2 for s in segments)
    if has_clarify and not has_dispatch:
        return "gate"
    return "dispatch"


@lru_cache(maxsize=1)
def build_graph():
    g: StateGraph = StateGraph(PlanState)
    g.add_node("segment", segment_node)
    g.add_node("classify", classify_node)
    g.add_node("correction", correction_node)
    g.add_node("dispatch", parallel_dispatch_workers_node)
    g.add_node("extract_fills", extract_slot_fills_node)
    g.add_node("gate", gate_node)
    g.add_node("conversation", conversation_node)
    g.add_node("integrator", response_integrator_node)

    g.add_edge(START, "segment")
    g.add_edge("segment", "classify")
    g.add_edge("classify", "correction")
    g.add_conditional_edges(
        "correction",
        _clarify_branch,
        {"dispatch": "dispatch", "gate": "gate"},
    )
    g.add_edge("dispatch", "extract_fills")
    g.add_edge("extract_fills", "gate")
    g.add_edge("gate", "conversation")
    g.add_edge("conversation", "integrator")
    g.add_edge("integrator", END)

    return g.compile()


async def run_turn(state: PlanState, user_input: str) -> PlanState:
    """한 턴 실행. state는 이전 턴의 누적 상태."""
    graph = build_graph()
    turn = state.get("turn", 0) + 1
    messages = list(state.get("messages") or [])
    messages.append(Message(role="user", content=user_input, turn=turn))

    next_state = {
        **state,
        "user_input": user_input,
        "turn": turn,
        "messages": messages,
        "turn_segments": [],
        "output_request": None,
        "pending_clarifications": [],
    }
    result: PlanState = await graph.ainvoke(next_state)

    # 어시스턴트 응답 적재
    answer = (result.get("pending_question") or "").strip()
    if answer:
        out_messages = list(result.get("messages") or messages)
        out_messages.append(Message(role="assistant", content=answer, turn=turn))
        result = {**result, "messages": out_messages}
    return result
