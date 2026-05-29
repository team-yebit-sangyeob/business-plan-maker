"""LangGraph 구성 (Fig.0 우선순위 토폴로지, spec v0.7.5):

  segment → classify
          → correction              (correction 라벨 세그먼트 처리)
          → clarify_gate            (clarify 라우트만 있고 워커 라우트 없으면 dispatch 우회)
            ├ dispatch+fills 경로   (워커 라우트 발견 → 리서치/RAG/비평 호출)
            └ skip 경로             (명확화 우선)
          → gate → conversation → integrator → END

처리 순서·분기는 별도 priority 필드 없이 세그먼트의 routes/utterance_types에서
직접 파생한다(워커 호출은 routes, 정정 처리는 utterance_types).

end-to-end trace 예시 (turn 5, "카카오는 빼자. 예산은 1억으로 가자."):
  segment      → [seg1 "카카오는 빼자"(hints=correction), seg2 "예산 1억으로 가자"]
  classify     → seg1=["correction"](routes=none), seg2=["claim"](routes=research/rag/critic)
  correction   → target "네이버·카카오" → "네이버" (correction_log에 기록)
  clarify_gate → clarify 라우트 없고 워커 라우트 있음 → "dispatch"
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


_WORKER_ROUTES = frozenset({"research", "rag", "critic"})


def _clarify_branch(state: PlanState) -> Literal["dispatch", "gate"]:
    """명확화(clarify 라우트)만 있고 부를 워커가 하나도 없으면 디스패치 우회.

    dispatch_node와 같은 기준(워커 라우트 유무)으로 판단해 일관성 유지 —
    부를 워커가 있으면(claim·question·opinion 등) 명확화가 섞여 있어도 dispatch로 보낸다.
    """
    segments = state.get("turn_segments") or []
    has_clarify = any("clarify" in (s.get("routes") or []) for s in segments)
    has_dispatch = any(_WORKER_ROUTES & set(s.get("routes") or []) for s in segments)
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
