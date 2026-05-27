"""LangGraph 구성. 흐름 (6장):

  segment → classify → correction → fill_and_validate → gate → conversation
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from common.schema import PlanState, initial_state
from agents.orchestrator.nodes.segment import segment_node
from agents.orchestrator.nodes.classify import classify_node
from agents.orchestrator.nodes.correction import correction_node
from agents.orchestrator.nodes.router import fill_and_validate_node
from agents.orchestrator.nodes.gate import gate_node
from agents.conversation.agent import conversation_node


@lru_cache(maxsize=1)
def build_graph():
    g: StateGraph = StateGraph(PlanState)
    g.add_node("segment", segment_node)
    g.add_node("classify", classify_node)
    g.add_node("correction", correction_node)
    g.add_node("fill_and_validate", fill_and_validate_node)
    g.add_node("gate", gate_node)
    g.add_node("conversation", conversation_node)

    g.add_edge(START, "segment")
    g.add_edge("segment", "classify")
    g.add_edge("classify", "correction")
    g.add_edge("correction", "fill_and_validate")
    g.add_edge("fill_and_validate", "gate")
    g.add_edge("gate", "conversation")
    g.add_edge("conversation", END)

    return g.compile()


async def run_turn(state: PlanState, user_input: str) -> PlanState:
    """한 턴 실행. state는 이전 턴의 누적 상태."""
    graph = build_graph()
    next_state = {
        **state,
        "user_input": user_input,
        "turn": state.get("turn", 0) + 1,
        "turn_segments": [],
        "output_request": None,
    }
    result = await graph.ainvoke(next_state)
    return result
