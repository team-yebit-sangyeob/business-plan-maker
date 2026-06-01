"""LangGraph Studio(`langgraph dev`) 진입점.

오케스트레이터 그래프를 '컴파일된 그래프' 변수로 노출한다. langgraph.json이
이 `graph`를 참조해 Studio에서 토폴로지 시각화·노드 스텝 디버깅에 사용한다.
프로덕션 경로(graph.py → run_turn)는 건드리지 않아 import-time 부작용을 분리한다.
"""
from agents.orchestrator.graph import build_graph

graph = build_graph()
