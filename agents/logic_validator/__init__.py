"""논리 검증(logic_validator) 워커 — 구 critic 클러스터.

RAG가 회수한 근거가 claim을 논리적으로 지지하는지 판정한다(verdict→agreement). 판정
엔진은 agents/validator/(run_validator)이며, 여기서는 오케스트레이터 결선용 어댑터만 export.
"""
from agents.logic_validator.worker import run_logic_validator

__all__ = ["run_logic_validator"]
