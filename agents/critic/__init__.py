"""비평(Critic) 에이전트 — 추론 점검 · 정합성 판단 (spec v0.7.5).

분류는 수행하지 않음. 라벨링된 발화 + 슬롯 상태(read-only)를 입력으로 받음.
"""
from agents.critic.stub import run_critic

__all__ = ["run_critic"]
