"""비평(Critic) — stub. 추론 점검 + 정합성 판단 (spec v0.7.5).

라벨링된 발화(`subject`)와 현재 슬롯 상태(read-only)를 입력으로 받음.
실 로직(논리 비약·전제 충돌 탐지)은 추후. 분류는 일절 수행하지 않음.
"""
from __future__ import annotations

from common.schema import ValidationReport
from common.schema.state import Slot


async def run_critic(subject: str, slots: dict[str, Slot]) -> ValidationReport:
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 비평(Critic) 에이전트가 아직 연결되지 않았습니다.",
        ],
        "sources": [],
        "agreement": "unknown",
        "cluster": "critic",
    }
