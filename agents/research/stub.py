"""리서치 클러스터 — stub. 실 구현은 research/{decomposer,searcher,evaluator,rewriter,reporter,curator}."""
from __future__ import annotations

from common.schema import ValidationReport


async def run_research(subject: str) -> ValidationReport:
    """9장 리서치 리포트 양식. stub은 고정 응답."""
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 외부 데이터 검증이 아직 연결되지 않았습니다.",
        ],
        "sources": ["stub://placeholder"],
        "agreement": "unknown",
    }
