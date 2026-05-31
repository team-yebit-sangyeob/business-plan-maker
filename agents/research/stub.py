"""리서치 클러스터 — mock/폴백 응답 헬퍼.

실 구현은 research_main.run_research (decomposer → searcher → reporter).
OPENAI 키가 없거나 BPM_LLM_MODE=mock 이거나 파이프라인이 실패하면 이 stub 응답으로
폴백한다 — 키 없이도 오케스트레이터 그래프가 끝까지 돌도록.
"""
from __future__ import annotations

from common.schema import ValidationReport


def stub_report(claim: str, *, error: str | None = None) -> ValidationReport:
    """키 없음/mock/실패 시 돌려줄 고정 ValidationReport."""
    if error:
        msg = f"[stub] 리서치 파이프라인 폴백: {error[:120]}"
    else:
        msg = "[stub] 외부 데이터 검증 비활성화 (OPENAI 키 없음 또는 mock 모드)."
    return {
        "subject": (claim or "")[:80],
        "findings": [msg],
        "sources": ["stub://placeholder"],
        "agreement": "unknown",
        "cluster": "research",
    }


async def run_research(subject: str) -> ValidationReport:
    """레거시 호환 진입점 — 문자열 주제를 받아 stub 응답.

    실 파이프라인은 agents.research.research_main.run_research 를 쓴다.
    """
    return stub_report(subject)
