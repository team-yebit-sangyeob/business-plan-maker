"""리서치 클러스터 — 실패 폴백 응답 헬퍼.

실 구현은 research_main.run_research (decomposer → searcher → reporter). 실 파이프라인이
실패하면(예: 검색 오류) 그래프를 죽이지 않도록 이 stub 응답으로 강등한다 — 데모용 가짜
데이터가 아니라 진단용 폴백이다(키 없으면 상위에서 실행 자체가 막힌다).
"""
from __future__ import annotations

from common.schema import ValidationReport


def stub_report(claim: str, *, error: str | None = None) -> ValidationReport:
    """리서치 파이프라인 실패 시 돌려줄 진단용 ValidationReport."""
    detail = f": {error[:120]}" if error else ""
    return {
        "subject": (claim or "")[:80],
        "findings": [f"[stub] 리서치 파이프라인 폴백{detail}"],
        "sources": ["stub://placeholder"],
        "agreement": "unknown",
        "cluster": "research",
    }
