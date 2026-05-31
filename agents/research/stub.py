"""리서치 클러스터 — mock/폴백 응답 헬퍼.

실 구현은 research_main.run_research (decomposer → searcher → reporter).
OPENAI 키가 없거나 BPM_LLM_MODE=mock 이거나 파이프라인이 실패하면 이 stub 응답으로
폴백한다 — 키 없이도 오케스트레이터 그래프가 끝까지 돌도록.
"""
from __future__ import annotations

from common.schema import ValidationReport


def stub_report(claim: str, *, error: str | None = None) -> ValidationReport:
    """키 없음/mock/실패 시 돌려줄 ValidationReport.

    error가 있으면 진단 메시지(파이프라인 실패 폴백). 없으면(mock/키 없음) 프론트엔드
    데모가 죽지 않게 발화 주제에 맞춘 '데모용' 외부 리서치 근거를 생성한다 — 실 검색이
    아니므로 [데모] 표식을 단다.
    """
    if error:
        return {
            "subject": (claim or "")[:80],
            "findings": [f"[stub] 리서치 파이프라인 폴백: {error[:120]}"],
            "sources": ["stub://placeholder"],
            "agreement": "unknown",
            "cluster": "research",
        }
    head = (claim or "").strip()[:36]
    return {
        "subject": (claim or "")[:80],
        "findings": [
            f"[데모] 공개 시장 자료상 '{head}' 관련 수요는 성장세지만 세그먼트별 편차가 큽니다.",
            "[데모] 유사 진입 사례에서 채널·현지화 비용이 핵심 변수로 지적됨.",
        ],
        "sources": ["[데모] industry_report_2025", "[데모] market_news"],
        "agreement": "partial",
        "cluster": "research",
    }


async def run_research(subject: str) -> ValidationReport:
    """레거시 호환 진입점 — 문자열 주제를 받아 stub 응답.

    실 파이프라인은 agents.research.research_main.run_research 를 쓴다.
    """
    return stub_report(subject)
