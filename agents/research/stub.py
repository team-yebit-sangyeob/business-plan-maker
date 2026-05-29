"""리서치 클러스터 — stub. 실 구현은 research/{decomposer,searcher,evaluator,rewriter,reporter,curator}.

NOTE(정합성): research_spec의 입력은 VerificationRequest(claim·claim_type·slot_context·
freshness_max_days …)다. 현 stub은 claim(subject)·claim_type까지 받고, slot_context·
freshness는 아직 미수용 — 실 구현 시 나머지 필드도 맞출 것.
claim_type은 검증 강도/쿼리 분해 단서로 쓰인다(예: market_fill이면 출처 4건+ 엄격 기준).
"""
from __future__ import annotations

from common.schema import ValidationReport
from common.schema.state import ClaimType


async def run_research(
    subject: str, claim_type: ClaimType | None = None
) -> ValidationReport:
    """외부 사실 검증 (기획서 9장 리포트 양식). stub은 고정 응답.

    실 구현 반환 예시:
        run_research("한국 게임 시장이 포화 상태다", "fact") →
        {
          "subject": "한국 게임 시장이 포화 상태다",
          "findings": ["2024 모바일 게임 신규 출시 -12%", "상위 10개 매출 점유 78%"],
          "sources": ["한국콘텐츠진흥원 2024 백서", "https://..."],
          "agreement": "partial",   # 사용자 주장과 부분 일치
          "cluster": "research",
        }
    """
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 외부 데이터 검증이 아직 연결되지 않았습니다.",
        ],
        "sources": ["stub://placeholder"],
        "agreement": "unknown",
        "cluster": "research",
    }
