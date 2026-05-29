"""RAG 클러스터 — stub. 회사 문서 저장소(B) 미연결.

NOTE(정합성): rag_spec은 입력이 RagQuery(intent·slot_context·hint_collections …)이고
검색 라우터가 정책/로드맵/재무/인사 컬렉션을 고르지만, 현 stub은 subject:str만 받는다.
실 구현 시 스키마를 맞출 것.
"""
from __future__ import annotations

from common.schema import ValidationReport


async def run_rag_check(subject: str) -> ValidationReport:
    """회사 내부 문서 정합성 근거 회수. stub은 고정 응답.

    실 구현 반환 예시:
        run_rag_check("B2B 영업 인프라가 강하다") →
        {
          "subject": "B2B 영업 인프라가 강하다",
          "findings": ["영업팀 12명, B2C 중심 운영", "B2B 전담 0명"],
          "sources": ["sales_org_2026.pdf", "q1_roadmap.md"],
          "agreement": "contradicts",   # 사용자 주장과 충돌 → 비평이 이 근거로 판단
          "cluster": "rag",
        }
    """
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 회사 문서 저장소(RAG)가 아직 연결되지 않았습니다.",
        ],
        "sources": ["stub://company-kb"],
        "agreement": "unknown",
        "cluster": "rag",
    }
