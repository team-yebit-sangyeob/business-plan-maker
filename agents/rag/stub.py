"""RAG 클러스터 — stub. 회사 문서 저장소(B) 미연결."""
from __future__ import annotations

from common.schema import ValidationReport


async def run_rag_check(subject: str) -> ValidationReport:
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 회사 문서 저장소(RAG)가 아직 연결되지 않았습니다.",
        ],
        "sources": ["stub://company-kb"],
        "agreement": "unknown",
        "cluster": "rag",
    }
