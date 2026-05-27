"""추론 점검 — stub. LLM 기반 논리 점검은 추후."""
from __future__ import annotations

from common.schema import ValidationReport


async def run_inference(subject: str) -> ValidationReport:
    return {
        "subject": subject[:80],
        "findings": [
            "[stub] 추론 점검 워커가 아직 연결되지 않았습니다.",
        ],
        "sources": [],
        "agreement": "unknown",
        "cluster": "inference",
    }
