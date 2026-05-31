"""리서치 클러스터 진입점 — 외부 사실 검증 파이프라인.

흐름: VerificationRequest → 분해기 → 검색·평가·재작성 루프 → 리포터 → ValidationReport.

dispatch가 research·rag를 asyncio.gather로 병렬 호출하므로, 동기인 OpenAI/Tavily
호출은 asyncio.to_thread로 감싸 이벤트 루프를 막지 않는다(병렬성 보존).
키가 없거나 mock이거나 파이프라인이 실패하면 stub 응답으로 폴백 — 그래프는 안 죽는다.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from common.schema import ValidationReport, VerificationRequest
from agents.research.stub import stub_report
from agents.research.decomposer import decompose
from agents.research.searcher import gather_evidence
from agents.research.reporter import write_report


logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

_DEFAULT_FRESHNESS_DAYS = 180


def _should_stub() -> bool:
    """키 없음 또는 mock 모드면 실 파이프라인을 돌리지 않는다."""
    if os.environ.get("BPM_LLM_MODE", "").strip().lower() == "mock":
        return True
    return not os.environ.get("OPENAI_API_KEY")


def _make_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _normalize(req: VerificationRequest | str) -> VerificationRequest:
    """문자열로 들어와도(레거시 호출) VerificationRequest로 정규화."""
    if isinstance(req, str):
        return {"claim": req}
    return req


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _run_pipeline(client: Any, req: VerificationRequest, claim: str) -> dict[str, Any]:
    """동기 파이프라인 본체 — to_thread에서 한 번에 돌린다(루프 1회 점유)."""
    freshness = req.get("freshness_max_days") or _DEFAULT_FRESHNESS_DAYS
    subqueries = decompose(client, req, model=MODEL)
    evidence = gather_evidence(client, subqueries, freshness_days=freshness, model=MODEL)
    return write_report(client, claim, evidence, model=MODEL)


async def run_research(req: VerificationRequest | str) -> ValidationReport:
    """외부 사실 검증 → ValidationReport(cluster="research")."""
    req = _normalize(req)
    claim = (req.get("claim") or "").strip()

    if _should_stub():
        return stub_report(claim)

    try:
        client = _make_client()
        data = await asyncio.to_thread(_run_pipeline, client, req, claim)
    except Exception as exc:  # 파이프라인 실패 시 그래프를 죽이지 않고 폴백.
        logger.warning("리서치 파이프라인 실패 → stub 폴백: %s", exc)
        return stub_report(claim, error=str(exc))

    findings = data.get("findings") or ["외부 근거를 확보하지 못했습니다."]
    return {
        "subject": claim[:80],
        "findings": findings,
        "sources": _dedupe(data.get("sources") or []),
        "agreement": data.get("agreement", "unknown"),
        "cluster": "research",
    }
