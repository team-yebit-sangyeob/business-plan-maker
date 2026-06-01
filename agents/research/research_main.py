"""리서치 클러스터 진입점 — 외부 사실 검증 파이프라인.

흐름: VerificationRequest → 분해기 → 검색·평가·재작성 루프 → 리포터 → ValidationReport.

dispatch가 research·rag를 asyncio.gather로 병렬 호출하므로, 동기인 OpenAI/Tavily
호출은 asyncio.to_thread로 감싸 이벤트 루프를 막지 않는다(병렬성 보존).
파이프라인이 실패하면 stub 응답으로 폴백 — 그래프는 안 죽는다(복원력). 키가 없으면 상위
(call_json·서버 기동)에서 실행 자체가 막히므로 여기엔 키리스/mock 분기가 없다.
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
from agents.research._util import traceable


logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

_DEFAULT_FRESHNESS_DAYS = 180


def _make_client() -> Any:
    """OpenAI 클라이언트. LangSmith가 있으면 wrap_openai로 감싸 호출을 트레이스에 남긴다."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:  # langsmith 미설치/래핑 실패 — 트레이싱만 포기, 파이프라인은 계속.
        return client


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


@traceable(name="research", run_type="chain")
async def run_research(req: VerificationRequest | str) -> ValidationReport:
    """외부 사실 검증 → ValidationReport(cluster="research")."""
    req = _normalize(req)
    claim = (req.get("claim") or "").strip()

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
