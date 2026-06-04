"""논리 검증 워커 (구 critic 클러스터) — RAG 근거가 claim을 논리적으로 지지하는지 판정.

dispatch의 2단계에서 호출된다. 1단계 RAG(rag_extractor) 결과(RagExtractorResult)를 받아
validator 엔진(agents/validator/run_validator)으로 verdict를 내고 ValidationReport로 매핑한다.
판정 엔진은 agents/validator/에 그대로 있고, 여기서는 오케스트레이터 seam(ValidationReport)에
맞춰 어댑트만 한다. 동기·블로킹인 run_validator는 asyncio.to_thread로 감싸 이벤트 루프를
막지 않는다(dispatch 병렬성 보존).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from common.schema import ValidationReport

if TYPE_CHECKING:
    from agents.rag.rag_extractor import RagExtractorResult


logger = logging.getLogger(__name__)

# validator verdict → ValidationReport.agreement.
# insufficient(관련은 있으나 결론 불충분)→partial, unrelated(논리 무관)→unknown.
_VERDICT_TO_AGREEMENT = {
    "supports": "confirms",
    "contradicts": "contradicts",
    "insufficient": "partial",
    "unrelated": "unknown",
}


def _report(
    subject: str, findings: list[str], sources: list[str], agreement: str
) -> ValidationReport:
    return {
        "subject": (subject or "")[:80],
        "findings": findings,
        "sources": sources,
        "agreement": agreement,  # type: ignore[typeddict-item]
        "cluster": "logic_validator",
        # 자기 출처는 만들지 않는다 — 판정 대상인 RAG 청크는 이미 rag 워커가 인용했다(중복 방지).
        # logic_validator는 agreement(판정)·findings(근거 추론)만 기여한다.
        "citations": [],
    }


async def run_logic_validator(
    subject: str,
    rag_result: Optional["RagExtractorResult"] = None,
) -> ValidationReport:
    """RAG 근거 ↔ claim 논리 지지 판정 → ValidationReport(cluster="logic_validator").

    rag_result는 dispatch가 1단계 RAG에서 받아 넘긴 RagExtractorResult. RAG 근거가 있을 때만
    dispatch가 이 워커를 부른다(research 전용 모드나 RAG가 못 찾은 세그먼트는 디스패치에서 제외).
    외부 리서치 근거판단은 research 워커가 자체 agreement로 따로 낸다 — 여기선 사내 RAG 근거만
    판정해 두 판단을 분리한다. rag_result=None은 정상 경로엔 오지 않지만 방어적으로 '근거 없음'을 낸다.
    """
    subject = (subject or "").strip()
    if rag_result is None:
        return _report(subject, ["검증할 RAG 근거가 없습니다."], [], "unknown")

    try:
        from agents.validator.validator import run_validator

        vres, _ = await asyncio.to_thread(run_validator, rag_result, None, False)
    except Exception as exc:  # 판정 실패도 그래프를 죽이지 않고 표시만 한다.
        logger.warning("logic_validator 실패 → 폴백: %s", exc)
        return _report(subject, [f"논리 검증을 완료하지 못했습니다: {exc}"], [], "unknown")

    reasoning = (vres.get("reasoning") or "").strip()
    evidence = [e.strip() for e in (vres.get("evidence_used") or []) if e and e.strip()]
    findings = [f for f in [reasoning, *evidence[:2]] if f] or ["논리 검증 결과가 비어 있습니다."]

    source_file = (rag_result.get("source_file") or "").strip()
    source_page = (rag_result.get("source_page") or "").strip()
    sources = [f"{source_file} (p.{source_page})".strip()] if source_file else []

    agreement = _VERDICT_TO_AGREEMENT.get(vres.get("verdict", ""), "unknown")
    return _report(subject, findings, sources, agreement)
