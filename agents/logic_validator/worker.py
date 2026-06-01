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
    }


def _format_research_evidence(report: Optional[ValidationReport]) -> Optional[str]:
    """research ValidationReport → validator 프롬프트용 텍스트(없으면 None).

    findings(list[str]) + sources(list[str])만 사람이 읽는 형태로 묶는다. agreement는
    reporter의 거친 자동 플래그라 validator 판정 편향을 줄이려 일부러 제외한다(findings/sources로
    충분). validator는 이 텍스트만 받으므로 ValidationReport 스키마에 의존하지 않는다.
    """
    if not report:
        return None
    findings = [f.strip() for f in (report.get("findings") or []) if f and f.strip()]
    sources = [s.strip() for s in (report.get("sources") or []) if s and s.strip()]
    if not findings and not sources:
        return None
    parts: list[str] = []
    if findings:
        parts.append("findings:\n" + "\n".join(f"- {f}" for f in findings))
    if sources:
        parts.append("sources:\n" + "\n".join(f"- {s}" for s in sources))
    return "\n\n".join(parts)


async def run_logic_validator(
    subject: str,
    rag_result: Optional["RagExtractorResult"] = None,
    research_report: Optional[ValidationReport] = None,
) -> ValidationReport:
    """RAG 근거(+선택적 research) ↔ claim 논리 지지 판정 → ValidationReport(cluster="logic_validator").

    rag_result는 dispatch가 1단계 RAG에서 받아 넘긴 RagExtractorResult. 없으면(RAG 근거
    미확보) 판정 대상이 없으므로 '근거 없음'으로 돌려준다(그래프는 막지 않음).
    research_report는 같은 세그먼트의 1단계 research 산출물(claim 라우트일 때만 존재). 있으면
    findings/sources를 텍스트로 묶어 validator에 보조 근거로 넘긴다(없으면 사내 근거만으로 판정).
    """
    subject = (subject or "").strip()
    if rag_result is None:
        return _report(subject, ["검증할 RAG 근거가 없습니다."], [], "unknown")

    try:
        from agents.validator.validator import run_validator

        research_evidence = _format_research_evidence(research_report)
        vres, _ = await asyncio.to_thread(
            run_validator, rag_result, research_evidence, False
        )
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
