"""RAG 워커 — 사내 벡터DB 근거 회수(retrieval).

dispatch의 1단계에서 호출된다. rag_extractor(run_rag_extractor)로 claim에 대한 사내 문서
근거를 찾아 ValidationReport(cluster="rag")로 매핑하고, 2단계 logic_validator가 판정에 쓰도록
RagExtractorResult 원본을 함께 돌려준다. 판정(agreement)은 logic_validator의 몫이므로 여기서는
agreement="unknown"으로 둔다(retrieval만). 동기·블로킹인 run_rag_extractor는 asyncio.to_thread로
감싸 이벤트 루프를 막지 않는다(dispatch 병렬성 보존).
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import Optional, Tuple, TYPE_CHECKING

from common.schema import Citation, ValidationReport

if TYPE_CHECKING:
    from agents.rag.rag_extractor import RagExtractorResult


logger = logging.getLogger(__name__)


def _report(
    subject: str,
    findings: list[str],
    sources: list[str],
    citations: Optional[list[Citation]] = None,
) -> ValidationReport:
    return {
        "subject": (subject or "")[:80],
        "findings": findings,
        "sources": sources,
        "agreement": "unknown",  # 판정은 logic_validator의 몫 — 여기선 retrieval만.
        "cluster": "rag",
        "citations": citations or [],
    }


async def run_rag_check(
    subject: str,
) -> Tuple[ValidationReport, Optional["RagExtractorResult"]]:
    """사내 문서에서 claim 근거를 회수 → (ValidationReport(cluster="rag"), RagExtractorResult).

    근거를 못 찾으면(FallbackRequired) 또는 실패하면 rag_result=None으로 돌려준다 —
    2단계 logic_validator는 None을 받으면 '근거 없음'으로 처리한다.
    """
    subject = (subject or "").strip()
    if not subject:
        return (_report(subject, ["검증할 발화가 비어 있습니다."], []), None)

    try:
        from agents.rag.rag_extractor import FallbackRequired, run_rag_extractor
    except Exception as exc:  # import 실패(의존성 등) — 그래프는 살린다.
        logger.warning("rag_extractor import 실패 → 폴백: %s", exc)
        return (_report(subject, [f"RAG 모듈 로드 실패: {exc}"], []), None)

    try:
        result, _, _ = await asyncio.to_thread(
            run_rag_extractor, subject, None, None, False
        )
    except FallbackRequired:
        return (_report(subject, ["사내 문서에서 관련 근거를 찾지 못했습니다."], []), None)
    except Exception as exc:  # 회수 실패도 그래프를 죽이지 않고 폴백.
        logger.warning("RAG 회수 실패 → 폴백: %s", exc)
        return (_report(subject, [f"RAG 회수 중 오류: {exc}"], []), None)

    highlight = (result.get("highlight") or "").strip()
    reason = (result.get("highlight_reason") or "").strip()
    findings = [f for f in [highlight, reason] if f] or ["근거 하이라이트가 비어 있습니다."]

    source_file = (result.get("source_file") or "").strip()
    source_page = (result.get("source_page") or "").strip()
    sources = [f"{source_file} (p.{source_page})".strip()] if source_file else []

    # 구조화 출처 — RagExtractorResult가 이미 쥐고 있던 파일/페이지/폴더/원문을 보존한다(평탄화 중단).
    # similarity_pct는 검색 툴 루프 안에서만 살고 RagExtractorResult엔 안 실리므로 score_kind="none".
    citations: list[Citation] = []
    if source_file:
        raw = (result.get("raw_source") or "").strip()
        citations.append(
            {
                "cluster": "rag",
                "title": source_file,
                "url": "",
                "snippet": highlight or raw[:200],
                "source_file": source_file,
                "page": source_page,
                "folder": (result.get("folder_searched") or "").strip(),
                "score": 0.0,
                "score_kind": "none",
                "accessed_at": date.today().isoformat(),
            }
        )

    return (_report(subject, findings, sources, citations), result)
