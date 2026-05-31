"""RAG 클러스터 — 단일 LLM 호출 시뮬레이션 stub.

실 구현은 6-서브 파이프라인(쿼리 분해 → 검색 라우터 → 벡터 검색 → 평가 → 재작성 →
컨텍스트 통합)으로 사내 문서(정책·로드맵·재무·인사)를 회수한다(rag_spec). 회사
문서 저장소(B 인프라)가 아직 없으므로, 여기서는 그 전체 흐름을 LLM 1회 호출로
'흉내'낸다 — 발화 주제에 대해 사내 자료에서 나왔을 법한 구체적 근거 + 그럴듯한
문서 출처를 생성. 프론트엔드가 실제처럼 end-to-end로 돌려볼 수 있게 하기 위함.

NOTE(정합성): rag_spec의 실 입력은 RagQuery(intent·slot_context·hint_collections …)
이지만, dispatch가 아직 subject:str만 넘기므로 시그니처는 그대로 둔다. 실 구현 시
스키마를 맞출 것. mock 모드(키 없음)에서는 mocks.py의 데모 핸들러가 응답한다.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from common.schema import ValidationReport
from agents.orchestrator.llm import call_json


logger = logging.getLogger(__name__)

_AGREEMENTS = {"confirms", "contradicts", "partial", "unknown"}

# 첫 줄이 mock 핸들러 key (mocks._rag와 일치해야 함).
_SYSTEM = """RAG 시뮬레이션
회사 내부 문서 저장소(정책·로드맵·재무·인사)를 검색하는 RAG 클러스터를 한 번의 호출로 시뮬레이션한다.
실제 문서 검색 대신, 발화 주제에 대해 회사 내부 자료에서 나왔을 법한 구체적 근거를 만들어낸다.

- findings: 회사 내부 관점의 사실 2~4개. 조직 규모·예산·로드맵 방향·과거 선례 등 구체 수치를 섞어라.
- sources: 그럴듯한 사내 문서 파일명 1~3개 (예: sales_org_2026.pdf, q1_roadmap.md, hr_headcount.xlsx).
- agreement: 사용자 주장과 회사 자료의 일치도 — confirms / contradicts / partial / unknown 중 하나.

이건 데모용 시뮬레이션이다. 단정적 사실로 오인되지 않게 '사내 추정' 톤을 유지하라.
JSON만 출력."""


class _RagOut(BaseModel):
    findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    agreement: str = "unknown"


def _fallback(subject: str) -> ValidationReport:
    return {
        "subject": subject[:80],
        "findings": ["[stub] 회사 문서 저장소(RAG) 시뮬레이션을 생성하지 못했습니다."],
        "sources": ["stub://company-kb"],
        "agreement": "unknown",
        "cluster": "rag",
    }


async def run_rag_check(subject: str) -> ValidationReport:
    """회사 내부 문서 정합성 근거 회수 — LLM 1회로 시뮬레이션.

    반환 예시 ("B2B 영업 인프라가 강하다"):
        {
          "subject": "B2B 영업 인프라가 강하다",
          "findings": ["영업팀 12명, B2C 중심 운영", "B2B 전담 0명"],
          "sources": ["sales_org_2026.pdf", "q1_roadmap.md"],
          "agreement": "contradicts",   # 사용자 주장과 충돌 → 비평이 이 근거로 판단
          "cluster": "rag",
        }
    """
    subject = (subject or "").strip()
    if not subject:
        return _fallback(subject)

    user = (
        "아래 발화 주제와 맞물리는 회사 내부 자료(조직·예산·로드맵·선례)의 근거를 시뮬레이션하라.\n\n"
        f"[발화 주제]\n{subject}"
    )
    try:
        out = await call_json(_SYSTEM, user, _RagOut)
    except Exception as exc:  # 그래프를 죽이지 않고 폴백.
        logger.warning("RAG 시뮬레이션 실패 → stub 폴백: %s", exc)
        return _fallback(subject)

    findings = [f.strip() for f in out.findings if f.strip()]
    if not findings:
        return _fallback(subject)
    agreement = out.agreement if out.agreement in _AGREEMENTS else "unknown"
    return {
        "subject": subject[:80],
        "findings": findings,
        "sources": [s.strip() for s in out.sources if s.strip()],
        "agreement": agreement,  # type: ignore[typeddict-item]
        "cluster": "rag",
    }
