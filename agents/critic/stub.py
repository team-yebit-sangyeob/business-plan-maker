"""비평(Critic) — 단일 LLM 호출 시뮬레이션 stub (spec v0.7.5).

실 구현은 추론 점검(전제→결론 비약·누락 변수)과 정합성 판단(근거 충돌)을 별도
모듈로 돌려 CritiqueResult(reasoning·consistency·severity)를 낸다(critic_spec). 여기서는
그 판단을 LLM 1회 호출로 '흉내'내 ValidationReport로 돌려준다 — 프론트엔드 end-to-end용.

입력은 라벨링된 발화(`subject`) + 현재 슬롯 상태(read-only) + 1단계 산출물
(research_report·rag_context). dispatch가 리서치·RAG를 먼저 끝낸 뒤 그 결과를 넘긴다
(critic_spec §6 "리서치·RAG 병렬 → 비평 후속"). 비평은 막지 않고 '표시'만 한다.

NOTE(정합성): 실 구현 시 입력 CriticInput(label·requested_modes·research_report·
rag_context …)·출력 CritiqueResult(severity[info/warn/critical])로 스키마를 맞출 것.
mock 모드(키 없음)에서는 mocks.py의 데모 핸들러가 응답한다.
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from common.schema import ValidationReport
from common.schema.state import Slot
from agents.orchestrator.llm import call_json


logger = logging.getLogger(__name__)

_AGREEMENTS = {"confirms", "contradicts", "partial", "unknown"}

# 첫 줄이 mock 핸들러 key (mocks._critic와 일치해야 함).
_SYSTEM = """비평 시뮬레이션
추론 비약과 정합성 충돌을 점검하는 비평(Critic) 에이전트를 한 번의 호출로 시뮬레이션한다.
입력: 사용자 발화 + 현재 슬롯 상태 + (선택) 리서치/RAG 근거.

- findings: 1~3개. (1) 추론 점검 — 전제에서 결론으로의 비약·누락 변수, (2) 정합성 — 발화가 동반 근거(리서치/RAG)나 다른 슬롯과 충돌하는 지점. 단정 말고 '의심'으로 표시.
- agreement: 발화와 근거의 정합도 — confirms / contradicts / partial / unknown 중 하나.
- sources: 보통 비움 — 비평은 새 출처를 만들지 않고 받은 근거를 가리킨다.

비평은 사용자를 가로막지 않는다. 다음 행동(질문할지·진행할지)은 오케스트레이터·대화의 몫.
JSON만 출력."""


class _CriticOut(BaseModel):
    findings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    agreement: str = "unknown"


def _fallback(subject: str, mode: str, basis: str) -> ValidationReport:
    return {
        "subject": subject[:80],
        "findings": [f"[stub] 비평(Critic) 시뮬레이션을 생성하지 못했습니다. 모드={mode}{basis}."],
        "sources": [],
        "agreement": "unknown",
        "cluster": "critic",
    }


def _filled_slot_lines(slots: dict[str, Slot]) -> str:
    lines = [
        f"- {name}: {s.get('value')}"
        for name, s in (slots or {}).items()
        if (s or {}).get("value")
    ]
    return "\n".join(lines) or "(채워진 슬롯 없음)"


def _evidence_block(label: str, report: Optional[ValidationReport]) -> str:
    if not report:
        return ""
    findings = "; ".join(report.get("findings") or [])
    return f"\n[{label} 근거]\n{findings}"


async def run_critic(
    subject: str,
    slots: dict[str, Slot],
    research_report: Optional[ValidationReport] = None,
    rag_context: Optional[ValidationReport] = None,
) -> ValidationReport:
    """추론 비약·전제 충돌 표시(막지는 않음) — LLM 1회로 시뮬레이션.

    research_report·rag_context가 있으면 정합성(consistency) 비교의 근거가 되고,
    없으면 추론(reasoning) 점검만 수행한다(슬롯 상태 자체의 양립 가능성).
    """
    subject = (subject or "").strip()
    received = [
        name
        for name, report in (("research", research_report), ("rag", rag_context))
        if report is not None
    ]
    mode = "정합성+추론" if received else "추론"
    basis = f" (근거 입력: {', '.join(received)})" if received else " (근거 입력 없음 — 추론만)"

    if not subject:
        return _fallback(subject, mode, basis)

    user = (
        f"[발화]\n{subject}\n\n"
        f"[현재 슬롯]\n{_filled_slot_lines(slots)}"
        + _evidence_block("리서치", research_report)
        + _evidence_block("회사(RAG)", rag_context)
        + f"\n\n[점검 모드] {mode}"
    )
    try:
        out = await call_json(_SYSTEM, user, _CriticOut)
    except Exception as exc:  # 그래프를 죽이지 않고 폴백.
        logger.warning("비평 시뮬레이션 실패 → stub 폴백: %s", exc)
        return _fallback(subject, mode, basis)

    findings = [f.strip() for f in out.findings if f.strip()]
    if not findings:
        return _fallback(subject, mode, basis)
    agreement = out.agreement if out.agreement in _AGREEMENTS else "unknown"
    return {
        "subject": subject[:80],
        "findings": findings,
        "sources": [s.strip() for s in out.sources if s.strip()],
        "agreement": agreement,  # type: ignore[typeddict-item]
        "cluster": "critic",
    }
