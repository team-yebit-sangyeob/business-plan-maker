"""계획서 작성 에이전트 — 하이브리드 진입점(async). (12장 Block 1)

세션 누적 근거(session_evidence)와 슬롯을 받아 한 편의 사업 계획서 마크다운을 만든다:
  1) citations.build_footnotes — 결정론 각주 번호·끝 출처목록(정확성 보장: LLM 미경유)
  2) narrative.write_narratives — 근거에 갇힌 LLM 섹션 서술(실패해도 골격으로 폴백)
  3) skeleton.render — 일관된 번호형 구조로 조립
출력 마크다운이 곧 산출물이다(현재 pdf_renderer는 stub이라 마크다운이 실물).

LLM(서술)이 죽어도 출처·구조는 결정론이라 그대로 나온다 — 출처 정확성은 LLM 상태와 무관하다.
"""
from __future__ import annotations

import logging
from datetime import datetime

from common.schema import PlanState
from common.schema.state import OPTIONAL_SLOTS
from agents.planner.citations import build_footnotes
from agents.planner.narrative import write_narratives
from agents.planner.skeleton import render


logger = logging.getLogger(__name__)


async def compose_markdown(state: PlanState) -> str:
    """PlanState → 사업 계획서 마크다운. 출력 게이트(Type 0 거절)는 호출자(plan 라우트)가 본다."""
    slots = state.get("slots") or {}
    records = state.get("session_evidence") or []
    correction_log = state.get("correction_log") or []

    foot = build_footnotes(records)
    missing = [n for n in OPTIONAL_SLOTS if not (slots.get(n) or {}).get("value")]

    # 서술은 LLM이지만 실패해도 출처(결정론)는 그대로 — 골격+인용만으로 폴백한다.
    try:
        narr = await write_narratives(slots, records, missing)
        narratives_by_slot = {s.slot: (s.prose or "").strip() for s in narr.sections}
        summary = (narr.summary or "").strip()
    except Exception as exc:  # 키 부재·파싱 실패 등 — 계획서 생성 자체는 막지 않는다.
        logger.warning("계획서 서술 생성 실패 → 결정론 골격으로 폴백: %s", exc)
        narratives_by_slot, summary = {}, ""

    return render(
        slots=slots,
        narratives_by_slot=narratives_by_slot,
        summary=summary,
        foot=foot,
        correction_log=correction_log,
        early=bool(missing),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
