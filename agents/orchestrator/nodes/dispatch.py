"""세그먼트 routes를 보고 리서치/RAG/비평(Critic) 워커를 호출 (Fig.0 ③).

spec v0.7.5: "검증" 단계는 사라지고 비평·리서치·RAG 호출로 분기.
비평(Critic)은 라벨링된 발화 + 슬롯 상태(read-only)를 함께 받는다.

2단계 디스패치 (critic_spec §6 "리서치·RAG 병렬 → 비평 후속"):
  1단계 — research·rag를 전 세그먼트 병렬(asyncio.gather)로 먼저 끝낸다.
  2단계 — critic은 같은 세그먼트의 1단계 산출물(research_report·rag_context)을
          입력으로 받아 호출한다. 정합성(consistency) 모드는 이 두 근거로
          '사용자 주장 ↔ 외부 사실/회사 문서'를 비교하기 때문.
추론(reasoning) 점검은 근거가 없어도 수행되므로, research/rag 라우트가 없는
세그먼트의 critic은 두 입력이 None인 채로 돌아간다(현 매트릭스엔 그런 조합 없음).
"""
from __future__ import annotations

import asyncio

from common.schema import PlanState, ValidationReport
from agents.research.stub import run_research
from agents.rag.stub import run_rag_check
from agents.critic.stub import run_critic


# 실제 워커를 가진 라우트. clarify/none은 디스패치 대상이 아님.
_WORKER_ROUTES = {"research", "rag", "critic"}


async def parallel_dispatch_workers_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    slots = state.get("slots") or {}

    # 디스패치 대상 세그먼트만 추림 (subject 비어있으면 제외).
    # '워커 라우트 유무'로 판단 — opinion(routes=rag·critic)도 기획서 매트릭스대로
    # 디스패치되도록. (명확화-only 턴은 graph의 _clarify_branch가 미리 우회)
    targets: list[tuple[str, list[str]]] = []
    for seg in segments:
        routes = seg.get("routes") or []
        if not (_WORKER_ROUTES & set(routes)):
            continue
        subject = (seg.get("canonical_text") or seg.get("text", "")).strip()
        if not subject:
            continue
        targets.append((subject, list(routes)))

    if not targets:
        return {}

    # --- 1단계: 리서치·RAG 병렬 (외부 사실 + 회사 문서) ---
    fact_specs: list[tuple[int, str]] = []  # (target_idx, route)
    fact_coros = []
    for idx, (subject, routes) in enumerate(targets):
        if "research" in routes:
            fact_specs.append((idx, "research"))
            fact_coros.append(run_research(subject))
        if "rag" in routes:
            fact_specs.append((idx, "rag"))
            fact_coros.append(run_rag_check(subject))

    fact_reports = list(await asyncio.gather(*fact_coros)) if fact_coros else []

    # 세그먼트별로 1단계 결과를 묶어 critic 입력으로 전달할 준비.
    research_by_idx: dict[int, ValidationReport] = {}
    rag_by_idx: dict[int, ValidationReport] = {}
    reports: list[ValidationReport] = []
    for (idx, route), report in zip(fact_specs, fact_reports):
        reports.append(report)
        if route == "research":
            research_by_idx[idx] = report
        else:
            rag_by_idx[idx] = report

    # --- 2단계: 비평 (1단계 산출물을 입력으로) ---
    critic_coros = []
    for idx, (subject, routes) in enumerate(targets):
        if "critic" in routes:
            critic_coros.append(
                run_critic(
                    subject,
                    slots,
                    research_report=research_by_idx.get(idx),
                    rag_context=rag_by_idx.get(idx),
                )
            )
    if critic_coros:
        reports.extend(await asyncio.gather(*critic_coros))

    if not reports:
        return {}

    existing = list(state.get("validation_reports") or [])
    existing.extend(reports)
    return {"validation_reports": existing}
