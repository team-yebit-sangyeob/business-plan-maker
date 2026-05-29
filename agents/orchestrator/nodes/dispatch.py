"""세그먼트 routes를 보고 리서치/RAG/비평(Critic) 워커를 병렬 호출 (Fig.0 ③).

spec v0.7.5: "검증" 단계는 사라지고 비평·리서치·RAG 호출로 분기.
비평(Critic)은 라벨링된 발화 + 슬롯 상태(read-only)를 함께 받는다.

NOTE(정합성): critic_spec은 비평 정합성 모드가 리서치/RAG 결과(research_report·
rag_context)를 입력으로 받는다고 정의한다(§6: 리서치·RAG 병렬 → 비평 후속).
현재 stub은 셋을 모두 동시 호출하고 run_critic에 slots만 넘긴다 — 비평을 실제
구현할 때 2단계(리서치·RAG 먼저 → 결과를 critic 입력으로)로 분리해야 한다.
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
    tasks: list[asyncio.Task[ValidationReport]] = []

    for seg in segments:
        # priority가 아니라 '워커 라우트 유무'로 판단 — opinion(priority 3, routes=rag·critic)도
        # 기획서 매트릭스대로 디스패치되도록. (명확화-only 턴은 graph의 _clarify_branch가 미리 우회)
        routes = seg.get("routes") or []
        if not (_WORKER_ROUTES & set(routes)):
            continue
        subject = seg.get("canonical_text") or seg.get("text", "")
        if not subject.strip():
            continue
        for route in seg.get("routes") or []:
            if route == "research":
                tasks.append(asyncio.create_task(run_research(subject)))
            elif route == "rag":
                tasks.append(asyncio.create_task(run_rag_check(subject)))
            elif route == "critic":
                tasks.append(asyncio.create_task(run_critic(subject, slots)))

    if not tasks:
        return {}

    new_reports = await asyncio.gather(*tasks)
    existing = list(state.get("validation_reports") or [])
    existing.extend(new_reports)
    return {"validation_reports": existing}
