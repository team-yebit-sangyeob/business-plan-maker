"""세그먼트 routes를 보고 리서치/RAG/추론 stub을 병렬 호출 (Fig.0 ③)."""
from __future__ import annotations

import asyncio

from common.schema import PlanState, ValidationReport
from agents.research.stub import run_research
from agents.rag.stub import run_rag_check
from agents.inference.stub import run_inference


_CLUSTER_DISPATCH = {
    "research": run_research,
    "rag": run_rag_check,
    "inference": run_inference,
}


async def parallel_verify_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    tasks: list[asyncio.Task[ValidationReport]] = []

    for seg in segments:
        if seg.get("priority") != 2:
            continue
        subject = seg.get("canonical_text") or seg.get("text", "")
        if not subject.strip():
            continue
        for route in seg.get("routes") or []:
            fn = _CLUSTER_DISPATCH.get(route)
            if fn is None:
                continue
            tasks.append(asyncio.create_task(fn(subject)))

    if not tasks:
        return {}

    new_reports = await asyncio.gather(*tasks)
    existing = list(state.get("validation_reports") or [])
    existing.extend(new_reports)
    return {"validation_reports": existing}
