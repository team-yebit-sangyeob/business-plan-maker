"""유형별 worker 호출 라우팅. 클러스터는 stub."""
from __future__ import annotations

from common.schema import PlanState, ValidationReport
from agents.research.stub import run_research
from agents.orchestrator.nodes.correction import collect_slot_fills


async def fill_and_validate_node(state: PlanState) -> dict:
    # 1) 슬롯 채움
    fills = collect_slot_fills(state)
    state = {**state, **fills}

    # 2) 검증 필요한 세그먼트만 골라 리서치 stub 호출
    reports: list[ValidationReport] = list(state.get("validation_reports") or [])
    for seg in state.get("turn_segments") or []:
        if seg["utterance_type"] in ("fact_claim", "hypothesis", "decision", "constraint"):
            r = await run_research(seg["text"])
            reports.append(r)

    return {"slots": fills["slots"], "validation_reports": reports}
