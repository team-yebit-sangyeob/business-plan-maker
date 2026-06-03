"""세그먼트 routes를 보고 리서치/RAG/논리검증(logic_validator) 워커를 호출 (Fig.0 ③).

spec v0.7.5: "검증" 단계는 사라지고 logic_validator·리서치·RAG 호출로 분기.

2단계 디스패치 (리서치·RAG 병렬 → 논리검증 후속):
  1단계 — research·rag를 전 세그먼트 병렬(asyncio.gather)로 먼저 끝낸다. RAG는 회수 결과
          (ValidationReport)와 함께 원본 RagExtractorResult를 돌려준다.
  2단계 — logic_validator는 같은 세그먼트의 1단계 RAG 산출물(RagExtractorResult)을 입력으로
          받아 claim ↔ 사내 근거의 논리적 지지 여부(verdict→agreement)를 판정한다.
라우트 매트릭스상 logic_validator는 항상 rag와 동반하므로(claim), 판정에 쓸 RAG
결과는 늘 존재한다. 만약 RAG가 근거를 못 찾으면(rag_result=None) '근거 없음'으로 흐른다.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from common.schema import EvidenceRecord, PlanState, ValidationReport, VerificationRequest
from agents.orchestrator.progress import emit

if TYPE_CHECKING:
    from agents.rag.rag_extractor import RagExtractorResult


# 실제 워커를 가진 라우트. clarify/none은 디스패치 대상이 아님.
_WORKER_ROUTES = {"research", "rag", "logic_validator"}


def _evidence_record(report: ValidationReport, target_slot: str | None, turn: int) -> EvidenceRecord:
    """ValidationReport + 슬롯 연결 → 세션 누적용 EvidenceRecord. target_slot은 dispatch
    시점의 세그먼트 힌트(없을 수 있음) — run_turn이 fill 확정 슬롯으로 백필한다."""
    return {
        "subject": report.get("subject", ""),
        "cluster": report.get("cluster", "research"),
        "findings": report.get("findings") or [],
        "agreement": report.get("agreement", "unknown"),
        "citations": report.get("citations") or [],
        "target_slot": target_slot,
        "turn": turn,
    }

# 리서치 검색 recency 힌트 기본값(일). 회사 조직처럼 빠르게 변하는 항목은 추후 세분화.
_RESEARCH_FRESHNESS_DAYS = 180


def _slot_context(slots: dict) -> dict:
    """채워진 슬롯만 발췌 — 리서치 분해기가 검증 방식을 정하는 단서."""
    return {name: s.get("value") for name, s in slots.items() if s.get("value")}


def _verification_request(
    subject: str, label: str, slots: dict, state: PlanState
) -> VerificationRequest:
    """세그먼트 1건 → 리서치 클러스터 입력 (research_spec VerificationRequest)."""
    return {
        "claim": subject,
        "utterance_label": label,
        "slot_context": _slot_context(slots),
        "freshness_max_days": _RESEARCH_FRESHNESS_DAYS,
        "session_id": state.get("session_id", ""),
        "turn_id": state.get("turn", 0),
    }


async def parallel_dispatch_workers_node(state: PlanState) -> dict:
    """워커 라우트 세그먼트에 리서치·RAG·논리검증을 디스패치한다 →
    {"turn_validation_reports","turn_evidence"}(없으면 {})."""
    # worker import는 함수 안에서 — 모듈 로드 시 agents.{research,rag,logic_validator} ↔
    # agents.orchestrator 패키지 순환 import를 피한다(import 순서 의존 크래시 방지).
    from agents.research import run_research
    from agents.rag import run_rag_check
    from agents.logic_validator import run_logic_validator

    segments = state.get("turn_segments") or []
    slots = state.get("slots") or {}

    # 디스패치 대상 세그먼트만 추림 (subject 비어있으면 제외).
    # '워커 라우트 유무'로 판단 — claim·question 등 워커 라우트가 있으면 매트릭스대로 디스패치.
    # (interaction(meta·recall)·correction·명확화-only 세그먼트는 워커 라우트가 없어 제외)
    # 튜플 4번째 = 세그먼트의 target_slot 힌트 — 근거를 슬롯에 연결하는 1차 단서(없으면 None).
    targets: list[tuple[str, list[str], str, str | None]] = []
    for seg in segments:
        routes = seg.get("routes") or []
        if not (_WORKER_ROUTES & set(routes)):
            continue
        subject = (seg.get("canonical_text") or seg.get("text", "")).strip()
        if not subject:
            continue
        labels = seg.get("utterance_types") or []
        label = labels[0] if labels else "claim"
        targets.append((subject, list(routes), label, seg.get("target_slot")))

    if not targets:
        return {}

    # --- 1단계: 리서치·RAG 병렬 (외부 사실 + 회사 문서) ---
    # 호출 직전에 agent_start를 발행 → 프론트가 '실행 중'을 실제 호출과 동시에 본다.
    fact_specs: list[tuple[int, str]] = []  # (target_idx, route)
    fact_coros = []
    for idx, (subject, routes, label, _slot) in enumerate(targets):
        if "research" in routes:
            fact_specs.append((idx, "research"))
            emit({"type": "agent_start", "cluster": "research", "subject": subject[:80]})
            fact_coros.append(
                run_research(_verification_request(subject, label, slots, state))
            )
        if "rag" in routes:
            fact_specs.append((idx, "rag"))
            emit({"type": "agent_start", "cluster": "rag", "subject": subject[:80]})
            fact_coros.append(run_rag_check(subject))

    fact_results = list(await asyncio.gather(*fact_coros)) if fact_coros else []

    # 1단계 결과 적재 + 결과 카드 발행. RAG는 (report, rag_result) 튜플 → rag_result는
    # 2단계 logic_validator 입력으로만 쓰고 프론트엔 안 보낸다(raw_source 등 대용량 제외).
    # research report도 idx로 보관해 2단계 logic_validator에 보조 근거로 넘긴다(claim 라우트일 때만).
    rag_result_by_idx: dict[int, "RagExtractorResult"] = {}
    research_report_by_idx: dict[int, ValidationReport] = {}
    reports: list[ValidationReport] = []
    # (report, target_idx) — 리포트를 세그먼트(→슬롯)로 되짚어 EvidenceRecord를 만든다.
    report_idx: list[tuple[ValidationReport, int]] = []
    for (idx, route), res in zip(fact_specs, fact_results):
        if route == "rag":
            report, rag_result = res
            if rag_result is not None:
                rag_result_by_idx[idx] = rag_result
        else:  # route == "research"
            report = res
            research_report_by_idx[idx] = report
        reports.append(report)
        report_idx.append((report, idx))
        emit({"type": "validation_report", **report})

    # --- 2단계: 논리검증 (1단계 RAG 산출물을 입력으로) ---
    lv_coros = []
    lv_idx: list[int] = []  # lv 리포트가 어느 타깃(→슬롯)에서 나왔는지
    for idx, (subject, routes, _label, _slot) in enumerate(targets):
        if "logic_validator" in routes:
            emit({"type": "agent_start", "cluster": "logic_validator", "subject": subject[:80]})
            lv_coros.append(
                run_logic_validator(
                    subject,
                    rag_result_by_idx.get(idx),
                    research_report_by_idx.get(idx),
                )
            )
            lv_idx.append(idx)
    if lv_coros:
        lv_reports = list(await asyncio.gather(*lv_coros))
        for j, report in enumerate(lv_reports):
            emit({"type": "validation_report", **report})
            reports.append(report)
            report_idx.append((report, lv_idx[j]))

    if not reports:
        return {}

    # turn_validation_reports = 이번 턴 dispatch 결과만(대화 보고·SSE 활동용, 매 턴 리셋).
    # turn_evidence = 같은 결과 + 슬롯 연결정보. run_turn이 session_evidence로 누적한다.
    turn = state.get("turn", 0)
    turn_evidence: list[EvidenceRecord] = [
        _evidence_record(report, targets[idx][3], turn) for report, idx in report_idx
    ]
    return {"turn_validation_reports": reports, "turn_evidence": turn_evidence}
