"""세그먼트 routes를 보고 리서치/RAG/논리검증(logic_validator) 워커를 호출 (Fig.0 ③).

spec v0.7.5: "검증" 단계는 사라지고 logic_validator·리서치·RAG 호출로 분기.

2단계 디스패치 (리서치·RAG 병렬 → 논리검증 후속):
  1단계 — research·rag를 전 세그먼트 병렬로 돌리되, asyncio.as_completed로 완료되는 대로
          결과 카드(validation_report)를 발행한다 — 먼저 끝난 워커(웹/사내문서)가 먼저 보인다.
          RAG는 회수 결과(ValidationReport)와 함께 원본 RagExtractorResult를 돌려준다.
  2단계 — logic_validator는 1단계 전체 완료 뒤 시작(배리어)하며, 같은 세그먼트의 1단계 RAG
          산출물(RagExtractorResult)을 입력으로 받아 claim ↔ 사내 근거의 논리적 지지 여부
          (verdict→agreement)를 판정한다.
라우트 매트릭스상 logic_validator는 claim에서 rag와 동반하지만, 사용자가 evidence_mode로
rag를 끄거나(research 전용) RAG가 근거를 못 찾으면 rag_result=None이 된다 — 그때는 research
리포트만으로, 둘 다 없으면 '근거 없음'으로 판정이 흐른다(logic_validator는 None을 허용한다).

evidence_mode(both/research/rag): 1단계에서 research/rag 디스패치를 거르는 사용자 토글.
both=둘 다, research=웹만, rag=사내문서만. logic_validator는 끄지 않는다(claim이면 항상 돈다).
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


async def _tagged(idx: int, route: str, coro):
    """워커 1건을 (idx, route, result)로 태깅 — as_completed가 누가 끝났는지 알게 한다."""
    return idx, route, await coro


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
    {"turn_validation_reports","turn_evidence"}(디스패치 대상이나 리포트가 없으면 {})."""
    # worker import는 함수 안에서 — 모듈 로드 시 agents.{research,rag,logic_validator} ↔
    # agents.orchestrator 패키지 순환 import를 피한다(import 순서 의존 크래시 방지).
    from agents.research import run_research
    from agents.rag import run_rag_check
    from agents.logic_validator import run_logic_validator

    segments = state.get("turn_segments") or []
    slots = state.get("slots") or {}
    # 근거 출처 범위 — 사용자가 프론트 토글로 고른다(both/research/rag). 라우트는 매트릭스가
    # 정한 그대로 두고, 여기서 디스패치 단계에만 거른다(derive_routes는 순수 유지).
    mode = state.get("evidence_mode", "both")

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
    # fact_specs는 디스패치(등장) 순서를 보존 — 발행은 완료순이어도 반환 리스트는 이 순서로 되돌린다.
    fact_specs: list[tuple[int, str]] = []  # (target_idx, route)
    fact_tasks = []
    for idx, (subject, routes, label, _slot) in enumerate(targets):
        if "research" in routes and mode != "rag":
            fact_specs.append((idx, "research"))
            emit({"type": "agent_start", "cluster": "research", "subject": subject[:80]})
            fact_tasks.append(
                _tagged(idx, "research", run_research(_verification_request(subject, label, slots, state)))
            )
        if "rag" in routes and mode != "research":
            fact_specs.append((idx, "rag"))
            emit({"type": "agent_start", "cluster": "rag", "subject": subject[:80]})
            fact_tasks.append(_tagged(idx, "rag", run_rag_check(subject)))

    # 완료 순으로 결과 카드를 즉시 발행 — 먼저 끝난 워커(웹/사내문서)가 먼저 보인다.
    # RAG는 (report, rag_result) 튜플 → rag_result는 2단계 logic_validator 입력으로만 쓰고
    # 프론트엔 안 보낸다(사내 원문은 report.citations의 snippet/raw_source로 별도 전달).
    # research report도 idx로 보관해 2단계 logic_validator에 보조 근거로 넘긴다(claim 라우트일 때만).
    rag_result_by_idx: dict[int, "RagExtractorResult"] = {}
    research_report_by_idx: dict[int, ValidationReport] = {}
    report_by_spec: dict[tuple[int, str], ValidationReport] = {}
    for coro in asyncio.as_completed(fact_tasks):
        idx, route, res = await coro
        if route == "rag":
            report, rag_result = res
            if rag_result is not None:
                rag_result_by_idx[idx] = rag_result
        else:  # route == "research"
            report = res
            research_report_by_idx[idx] = report
        report_by_spec[(idx, route)] = report
        emit({"type": "validation_report", **report})

    # 반환용 리스트는 디스패치 순서로 재구성(결정론) — 발행은 완료순이지만 다운스트림은 안정순으로.
    # (report, target_idx) — 리포트를 세그먼트(→슬롯)로 되짚어 EvidenceRecord를 만든다.
    reports: list[ValidationReport] = []
    report_idx: list[tuple[ValidationReport, int]] = []
    for (idx, route) in fact_specs:
        report = report_by_spec[(idx, route)]
        reports.append(report)
        report_idx.append((report, idx))

    # --- 2단계: 논리검증 (1단계 RAG 산출물을 입력으로) ---
    # 2단계는 1단계 전체 완료 뒤 시작(배리어 유지) — lv는 같은 idx의 RAG·리서치 산출물이 필요.
    # lv가 여럿이면 1단계처럼 완료순으로 발행하고, 반환은 lv_idx(디스패치) 순서로 되돌린다.
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
        lv_tasks = [
            _tagged(lv_idx[j], "logic_validator", c) for j, c in enumerate(lv_coros)
        ]
        lv_by_idx: dict[int, ValidationReport] = {}
        for coro in asyncio.as_completed(lv_tasks):
            tgt_idx, _route, report = await coro
            emit({"type": "validation_report", **report})
            lv_by_idx[tgt_idx] = report
        for tgt_idx in lv_idx:
            reports.append(lv_by_idx[tgt_idx])
            report_idx.append((lv_by_idx[tgt_idx], tgt_idx))

    if not reports:
        return {}

    # turn_validation_reports = 이번 턴 dispatch 결과만(대화 보고·SSE 활동용, 매 턴 리셋).
    # turn_evidence = 같은 결과 + 슬롯 연결정보. run_turn이 session_evidence로 누적한다.
    turn = state.get("turn", 0)
    turn_evidence: list[EvidenceRecord] = [
        _evidence_record(report, targets[idx][3], turn) for report, idx in report_idx
    ]
    return {"turn_validation_reports": reports, "turn_evidence": turn_evidence}
