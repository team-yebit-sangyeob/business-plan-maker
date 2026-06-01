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

from common.schema import PlanState, ValidationReport, VerificationRequest
from agents.research import run_research
from agents.rag.stub import run_rag_check
from agents.critic.stub import run_critic
from agents.orchestrator.progress import emit
from agents.orchestrator.llm import _resolve_mode


# 실제 워커를 가진 라우트. clarify/none은 디스패치 대상이 아님.
_WORKER_ROUTES = {"research", "rag", "critic"}

# 리서치 검색 recency 힌트 기본값(일). 회사 조직처럼 빠르게 변하는 항목은 추후 세분화.
_RESEARCH_FRESHNESS_DAYS = 180

# mock 연출용 단계 간 지연(초). live(실 호출)에는 적용하지 않는다 — 추가 지연 0.
_MOCK_PACE_SECONDS = 0.12


async def _pace(is_mock: bool) -> None:
    """mock에서만 단계 사이 잠깐 멈춰 '실행 중 → 결과' 전환이 화면에 보이게 한다."""
    if is_mock:
        await asyncio.sleep(_MOCK_PACE_SECONDS)


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
    segments = state.get("turn_segments") or []
    slots = state.get("slots") or {}

    # 디스패치 대상 세그먼트만 추림 (subject 비어있으면 제외).
    # '워커 라우트 유무'로 판단 — opinion(routes=rag·critic)도 기획서 매트릭스대로
    # 디스패치되도록. (명확화-only 턴은 graph의 _clarify_branch가 미리 우회)
    targets: list[tuple[str, list[str], str]] = []
    for seg in segments:
        routes = seg.get("routes") or []
        if not (_WORKER_ROUTES & set(routes)):
            continue
        subject = (seg.get("canonical_text") or seg.get("text", "")).strip()
        if not subject:
            continue
        labels = seg.get("utterance_types") or []
        label = labels[0] if labels else "claim"
        targets.append((subject, list(routes), label))

    if not targets:
        return {}

    is_mock = _resolve_mode() == "mock"

    # --- 1단계: 리서치·RAG 병렬 (외부 사실 + 회사 문서) ---
    # 호출 직전에 agent_start를 발행 → 프론트가 '실행 중'을 실제 호출과 동시에 본다.
    fact_specs: list[tuple[int, str]] = []  # (target_idx, route)
    fact_coros = []
    for idx, (subject, routes, label) in enumerate(targets):
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

    if fact_coros:
        await _pace(is_mock)  # '실행 중' 카드가 잠깐 보이도록 (mock 한정)
        fact_reports = list(await asyncio.gather(*fact_coros))
    else:
        fact_reports = []

    # 세그먼트별로 1단계 결과를 묶어 critic 입력으로 전달할 준비 + 결과 카드 발행.
    research_by_idx: dict[int, ValidationReport] = {}
    rag_by_idx: dict[int, ValidationReport] = {}
    reports: list[ValidationReport] = []
    for (idx, route), report in zip(fact_specs, fact_reports):
        reports.append(report)
        emit({"type": "validation_report", **report})
        if route == "research":
            research_by_idx[idx] = report
        else:
            rag_by_idx[idx] = report

    # --- 2단계: 비평 (1단계 산출물을 입력으로) ---
    critic_coros = []
    for idx, (subject, routes, _label) in enumerate(targets):
        if "critic" in routes:
            emit({"type": "agent_start", "cluster": "critic", "subject": subject[:80]})
            critic_coros.append(
                run_critic(
                    subject,
                    slots,
                    research_report=research_by_idx.get(idx),
                    rag_context=rag_by_idx.get(idx),
                )
            )
    if critic_coros:
        await _pace(is_mock)  # 1단계 결과 해소 후 비평 '실행 중'이 보이도록 (mock 한정)
        critic_reports = list(await asyncio.gather(*critic_coros))
        for report in critic_reports:
            emit({"type": "validation_report", **report})
        reports.extend(critic_reports)

    if not reports:
        return {}

    existing = list(state.get("validation_reports") or [])
    existing.extend(reports)
    # turn_validation_reports = 이번 턴 것만(대화 보고·SSE 활동용). validation_reports는 누적.
    return {"validation_reports": existing, "turn_validation_reports": reports}
