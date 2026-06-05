"""LangGraph 구성 (Fig.0 우선순위 토폴로지, spec v0.7.5):

  confirm_resolve → segment → classify
  (보류된 슬롯 확인 해소)
          → correction              (correction 라벨 세그먼트 처리)
          → _clarify_branch         (clarify 라우트만 있고 워커 라우트 없으면 dispatch·fills 우회)
            ├ dispatch+fills 경로   (워커 라우트 발견 → 리서치/RAG/논리검증 호출)
            └ skip 경로             (명확화 우선 → conversation 직행)
          → conversation → integrator → END

계획서 생성은 그래프 밖이다: 채팅(이 그래프)은 슬롯을 채우고 답할 뿐, 계획서는 명시적
버튼(POST /plan)에서만 합성한다(필수 슬롯 게이트는 그 라우트가 required_missing으로 본다).

처리 순서와 분기는 별도 priority 필드 없이 세그먼트의 routes/utterance_types에서
바로 파생한다(워커 호출은 routes, 정정 처리는 utterance_types).

end-to-end trace 예시 (turn 5, "카카오는 빼자. 예산은 1억으로 가자."):
  segment        → [seg1 "카카오는 빼자", seg2 "예산 1억으로 가자"]
  classify       → seg1=["correction"](routes=none), seg2=["claim"](routes=research/rag/logic_validator)
  correction     → target "네이버·카카오" → "네이버" (correction_log에 기록)
  _clarify_branch→ clarify 라우트 없고 워커 라우트 있음 → "dispatch"
  dispatch       → seg2 canonical: 1단계 research·rag 병렬 → 2단계 logic_validator(1단계 RAG 산출물 입력)
                   → turn_validation_reports 적재
  extract_fills  → 빈 슬롯에 "예산 1억" 채울 수 있으면 resources 등에 반영
  conversation   → _build_intents(acknowledge·report_findings·ask_slot)를 자연어 한 응답으로 → pending_question
  integrator     → pass-through: 이번 턴 pending_clarifications만 기록(LLM 없음)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

from common.schema import EvidenceRecord, Message, PlanState, Segment
from agents.orchestrator.nodes.segment import segment_node
from agents.orchestrator.nodes.classify import classify_node
from agents.orchestrator.nodes.correction import (
    correction_node,
    extract_slot_fills_node,
)
from agents.orchestrator.nodes.confirm import confirm_resolve_node
from agents.orchestrator.nodes.dispatch import parallel_dispatch_workers_node
from agents.orchestrator.nodes.integrator import response_integrator_node
from agents.orchestrator.progress import emit
from agents.conversation.agent import conversation_node


_WORKER_ROUTES = frozenset({"research", "rag", "logic_validator"})


# 전이 오류(429·timeout·5xx 등) 한정 재시도 — LLM 노드에만 단다. 노드는 순수(state→LLM→dict)라
# 재시도가 안전하다. dispatch엔 절대 달지 않는다: 재시도가 research·rag·logic_validator 워커를
# 재호출해 비멱등 실행·SSE 카드 중복을 부른다. (기본값 backoff_factor=2.0·jitter=True를 그대로 쓴다.)
_LLM_RETRY = RetryPolicy(max_attempts=3)


# 노드 단계(stage) 진행 라벨 — 노드 시작 직전 emit해 프론트가 "지금 뭐 하는 중"을 본다.
# dispatch는 제외(agent_start/validation_report가 더 풍부 — 중복 방지). integrator도 제외(즉시 통과).
_STAGE_LABELS: dict[str, str] = {
    "confirm_resolve": "확인 정리하기",
    "segment": "발화 분석",
    "classify": "유형 분류",
    "correction": "정정 반영",
    "extract_fills": "슬롯 채우기",
    "conversation": "답변 작성",
}


def _staged(name: str, fn):
    """노드를 감싸 시작 직전 stage 이벤트를 emit한다(라벨 있을 때만).

    emit은 progress.py의 ContextVar emitter로 흐른다 — 미설정(테스트·직접 호출) 시 no-op이라
    노드 동작엔 영향이 없다. dispatch의 agent_start와 같은 "호출 직전 발행" 패턴.
    """
    label = _STAGE_LABELS.get(name)

    async def wrapped(state: PlanState) -> dict:
        if label:
            emit({"type": "stage", "node": name, "label": label})
        return await fn(state)

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


def _post_confirm_branch(state: PlanState) -> Literal["segment", "conversation"]:
    """confirm_resolve가 발화를 순수 확인 답으로 소비했으면(accept/pick/reject + 추가내용 없음)
    파이프라인을 건너뛰고 conversation 직행 — 열린 제안에 대한 답은 새 리서치 주문이 아니라
    이미 confirm_resolve가 슬롯에 반영했다. revise·unrelated·추가내용 섞인 답은 segment로 흘려
    정상 처리한다(상태 주입을 받은 classify가 잔여 확인절을 meta로 떨궈 재디스패치를 막는다)."""
    return "conversation" if state.get("confirmation_consumed") else "segment"


def _clarify_branch(state: PlanState) -> Literal["dispatch", "conversation"]:
    """명확화(clarify 라우트)만 있고 부를 워커가 하나도 없으면 디스패치·슬롯채움 우회.

    dispatch_node와 같은 기준(워커 라우트 유무)으로 판단해 일관성 유지 —
    부를 워커가 있으면(claim·question 등) 명확화가 섞여 있어도 dispatch로 보낸다.
    """
    segments = state.get("turn_segments") or []
    has_clarify = any("clarify" in (s.get("routes") or []) for s in segments)
    has_dispatch = any(_WORKER_ROUTES & set(s.get("routes") or []) for s in segments)
    if has_clarify and not has_dispatch:
        return "conversation"
    return "dispatch"


@lru_cache(maxsize=1)
def build_graph():
    """노드·엣지를 결선한 LangGraph를 컴파일해 돌려준다(프로세스당 1회 캐시)."""
    g: StateGraph = StateGraph(PlanState)
    # LLM 노드엔 _LLM_RETRY를 단다(전이 오류 재시도). dispatch·integrator는 제외 —
    # dispatch는 제외 워커(research·rag·logic_validator) 재호출 위험, integrator는 LLM 없는 통과.
    g.add_node("confirm_resolve", _staged("confirm_resolve", confirm_resolve_node), retry_policy=_LLM_RETRY)
    g.add_node("segment", _staged("segment", segment_node), retry_policy=_LLM_RETRY)
    g.add_node("classify", _staged("classify", classify_node), retry_policy=_LLM_RETRY)
    g.add_node("correction", _staged("correction", correction_node), retry_policy=_LLM_RETRY)
    g.add_node("dispatch", _staged("dispatch", parallel_dispatch_workers_node))
    g.add_node("extract_fills", _staged("extract_fills", extract_slot_fills_node), retry_policy=_LLM_RETRY)
    g.add_node("conversation", _staged("conversation", conversation_node), retry_policy=_LLM_RETRY)
    g.add_node("integrator", _staged("integrator", response_integrator_node))

    g.add_edge(START, "confirm_resolve")
    # 순수 확인 답이면 segment 이하 우회(conversation 직행), 아니면 평소대로 segment.
    g.add_conditional_edges(
        "confirm_resolve",
        _post_confirm_branch,
        {"segment": "segment", "conversation": "conversation"},
    )
    g.add_edge("segment", "classify")
    g.add_edge("classify", "correction")
    g.add_conditional_edges(
        "correction",
        _clarify_branch,
        {"dispatch": "dispatch", "conversation": "conversation"},
    )
    g.add_edge("dispatch", "extract_fills")
    g.add_edge("extract_fills", "conversation")
    g.add_edge("conversation", "integrator")
    g.add_edge("integrator", END)

    return g.compile()


def _merge_session_evidence(
    prev: list[EvidenceRecord],
    turn_evidence: list[EvidenceRecord],
    segments: list[Segment],
) -> list[EvidenceRecord]:
    """이번 턴 근거를 세션 누적분에 합친다 — 슬롯 백필 + 중복 제거.

    백필: dispatch 시점엔 세그먼트 힌트만 있을 수 있다(target_slot=None 가능). extract_fills는
      dispatch '뒤'에 돌며 슬롯을 확정하므로, 그 확정 슬롯을 turn_segments에서 subject로 되짚어
      더 정확한 값으로 덮는다(없으면 dispatch 힌트 유지).
    중복 제거: 키 (subject, cluster), last-write-wins. 같은 claim을 다음 턴 재검증해도 최신 1건만.
    """
    slot_by_subject = {
        (s.get("canonical_text") or s.get("text", "")).strip()[:80]: s.get("target_slot")
        for s in (segments or [])
        if s.get("target_slot")
    }
    enriched: list[EvidenceRecord] = []
    for rec in turn_evidence:
        final_slot = slot_by_subject.get(rec.get("subject", "")) or rec.get("target_slot")
        enriched.append({**rec, "target_slot": final_slot})

    merged: dict[tuple[str, str], EvidenceRecord] = {}
    for rec in [*prev, *enriched]:
        merged[(rec.get("subject", ""), rec.get("cluster", ""))] = rec  # 뒤(=최신)가 이김
    return list(merged.values())


async def run_turn(
    state: PlanState, user_input: str, evidence_mode: str = "both"
) -> PlanState:
    """한 턴 실행. state는 이전 턴의 누적 상태. evidence_mode는 이번 턴 dispatch의
    근거 출처 범위(both/research/rag) — 프론트 토글 값을 매 턴 반영한다."""
    graph = build_graph()
    turn = state.get("turn", 0) + 1
    messages = list(state.get("messages") or [])
    messages.append(Message(role="user", content=user_input, turn=turn))

    # 턴 시작 시점의 열린 제안 스냅샷 — confirm_resolve가 라이브 큐를 pop해도 segment·classify가
    # "이번 발화가 무엇에 대한 답인가"를 보게 박아둔다(confirmation_consumed는 매 턴 False로 리셋).
    pending = state.get("pending_confirmations") or []
    open_proposal = pending[0] if pending else None

    next_state = {
        **state,
        "user_input": user_input,
        "turn": turn,
        "messages": messages,
        "turn_segments": [],
        "pending_clarifications": [],
        "incomplete_fills": [],
        "turn_validation_reports": [],
        "turn_evidence": [],
        "evidence_mode": evidence_mode,
        "open_proposal": open_proposal,
        "confirmation_consumed": False,
    }
    result: PlanState = await graph.ainvoke(next_state)

    # 이번 턴 근거를 세션 누적분(session_evidence)으로 합친다 — 계획서가 출처를 인용하는 원천.
    # (turn_validation_reports와 달리 턴을 넘어 살아남는다. messages·correction_log와 같은 위치에서 누적.)
    session_evidence = _merge_session_evidence(
        list(result.get("session_evidence") or []),
        list(result.get("turn_evidence") or []),
        result.get("turn_segments") or [],
    )
    result = {**result, "session_evidence": session_evidence}

    # 어시스턴트 응답 적재
    answer = (result.get("pending_question") or "").strip()
    if answer:
        out_messages = list(result.get("messages") or messages)
        out_messages.append(Message(role="assistant", content=answer, turn=turn))
        result = {**result, "messages": out_messages}
    return result
