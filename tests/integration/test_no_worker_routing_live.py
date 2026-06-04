"""워커 미호출 — 실 LLM 분류까지 검증.

단위(test_no_worker_dispatch.py)는 call_json을 스텁해 '유형 → 라우트' 계약만 고정한다 —
"meta면 워커 0"은 보장하지만 "'응 다음으로'가 실제로 meta로 분류되는지"(LLM 판단)는 검증
못 한다. 그 빈틈을 실 LLM으로 닫는다: 현실적인 발화를 진짜 segment_node→classify_node에
태워, 파생된 routes에 워커 라우트(research/rag/logic_validator)가 없음을 단언한다.

dispatch는 호출하지 않는다 — 워커 호출 여부는 routes로 이미 결정나므로, 실제 워커(웹/사내)를
부르지 않고 분류 단계만 검증한다(비용·외부 의존 회피). 양성 대조로 검증 가능한 claim은 워커
라우트가 잡힘을 함께 확인해, '미호출' 단언이 공허하지 않게 한다.

OPENAI_API_KEY가 없으면 스킵(call_json은 mock 모드 없음). LLM 응답은 비결정적이라 정확
라벨이 아니라 '워커 라우트 유무'로 느슨하게 단언한다.
"""
import asyncio
import os

import pytest

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)

from common.schema.state import initial_state
from agents.orchestrator.nodes.segment import segment_node
from agents.orchestrator.nodes.classify import classify_node

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="실 LLM 호출 — OPENAI_API_KEY 필요(call_json은 mock 모드 없음)",
)

_WORKER_ROUTES = {"research", "rag", "logic_validator"}


def _route_real(user_input, *, messages=None, session_evidence=None,
                open_proposal=None, last_asked_slot=None, slots=None):
    """실 LLM으로 segment→classify를 돌려 분류된 세그먼트를 돌려준다(dispatch는 안 탐)."""
    state = initial_state()
    state["turn"] = 2
    state["user_input"] = user_input
    if messages:
        state["messages"] = messages
    if session_evidence:
        state["session_evidence"] = session_evidence
    if open_proposal:
        state["open_proposal"] = open_proposal
    if last_asked_slot:
        state["last_asked_slot"] = last_asked_slot
    if slots:
        state["slots"] = {**state["slots"], **slots}

    async def _run():
        seg_out = await segment_node(state)
        state["turn_segments"] = seg_out["turn_segments"]
        cls_out = await classify_node(state)
        return cls_out["turn_segments"]

    return asyncio.run(_run())


def _has_worker_route(segments):
    return any(_WORKER_ROUTES & set(s.get("routes") or []) for s in segments)


# 직전 대화 맥락 — recall·reason 분류의 현실적 근거.
_COFFEE_MESSAGES = [
    {"role": "user", "content": "커피로 사업 계획 짜고 싶어", "turn": 1},
    {"role": "assistant", "content": "국내 저가 커피 시장은 점포 포화 상태야. 어떤 문제를 풀려는 거야?", "turn": 1},
]
_WEBTOON_MESSAGES = [
    {"role": "user", "content": "웹툰 IP가 일본에서 통할 거 같아", "turn": 1},
    {"role": "assistant", "content": "일본 웹툰 시장이 성장 중이라 가능성 있어 보여. 타겟은 누구로 할까?", "turn": 1},
]
# 이미 '정한' 결정을 되묻는 순수 recall — 주장 재확인(재검증 가능)과 달리 워커가 필요 없다.
_DECIDED_TARGET_MESSAGES = [
    {"role": "user", "content": "타겟은 네이버 웹툰 콘텐츠팀으로 가자", "turn": 1},
    {"role": "assistant", "content": "좋아, 타겟을 네이버 웹툰 콘텐츠팀으로 정해뒀어. 다음은 솔루션을 볼까?", "turn": 1},
]
_COFFEE_EVIDENCE = [{
    "subject": "국내 저가 커피 시장", "cluster": "research",
    "findings": ["점포 포화로 가맹점 수익성 악화", "프리미엄·해외 진출로 돌파구 모색"],
    "agreement": "confirms", "citations": [], "target_slot": None, "turn": 1,
}]


def test_live_meta_progress_no_worker():
    segs = _route_real("응 좋아 다음으로 넘어가자", last_asked_slot="problem")
    assert not _has_worker_route(segs), segs


def test_live_recall_no_worker():
    # 이미 '정한' 결정을 되묻는 순수 recall — 대화 이력에서 답하면 되니 워커가 필요 없다.
    # (주장 재확인 "아까 X 된다며?"는 재검증 가능이라 question으로도 빠질 수 있어 제외.)
    segs = _route_real("우리 타겟 누구로 정했었지?", messages=_DECIDED_TARGET_MESSAGES)
    assert not _has_worker_route(segs), segs


def test_live_tool_help_no_worker():
    segs = _route_real("솔루션 슬롯이 뭐하는 칸이야?")
    assert not _has_worker_route(segs), segs


def test_live_reason_request_no_worker():
    # 이번 픽스의 제안요청 — 새 리서치가 아니라 가진 근거로 추론. 워커 라우트가 없어야 한다.
    segs = _route_real("네 생각엔 우리 문제가 뭐야?",
                       messages=_COFFEE_MESSAGES, session_evidence=_COFFEE_EVIDENCE,
                       last_asked_slot="problem")
    assert not _has_worker_route(segs), segs


def test_live_out_of_scope_no_worker():
    segs = _route_real("오늘 서울 날씨 어때?", messages=_COFFEE_MESSAGES)
    # 스코프밖이면 in_scope=False로 routes가 ["none"]으로 덮인다.
    assert not _has_worker_route(segs), segs
    assert all(s.get("in_scope") is False for s in segs), segs


def test_live_confirmation_answer_no_worker():
    # 열린 제안에 대한 답은 classify 백스톱에서도 meta로 잡혀 워커 라우트가 없어야 한다
    # (실제 우회는 confirm_resolve가 먼저 하지만, classify가 새 주장으로 오인하지 않는지 확인).
    open_proposal = {
        "value": "AI 자동 검수 도구", "proposed_slot": "solution",
        "candidate_slots": ["solution"], "confirm_kind": "commit",
        "previous_value": "", "adequate": True,
    }
    segs = _route_real("응 그걸로 넣어", open_proposal=open_proposal)
    assert not _has_worker_route(segs), segs


def test_live_claim_does_dispatch_positive_control():
    # 양성 대조 — 검증 가능한 claim은 워커 라우트가 잡혀야 한다(미호출 단언이 공허하지 않음).
    segs = _route_real("일본 웹툰 시장은 2023년에도 성장했다", messages=_WEBTOON_MESSAGES)
    assert _has_worker_route(segs), segs
