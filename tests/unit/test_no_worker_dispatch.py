"""워커 미호출 보장 — 리서치·RAG·논리검증이 '안 불려야 하는' 상황들의 회귀 테스트.

워커 호출은 LLM이 아니라 코드가 결정론으로 정한다(classify의 derive_routes → 라우트,
graph의 분기 → dispatch 진입 여부, dispatch 노드 → 실제 호출). LLM이 '무슨 발화인가'만
정하고 '누구를 부를지'는 코드라, 일단 유형이 정해지면 워커 미호출은 결정론으로 검증된다.

4개 레이어로 가른다:
  1) derive_routes 매트릭스(순수)            — interaction·correction 유형은 워커 라우트가 없다.
  2) dispatch 노드 + 워커 스파이             — 워커 라우트 없는/빈 subject 세그먼트는 호출 0,
                                              evidence_mode 토글이 한쪽 워커를 끈다.
  3) graph 분기(순수)                        — clarify-only·순수 확인답이면 dispatch를 우회한다.
  4) run_turn 멀티턴 해네스(실 그래프+스파이) — interaction·확인·명확화·스코프밖 턴을 이어서
                                              돌려도 워커 호출이 0으로 유지된다(양성 대조 포함).

dispatch는 워커를 함수 안에서 import한다(순환 회피) — 스파이 패치 타깃은 원천 패키지
(agents.research.run_research / agents.rag.run_rag_check / agents.logic_validator.run_logic_validator).
"""
import asyncio

import pytest

import agents.orchestrator.graph as graph_mod  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)

from common.schema.state import initial_state
from agents.orchestrator.nodes.classify import derive_routes
from agents.orchestrator.nodes.dispatch import parallel_dispatch_workers_node
from agents.orchestrator.graph import _clarify_branch, _post_confirm_branch, run_turn

# 노드별 LLM 출력 스키마 — 멀티턴 해네스가 스텁 응답을 만들 때 쓴다.
from agents.orchestrator.nodes.segment import SegmentOut, SegmentItem
from agents.orchestrator.nodes.classify import ClassifyOut, ClassifyItem
from agents.orchestrator.nodes.correction import CorrectionOut, FillOut
from agents.orchestrator.nodes.confirm import ConfirmOut
from agents.conversation.agent import ConversationOut


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def _seg(text, types, routes, *, in_scope=True):
    return {
        "text": text,
        "canonical_text": text,
        "utterance_types": list(types),
        "in_scope": in_scope,
        "target_slot": None,
        "routes": list(routes),
    }


def _install_worker_spies(monkeypatch, calls):
    """research/rag/logic_validator를 호출 기록 스파이로 갈아끼운다(미호출 검증용)."""

    async def fake_research(req):
        calls.setdefault("research", []).append(req)
        return {"subject": req.get("claim", ""), "cluster": "research",
                "findings": ["외부 사실 X"], "agreement": "confirms", "citations": []}

    async def fake_rag(subject):
        calls.setdefault("rag", []).append(subject)
        report = {"subject": subject, "cluster": "rag", "findings": [],
                  "agreement": "unknown", "citations": []}
        return report, None  # rag_result=None → logic_validator는 자연히 안 돈다

    async def fake_lv(*args, **kwargs):
        calls.setdefault("lv", []).append((args, kwargs))
        return {"subject": args[0] if args else "", "cluster": "logic_validator",
                "findings": [], "agreement": "unknown", "citations": []}

    monkeypatch.setattr("agents.research.run_research", fake_research)
    monkeypatch.setattr("agents.rag.run_rag_check", fake_rag)
    monkeypatch.setattr("agents.logic_validator.run_logic_validator", fake_lv)


# ===========================================================================
# 레이어 1 — derive_routes 매트릭스(순수): interaction·correction은 워커 라우트 0
# ===========================================================================

_WORKER_ROUTES = {"research", "rag", "logic_validator"}


@pytest.mark.parametrize("types", [
    ["meta"],
    ["recall"],
    ["tool_help"],
    ["reason"],
    ["correction"],
    # interaction이 content와 섞여도, classify_node가 content를 떨궈 단독으로 남긴 뒤의 입력 형태
    ["meta"],
])
def test_derive_routes_interaction_has_no_worker(types):
    routes = derive_routes(types)
    assert not (_WORKER_ROUTES & set(routes)), (types, routes)
    assert routes == ["none"]


def test_derive_routes_clarification_only_has_no_worker():
    # 명확화는 clarify 라우트만 — 워커 라우트가 아니라서 dispatch 대상이 아니다.
    routes = derive_routes(["clarification_needed"])
    assert not (_WORKER_ROUTES & set(routes))
    assert routes == ["clarify"]


# ===========================================================================
# 레이어 2 — dispatch 노드 + 스파이: 워커 라우트 없는 세그먼트는 호출 0
# ===========================================================================

def _dispatch_state(segments, *, evidence_mode="both"):
    st = initial_state()
    st["turn"] = 1
    st["evidence_mode"] = evidence_mode
    st["turn_segments"] = segments
    return st


@pytest.mark.parametrize("seg", [
    _seg("응 다음", ["meta"], ["none"]),
    _seg("아까 일본 된다며?", ["recall"], ["none"]),
    _seg("솔루션 슬롯이 뭐야?", ["tool_help"], ["none"]),
    _seg("여기서 문제점 추론해봐", ["reason"], ["none"]),
    _seg("그건 취소하자", ["correction"], ["none"]),
    _seg("음 좀 막연하네", ["clarification_needed"], ["clarify"]),
    _seg("오늘 날씨 어때?", ["question"], ["none"], in_scope=False),  # 스코프밖 → routes 덮임
])
def test_dispatch_skips_when_no_worker_route(monkeypatch, seg):
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)

    out = asyncio.run(parallel_dispatch_workers_node(_dispatch_state([seg])))

    assert calls == {}, calls  # 어떤 워커도 안 불렸다
    assert out == {}  # 디스패치 결과 없음


def test_dispatch_skips_empty_subject_even_with_worker_route(monkeypatch):
    # claim이라 워커 라우트는 있으나 canonical_text가 비면 subject가 없어 디스패치 제외.
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    blank = _seg("   ", ["claim"], ["research", "rag", "logic_validator"])

    out = asyncio.run(parallel_dispatch_workers_node(_dispatch_state([blank])))

    assert calls == {}, calls
    assert out == {}


def test_dispatch_skips_all_when_turn_is_all_interaction(monkeypatch):
    # 한 턴에 interaction 세그먼트가 여럿이어도(meta+recall+correction) 워커는 0.
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    segs = [
        _seg("응", ["meta"], ["none"]),
        _seg("아까 뭐랬지", ["recall"], ["none"]),
        _seg("그건 빼자", ["correction"], ["none"]),
    ]

    out = asyncio.run(parallel_dispatch_workers_node(_dispatch_state(segs)))

    assert calls == {}, calls
    assert out == {}


def test_dispatch_evidence_mode_research_skips_rag(monkeypatch):
    # 토글 research → claim이라도 RAG는 안 부른다(웹 리서치만).
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    seg = _seg("일본 웹툰 시장은 성장 중이다", ["claim"], ["research", "rag", "logic_validator"])

    asyncio.run(parallel_dispatch_workers_node(_dispatch_state([seg], evidence_mode="research")))

    assert calls.get("research")  # 웹 리서치는 돈다
    assert "rag" not in calls  # 사내 RAG는 안 돈다
    assert "lv" not in calls  # RAG가 없으니 논리검증도 안 돈다


def test_dispatch_evidence_mode_rag_skips_research(monkeypatch):
    # 토글 rag → 웹 리서치는 안 부른다(사내 문서만).
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    seg = _seg("우리 회사 일본 진출한 적 있어?", ["question"], ["research", "rag"])

    asyncio.run(parallel_dispatch_workers_node(_dispatch_state([seg], evidence_mode="rag")))

    assert "research" not in calls  # 웹 리서치는 안 돈다
    assert calls.get("rag")  # 사내 RAG만 돈다


# ===========================================================================
# 레이어 3 — graph 분기(순수): dispatch 진입 자체를 막는 두 길목
# ===========================================================================

def test_clarify_branch_skips_dispatch_when_clarify_only():
    st = {"turn_segments": [_seg("막연하네", ["clarification_needed"], ["clarify"])]}
    assert _clarify_branch(st) == "conversation"


def test_clarify_branch_dispatches_when_worker_route_present():
    # 명확화가 섞여도 워커 라우트(claim)가 하나라도 있으면 dispatch로 간다.
    st = {"turn_segments": [
        _seg("막연하네", ["clarification_needed"], ["clarify"]),
        _seg("일본 시장 성장 중", ["claim"], ["research", "rag", "logic_validator"]),
    ]}
    assert _clarify_branch(st) == "dispatch"


def test_post_confirm_branch_skips_pipeline_when_consumed():
    # 순수 확인답을 confirm_resolve가 소비하면 segment 이하(→dispatch)를 통째로 우회.
    assert _post_confirm_branch({"confirmation_consumed": True}) == "conversation"


def test_post_confirm_branch_continues_when_not_consumed():
    assert _post_confirm_branch({"confirmation_consumed": False}) == "segment"
    assert _post_confirm_branch({}) == "segment"


# ===========================================================================
# 레이어 4 — run_turn 멀티턴 해네스(실 그래프 + 워커 스파이)
# ===========================================================================

class _Script:
    """이번 턴 노드 LLM이 낼 결과를 테스트가 정한다(턴마다 갱신)."""
    def __init__(self):
        self.text = ""
        self.types = ["meta"]
        self.in_scope = True
        self.confirm = None  # ConfirmOut | None (pending 없으면 confirm 노드가 no-op이라 안 쓰임)
        self.schemas_seen: list[str] = []


def _install_node_stubs(monkeypatch, script):
    """segment/classify/correction/fills/confirm/conversation의 call_json을 스텁한다.

    스키마 클래스명으로 어느 노드인지 가른다 — 한 함수로 6개 노드를 모두 받는다.
    workers는 별도 스파이(_install_worker_spies)로 감시한다.
    """
    async def fake_call_json(system, user, schema, *, reasoning_effort=None):
        name = schema.__name__
        script.schemas_seen.append(name)
        if name == "SegmentOut":
            return SegmentOut(segments=[SegmentItem(text=script.text, canonical_text=script.text)])
        if name == "ClassifyOut":
            return ClassifyOut(items=[ClassifyItem(
                canonical_text=script.text,
                utterance_types=list(script.types),
                in_scope=script.in_scope,
            )])
        if name == "CorrectionOut":
            return CorrectionOut(actions=[])
        if name == "FillOut":
            return FillOut(fills=[])
        if name == "ConfirmOut":
            return script.confirm
        if name == "ConversationOut":
            return ConversationOut(message="(테스트 응답)")
        raise AssertionError(f"예상 못한 스키마: {name}")

    for mod in (
        "agents.orchestrator.nodes.segment",
        "agents.orchestrator.nodes.classify",
        "agents.orchestrator.nodes.correction",
        "agents.orchestrator.nodes.confirm",
        "agents.conversation.agent",
    ):
        monkeypatch.setattr(f"{mod}.call_json", fake_call_json)


# (발화, 유형, in_scope) — 어느 것도 워커를 부르면 안 되는 interaction/스코프밖 멀티턴.
_NO_WORKER_TURNS = [
    ("응 다음으로", ["meta"], True),
    ("아까 일본 된다고 했잖아?", ["recall"], True),
    ("솔루션 슬롯이 뭐하는 칸이야?", ["tool_help"], True),
    ("여기서 문제점 좀 추론해봐", ["reason"], True),
    ("네 생각엔 문제가 뭐야?", ["reason"], True),  # 이번 픽스의 제안요청도 워커 없이 처리
    ("그건 취소하자", ["correction"], True),
    ("음 그건 좀 막연하네", ["clarification_needed"], True),
    ("오늘 서울 날씨 어때?", ["question"], False),  # 스코프밖 → routes 덮여 워커 0
]


def test_multiturn_interaction_never_calls_workers(monkeypatch):
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    script = _Script()
    _install_node_stubs(monkeypatch, script)

    state = initial_state()
    for i, (text, types, in_scope) in enumerate(_NO_WORKER_TURNS):
        script.text, script.types, script.in_scope = text, types, in_scope
        state = asyncio.run(run_turn(state, text))
        # 매 턴 누적 호출이 0으로 유지된다.
        assert calls == {}, f"턴 {i+1} '{text}' 후 워커 호출됨: {calls}"
        # 응답은 매 턴 생성된다(파이프라인이 conversation까지 도달).
        assert state.get("pending_question")

    assert calls == {}


def test_multiturn_confirmation_answer_skips_pipeline(monkeypatch):
    # 열린 제안에 "응"으로 답하면 confirm_resolve가 소비 → segment/classify/dispatch 통째 우회.
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    script = _Script()
    _install_node_stubs(monkeypatch, script)
    script.confirm = ConfirmOut(decision="accept", slot=None, has_additional_content=False)

    state = initial_state()
    state["pending_confirmations"] = [{
        "value": "AI 자동 검수 도구",
        "proposed_slot": "solution",
        "candidate_slots": ["solution"],
        "source_text": "AI 자동 검수 도구",
        "reason": "",
        "attempts": 0,
        "confirm_kind": "commit",
        "previous_value": "",
        "adequate": True,
    }]

    state = asyncio.run(run_turn(state, "응 그걸로"))

    assert calls == {}  # 확인답은 새 리서치 주문이 아니다 — 워커 0
    # 순수 확인답 소비 → segment/classify까지 안 갔다(SegmentOut·ClassifyOut 요청 없음).
    assert "SegmentOut" not in script.schemas_seen
    assert "ClassifyOut" not in script.schemas_seen


def test_multiturn_harness_detects_dispatch_for_claim(monkeypatch):
    # 양성 대조 — 검증 가능한 claim(in_scope) 턴은 실제로 워커를 부른다.
    # (이게 통과해야 위의 '미호출' 단언들이 공허하지 않다.)
    calls: dict = {}
    _install_worker_spies(monkeypatch, calls)
    script = _Script()
    _install_node_stubs(monkeypatch, script)
    script.text, script.types, script.in_scope = "일본 웹툰 시장은 성장 중이다", ["claim"], True

    asyncio.run(run_turn(initial_state(), "일본 웹툰 시장은 성장 중이다"))

    assert calls.get("research"), "claim 턴인데 리서치가 안 불렸다 — 해네스가 dispatch를 못 잡는다"
    assert calls.get("rag")
