"""dispatch 2단계 게이트 테스트 — logic_validator는 RAG 근거가 있을 때만 돈다.

회귀 배경: research 전용 모드(RAG 끔)에서도 logic_validator가 무조건 디스패치돼
rag_result=None을 받고 "검증할 RAG 근거가 없습니다"(판단 보류) 카드를 냈다. 픽스는
① logic_validator를 같은 idx의 rag_result가 있을 때만 호출(research 전용·RAG 미회수 시 제외),
② research를 logic_validator 판정에 섞지 않음(근거판단 분리) — run_logic_validator는 (subject, rag_result) 2인자.

dispatch는 워커를 함수 안에서 import한다(순환 import 회피) — 패치 타깃은 원천 패키지
(agents.research.run_research / agents.rag.run_rag_check / agents.logic_validator.run_logic_validator).
"""
import asyncio

import agents.orchestrator.graph  # noqa: F401  -- 순환 import 순서 고정(먼저 로드)

from common.schema.state import initial_state
from agents.orchestrator.nodes.dispatch import parallel_dispatch_workers_node


def _claim_seg(text):
    return {
        "text": text,
        "canonical_text": text,
        "utterance_types": ["claim"],
        "in_scope": True,
        "target_slot": None,
        "routes": ["research", "rag", "logic_validator"],
    }


def _report(subject, *, cluster, agreement="unknown", findings=None):
    return {
        "subject": subject,
        "cluster": cluster,
        "findings": findings or [],
        "agreement": agreement,
        "citations": [],
    }


def _install_workers(monkeypatch, *, rag_result, calls):
    """research/rag/logic_validator 워커를 스텁하고 호출을 calls에 기록한다."""

    async def fake_research(req):
        calls.setdefault("research", []).append(req)
        return _report(req["claim"], cluster="research", agreement="confirms", findings=["외부 사실 X"])

    async def fake_rag(subject):
        calls.setdefault("rag", []).append(subject)
        return _report(subject, cluster="rag"), rag_result

    async def fake_lv(*args, **kwargs):
        calls.setdefault("lv", []).append((args, kwargs))
        return _report(args[0] if args else "", cluster="logic_validator")

    monkeypatch.setattr("agents.research.run_research", fake_research)
    monkeypatch.setattr("agents.rag.run_rag_check", fake_rag)
    monkeypatch.setattr("agents.logic_validator.run_logic_validator", fake_lv)


def _state(evidence_mode):
    st = initial_state()
    st["turn"] = 1
    st["evidence_mode"] = evidence_mode
    st["turn_segments"] = [_claim_seg("일본 웹툰 시장은 성장 중이다")]
    return st


def test_lv_skipped_in_research_only_mode(monkeypatch):
    # research 전용: rag 자체가 안 돌고(rag_result 없음) → logic_validator도 안 돈다.
    calls: dict = {}
    _install_workers(monkeypatch, rag_result=None, calls=calls)

    out = asyncio.run(parallel_dispatch_workers_node(_state("research")))

    assert calls.get("research")  # 웹 리서치는 돈다
    assert "rag" not in calls  # 사내 RAG는 안 돈다(토글)
    assert "lv" not in calls  # 논리검증도 안 돈다 → "근거 없음" 카드 없음
    clusters = {r.get("cluster") for r in out.get("turn_validation_reports", [])}
    assert "logic_validator" not in clusters


def test_lv_skipped_when_rag_finds_nothing(monkeypatch):
    # both 모드라도 RAG가 근거를 못 찾으면(rag_result=None) logic_validator는 제외.
    calls: dict = {}
    _install_workers(monkeypatch, rag_result=None, calls=calls)

    out = asyncio.run(parallel_dispatch_workers_node(_state("both")))

    assert calls.get("rag")  # rag는 디스패치됨(both)
    assert "lv" not in calls  # 그러나 근거가 없어 논리검증은 생략
    clusters = {r.get("cluster") for r in out.get("turn_validation_reports", [])}
    assert "logic_validator" not in clusters


def test_lv_runs_rag_only_when_rag_present(monkeypatch):
    # RAG 근거가 있으면 logic_validator가 (subject, rag_result) 2인자로만 호출된다 — research 미합류.
    rag_result = {"qk": "...", "claim": "일본 웹툰 고성장", "highlight": "사내 +18%"}
    calls: dict = {}
    _install_workers(monkeypatch, rag_result=rag_result, calls=calls)

    asyncio.run(parallel_dispatch_workers_node(_state("both")))

    assert len(calls.get("lv", [])) == 1
    args, kwargs = calls["lv"][0]
    assert args == ("일본 웹툰 시장은 성장 중이다", rag_result)  # 정확히 2인자
    assert not kwargs  # research_report 등 추가 인자 없음(근거판단 분리)
