"""리서치 클러스터 단위 테스트.

openai/tavily 미설치 환경에서도 돌도록, OpenAI Responses client는 가짜 객체를
주입하고 web_search는 monkeypatch한다. (실 LLM/검색 호출 없음.)
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agents.research import run_research
from agents.research import research_main
from agents.research.provider import web_search, _freshness_to_time_range
from agents.research.decomposer import decompose
from agents.research.searcher import gather_evidence
from agents.research.reporter import write_report
import agents.research.searcher as searcher_mod


# --- 가짜 OpenAI Responses client ------------------------------------------

class _Item:
    def __init__(self, *, type, name=None, arguments=None, call_id=None):
        self.type = type
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _Resp:
    def __init__(self, *, output_text="", output=None, id="resp", status="completed"):
        self.output_text = output_text
        self.output = output or []
        self.id = id
        self.status = status


class _FakeResponses:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.responses = _FakeResponses(scripted)


def _fn_call(query, call_id="c1"):
    return _Resp(
        output=[
            _Item(
                type="function_call",
                name="web_search",
                arguments=json.dumps({"query": query}),
                call_id=call_id,
            )
        ]
    )


# --- provider --------------------------------------------------------------

def test_freshness_buckets():
    assert _freshness_to_time_range(None) is None
    assert _freshness_to_time_range(0) is None
    assert _freshness_to_time_range(1) == "day"
    assert _freshness_to_time_range(7) == "week"
    assert _freshness_to_time_range(30) == "month"
    assert _freshness_to_time_range(180) == "year"


def test_web_search_mock_mode(monkeypatch):
    monkeypatch.setenv("BPM_LLM_MODE", "mock")
    hits = web_search("한국 게임 시장", max_results=3)
    assert len(hits) == 1
    assert hits[0]["url"] and "score" in hits[0]


# --- decomposer ------------------------------------------------------------

def test_decompose_parses_subqueries():
    client = FakeClient(
        [_Resp(output_text='{"subqueries": ["게임 시장 성장률 2024", "신규 진입 현황"]}')]
    )
    subs = decompose(client, {"claim": "게임 시장 포화", "slot_context": {}}, model="m")
    assert subs == ["게임 시장 성장률 2024", "신규 진입 현황"]


def test_decompose_fallback_to_claim_on_empty():
    client = FakeClient([_Resp(output_text='{"subqueries": []}')])
    subs = decompose(client, {"claim": "게임 시장 포화"}, model="m")
    assert subs == ["게임 시장 포화"]


# --- searcher --------------------------------------------------------------

def test_searcher_tool_loop(monkeypatch):
    calls = {}

    def fake_web_search(query, **kw):
        calls["query"] = query
        calls["kw"] = kw
        return [{"title": "t", "url": "https://a", "content": "c", "score": 0.9}]

    monkeypatch.setattr(searcher_mod, "web_search", fake_web_search)

    client = FakeClient(
        [
            _fn_call("게임 시장 성장률"),
            _Resp(
                output_text='{"evidence":[{"snippet":"+5% YoY","url":"https://a","title":"t","score":0.9}]}'
            ),
        ]
    )
    ev = gather_evidence(client, ["게임 시장 성장률"], freshness_days=180, model="m")
    assert calls["query"] == "게임 시장 성장률"
    assert calls["kw"].get("freshness_days") == 180
    assert ev and ev[0]["url"] == "https://a" and ev[0]["score"] == 0.9


def test_searcher_empty_on_no_subqueries():
    client = FakeClient([])
    assert gather_evidence(client, [], freshness_days=None, model="m") == []


# --- reporter --------------------------------------------------------------

@pytest.mark.parametrize("verdict", ["confirms", "contradicts", "partial", "unknown"])
def test_reporter_agreement_passthrough(verdict):
    client = FakeClient(
        [
            _Resp(
                output_text=json.dumps(
                    {"findings": ["f"], "sources": ["https://a"], "agreement": verdict}
                )
            )
        ]
    )
    out = write_report(client, "claim", [], model="m")
    assert out["agreement"] == verdict


def test_reporter_invalid_agreement_defaults_unknown():
    client = FakeClient(
        [_Resp(output_text='{"findings":["f"],"sources":[],"agreement":"yes"}')]
    )
    out = write_report(client, "claim", [], model="m")
    assert out["agreement"] == "unknown"


# --- run_research (stub paths) ---------------------------------------------

def test_run_research_stub_in_mock_mode(monkeypatch):
    monkeypatch.setenv("BPM_LLM_MODE", "mock")
    r = asyncio.run(run_research({"claim": "한국 게임 시장이 포화 상태다"}))
    assert r["cluster"] == "research"
    assert r["agreement"] == "unknown"
    assert r["subject"] == "한국 게임 시장이 포화 상태다"


def test_run_research_accepts_legacy_string(monkeypatch):
    monkeypatch.setenv("BPM_LLM_MODE", "mock")
    r = asyncio.run(run_research("게임 시장 포화"))
    assert r["cluster"] == "research" and r["subject"] == "게임 시장 포화"


# --- run_research (full pipeline with fake client) -------------------------

def test_run_research_end_to_end(monkeypatch):
    monkeypatch.delenv("BPM_LLM_MODE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_web_search(query, **kw):
        return [{"title": "콘텐츠진흥원", "url": "https://a", "content": "+5% YoY", "score": 0.9}]

    monkeypatch.setattr(searcher_mod, "web_search", fake_web_search)

    scripted = [
        _Resp(output_text='{"subqueries": ["게임 시장 성장률 2024"]}'),  # decompose
        _fn_call("게임 시장 성장률 2024"),                                # searcher turn 1
        _Resp(
            output_text='{"evidence":[{"snippet":"+5% YoY","url":"https://a","title":"t","score":0.9}]}'
        ),  # searcher final
        _Resp(
            output_text='{"findings":["2024 +5% 성장"],"sources":["https://a","https://a","https://b"],"agreement":"contradicts"}'
        ),  # reporter
    ]
    monkeypatch.setattr(research_main, "_make_client", lambda: FakeClient(scripted))

    r = asyncio.run(
        run_research(
            {"claim": "게임 시장이 포화 상태다", "slot_context": {}, "freshness_max_days": 180}
        )
    )
    assert r["cluster"] == "research"
    assert r["agreement"] == "contradicts"
    assert r["findings"] == ["2024 +5% 성장"]
    assert r["sources"] == ["https://a", "https://b"]  # 중복 제거
    assert r["subject"] == "게임 시장이 포화 상태다"
