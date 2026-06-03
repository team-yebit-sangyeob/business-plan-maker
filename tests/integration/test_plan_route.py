"""POST /plan 통합 — 라우트 결선(async) + 카드/마크다운 산출 + Type 0 거절.

LLM(서술)은 가짜로 바꿔 키 없이도 결정론적으로 돈다 — 검증 대상은 라우트 배선과 출처 정확성.
"""
from fastapi.testclient import TestClient

import agents.planner.compose as compose
from agents.planner.narrative import NarrativeOut, SectionNarrative
from api_server.main import app
from api_server.session_store import get_store


def _seed_filled_session():
    store = get_store()
    sid, state = store.create()
    slots = state["slots"]
    for n in ("problem", "target", "goal"):
        slots[n] = {"value": f"값-{n}", "source_label": "user", "status": "filled"}
    slots["market"] = {"value": "국내 웹툰 시장 성장세", "source_label": "user", "status": "filled"}
    state["session_evidence"] = [{
        "subject": "시장 1.8조", "cluster": "research", "findings": ["1.8조"], "agreement": "confirms",
        "citations": [{"cluster": "research", "title": "콘진원 2024 백서", "url": "https://k/x",
                       "snippet": "매출 1.8조", "score": 0.82, "score_kind": "relevance",
                       "accessed_at": "2026-06-03"}],
        "target_slot": "market", "turn": 2,
    }]
    store.update(sid, state)
    return store, sid


def test_plan_route_returns_card_and_cited_markdown(monkeypatch):
    async def fake(slots, records, missing):
        return NarrativeOut(summary="웹툰 감수 사업이다.",
                            sections=[SectionNarrative(slot="market", prose="국내 웹툰 시장은 성장세다.")])

    monkeypatch.setattr(compose, "write_narratives", fake)
    store, sid = _seed_filled_session()

    resp = TestClient(app).post("/plan", json={"session_id": sid})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["plan_id"]
    assert body["download_url"].endswith("/download")

    md = store.get_pdf(body["plan_id"])["markdown"]
    assert "# 사업 계획서" in md
    assert "## 근거 및 출처" in md
    assert "[1] (리서치) 콘진원 2024 백서" in md
    assert "국내 웹툰 시장은 성장세다. [1]" in md


def test_plan_route_rejects_when_required_slots_missing():
    store = get_store()
    sid, state = store.create()  # 전부 빈 슬롯 → 필수 미달(Type 0)
    store.update(sid, state)

    resp = TestClient(app).post("/plan", json={"session_id": sid})
    assert resp.status_code == 400
    assert resp.json()["detail"]["code"] == "required_slots_missing"
