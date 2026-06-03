"""compose(하이브리드 진입점) — groundedness 가드 + 서술 실패 폴백.

핵심 보장: LLM 서술이 출처 번호를 환각해도(또는 아예 죽어도) 최종 출처는 코드(citations)가 찍는다.
LLM은 호출하지 않는다 — write_narratives를 가짜로 바꿔 결정론적으로 검증.
"""
import asyncio

import agents.planner.compose as compose
from agents.planner.narrative import NarrativeOut, SectionNarrative
from common.schema import ALL_SLOTS


def _state():
    slots = {n: {"value": None, "source_label": "empty", "status": "empty"} for n in ALL_SLOTS}
    for n in ("problem", "target", "goal", "market"):
        slots[n] = {"value": f"값-{n}", "source_label": "user", "status": "filled"}
    records = [{
        "subject": "시장 1.8조", "cluster": "research", "findings": ["1.8조"], "agreement": "confirms",
        "citations": [{"cluster": "research", "title": "콘진원", "url": "https://k/x",
                       "snippet": "1.8조", "score": 0.8, "score_kind": "relevance",
                       "accessed_at": "2026-06-03"}],
        "target_slot": "market", "turn": 2,
    }]
    return {"slots": slots, "session_evidence": records, "output_request": "type2", "correction_log": []}


def test_groundedness_guard_strips_hallucinated_marker(monkeypatch):
    async def fake(slots, records, missing):
        # 모델이 잘못된 번호 [99]를 끼워 넣은 상황
        return NarrativeOut(summary="요약", sections=[
            SectionNarrative(slot="market", prose="국내 웹툰 시장은 성장세다 [99].")])

    monkeypatch.setattr(compose, "write_narratives", fake)
    md = asyncio.run(compose.compose_markdown(_state()))
    assert "[99]" not in md
    assert "국내 웹툰 시장은 성장세다. [1]" in md   # 환각 번호 제거 + 코드 번호 부착
    assert '[1] (리서치) 콘진원' in md


def test_narrative_failure_falls_back_to_deterministic(monkeypatch):
    async def boom(slots, records, missing):
        raise RuntimeError("LLM down / 키 없음")

    monkeypatch.setattr(compose, "write_narratives", boom)
    md = asyncio.run(compose.compose_markdown(_state()))
    # 서술이 죽어도 구조·출처는 그대로 — 값줄에 마커 + 끝 출처목록 정확.
    assert "## 근거 및 출처" in md
    assert "[1] (리서치) 콘진원" in md
    assert "값-market [1]" in md


def test_compose_is_async():
    assert asyncio.iscoroutinefunction(compose.compose_markdown)
