"""웹 검색 추상화 — 프로바이더 교체를 1파일에 가둔다.

기본 프로바이더는 Tavily. `RESEARCH_SEARCH_PROVIDER` 환경변수로 선택한다.
Exa로 갈아끼우려면 `_exa_search`만 구현하고 env를 바꾸면 끝 — searcher/reporter는
`web_search`의 반환 타입(SearchHit)만 보므로 상위 코드는 손대지 않는다.

키가 없거나 BPM_LLM_MODE=mock 이면 결정적 mock hit을 돌려준다 — 키 없이도
오케스트레이터 그래프가 끝까지 돌도록(orchestrator/llm.py 의 mock 철학과 동일).
"""
from __future__ import annotations

import os
from typing import Any, TypedDict

from agents.research._util import traceable


class SearchHit(TypedDict):
    title: str
    url: str
    content: str   # 본문 스니펫 (LLM 컨텍스트 주입용)
    score: float   # 관련도 0~1 (프로바이더 제공, 없으면 0.0)


def _freshness_to_time_range(freshness_days: int | None) -> str | None:
    """검색 recency 힌트(일) → Tavily time_range 버킷. None이면 기간 제한 없음."""
    if not freshness_days or freshness_days <= 0:
        return None
    if freshness_days <= 2:
        return "day"
    if freshness_days <= 10:
        return "week"
    if freshness_days <= 45:
        return "month"
    return "year"


def _is_mock() -> bool:
    return os.environ.get("BPM_LLM_MODE", "").strip().lower() == "mock"


def _mock_hits(query: str, max_results: int) -> list[SearchHit]:
    hit: SearchHit = {
        "title": f"[mock] {query[:40]} 관련 자료",
        "url": "https://example.com/mock",
        "content": f"[mock] '{query}' 외부 검색이 비활성화돼 있습니다 (검색 키 없음 또는 mock 모드).",
        "score": 0.0,
    }
    return [hit][:max_results]


def _tavily_search(
    query: str, max_results: int, freshness_days: int | None, topic: str
) -> list[SearchHit]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    kwargs: dict[str, Any] = {
        "search_depth": "advanced",
        "include_answer": False,
        "max_results": max_results,
        "topic": topic,
    }
    time_range = _freshness_to_time_range(freshness_days)
    if time_range:
        kwargs["time_range"] = time_range

    resp = client.search(query, **kwargs)
    hits: list[SearchHit] = []
    for r in resp.get("results", []):
        hits.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": float(r.get("score", 0.0) or 0.0),
            }
        )
    return hits


@traceable(name="research.web_search", run_type="tool")
def web_search(
    query: str,
    *,
    max_results: int = 5,
    freshness_days: int | None = None,
    topic: str = "general",
) -> list[SearchHit]:
    """한 쿼리로 웹을 검색해 SearchHit 리스트를 반환 (동기).

    상위에서 asyncio.to_thread로 감싸 호출한다(이벤트 루프 비차단).
    """
    provider = os.environ.get("RESEARCH_SEARCH_PROVIDER", "tavily").strip().lower()

    if _is_mock():
        return _mock_hits(query, max_results)

    if provider == "tavily":
        if not os.environ.get("TAVILY_API_KEY"):
            return _mock_hits(query, max_results)
        return _tavily_search(query, max_results, freshness_days, topic)

    if provider == "exa":
        # Exa 어댑터 자리 — 인터페이스(SearchHit)는 고정. 구현 시 여기만 채우면 된다.
        raise NotImplementedError(
            "Exa 검색 어댑터는 아직 미구현입니다. provider.py 의 _exa_search를 추가하세요."
        )

    raise ValueError(f"알 수 없는 검색 프로바이더: {provider!r} (tavily | exa)")
