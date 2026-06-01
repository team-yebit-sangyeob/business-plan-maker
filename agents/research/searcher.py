"""검색·평가·재작성 루프 — web_search 툴을 가진 단일 에이전트.

rag_extractor._run_agent 패턴을 복제하되 툴이 search_vector_db 대신 web_search다.
결과 score가 낮거나 주제와 안 맞으면 에이전트가 스스로 쿼리를 재작성해 재검색한다
(별도 rewriter 단위 없이 루프 안에서 흡수). max_turns 초과 시 빈 근거 반환 →
리포터가 agreement=unknown 처리.
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from agents.research._util import parse_json_block, today_iso, traceable
from agents.research.provider import web_search


MAX_SEARCH_TURNS = 8


class Evidence(TypedDict):
    snippet: str
    url: str
    title: str
    score: float


_WEB_SEARCH_TOOL = [
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "주어진 한국어 쿼리로 웹을 검색해 제목·URL·본문 스니펫·관련도(score)를 돌려줍니다. "
            "score가 낮거나(0.5 미만) 결과가 주제와 안 맞으면 쿼리를 바꿔 다시 호출하세요."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색할 한국어 쿼리"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }
]

_SEARCHER_SYSTEM = f"""당신은 웹 사실 검증 검색 에이전트입니다.

주어진 sub-query들로 web_search 툴을 호출해 주장 검증에 쓸 근거를 모으세요.

[절차]
1. 각 sub-query로 web_search를 호출하세요.
2. score가 낮거나(0.5 미만) 결과가 주제와 안 맞으면 쿼리를 재작성해 다시 검색하세요.
3. 시점이 중요한 주제(시장 규모·트렌드·통계 등)는 입력의 '오늘 날짜'를 기준으로 최신 자료를 우선하세요. 관련도가 비슷하면 더 최근(연·월이 빠른) 자료를 고르고, 옛 연도 자료는 후순위로 미루세요.
4. 총 {MAX_SEARCH_TURNS}회 이내로 검색하세요. 쓸 만한 근거가 모이면 멈추세요.
5. 수집한 근거를 관련도 높은 순으로 최대 6개까지 아래 JSON으로 출력하세요.

검색 결과의 content를 통째로 길게 붙이지 말고, 주장과 직접 관련된 부분만 한 줄로 추리세요. 가능하면 근거의 시점(연·월)을 snippet에 드러내세요.

[최종 출력 — 반드시 JSON]
{{"evidence": [{{"snippet": "근거 한 줄 요약/인용", "url": "출처 URL", "title": "출처 제목", "score": 0.0}}]}}"""


def _norm_evidence(e: dict[str, Any]) -> Evidence:
    return {
        "snippet": str(e.get("snippet", "")).strip(),
        "url": str(e.get("url", "")).strip(),
        "title": str(e.get("title", "")).strip(),
        "score": float(e.get("score", 0.0) or 0.0),
    }


@traceable(name="research.gather_evidence", run_type="chain")
def gather_evidence(
    client: Any,
    subqueries: list[str],
    *,
    freshness_days: int | None,
    model: str,
    max_turns: int = MAX_SEARCH_TURNS,
) -> list[Evidence]:
    """web_search 툴 루프로 근거를 모은다 (동기). 근거 없으면 빈 리스트."""
    if not subqueries:
        return []

    user = (
        f"오늘 날짜: {today_iso()}\n"
        "sub-queries:\n" + "\n".join(f"- {q}" for q in subqueries)
    )
    previous_response_id: str | None = None
    current_input: list[dict[str, Any]] = [{"role": "user", "content": user}]

    for _ in range(max_turns):
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": _SEARCHER_SYSTEM,
            "input": current_input,
            "tools": _WEB_SEARCH_TOOL,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
        previous_response_id = response.id

        tool_outputs = []
        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                hits = web_search(args.get("query", ""), freshness_days=freshness_days)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(hits, ensure_ascii=False),
                    }
                )

        # 툴 호출이 없으면 에이전트가 최종 JSON을 낸 것.
        if not tool_outputs:
            data = parse_json_block(response.output_text)
            return [
                _norm_evidence(e)
                for e in (data.get("evidence") or [])
                if isinstance(e, dict)
            ][:6]

        current_input = tool_outputs

    # max_turns 초과 — 근거 없이 반환 (리포터가 unknown 처리).
    return []
