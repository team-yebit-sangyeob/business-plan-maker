"""분해기 — claim + slot_context → 검증 가능한 한국어 sub-query 1~3개.

가설·결론의 비약 자체는 검증 대상이 아니다(그건 비평 몫). 여기서는 웹에서
사실로 확인 가능한 '전제'만 골라 검색 쿼리로 만든다.
"""
from __future__ import annotations

import json
from typing import Any

from common.schema import VerificationRequest
from agents.research._util import parse_json_block, today_iso, traceable


_DECOMPOSER_SYSTEM = """당신은 사실 검증 쿼리 분해 전문가입니다.

사용자 주장(claim)과 슬롯 컨텍스트(slot_context)를 보고, 웹에서 검증 가능한
한국어 검색 쿼리를 1~3개 만드세요.

[규칙]
- 검증 가능한 '사실·전제'만 쿼리로 만드세요. "우리 IP가 통할 것이다" 같은 결론·비약 자체는 쿼리로 만들지 마세요(검증 대상은 그 밑의 전제: "일본 웹툰 시장 성장", "한류 콘텐츠 수용도").
- slot_context(타겟·시장 등)가 있으면 단서로 활용해 쿼리를 구체화하세요.
- 각 쿼리는 독립적으로 검색 가능한 짧은 한국어 문장 또는 키워드 조합.
- 검색은 입력의 '오늘 날짜' 기준으로 가장 최근 데이터를 노립니다. 시장 규모·트렌드·통계처럼 시점이 중요한 주제는 올해 또는 직전 12개월 연도/기간을 쿼리에 명시하고, 철 지난 옛 연도는 쓰지 마세요.

반드시 아래 JSON 형식으로만 출력하세요:
{"subqueries": ["쿼리1", "쿼리2"]}"""


@traceable(name="research.decompose", run_type="chain")
def decompose(client: Any, req: VerificationRequest, *, model: str) -> list[str]:
    """claim/slot_context를 보고 검색용 sub-query 리스트를 생성한다 (동기)."""
    claim = (req.get("claim") or "").strip()
    slot_context = req.get("slot_context") or {}

    user = (
        f"오늘 날짜: {today_iso()}\n"
        f"claim: {claim}\n"
        f"slot_context: {json.dumps(slot_context, ensure_ascii=False)}"
    )
    resp = client.responses.create(
        model=model,
        instructions=_DECOMPOSER_SYSTEM,
        input=[{"role": "user", "content": user}],
    )
    data = parse_json_block(resp.output_text)
    subs = [
        s.strip()
        for s in (data.get("subqueries") or [])
        if isinstance(s, str) and s.strip()
    ]
    # 분해 실패 시 claim 자체를 단일 쿼리로 폴백.
    return subs[:3] or ([claim] if claim else [])
