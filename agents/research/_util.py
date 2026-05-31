"""리서치 스테이지 공용 유틸 — LLM JSON 출력 파싱."""
from __future__ import annotations

import json
from typing import Any


def parse_json_block(text: str) -> dict[str, Any]:
    """LLM 응답에서 JSON 객체를 추출한다. ```json 코드펜스가 있으면 벗긴다.

    파싱 실패 시 빈 dict 반환 (상위에서 기본값 처리) — 그래프가 죽지 않도록.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = "\n".join(line for line in s.splitlines() if not line.startswith("```")).strip()
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}
