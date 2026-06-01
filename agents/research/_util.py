"""리서치 스테이지 공용 유틸 — LLM JSON 출력 파싱·날짜·트레이싱."""
from __future__ import annotations

import json
from datetime import date
from typing import Any


try:  # LangSmith 있으면 실제 트레이서, 없으면 무동작 데코레이터로 폴백(그래프 안 죽게).
    from langsmith import traceable
except ImportError:  # pragma: no cover - langsmith는 보통 langgraph의 전이 의존성으로 깔림.
    def traceable(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        """`@traceable` / `@traceable(...)` 둘 다 받는 무동작 데코레이터."""
        def _wrap(fn: Any) -> Any:
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _wrap


def today_iso() -> str:
    """오늘 날짜(ISO, 예: 2026-06-01) — 검색 쿼리 최신성의 기준점.

    분해기·검색기 입력에 주입해, 모델이 학습 컷오프가 아닌 '오늘'을 기준으로
    최신 연도/기간을 쿼리에 넣도록 한다.
    """
    return date.today().isoformat()


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
