"""LLM 호출 헬퍼 — pydantic JSON 강제 출력.

`BPM_LLM_MODE=mock` 또는 키 없음 → 등록된 mock 응답 반환 (테스트용).
실 호출은 langchain-openai ChatOpenAI.
"""
from __future__ import annotations

import json
import os
from typing import Awaitable, Callable, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def _is_mock_mode() -> bool:
    if os.environ.get("BPM_LLM_MODE", "").lower() == "mock":
        return True
    if not os.environ.get("OPENAI_API_KEY"):
        return True
    return False


# --- mock 등록 -------------------------------------------------------------
# key = (schema_qualname, marker) — marker는 system prompt 첫 줄 또는 명시 키.
_MOCK_HANDLERS: dict[str, Callable[[str, str], dict]] = {}


def register_mock(key: str, handler: Callable[[str, str], dict]) -> None:
    """테스트에서 LLM 응답을 가짜로 등록. key는 system prompt 첫 줄과 일치."""
    _MOCK_HANDLERS[key] = handler


def clear_mocks() -> None:
    _MOCK_HANDLERS.clear()


def _mock_call(system: str, user: str, schema: type[T]) -> T:
    first_line = system.strip().splitlines()[0] if system else ""
    handler = _MOCK_HANDLERS.get(first_line)
    if handler is None:
        # 등록 안 됐으면 빈 객체로 시도
        try:
            return schema()  # type: ignore[call-arg]
        except ValidationError as exc:  # 필수 필드 있으면 명시적 실패
            raise RuntimeError(
                f"mock LLM 응답이 등록되지 않았고 schema {schema.__name__} 기본값으로도 만들 수 없음: {exc}"
            ) from exc
    payload = handler(system, user)
    return schema.model_validate(payload)


# --- 실 호출 ---------------------------------------------------------------
async def call_json(system: str, user: str, schema: type[T]) -> T:
    """system/user 프롬프트로 LLM 호출 → schema 인스턴스 반환."""
    if _is_mock_mode():
        return _mock_call(system, user, schema)

    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("BPM_LLM_MODEL", "gpt-5-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    json_schema_hint = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    augmented_system = (
        system.strip()
        + "\n\n반드시 다음 JSON 스키마에 맞는 JSON 객체만 출력하라:\n"
        + json_schema_hint
    )

    last_error: Exception | None = None
    for _ in range(2):
        try:
            resp = await llm.ainvoke(
                [
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user},
                ]
            )
            raw = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = json.loads(raw)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            continue
    raise RuntimeError(f"LLM JSON 파싱 실패: {last_error}")
