"""LLM 호출 헬퍼 — pydantic JSON 강제 출력.

모드 결정 (`_resolve_mode`):
- `BPM_LLM_MODE=mock`            → mock (등록된 핸들러 → 없으면 스키마 유효 기본값).
- `BPM_LLM_MODE=live`            → 실 호출. 키 없으면 명확한 에러.
- 미설정                         → 키 있으면 live, 없으면 mock 으로 추론.

mock 모드는 핸들러가 없어도 절대 죽지 않는다 — 필수 필드를 타입별 기본값으로
채워 스키마 유효 객체를 만든다. 덕분에 키 없이도 그래프 흐름을 끝까지 돌릴 수 있다.
"""
from __future__ import annotations

import json
import logging
import os
import types
from typing import Callable, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _resolve_mode() -> str:
    """'mock' 또는 'live' 반환."""
    mode = os.environ.get("BPM_LLM_MODE", "").strip().lower()
    if mode == "mock":
        return "mock"
    if mode == "live":
        return "live"
    # 미설정 → 키 유무로 추론
    return "live" if os.environ.get("OPENAI_API_KEY") else "mock"


# --- mock 등록 -------------------------------------------------------------
# key = system prompt 첫 줄.
_MOCK_HANDLERS: dict[str, Callable[[str, str], dict]] = {}


def register_mock(key: str, handler: Callable[[str, str], dict]) -> None:
    """테스트에서 LLM 응답을 가짜로 등록. key는 system prompt 첫 줄과 일치."""
    _MOCK_HANDLERS[key] = handler


def clear_mocks() -> None:
    """명시적(테스트) 핸들러만 비운다. 빌트인 데모 핸들러는 유지."""
    _MOCK_HANDLERS.clear()


_DEFAULT_HANDLERS: dict[str, Callable[[str, str], dict]] | None = None


def _default_handler(first_line: str) -> Callable[[str, str], dict] | None:
    """빌트인 데모 핸들러 조회 (지연 import — 순환 참조 회피)."""
    global _DEFAULT_HANDLERS
    if _DEFAULT_HANDLERS is None:
        try:
            from agents.orchestrator.mocks import DEFAULT_MOCK_HANDLERS

            _DEFAULT_HANDLERS = DEFAULT_MOCK_HANDLERS
        except Exception:  # pragma: no cover - 방어
            _DEFAULT_HANDLERS = {}
    return _DEFAULT_HANDLERS.get(first_line)


def _zero_for(annotation) -> object:
    """타입 어노테이션에 맞는 '빈 값'을 만든다 (필수 필드 채움용)."""
    origin = get_origin(annotation)
    if origin in (list, set, frozenset, tuple):
        return []
    if origin is dict:
        return {}
    if origin in (Union, getattr(types, "UnionType", ())):
        args = get_args(annotation)
        if type(None) in args:  # Optional → None
            return None
        non_none = [a for a in args if a is not type(None)]
        return _zero_for(non_none[0]) if non_none else None
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is str:
        return ""
    return None


def _best_effort_default(schema: type[T]) -> T:
    """핸들러 미등록 시 — 필수 필드만 타입별 기본값으로 채워 유효 객체 생성."""
    data: dict[str, object] = {}
    for name, field in schema.model_fields.items():
        if field.is_required():
            data[name] = _zero_for(field.annotation)
    return schema.model_validate(data)


def _mock_call(system: str, user: str, schema: type[T]) -> T:
    first_line = system.strip().splitlines()[0] if system else ""
    # 우선순위: 명시적(테스트) 핸들러 → 빌트인 데모 핸들러 → 빈 기본값
    handler = _MOCK_HANDLERS.get(first_line) or _default_handler(first_line)
    if handler is not None:
        try:
            return schema.model_validate(handler(system, user))
        except ValidationError:
            logger.warning("mock 핸들러 응답이 %s 스키마에 안 맞음 → 빈 기본값.", schema.__name__)

    # 핸들러 없거나 검증 실패 → 죽지 않고 빈 기본값 (그래프 흐름 확인용)
    try:
        return schema()  # type: ignore[call-arg]
    except ValidationError:
        return _best_effort_default(schema)


# --- 실 호출 ---------------------------------------------------------------
async def call_json(system: str, user: str, schema: type[T]) -> T:
    """system/user 프롬프트로 LLM 호출 → schema 인스턴스 반환."""
    if _resolve_mode() == "mock":
        return _mock_call(system, user, schema)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "BPM_LLM_MODE=live 인데 OPENAI_API_KEY가 없습니다. "
            ".env에 키를 넣거나 BPM_LLM_MODE=mock으로 실행하세요."
        )

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
