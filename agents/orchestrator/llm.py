"""LLM 호출 헬퍼 — pydantic JSON 강제 출력.

mock/live 모드 분기는 없다. `OPENAI_API_KEY`가 없으면 실행 자체가 막힌다(fail-fast) —
`call_json`은 키가 없으면 즉시 RuntimeError를 던진다. 모델은 `BPM_LLM_MODEL`로 교체 가능.
"""
from __future__ import annotations

import json
import logging
import os
from typing import TypeVar

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def call_json(system: str, user: str, schema: type[T]) -> T:
    """system/user 프롬프트로 LLM 호출 → schema 인스턴스 반환.

    키가 없으면 RuntimeError. 응답 JSON이 스키마에 안 맞으면 1회 재시도 후 실패한다.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 없습니다 — 이 앱은 키 없이 실행되지 않습니다. "
            ".env에 OPENAI_API_KEY를 넣으세요."
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
