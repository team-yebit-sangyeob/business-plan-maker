"""LLM 호출 헬퍼 — pydantic 구조화 출력.

mock/live 모드 분기는 없다. `OPENAI_API_KEY`가 없으면 실행 자체가 막힌다(fail-fast) —
`call_json`은 키가 없으면 즉시 RuntimeError를 던진다. 모델은 `BPM_LLM_MODEL`로 교체 가능
(키 확인·모델 읽기는 `common/config.py`의 `require_openai_key`·`orchestrator_model` 경유).

추론 강도는 `BPM_LLM_REASONING`(기본 low)으로 조절한다 — 기본 모델 gpt-5.4-mini는 추론
모델이라 reasoning_effort 미설정이면 서버 기본(=medium) 추론으로 돌아 느리다. 추론 모델일
때만 ChatOpenAI에 reasoning_effort로 넘긴다(비추론 모델은 이 파라미터를 거부).

구조화 출력은 LangChain `with_structured_output`이 맡는다 — 스키마 변환·함수콜 강제·파싱·
pydantic 검증을 한 번에. (수동 JSON 스키마 주입+json.loads+model_validate를 대체.)
"""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from common.config import (
    orchestrator_model,
    orchestrator_reasoning_effort,
    require_openai_key,
)


T = TypeVar("T", bound=BaseModel)


async def call_json(
    system: str,
    user: str,
    schema: type[T],
    *,
    reasoning_effort: str | None = None,
) -> T:
    """system/user 프롬프트로 LLM 호출 → schema 인스턴스 반환.

    키가 없으면 RuntimeError. `with_structured_output`이 함수콜로 스키마를 강제하고 검증된
    pydantic 인스턴스를 돌려준다. 전이 오류(429·timeout) 재시도는 그래프 노드의 RetryPolicy가
    일원화하므로 여기선 따로 두지 않는다(planner의 호출은 그래프 밖이라 전이 오류가 그대로
    전파되지만, 기존에도 그 루프는 검증 오류만 잡았을 뿐 전이 오류는 전파했다 — 동작 동일).

    reasoning_effort: 노드별로 추론 강도를 낮추고 싶을 때만 넘긴다(기본은
    BPM_LLM_REASONING). 추론 모델(gpt-5·o계열)에만 적용된다.
    """
    require_openai_key("app")

    from langchain_openai import ChatOpenAI

    model = orchestrator_model()
    effort = reasoning_effort or orchestrator_reasoning_effort()
    kwargs: dict = {"model": model, "temperature": 0}
    # 추론 모델일 때만 reasoning_effort를 보낸다 — 비추론 모델은 이 파라미터를 거부한다.
    # gpt-5 비-chat은 langchain-openai가 temperature를 자동 제거하므로 둘을 같이 둬도 안전하다.
    if model.startswith(("gpt-5", "o1", "o3", "o4")) and "chat" not in model:
        kwargs["reasoning_effort"] = effort

    llm = ChatOpenAI(**kwargs).with_structured_output(schema, method="function_calling")
    return await llm.ainvoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
