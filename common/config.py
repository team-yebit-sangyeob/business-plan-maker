"""환경설정 단일 표면 — 흩어져 있던 env/모델/키 읽기를 한 곳에 모은다.

호출부는 os.environ을 직접 읽지 않고 여기 접근자만 쓴다. 각 접근자는 종전 인라인
읽기와 같은 env 변수·같은 기본값을 반환한다(동작 불변, 중앙화만 한다).

[모델 노브가 둘인 이유]
오케스트레이터와 리서치는 LLM 호출 표면이 다르다 — 오케는 LangChain ChatOpenAI(JSON 모드),
리서치는 OpenAI Responses API를 직접 쓴다. 그래서 모델 env도 의도적으로 갈라져 있다:
  - BPM_LLM_MODEL  → 오케스트레이터(call_json)
  - OPENAI_MODEL   → 리서치 파이프라인
두 기본값이 다른 것도 이 분리에서 온다. 하나로 합치지 않고 둘을 그대로 둔다.

키가 없으면 mock 경로 없이 fail-fast 한다 — require_* 가 명확한 RuntimeError를 던지고,
상위(서버 기동·call_json·run_research)가 받아 처리한다.
"""
from __future__ import annotations

import os
from typing import Literal


# --- 모델 -------------------------------------------------------------------
def orchestrator_model() -> str:
    """오케스트레이터 LLM(call_json) 모델명 — BPM_LLM_MODEL."""
    return os.environ.get("BPM_LLM_MODEL", "gpt-5-mini")


def research_model() -> str:
    """리서치 파이프라인 모델명 — OPENAI_MODEL(오케와 별개)."""
    return os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


# --- OpenAI 키 --------------------------------------------------------------
def openai_api_key() -> str | None:
    """OpenAI API 키(없으면 None) — 리서치 클라이언트 생성자에 넘긴다."""
    return os.getenv("OPENAI_API_KEY")


# 키 부재 시 거부 문구는 호출 맥락마다 다르게 노출돼 왔다(사용자 노출 문구라 합치지 않는다).
_OPENAI_KEY_MESSAGES: dict[str, str] = {
    "app": (
        "OPENAI_API_KEY가 없습니다 — 이 앱은 키 없이 실행되지 않습니다. "
        ".env에 OPENAI_API_KEY를 넣으세요."
    ),
    "server": (
        "OPENAI_API_KEY가 없습니다 — api_server는 키 없이 기동할 수 없습니다. "
        ".env에 OPENAI_API_KEY를 넣으세요."
    ),
}


def require_openai_key(context: Literal["app", "server"]) -> None:
    """OPENAI_API_KEY 존재를 강제한다 — 없으면 맥락별 문구로 RuntimeError(fail-fast)."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(_OPENAI_KEY_MESSAGES[context])


# --- 리서치 웹 검색 ---------------------------------------------------------
def search_provider() -> str:
    """웹 검색 프로바이더 선택 — RESEARCH_SEARCH_PROVIDER(기본 tavily)."""
    return os.environ.get("RESEARCH_SEARCH_PROVIDER", "tavily").strip().lower()


def tavily_api_key() -> str | None:
    """Tavily API 키(없으면 None)."""
    return os.environ.get("TAVILY_API_KEY")


def require_tavily_key() -> str:
    """Tavily 키 존재를 강제한다 — 없으면 RuntimeError(상위 run_research가 폴백)."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError(
            "TAVILY_API_KEY가 없습니다 — 리서치 웹 검색을 실행할 수 없습니다."
        )
    return key
