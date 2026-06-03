"""환경설정 표면 — 흩어져 있던 env/모델/키 읽기를 한 곳에 모은다.

오케스트레이터·리서치 진입점·검색 프로바이더의 env 읽기를 여기 접근자 뒤로 모았다.
이 호출부들은 os.environ을 직접 읽지 않는다. 각 접근자는 종전 인라인 읽기와 같은 env
변수·같은 기본값을 반환한다(동작 불변, 중앙화만 한다).

범위 밖: rag·validator 워커(agents/rag·agents/validator)는 아직 OPENAI_MODEL·OPENAI_API_KEY·
DOCS_BASE_PATH 등을 직접 읽는다 — 이번 정리 범위가 아니라 이 표면을 거치지 않는다. 그래서
OPENAI_MODEL은 research_model()과 rag 양쪽에서 읽힌다(같은 기본값).

[모델 노브가 둘인 이유]
오케스트레이터와 리서치는 LLM 호출 표면이 다르다 — 오케는 LangChain ChatOpenAI(JSON 모드),
리서치는 OpenAI Responses API를 직접 쓴다. 그래서 모델 env도 의도적으로 갈라져 있다:
  - BPM_LLM_MODEL  → 오케스트레이터(call_json)
  - OPENAI_MODEL   → 리서치 파이프라인(rag·validator도 같은 env를 직접 읽는다)
둘은 같은 gpt-5.4-mini 기본을 쓰지만 표면(LangChain vs Responses API)이 달라 노브를 합치지
않고 그대로 둔다.

[추론 강도 노브]
오케 기본 모델(gpt-5.4-mini)은 추론 모델이라 reasoning_effort 미설정이면 서버 기본(=medium)
추론으로 돌아 한 턴(segment·classify·gate·conversation 등 순차 호출)이 느려진다. 그래서
BPM_LLM_REASONING(기본 low)으로 추론 강도를 낮춰 지연을 줄인다 — call_json이 추론 모델일 때만
ChatOpenAI에 reasoning_effort로 넘긴다.

[키 부재 처리는 키마다 다르다]
  - OPENAI_API_KEY 없음 → 하드 차단: require_openai_key가 RuntimeError로 기동/호출을 막는다.
  - TAVILY_API_KEY 없음 → 소프트 폴백: require_tavily_key가 RuntimeError를 던지지만 상위
    run_research가 잡아 stub_report로 강등한다(그래프는 안 죽는다).
어느 쪽도 mock 경로는 없다.
"""
from __future__ import annotations

import os
from typing import Literal


# --- 모델 -------------------------------------------------------------------
def orchestrator_model() -> str:
    """오케스트레이터 LLM(call_json) 모델명 — BPM_LLM_MODEL."""
    return os.environ.get("BPM_LLM_MODEL", "gpt-5.4-mini")


def research_model() -> str:
    """리서치 파이프라인 모델명 — OPENAI_MODEL(오케와 별개)."""
    return os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")


def orchestrator_reasoning_effort() -> str:
    """오케스트레이터 LLM 추론 강도 — BPM_LLM_REASONING (기본 low).

    추론 모델(gpt-5·o계열)에만 의미가 있다 — call_json이 비추론 모델엔 넘기지 않는다.
    minimal/low/medium/high 중 하나. 기본 low로 지연을 줄이되 분류 품질은 유지한다.
    """
    return os.environ.get("BPM_LLM_REASONING", "low")


# --- OpenAI 키 --------------------------------------------------------------
def openai_api_key() -> str | None:
    """OpenAI API 키(없으면 None) — 리서치 클라이언트 생성자에 넘긴다."""
    return os.environ.get("OPENAI_API_KEY")


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
    """OPENAI_API_KEY 존재를 강제한다 — 없으면 맥락별 문구로 RuntimeError(fail-fast).

    context로 문구를 가르는 건 키 부재가 두 진입점(앱 실행 vs api_server 기동)에서 다르게
    노출돼 왔기 때문 — _OPENAI_KEY_MESSAGES 참고. Tavily는 진입점이 하나라 분기가 없다.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(_OPENAI_KEY_MESSAGES[context])


# --- 리서치 웹 검색 ---------------------------------------------------------
def search_provider() -> str:
    """웹 검색 프로바이더 선택 — RESEARCH_SEARCH_PROVIDER(기본 tavily)."""
    return os.environ.get("RESEARCH_SEARCH_PROVIDER", "tavily").strip().lower()


def tavily_api_key() -> str | None:
    """Tavily API 키(없으면 None) — _tavily_search의 TavilyClient 생성자에 넘긴다(강제는 require_tavily_key)."""
    return os.environ.get("TAVILY_API_KEY")


def require_tavily_key() -> None:
    """Tavily 키 존재를 강제한다 — 없으면 RuntimeError(상위 run_research가 stub으로 폴백).

    require_openai_key와 대칭으로 값을 돌려주지 않는다(가드 전용). 실제 키 읽기는
    _tavily_search가 tavily_api_key()로 한다 — 검증과 사용 지점을 분리해 둔다.
    """
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError(
            "TAVILY_API_KEY가 없습니다 — 리서치 웹 검색을 실행할 수 없습니다."
        )
