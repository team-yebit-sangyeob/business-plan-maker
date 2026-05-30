"""
claim_router_validation.py

자연어 입력에서 검색에 적합한 claim을 추출하는 에이전트 모듈.
검색 경로(web/internal) 결정은 상위 오케스트레이터에서 처리한다.

참고 파일:
- claim_router_validation.ipynb (LangGraph 기반 라우터 검증 구조)
- exp_openai_sdk.ipynb (OpenAI Responses API + 툴 에이전트 패턴)
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# experiments 루트의 .env 로드 (agents/rag 기준 두 단계 위)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 공통 설정 ─────────────────────────────────────────────────────────────────
# OPENAI_MODEL 환경변수로 모델을 교체할 수 있음 (예: o4-mini, gpt-4o)
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─── Pydantic 스키마 ────────────────────────────────────────────────────────────

class ClaimExtractionResult(BaseModel):
    """Agent 1 출력: 자연어에서 추출한 정제된 claim과 검색어"""
    claim: str = Field(description="검색에 적합하게 정제된 핵심 주장 문장")
    normalized_claim: str = Field(description="검색 최적화를 위해 정규화된 claim 표현")
    topic: str = Field(description="claim의 핵심 주제어 (짧게)")
    keywords: List[str] = Field(description="검색에 효과적인 한국어 핵심 키워드 3개")
    search_ready: bool = Field(
        description="사실 기반으로 검증 가능한 claim이면 True, 의견·질문·명령이면 False"
    )


# ─── LangGraph 상태 정의 ────────────────────────────────────────────────────────

class RouterState(TypedDict):
    """Claim 추출 파이프라인의 전체 실행 상태"""
    user_input: str                           # 사용자 원본 자연어 입력
    claim_result: Optional[Dict[str, Any]]    # ClaimExtractionResult 직렬화 dict
    error: Optional[str]                      # 파이프라인 오류 메시지


# ─── Agent 1: 검색에 적합한 Claim 추출 ─────────────────────────────────────────

_CLAIM_EXTRACTOR_SYSTEM = """당신은 팩트체크 및 리서치를 위한 claim 추출 전문가입니다.

사용자가 입력한 자연어를 분석하여 다음을 수행하세요.

[수행 절차]
1. 입력 문장이 사실 기반으로 외부/내부 자료로 검증 가능한 knowledge(claim)인지 판단하세요.
   - search_ready=true: 검증 가능한 사실 기반 주장
   - search_ready=false: 순수 의견, 감정 표현, 질문, 명령, 단순 요청
2. search_ready=true인 경우만 아래를 채우세요:
   - claim: 검색에 적합하게 정제된 핵심 주장 문장
   - normalized_claim: 검색 최적화를 위해 간결하게 정규화된 표현
   - topic: 핵심 주제어 (10자 이내)
   - keywords: 검색에 효과적인 한국어 키워드 3개 (구체적이고 검색 친화적으로)
3. search_ready=false이면 claim/normalized_claim/topic은 빈 문자열, keywords는 빈 배열로 두세요.

반드시 아래 JSON 형식으로만 출력하세요:
{
  "claim": "정제된 핵심 주장 문장",
  "normalized_claim": "정규화된 표현",
  "topic": "핵심 주제어",
  "keywords": ["키워드1", "키워드2", "키워드3"],
  "search_ready": true
}"""


def extract_claim_from_text(user_input: str) -> tuple[ClaimExtractionResult, dict]:
    """
    자연어 입력에서 검색에 적합한 claim과 키워드를 추출한다.

    :param user_input: 사용자가 입력한 원본 자연어 문장
    :return: (ClaimExtractionResult, 직렬화된 response dict)
    """
    response = client.responses.create(
        model=MODEL,
        instructions=_CLAIM_EXTRACTOR_SYSTEM,
        input=[{"role": "user", "content": user_input}],
    )
    try:
        serialized = response.model_dump()
    except Exception:
        serialized = {"_raw": str(response)}

    text = response.output_text.strip()
    # JSON 블록이 코드 펜스로 감싸져 있을 경우 제거
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()

    data = json.loads(text)
    result = ClaimExtractionResult(**data)
    return result, serialized


# ─── LangGraph 노드 함수 ────────────────────────────────────────────────────────

def _claim_extractor_node(state: RouterState) -> Dict[str, Any]:
    """
    Agent 1 노드: 자연어 입력에서 claim과 키워드를 추출한다.
    """
    try:
        result, _ = extract_claim_from_text(state["user_input"])
        return {"claim_result": result.model_dump(), "error": None}
    except Exception as e:
        import traceback
        print(f"[claim_router_validation] claim 추출 오류: {e}")
        traceback.print_exc()
        return {"claim_result": None, "error": f"claim 추출 오류: {e}"}


# ─── LangGraph 그래프 구성 ──────────────────────────────────────────────────────

def build_router_graph():
    """
    Claim 추출 파이프라인을 구성하고 컴파일한다.
    검색 경로 결정은 상위 오케스트레이터에서 처리한다.

    flow:
        START → claim_extractor → END
    """
    graph = StateGraph(RouterState)

    graph.add_node("claim_extractor", _claim_extractor_node)

    graph.add_edge(START, "claim_extractor")
    graph.add_edge("claim_extractor", END)

    return graph.compile()


# 모듈 임포트 시 그래프를 한 번만 컴파일하여 재사용
router_app = build_router_graph()


# ─── 파이프라인 실행 함수 ───────────────────────────────────────────────────────

def run_router_pipeline(user_input: str) -> RouterState:
    """
    자연어 입력을 받아 claim 추출 파이프라인을 실행한다.

    :param user_input: 사용자 자연어 입력 문장
    :return: RouterState (user_input, claim_result, error)
    """
    initial_state: RouterState = {
        "user_input": user_input,
        "claim_result": None,
        "error": None,
    }
    return router_app.invoke(initial_state)
