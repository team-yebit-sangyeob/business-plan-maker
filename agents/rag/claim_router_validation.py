"""
claim_router_validation.py

자연어 입력에서 검색에 적합한 claim을 추출하고
웹 또는 사내 문서 검색으로 라우팅을 결정하는 에이전트 모듈.

참고 파일:
- claim_router_validation.ipynb (LangGraph 기반 라우터 검증 구조)
- exp_openai_sdk.ipynb (OpenAI Responses API + 툴 에이전트 패턴)
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Literal, Optional, TypedDict

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


class SearchRouteDecision(BaseModel):
    """Agent 2 출력: 1차 검색 경로 결정"""
    route: Literal["internal", "web", "both"] = Field(
        description="검색 출처: internal(사내문서) | web(웹검색) | both(둘 다)"
    )
    reason: str = Field(description="해당 route를 선택한 근거 (짧고 명확하게)")


# ─── LangGraph 상태 정의 ────────────────────────────────────────────────────────

class RouterState(TypedDict):
    """라우터 파이프라인의 전체 실행 상태"""
    user_input: str                           # 사용자 원본 자연어 입력
    claim_result: Optional[Dict[str, Any]]    # ClaimExtractionResult 직렬화 dict
    route_decision: Optional[Dict[str, Any]]  # SearchRouteDecision 직렬화 dict
    error: Optional[str]                      # 파이프라인 오류 메시지


# ─── Agent 1: 검색에 적합한 Claim 추출 ─────────────────────────────────────────

_CLAIM_EXTRACTOR_SYSTEM = """당신은 팩트체크 및 리서치를 위한 claim 추출 전문가입니다.

사용자가 입력한 자연어를 분석하여 다음을 수행하세요.

[수행 절차]
1. 입력 문장이 사실 기반으로 외부/내부 자료로 검증 가능한 knowledge(claim)인지 판단하세요.
   - search_ready=True: 검증 가능한 사실 기반 주장
   - search_ready=False: 순수 의견, 감정 표현, 질문, 명령, 단순 요청
2. search_ready=True인 경우만 아래를 채우세요:
   - claim: 검색에 적합하게 정제된 핵심 주장 문장
   - normalized_claim: 검색 최적화를 위해 간결하게 정규화된 표현
   - topic: 핵심 주제어 (10자 이내)
   - keywords: 검색에 효과적인 한국어 키워드 3개 (구체적이고 검색 친화적으로)
3. search_ready=False이면 claim/normalized_claim/topic/keywords는 빈 문자열/빈 리스트로 두세요.

반드시 지정된 JSON 스키마에 맞게만 응답하세요."""


def extract_claim_from_text(user_input: str) -> tuple[ClaimExtractionResult, dict]:
    """
    자연어 입력에서 검색에 적합한 claim과 키워드를 추출한다.

    :param user_input: 사용자가 입력한 원본 자연어 문장
    :return: (ClaimExtractionResult, 직렬화된 response dict)
    """
    # structured output을 사용해 Pydantic 모델로 안전하게 파싱
    response = client.responses.parse(
        model=MODEL,
        instructions=_CLAIM_EXTRACTOR_SYSTEM,
        input=user_input,
        text_format=ClaimExtractionResult,
    )
    try:
        serialized = response.model_dump()
    except Exception:
        serialized = {"_raw": str(response)}
    return response.output_parsed, serialized


# ─── Agent 2: 검색 경로 결정 (1차 LLM 라우팅) ──────────────────────────────────

_SEARCH_ROUTER_SYSTEM = """당신은 팩트체크/데스크리서치용 검색 경로 라우터입니다.

입력된 normalized_claim, topic, keywords를 분석하여 어디서 검색할지 결정하세요.

[판단 기준]
- internal: 사내문서 검색이 적합한 경우
  · 내부 기획서, 사업계획서, 사내 운영 방침, 내부 전략/가설
  · 사내에서만 확인 가능한 내용
- web: 웹 검색이 적합한 경우
  · 외부 통계, 법/정책, 국가별 규제, 공개 보고서, 논문, 시장 자료, 최신 정보
  · 공개 자료로 충분히 검증 가능한 내용
- both: 사내문서와 웹 검색을 모두 해야 하는 경우
  · 내부 기획의 타당성을 외부 자료로 검증해야 하는 경우
  · 내부 주장과 외부 공개 근거를 비교해야 하는 경우

[중요 규칙]
- none은 절대 선택하지 마세요.
- 판단이 애매하면 반드시 web을 선택하세요.
- 이 단계에 들어온 claim은 이미 search_ready=True입니다.

반드시 지정된 JSON 스키마에 맞게만 응답하세요."""


def decide_search_route(
    normalized_claim: str,
    topic: str,
    keywords: List[str],
) -> SearchRouteDecision:
    """
    claim 정보를 기반으로 1차 검색 경로(internal/web/both)를 결정한다.

    :param normalized_claim: 정규화된 claim 문장
    :param topic: claim의 핵심 주제어
    :param keywords: 검색 키워드 목록
    :return: SearchRouteDecision (route, reason)
    """
    prompt = (
        f"normalized_claim: {normalized_claim}\n"
        f"topic: {topic}\n"
        f"keywords: {', '.join(keywords)}"
    )
    # structured output으로 Pydantic 모델에 안전하게 파싱
    response = client.responses.parse(
        model=MODEL,
        instructions=_SEARCH_ROUTER_SYSTEM,
        input=prompt,
        text_format=SearchRouteDecision,
    )
    return response.output_parsed


# ─── LangGraph 노드 함수 ────────────────────────────────────────────────────────

def _claim_extractor_node(state: RouterState) -> Dict[str, Any]:
    """
    Agent 1 노드: 자연어 입력에서 claim과 키워드를 추출한다.
    오류 발생 또는 search_ready=False이면 이후 라우팅 노드를 건너뛴다.
    """
    try:
        result, _ = extract_claim_from_text(state["user_input"])
        return {"claim_result": result.model_dump(), "error": None}
    except Exception as e:
        # 추출 실패 시 오류를 기록하고 파이프라인을 중단
        return {"claim_result": None, "error": f"claim 추출 오류: {e}"}


def _search_router_node(state: RouterState) -> Dict[str, Any]:
    """
    Agent 2 노드: 추출된 claim을 기반으로 검색 경로를 결정한다.

    1차: LLM 기반 라우팅 (internal/web/both)
    (개발 예정) 2차: 검색어와 벡터DB 파일명의 코사인 유사도 기반 라우팅 보정
                     - 임계점 이상이면 internal, 미만이면 web으로 1차 결과를 보정
    """
    claim_result = state.get("claim_result")
    if not claim_result:
        # claim이 없으면 라우팅 불가 → None 반환
        return {"route_decision": None}

    try:
        decision = decide_search_route(
            normalized_claim=claim_result.get("normalized_claim", ""),
            topic=claim_result.get("topic", ""),
            keywords=claim_result.get("keywords", []),
        )
        return {"route_decision": decision.model_dump()}
    except Exception as e:
        return {"route_decision": None, "error": f"라우팅 결정 오류: {e}"}


def _should_route(state: RouterState) -> str:
    """
    조건부 엣지: claim 추출 결과에 따라 다음 노드를 결정한다.
    - search_ready=True → 검색 경로 결정 노드 실행
    - search_ready=False 또는 오류 → 파이프라인 종료
    """
    claim_result = state.get("claim_result")
    if not claim_result or not claim_result.get("search_ready", False):
        return "end"
    return "route"


# ─── LangGraph 그래프 구성 ──────────────────────────────────────────────────────

def build_router_graph():
    """
    Claim 추출 → 검색 경로 결정 순서로 실행되는 LangGraph를 구성하고 컴파일한다.

    flow:
        START → claim_extractor
              → (search_ready=True)  → search_router → END
              → (search_ready=False) → END
    """
    graph = StateGraph(RouterState)

    graph.add_node("claim_extractor", _claim_extractor_node)
    graph.add_node("search_router", _search_router_node)

    graph.add_edge(START, "claim_extractor")
    graph.add_conditional_edges(
        "claim_extractor",
        _should_route,
        {"route": "search_router", "end": END},
    )
    graph.add_edge("search_router", END)

    return graph.compile()


# 모듈 임포트 시 그래프를 한 번만 컴파일하여 재사용
router_app = build_router_graph()


# ─── 파이프라인 실행 함수 ───────────────────────────────────────────────────────

def run_router_pipeline(user_input: str) -> RouterState:
    """
    자연어 입력을 받아 claim 추출 + 검색 경로 결정 전체 파이프라인을 실행한다.

    :param user_input: 사용자 자연어 입력 문장
    :return: RouterState (user_input, claim_result, route_decision, error)
    """
    initial_state: RouterState = {
        "user_input": user_input,
        "claim_result": None,
        "route_decision": None,
        "error": None,
    }
    return router_app.invoke(initial_state)


def run_router_pipeline_for_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    이미 추출된 claim 목록에 대해 검색 경로 결정만 실행한다.
    claim_extractor_output.json을 직접 입력받는 배치 처리 시 사용한다.

    :param claims: ClaimExtractorOutput.claims 리스트
                   (claim_id, normalized_claim, topic, keywords, search_ready 등 포함)
    :return: 각 claim에 "route_decision" 키가 추가된 결과 리스트
    """
    results = []
    for claim in claims:
        if not claim.get("search_ready", False):
            # search_ready가 아닌 claim은 라우팅 없이 통과
            results.append({**claim, "route_decision": None, "skip_reason": "search_ready=False"})
            continue

        try:
            decision = decide_search_route(
                normalized_claim=claim.get("normalized_claim", ""),
                topic=claim.get("topic", ""),
                keywords=claim.get("keywords", []),
            )
            results.append({**claim, "route_decision": decision.model_dump()})
        except Exception as e:
            results.append({**claim, "route_decision": None, "error": str(e)})

    return results
