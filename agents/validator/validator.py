"""
validator.py

RAG 파이프라인이 가져온 근거(highlight / raw_source)가 원문 claim을
논리적으로 뒷받침하는지 평가하는 에이전트 모듈.

─────────────────────────────────────────────────────────────
[에이전트 vs 툴 구분]

  에이전트(Agent): LLM이 대화 루프를 통해 스스로 판단하고 행동을 결정하는 주체.
    - ValidatorAgent : claim과 증거의 논리적 관계를 평가하고,
                       필요하다고 판단하면 추가 검색을 자율적으로 수행한 뒤
                       최종 verdict를 출력한다.

  툴(Tool): 에이전트가 외부 환경과 상호작용하기 위해 호출하는 함수.
    - search_vector_db : 지정 폴더의 Chroma 벡터DB에서 키워드로 청크를 검색한다.
                         에이전트 자신이 "더 많은 증거가 필요하다"고 판단할 때만
                         선택적으로 호출한다. 강제 호출이 아님.

  실제 함수 구현(`_tool_search_vector_db`, `_dispatch_tool`, `_run_agent`)은
  rag_extractor.py에서 import해 재사용한다. 코드 중복을 방지하기 위함.
─────────────────────────────────────────────────────────────

파이프라인:
  1) run_validator(rag_result)
       └─ ValidatorAgent.run(...)
            ├─ [초기 평가] claim + highlight + raw_source를 읽고 verdict 시도
            ├─ [선택적 추가 검색] 증거가 모호하면 search_vector_db 호출 (최대 MAX_VALIDATOR_TURNS회)
            └─ [최종 출력] verdict / confidence / reasoning / evidence_used

예외 처리:
  - MAX_VALIDATOR_TURNS 초과 → FallbackRequired catch → 기본값 반환 (크래시 없음)
  - JSON 파싱 실패        → json.JSONDecodeError catch → 기본값 반환

참고:
  - rag_extractor.py : 공유 툴 구현 및 _run_agent 헬퍼
  - 평가 지표(verdict, confidence) 도출 방식은 추후 세부 검증 예정
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv

# ─── 공유 유틸리티 import ────────────────────────────────────────────────────────
# rag_extractor.py 에 구현된 툴 함수·에이전트 헬퍼를 그대로 재사용한다.
# 동일 로직을 여기에 복사하지 않는다.
from agents.rag.rag_extractor import (
    FallbackRequired,
    MODEL,
    MAX_AGENT_TURNS,
    SEARCH_HIGHLIGHT_TOOLS,  # search_vector_db JSON 스키마 — 툴 정의 재사용
    RagExtractorResult,
    _dispatch_tool,          # 툴 이름 → 실제 함수 라우팅
    _run_agent,              # OpenAI Responses API 기반 툴 루프 헬퍼
    _serialize_response,     # SDK Response 객체 → dict 변환
    client,
)

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 설정 ───────────────────────────────────────────────────────────────────────

# ValidatorAgent 가 실행할 수 있는 최대 툴 호출 횟수.
# SearchHighlightAgent(12)보다 낮게 설정 — 초기 증거가 이미 제공되므로
# 추가 탐색 횟수는 적어도 충분하다.
MAX_VALIDATOR_TURNS: int = int(os.getenv("MAX_VALIDATOR_TURNS", "6"))


# ─── 결과 타입 정의 ─────────────────────────────────────────────────────────────

class ValidatorResult(TypedDict):
    """
    ValidatorAgent 의 최종 평가 결과.

    verdict (str):
        "supports"     - 증거가 claim 을 직접 논리적으로 뒷받침함
        "contradicts"  - 증거가 claim 을 논리적으로 반박하거나 약화시킴
        "insufficient" - 관련 주제이나 claim 확인/반박에 불충분
        "unrelated"    - 의미적으로 유사하지만 claim 과 논리적 관련이 없음

    confidence (float):
        LLM 이 스스로 평가한 확신 수준 (0.0 ~ 1.0).
        시스템 프롬프트에 구간별 앵커를 제공해 보정한다.
        ※ 수학적 계산이 아닌 LLM 생성값이며, 추후 별도 검증 예정.

    reasoning (str):
        verdict 의 근거가 되는 자연어 설명.
        confidence 단독 출력 시 LLM 이 과신하는 경향이 있으므로
        reasoning 을 함께 강제해 자기 일관성을 높인다.

    evidence_used (List[str]):
        에이전트가 검토한 청크들의 짧은 요약 목록.
        초기 증거 외에 추가 검색을 했다면 그 결과도 포함된다.

    additional_searches (int):
        초기 증거(RAG 출력) 외에 에이전트가 추가로 search_vector_db 를
        호출한 횟수. turn_log 에서 tool_calls 가 비어 있지 않은 턴 수로 계산한다.
    """
    verdict: str
    confidence: float
    reasoning: str
    evidence_used: List[str]
    additional_searches: int


# ─── 기본 실패 결과 ──────────────────────────────────────────────────────────────

def _default_validator_result(reason: str) -> ValidatorResult:
    """
    에이전트 실행 실패(MAX_VALIDATOR_TURNS 초과, JSON 파싱 오류 등) 시
    파이프라인 크래시 없이 반환할 안전한 기본값을 생성한다.

    verdict 를 "insufficient" 로 설정해 "평가 불가" 상태를 표현한다.
    """
    return ValidatorResult(
        verdict="insufficient",
        confidence=0.0,
        reasoning=f"[ValidatorAgent 실행 실패] {reason}",
        evidence_used=[],
        additional_searches=0,
    )


# ─── 시스템 프롬프트 ─────────────────────────────────────────────────────────────
#
# 프롬프트 설계 원칙:
#   1. 역할 명확화  : RAG 가 "의미적으로 유사한" 청크를 가져왔지만
#                    그것이 논리적 지지를 보장하지 않음을 LLM 에 명시한다.
#   2. 자율 검색 허용: 초기 증거가 모호하면 search_vector_db 를 선택적으로 호출 가능.
#                    강제가 아니며, 명확한 증거가 있으면 바로 verdict 를 출력한다.
#   3. 최대 검색 횟수: 루프 방지를 위해 최대 호출 횟수를 프롬프트에 명시한다.
#   4. verdict 기준 : 4가지 레이블의 논리적 조건을 정의한다.
#   5. confidence 앵커: 수치 범위별 의미를 제공해 LLM 의 과신을 억제한다.
#   6. reasoning 강제: verdict + confidence 만 출력하면 근거가 없어 검증 불가.
#                    reasoning 을 함께 작성하도록 강제한다.
#   7. JSON 출력 강제: 파싱 가능한 구조화 출력을 보장한다.

_VALIDATOR_SYSTEM = f"""당신은 claim 검증 전문가입니다.

[배경]
아래에서 제공하는 증거(evidence)는 임베딩 기반 유사도 검색으로 가져온 청크입니다.
의미적으로 유사하다는 것이 논리적으로 뒷받침한다는 뜻은 아닙니다.
같은 주제를 다루지만 claim 과 반대되거나 맥락이 다를 수 있습니다.

[당신의 임무]
1. claim 과 제공된 초기 증거(highlight, raw_source)를 주의 깊게 읽으세요.
2. 증거가 충분히 명확하면 바로 verdict 를 출력하세요.
3. 증거가 모호하거나 불충분하면 search_vector_db 툴을 호출해 추가 증거를 수집하세요.
   - 최대 {MAX_VALIDATOR_TURNS}회 추가 검색 가능합니다.
   - 추가 검색 없이도 판단이 가능하면 굳이 호출하지 않아도 됩니다.
4. 모든 증거를 검토한 뒤 아래 JSON 형식으로 verdict 를 출력하세요.

[verdict 정의]
- "supports"     : 증거가 claim 을 직접 논리적으로 뒷받침함.
                   claim 이 사실이거나 더 신뢰할 수 있게 만드는 증거.
- "contradicts"  : 증거가 claim 을 논리적으로 반박하거나 약화시킴.
                   claim 과 증거가 동시에 참일 수 없거나, 증거가 claim 을 크게 약화시킴.
- "insufficient" : 관련 주제이나 claim 을 확인하거나 반박하기에 너무 일반적·단편적임.
                   더 구체적인 증거가 있어야 판단 가능한 경우.
- "unrelated"    : 의미적으로 유사하지만 claim 의 핵심 주장과 논리적 관련이 없음.
                   다른 개념을 다루거나 맥락이 완전히 다른 경우.

[confidence 기준]
- 0.9 – 1.0 : 직접적이고 명시적인 근거가 있음
- 0.7 – 0.89: 강한 암묵적 지지 또는 반박
- 0.5 – 0.69: 시사적이나 결정적이지 않음
- 0.3 – 0.49: 추론이 억지스러움
- 0.0 – 0.29: 거의 추측 수준

[출력 — 반드시 JSON 형식]
{{
  "verdict": "supports|contradicts|insufficient|unrelated",
  "confidence": 0.0-1.0,
  "reasoning": "claim 과 증거의 논리적 관계를 설명하는 문장",
  "evidence_used": ["검토한 청크 요약 1", "검토한 청크 요약 2", ...]
}}"""


# ─── ValidatorAgent ──────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    RAG 파이프라인이 가져온 증거를 받아 claim 에 대한 논리적 지지 여부를 평가하는 에이전트.

    내부적으로 _run_agent() 헬퍼를 통해 OpenAI Responses API 와 멀티턴 대화를 수행한다.
    필요하다고 판단하면 search_vector_db 툴을 자율적으로 호출해 추가 증거를 수집한다.
    """

    def __init__(self, max_turns: int = MAX_VALIDATOR_TURNS) -> None:
        # max_turns: 에이전트가 실행할 수 있는 최대 툴 호출 횟수.
        # _run_agent() 내부의 하드 컷오프 역할을 한다.
        self.max_turns = max_turns

    def run(
        self,
        claim: str,
        qk: str,
        highlight: str,
        highlight_reason: str,
        raw_source: str,
        keywords: List[str],
        source_file: str,
        source_page: str,
        verbose: bool = True,
    ) -> Tuple[ValidatorResult, List[dict], List[dict]]:
        """
        ValidatorAgent 의 메인 실행 메서드.

        [실행 흐름]
        1. 모든 RAG 출력 필드를 하나의 유저 메시지로 조합한다.
        2. _run_agent() 를 호출해 에이전트 루프를 시작한다.
           - 에이전트는 초기 증거를 읽고 verdict 를 결정하거나,
             search_vector_db 를 호출해 추가 증거를 수집한 뒤 결정한다.
           - max_turns 초과 시 FallbackRequired 발생.
        3. 에이전트의 최종 텍스트 출력을 JSON 으로 파싱한다.
        4. turn_log 에서 additional_searches 를 계산한다.
           (tool_calls 가 비어 있지 않은 턴 = 에이전트가 툴을 호출한 턴)
        5. ValidatorResult 를 반환한다.

        :param claim:          검증할 핵심 claim 문장
        :param qk:             원문 입력 텍스트
        :param highlight:      RAG 가 생성한 ~100자 하이라이트 요약
        :param highlight_reason: RAG 가 해당 청크를 선택한 이유
        :param raw_source:     원본 청크 전문
        :param keywords:       RAG 검색에 사용된 키워드 목록
        :param source_file:    출처 파일명
        :param source_page:    출처 페이지
        :param verbose:        True 이면 턴별 진행 상황 출력
        :return: (ValidatorResult, response_trace, turn_log)
                 response_trace : 각 턴의 직렬화된 OpenAI SDK Response 목록 (디버그용)
                 turn_log       : 각 턴의 구조화된 툴 호출 로그
        """
        if verbose:
            print("[ValidatorAgent] 평가 시작...")

        # 에이전트에게 전달할 유저 메시지 구성.
        # highlight 와 raw_source 를 분리 제공해
        # RAG 의 highlight_reason 이 실제로 유효한지도 간접 검증 가능하게 한다.
        user_msg = (
            f"[CLAIM TO VERIFY]\n{claim}\n\n"
            f"[ORIGINAL QUERY]\n{qk}\n\n"
            f"[RETRIEVED EVIDENCE]\n"
            f"Highlight: {highlight}\n"
            f"Selection reason: {highlight_reason}\n\n"
            f"[FULL SOURCE CHUNK]\n{raw_source}\n\n"
            f"[SOURCE METADATA]\n"
            f"File: {source_file}  |  Page: {source_page}\n"
            f"Keywords used in retrieval: {', '.join(keywords)}"
        )

        # _run_agent() 는 rag_extractor.py 에 구현된 공통 에이전트 루프 헬퍼.
        # SEARCH_HIGHLIGHT_TOOLS 를 전달해 search_vector_db 툴을 사용 가능하게 한다.
        text, response_trace, turn_log = _run_agent(
            system=_VALIDATOR_SYSTEM,
            tools=SEARCH_HIGHLIGHT_TOOLS,
            user_msg=user_msg,
            max_turns=self.max_turns,
            verbose=verbose,
        )

        # 에이전트 출력이 코드 펜스(```)로 감싸져 있으면 제거한다.
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith("```")
            ).strip()

        raw = json.loads(text)

        # additional_searches: 에이전트가 초기 증거 외에 추가로 툴을 호출한 턴 수.
        # turn_log 의 각 엔트리에서 tool_calls 가 비어 있지 않으면 툴 호출이 있었던 턴이다.
        additional_searches = sum(
            1 for entry in turn_log if entry.get("tool_calls")
        )

        result = ValidatorResult(
            verdict=raw.get("verdict", "insufficient"),
            confidence=float(raw.get("confidence", 0.0)),
            reasoning=raw.get("reasoning", ""),
            evidence_used=raw.get("evidence_used", []),
            additional_searches=additional_searches,
        )

        if verbose:
            print(f"  verdict: {result['verdict']} (confidence: {result['confidence']})")
            if additional_searches:
                print(f"  추가 검색 횟수: {additional_searches}")

        return result, response_trace, turn_log


# ─── 진입점: run_validator ───────────────────────────────────────────────────────

def run_validator(
    rag_result: RagExtractorResult,
    verbose: bool = True,
) -> Tuple[ValidatorResult, dict]:
    """
    run_rag_extractor() 의 출력을 받아 ValidatorAgent 를 실행하는 진입점 함수.

    [역할]
    - RagExtractorResult TypedDict 에서 필드를 꺼내 ValidatorAgent.run() 에 전달한다.
    - 에이전트 실행 중 발생하는 모든 예외를 여기서 처리해 호출자에게 크래시가 전파되지 않게 한다.

    [예외 처리 전략]
    - FallbackRequired : MAX_VALIDATOR_TURNS 초과 시 _run_agent() 에서 발생.
                         "평가 불가" 기본값을 반환하고 예외를 외부로 전파하지 않는다.
                         validator 는 판단 레이어이므로 파이프라인 레벨 fallback 을 트리거하지 않는다.
    - json.JSONDecodeError: 에이전트 출력이 유효한 JSON 이 아닐 때 발생.
                            동일하게 기본값을 반환한다.
    - Exception (기타): 예기치 않은 오류도 기본값으로 처리해 안전하게 종료한다.

    :param rag_result: run_rag_extractor() 가 반환한 RagExtractorResult
    :param verbose:    True 이면 ValidatorAgent 내부 진행 상황 출력
    :return: (ValidatorResult, meta_dict)
             meta_dict = {
               "validator_response_trace": [...],  # 디버그용 SDK Response 목록
               "validator_turn_log": [...],         # 구조화된 턴별 툴 호출 로그
             }
    """
    agent = ValidatorAgent()

    try:
        validator_result, response_trace, turn_log = agent.run(
            claim=rag_result["claim"],
            qk=rag_result["qk"],
            highlight=rag_result["highlight"],
            highlight_reason=rag_result["highlight_reason"],
            raw_source=rag_result["raw_source"],
            keywords=rag_result["keywords"],
            source_file=rag_result["source_file"],
            source_page=rag_result["source_page"],
            verbose=verbose,
        )

    except FallbackRequired as e:
        # MAX_VALIDATOR_TURNS 초과: 에이전트가 제한 내에 verdict 를 완성하지 못한 경우.
        if verbose:
            print(f"  [ValidatorAgent] FallbackRequired: {e}")
        validator_result = _default_validator_result(str(e))
        response_trace, turn_log = [], []

    except json.JSONDecodeError as e:
        # 에이전트 출력이 JSON 으로 파싱되지 않은 경우.
        if verbose:
            print(f"  [ValidatorAgent] JSON 파싱 실패: {e}")
        validator_result = _default_validator_result(f"JSON 파싱 오류: {e}")
        response_trace, turn_log = [], []

    except Exception as e:
        # 기타 예기치 않은 오류.
        if verbose:
            print(f"  [ValidatorAgent] 예외 발생: {e}")
        validator_result = _default_validator_result(f"예외: {e}")
        response_trace, turn_log = [], []

    meta = {
        "validator_response_trace": response_trace,
        "validator_turn_log": turn_log,
    }
    return validator_result, meta
