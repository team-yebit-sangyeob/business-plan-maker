"""세그먼트 다중 라벨 분류 + 라우팅 결정 (기획서 5장 matrix).

흐름: segment_node가 만든 세그먼트마다 LLM이 발화 유형(다중 가능)을 고르고,
derive_routes/derive_priority가 그 라벨을 결정론적으로 워커 라우트·우선순위로
변환한다. LLM이 matrix를 어겨도 최종 routes는 derive_routes가 강제한다
(LLM은 '무슨 유형인가'만, '누구를 부를까'는 코드가 결정).

worked example
--------------
입력 세그먼트 canonical_text:
    "웹툰 IP가 일본 시장에서 통할 것이다"
LLM utterance_types →  ["claim"]
derive_claim_type  →  "hypothesis_premise"            # 가설 결론이 아닌, 받치는 전제를 검증
derive_routes      →  ["research", "rag", "critic"]   # 전제는 검색, 회사 적합성은 RAG, 비약은 비평
derive_priority    →  2                                # 검증 필요 = dispatch 단계

입력 세그먼트:
    "웹툰 시장 규모가 어떻게 돼? 그리고 타겟은 네이버로 가자"
LLM utterance_types →  ["question", "claim"]           # 한 문장에 두 유형
derive_routes      →  ["research", "rag", "critic"]    # 두 유형의 라우트 합집합
derive_priority    →  2
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from common.schema import PlanState
from common.schema.state import ClaimType, Route, UtteranceType
from agents.orchestrator.llm import call_json


_SYSTEM = """오케스트레이터 다중라벨 분류
각 세그먼트의 canonical_text를 보고 6개 발화 유형 중 해당하는 것을 모두 고른다(다중 라벨 허용).

유형:
- clarification_needed: 모호/추상, 추가 질문 필요
- claim: 검증 가능한 내용 발화 — 외부 사실 주장·가설·결정·제약을 모두 포함 (예: "게임 시장 포화" / "일본에서 통할 거 같다" / "타겟은 네이버로 가자" / "예산 1억, 6개월")
- opinion: 주관 선호 (예: "B2B가 우리 색깔")
- correction: 정정·취소 (예: "아니, 빼자")
- question: 사용자가 정보를 물어봄 (예: "웹툰 시장 규모가 어떻게 돼?")
- meta: 단순응답·진행 신호 (예: "다음", "뽑아줘")

여러 유형이 한 세그먼트에 동시에 해당할 수 있다 — 예: "시장 규모 어때? 타겟은 네이버로 가자" 같은 한 문장이면 question+claim. 검증할 게 없는 주관 선호면 opinion 단독으로 둔다.

claim_type (utterance_types에 "claim"이 있을 때만 채우고, 없으면 null):
- fact: 검증 가능한 외부 사실 (예: "게임 시장 포화 상태래")
- hypothesis_premise: 가설을 받치는 검증 가능한 전제 (예: "일본 웹툰 시장 성장 중"). 가설 결론 자체("우리 IP가 통할 거야")는 검증 대상이 아니다 — 받치는 전제만 고른다.
- decision_context: 결정의 배경이 되는 사실 (예: "타겟은 네이버로 가자" → "네이버 콘텐츠 운영 조직 현황")
- market_fill: 자동 채움 맥락에서 시장 규모·근거를 찾는 경우
claim이 여러 성격을 겸하면 가장 핵심인 하나만 고른다.

구분 가이드 — 헷갈리는 경계:
- claim vs opinion: 외부 데이터·회사 자료로 맞다/틀리다를 따질 수 있으면 claim, 순수 취향·가치판단이면 opinion.
  · "B2B 시장이 더 커" → claim (시장 규모는 검증 가능)
  · "난 B2B가 더 끌려" → opinion (주관)
  · "B2B가 우리 색깔에 맞아, 영업 인프라도 강하니까" → opinion (근거를 댔어도 '우리 색깔'은 취향 판단 — 단, 근거의 사실 여부는 회사 자료로 비평이 따짐)
  · "타겟은 네이버로 가자" → claim (결정 = 향후 슬롯에 박히는 약속, decision_context)
- correction은 '이전에 정한 것을 무르거나 바꿀 때'만. 정정 키워드가 있어도 새 진술이면 claim:
  · "아 카카오는 빼자" → correction (앞서 넣은 타겟을 무름)
  · "네이버 말고 카카오로 가자" → correction (대상 교체)
  · "말고 또 뭐가 있을까?" → question (정정 아님 — 키워드만 같음)
  · "바꾸는 게 어렵진 않아" → claim/opinion (정정 의도 없음)
- 한 세그먼트 다중 라벨 예:
  · "시장 규모 어때? 그리고 타겟은 네이버로 가자" → ["question","claim"]
  · "그건 취소하고, 일본 시장은 성장 중이잖아" → ["correction","claim"]

priority:
- 0: correction 포함
- 1: clarification_needed 포함 (correction 없을 때)
- 2: claim/question 중 하나라도 포함
- 3: opinion/meta만 있을 때

JSON만 출력."""


class ClassifyItem(BaseModel):
    canonical_text: str
    utterance_types: list[str] = Field(default_factory=list)
    claim_type: str | None = None  # "claim" 유형일 때만 의미, 그 외 무시


class ClassifyOut(BaseModel):
    items: list[ClassifyItem]


# 5장 matrix — 유형 → 활성 클러스터
# ● = 항상, △ = 검증 가능 정보면 (간단화: 일단 항상 호출), — = 스킵
_ROUTE_MATRIX: dict[str, set[Route]] = {
    "clarification_needed": {"clarify"},
    "claim": {"research", "rag", "critic"},  # 사실·가설·결정·제약 통합 — 전제는 리서치, 회사 적합성은 RAG, 비약은 비평
    "opinion": {"rag", "critic"},
    "question": {"research", "rag"},  # 외부 사실이면 리서치, 회사 내부 사안이면 RAG (둘 다 발동, 답 찾은 쪽이 응답)
    "correction": set(),  # correction 노드가 처리
    "meta": set(),
}


_VALID_TYPES: set[str] = set(_ROUTE_MATRIX.keys())

_VALID_CLAIM_TYPES: set[str] = {
    "fact",
    "hypothesis_premise",
    "decision_context",
    "market_fill",
}


def derive_claim_type(
    utterance_types: list[str], raw_claim_type: str | None
) -> ClaimType | None:
    """claim 세그먼트에만 claim_type을 부여(검증). 그 외 유형이면 None.

    LLM이 claim인데 claim_type을 안 주거나 알 수 없는 값이면 "fact"로 보수 처리
    — 리서치가 일반 사실 검증으로 다루는 게 오분류 시 가장 안전(전제 누락보다 낫다).
    """
    if "claim" not in utterance_types:
        return None
    if raw_claim_type in _VALID_CLAIM_TYPES:
        return raw_claim_type  # type: ignore[return-value]
    return "fact"


def derive_routes(utterance_types: list[str]) -> list[Route]:
    """다중 라벨 → 발동 워커 라우트(합집합). 매트릭스가 단일 출처.

    예: ["claim","question"] → {research,rag,critic} 합쳐서
        ["research","rag","critic"] (order대로 정렬).
        ["meta"] → 라우트 없음 → ["none"].
    """
    routes: set[Route] = set()
    for t in utterance_types:
        routes |= _ROUTE_MATRIX.get(t, set())
    if not routes:
        return ["none"]
    # 안정적 정렬 — 같은 라벨 집합이면 항상 같은 순서로 나오게(테스트·캐시 친화)
    order: list[Route] = ["clarify", "research", "rag", "critic", "none"]
    return [r for r in order if r in routes]


def derive_priority(utterance_types: list[str]) -> int:
    """다중 라벨 → 턴 내 처리 우선순위(낮을수록 먼저). 기획서 6장 우선순위 규칙.

    한 세그먼트에 여러 유형이 섞이면 가장 급한 것을 따른다(정정 > 명확화 > 검증 > 가벼움).
    예: ["correction","claim"]            → 0  (상태부터 맞춰야 하므로 정정 우선)
        ["clarification_needed"]          → 1  (모호한 채 검증하면 엉뚱한 걸 검증)
        ["claim"] / ["question"]          → 2  (리서치·RAG·비평 디스패치 대상)
        ["opinion"] / ["meta"]            → 3  (가벼움, 마지막)
    """
    if "correction" in utterance_types:
        return 0
    if "clarification_needed" in utterance_types:
        return 1
    if "claim" in utterance_types or "question" in utterance_types:
        return 2
    return 3


async def classify_node(state: PlanState) -> dict:
    segments = list(state.get("turn_segments") or [])
    if not segments:
        return {"turn_segments": []}

    texts = [s.get("canonical_text") or s.get("text", "") for s in segments]
    payload = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    out = await call_json(_SYSTEM, payload, ClassifyOut)

    # LLM 결과를 인덱스로 매칭 (개수 어긋나면 fallback)
    llm_items = out.items if len(out.items) == len(segments) else None

    for idx, seg in enumerate(segments):
        # segment 노드가 hints로 미리 박은 라벨(correction/clarification_needed/question/meta)은
        # 신뢰도가 높아 보존하고, classify LLM 결과를 그 위에 합친다.
        # 예: segment가 ["correction"]을 박고 LLM이 ["claim"]을 더하면 → ["correction","claim"]
        prior = list(seg.get("utterance_types") or [])
        types: list[str] = list(prior)
        if llm_items is not None:
            for t in llm_items[idx].utterance_types:
                if t in _VALID_TYPES and t not in types:
                    types.append(t)
        if not types:
            # LLM 개수 불일치(llm_items=None)이거나 빈 결과일 때의 안전 기본값.
            # opinion은 RAG+비평만 타서(리서치 비용 0) 오분류 시 가장 피해가 적다.
            types = ["opinion"]
        # 알 수 없는 라벨은 버리고(타입 캐스팅), routes·priority는 코드가 매트릭스로 재계산
        seg["utterance_types"] = [t for t in types if t in _VALID_TYPES]  # type: ignore[assignment]
        seg["priority"] = derive_priority(seg["utterance_types"])
        seg["routes"] = derive_routes(seg["utterance_types"])
        raw_ct = llm_items[idx].claim_type if llm_items is not None else None
        seg["claim_type"] = derive_claim_type(seg["utterance_types"], raw_ct)

    return {"turn_segments": segments}
