"""세그먼트 다중 라벨 분류 + 라우팅 결정 (기획서 5장 matrix).

흐름: segment_node가 만든 세그먼트마다 LLM이 발화 유형(다중 가능)을 고르고,
derive_routes/derive_priority가 그 라벨을 결정론적으로 워커 라우트·우선순위로
변환한다. LLM이 matrix를 어겨도 최종 routes는 derive_routes가 강제한다
(LLM은 '무슨 유형인가'만, '누구를 부를까'는 코드가 결정).

worked example
--------------
입력 세그먼트 canonical_text:
    "웹툰 IP가 일본 시장에서 통할 것이다"
LLM utterance_types →  ["hypothesis"]
derive_routes      →  ["research", "rag", "critic"]   # 전제는 검색, 회사 적합성은 RAG, 비약은 비평
derive_priority    →  2                                # 검증 필요 = dispatch 단계

입력 세그먼트:
    "타겟은 네이버로 정했고 예산은 1억"
LLM utterance_types →  ["decision", "constraint"]      # 한 문장에 두 유형
derive_routes      →  ["research", "rag", "critic"]    # 두 유형의 라우트 합집합
derive_priority    →  2
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from common.schema import PlanState
from common.schema.state import Route, UtteranceType
from agents.orchestrator.llm import call_json


_SYSTEM = """오케스트레이터 다중라벨 분류
각 세그먼트의 canonical_text를 보고 9개 발화 유형 중 해당하는 것을 모두 고른다(다중 라벨 허용).

유형:
- clarification_needed: 모호/추상, 추가 질문 필요
- fact_claim: 외부 사실 주장 (예: "게임 시장 포화")
- opinion: 주관 선호 (예: "B2B가 우리 색깔")
- hypothesis: 검증 가능한 가설 (예: "일본에서 통할 거 같다")
- decision: 결정·약속 (예: "타겟은 네이버로 가자")
- constraint: 숫자·기한·인원 등 제약 (예: "예산 1억, 6개월")
- correction: 정정·취소 (예: "아니, 빼자")
- question: 사용자가 정보를 물어봄 (예: "웹툰 시장 규모가 어떻게 돼?")
- meta: 단순응답·진행 신호 (예: "다음", "뽑아줘")

여러 유형이 한 세그먼트에 동시에 해당할 수 있다 — 예: "타겟 네이버로 정했어, 예산 1억" 같은 한 문장이면 decision+constraint. 의견에 정합성 확인이 필요하면 opinion 단독으로 둔다.

priority:
- 0: correction 포함
- 1: clarification_needed 포함 (correction 없을 때)
- 2: fact_claim/hypothesis/decision/constraint/question 중 하나라도 포함
- 3: opinion/meta만 있을 때

JSON만 출력."""


class ClassifyItem(BaseModel):
    canonical_text: str
    utterance_types: list[str] = Field(default_factory=list)


class ClassifyOut(BaseModel):
    items: list[ClassifyItem]


# 5장 matrix — 유형 → 활성 클러스터
# ● = 항상, △ = 검증 가능 정보면 (간단화: 일단 항상 호출), — = 스킵
_ROUTE_MATRIX: dict[str, set[Route]] = {
    "clarification_needed": {"clarify"},
    "fact_claim": {"research"},
    "opinion": {"rag", "critic"},
    "hypothesis": {"research", "rag", "critic"},
    "decision": {"research", "rag", "critic"},
    "constraint": {"research", "rag", "critic"},  # 리서치△=예산·기한이 업계 평균 대비 현실적인지(T.02)
    "question": {"research", "rag"},  # 외부 사실이면 리서치, 회사 내부 사안이면 RAG (둘 다 발동, 답 찾은 쪽이 응답)
    "correction": set(),  # correction 노드가 처리
    "meta": set(),
}


_VALID_TYPES: set[str] = set(_ROUTE_MATRIX.keys())


def derive_routes(utterance_types: list[str]) -> list[Route]:
    """다중 라벨 → 발동 워커 라우트(합집합). 매트릭스가 단일 출처.

    예: ["decision","constraint"] → {research,rag,critic} 합쳐서
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
    예: ["correction","decision"]        → 0  (상태부터 맞춰야 하므로 정정 우선)
        ["clarification_needed"]          → 1  (모호한 채 검증하면 엉뚱한 걸 검증)
        ["decision"] / ["question"]       → 2  (리서치·RAG·비평 디스패치 대상)
        ["opinion"] / ["meta"]            → 3  (가벼움, 마지막)
    """
    if "correction" in utterance_types:
        return 0
    if "clarification_needed" in utterance_types:
        return 1
    if any(
        t in utterance_types
        for t in ("fact_claim", "hypothesis", "decision", "constraint", "question")
    ):
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
        # 예: segment가 ["correction"]을 박고 LLM이 ["decision"]을 더하면 → ["correction","decision"]
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

    return {"turn_segments": segments}
