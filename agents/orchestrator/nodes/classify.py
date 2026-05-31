"""세그먼트 다중 라벨 분류 + 라우팅 결정 (기획서 5장 matrix).

흐름: segment_node가 만든 세그먼트마다 LLM이 발화 유형(다중 가능)을 고르고,
derive_routes가 그 라벨을 결정론적으로 워커 라우트로 변환한다. LLM이 matrix를
어겨도 최종 routes는 derive_routes가 강제한다
(LLM은 '무슨 유형인가'만, '누구를 부를까'는 코드가 결정).
처리 순서·분기는 별도 priority 필드 없이 routes/utterance_types에서 직접 파생.

worked example
--------------
입력 세그먼트 canonical_text:
    "웹툰 IP가 일본 시장에서 통할 것이다"
LLM utterance_types →  ["claim"]
derive_routes      →  ["research", "rag", "critic"]   # 전제는 검색, 회사 적합성은 RAG, 비약은 비평
(주장을 어떻게 분해·검증할지는 리서치 클러스터의 쿼리 분해기 몫 — 오케는 라우팅까지만)

입력 세그먼트:
    "웹툰 시장 규모가 어떻게 돼? 그리고 타겟은 네이버로 가자"
LLM utterance_types →  ["question", "claim"]           # 한 문장에 두 유형
derive_routes      →  ["research", "rag", "critic"]    # 두 유형의 라우트 합집합
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from common.schema import PlanState
from common.schema.state import Route, UtteranceType
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

구분 가이드 — 헷갈리는 경계:
- claim vs opinion: 핵심 발화가 참/거짓을 따질 사실 주장이면 claim, 가치판단·선호면 opinion.
  의견의 근거도 RAG·비평으로 점검되지만 근거의 검증 가능성은 분류를 가르지 않는다 — 둘의 라우팅 차이는 리서치(외부 웹) 발동 여부뿐.
  · "B2B 시장이 더 커" → claim (시장 규모는 외부 사실)
  · "난 B2B가 더 끌려" → opinion (주관)
  · "B2B가 우리 색깔에 맞아, 영업 인프라도 강하니까" → opinion ('우리 색깔'은 가치판단 — 근거('영업 인프라 강함')는 RAG가 따지지만 라벨은 opinion)
  · "타겟은 네이버로 가자" → claim (결정 = 향후 슬롯에 박히는 약속)
  · 단, 슬롯 통과조건 미달의 모호한 답변("월 매출 잘 나오게")은 claim이 아니라 clarification_needed로 우선 라벨.
- correction은 '이전에 정한 것을 무르거나 바꿀 때'만. 정정 키워드가 있어도 새 진술이면 claim:
  · "아 카카오는 빼자" → correction (앞서 넣은 타겟을 무름)
  · "네이버 말고 카카오로 가자" → correction (대상 교체)
  · "말고 또 뭐가 있을까?" → question (정정 아님 — 키워드만 같음)
  · "바꾸는 게 어렵진 않아" → claim/opinion (정정 의도 없음)
- 한 세그먼트 다중 라벨 예:
  · "시장 규모 어때? 그리고 타겟은 네이버로 가자" → ["question","claim"]
  · "그건 취소하고, 일본 시장은 성장 중이잖아" → ["correction","claim"]

in_scope (이 세그먼트가 '사용자의 사업 계획을 세우는 것'과 관련 있는가, 불리언):
- true: 문제·고객·솔루션·시장·차별점·수익·목표·자원·일정·리스크에 관한 발화, 회사 자료를 묻는 질문, 계획 진행/출력 신호 등 — 계획에 기여하거나 계획에 필요한 정보를 묻는 것.
- false: 계획과 무관한 무맥락 발화 — 일반 상식·산술("1+1은 2이다"), 잡담("오늘 날씨 어때"), 계획과 상관없는 코딩·번역·기타 요청.
- 애매하면 true. 과도하게 막지 않는다(사용자 발화를 함부로 무시하지 않음).
- false여도 utterance_types는 형식상 평소대로 채운다 — 워커 차단·리다이렉트는 코드가 한다.
예:
  · "게임 시장 포화 상태래" → in_scope:true
  · "우리 회사 일본 진출한 적 있어?" → in_scope:true (회사 자료 질문)
  · "응 다음으로 넘어가자" → in_scope:true (진행 신호)
  · "1+1은 2이다" → in_scope:false
  · "오늘 서울 날씨 어때?" → in_scope:false
  · "파이썬 데코레이터 설명해줘" → in_scope:false

JSON만 출력."""


class ClassifyItem(BaseModel):
    canonical_text: str
    utterance_types: list[str] = Field(default_factory=list)
    in_scope: bool = True  # 사업 계획과 관련 있는 발화인가. 기본 True(애매하면 통과)


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
        # 알 수 없는 라벨은 버리고(타입 캐스팅), routes는 코드가 매트릭스로 재계산
        seg["utterance_types"] = [t for t in types if t in _VALID_TYPES]  # type: ignore[assignment]
        seg["routes"] = derive_routes(seg["utterance_types"])

        # 스코프 가드 — 무맥락/잡담은 워커를 코드가 막는다. LLM이 매트릭스대로
        # claim→research를 줘도, in_scope=false면 routes를 ["none"]으로 덮어쓴다
        # ("LLM은 무슨 발화인지/관련 있는지만, 누구를 부를지는 코드"라는 분리 유지).
        # 개수 불일치(llm_items=None)면 보수적으로 in_scope=true(과차단 방지).
        in_scope = llm_items[idx].in_scope if llm_items is not None else True
        seg["in_scope"] = in_scope
        if not in_scope:
            seg["routes"] = ["none"]  # 리서치·RAG·비평 디스패치 안 됨 (clarify/dispatch 분기도 안 탐)

    return {"turn_segments": segments}
