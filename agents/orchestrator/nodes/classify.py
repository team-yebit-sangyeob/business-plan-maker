"""세그먼트 다중 라벨 분류 + 라우팅 결정 (기획서 5장 matrix).

흐름: segment_node가 만든 세그먼트마다 LLM이 발화 유형(다중 가능)을 고르고,
derive_routes가 그 라벨을 결정론적으로 워커 라우트로 변환한다. LLM이 matrix를
어겨도 최종 routes는 derive_routes가 강제한다
(LLM은 '무슨 유형인가'만, '누구를 부를까'는 코드가 결정).
처리 순서와 분기는 별도 priority 필드 없이 routes/utterance_types에서 바로 파생한다.

예시
----
입력 세그먼트 canonical_text:
    "웹툰 IP가 일본 시장에서 통할 것이다"
LLM utterance_types →  ["claim"]
derive_routes      →  ["research", "rag", "logic_validator"]   # 전제는 검색, 회사 적합성은 RAG, 논리는 logic_validator
(주장을 어떻게 분해·검증할지는 리서치 클러스터의 쿼리 분해기 몫 — 오케는 라우팅까지만)

입력 세그먼트:
    "웹툰 시장 규모가 어떻게 돼? 그리고 타겟은 네이버로 가자"
LLM utterance_types →  ["question", "claim"]           # 한 문장에 두 유형
derive_routes      →  ["research", "rag", "logic_validator"]    # 두 유형의 라우트 합집합
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from common.schema import PlanState
from common.schema.state import (
    Route,
    conversation_state_text,
    recent_history,
    slot_snapshot_text,
)
from agents.orchestrator.llm import call_json


_SYSTEM = """오케스트레이터 다중라벨 분류
각 세그먼트의 canonical_text를 보고, 먼저 상호작용(interaction)인지 내용(content)인지 가른다. 그다음 해당하는 유형을 모두 고른다(다중 라벨 허용).

상호작용(interaction) — 워커를 부르지 않고 conversation이 대화로 받는다:
- meta: 단순 응답이나 진행 신호 (예: "응 다음", "ok", "좋아")
- recall: 직전 대화에 이미 나온 내용을 다시 묻거나 확인하는 되묻기 ([최근 대화]에 있는 걸 되묻는 경우. 예: "아까 일본 된다며?", "방금 뭐랬지?", "우리 타겟 뭐로 정했지?")
- tool_help: 이 도구·슬롯·사용법 자체를 묻는 메타질문 (사업 내용이 아니라 '이 도구가 어떻게 동작하나'를 묻는다. 예: "솔루션 슬롯이 뭐야?", "슬롯이 뭔데?", "이거 어떻게 쓰는 거야?", "넌 뭐 할 수 있어?", "왜 자꾸 물어봐?")
- reason: 이미 모았거나 대화에서 논의된 내용을 바탕으로 결론·문제점·시사점을 추론·도출·종합해달라는 요청 (새 외부 정보 탐색이 아니라 '가진 것'을 종합. 예: "여기서 문제점 추론해봐", "방금 분석에서 도출할 만한 게 뭐야?", "위 내용 종합해줘", "이걸로 시사점 정리해줘"). 사용자가 어시스턴트에게 직접 의견·판단·제안·결정을 청하는 발화도 reason이다 (예: "네가 생각하는 문제는 뭐야?", "넌 어떻게 봐?", "네 생각엔?", "추천해줘", "정해줘", "제안해봐") — 새 외부정보를 물은 게 아니라 '가진 것으로 종합해 제안'하라는 요청이다.

내용(content) — 검증·명확화·정정·정보탐색이 필요하다:
- clarification_needed: 모호하거나 추상적이라 추가 질문이 필요함
- claim: 검증할 수 있는 내용 발화 — 외부 사실 주장, 가설, 결정, 제약, 근거 있는 가치판단을 모두 포함 (예: "게임 시장 포화" / "일본에서 통할 거 같다" / "타겟은 네이버로 가자" / "예산 1억, 6개월" / "B2B가 우리 색깔, 영업 인프라도 강하니까")
- correction: 정정이나 취소 (예: "아니, 빼자")
- question: 대화에 없던 새 정보를 물어봄 (예: "웹툰 시장 규모가 어떻게 돼?")

여러 유형이 한 세그먼트에 동시에 해당할 수 있다 — 예: "시장 규모 어때? 타겟은 네이버로 가자" 같은 한 문장이면 question+claim. 단, recall·meta·tool_help·reason은 단독으로 둔다(되묻기·진행신호·도구질문·추론요청은 그 자체가 발화의 핵심).

[대화 상태]·[현재 슬롯]을 함께 본다 — 이번 발화가 '직전에 시스템이 물은 것에 대한 답'인지부터 가린다:
- 열린 제안에 대한 답: [대화 상태]에 확인 대기가 있고, 어떤 세그먼트가 그 제안에 대한 답이면(수락 "이대로 넣어"/"응", 거부 "빼"/"아니", 슬롯 선택 "솔루션에 넣어") meta로 둔다 — 새 주장이 아니라 확인 응답이라 워커를 부르지 않는다. 그 답에 딸린 새 내용("응 넣고, 타겟은 20대야"의 "타겟은 20대")은 별개 세그먼트로 평소대로 분류한다.
- 직전 질문 슬롯에 대한 답: [대화 상태]에 직전 질문 슬롯이 있고 짧은 답(고유명사·수치·짧은 구)이 오면, 짧다는 이유로 clarification_needed로 떨구지 말고 claim으로 둔다(그 슬롯을 채울 답이다). 단 그 슬롯 기준에 비해 내용이 공허하면 아래대로 clarification_needed.

구분 가이드 — 헷갈리는 경계:
- recall vs question: 이미 [최근 대화]에 나온 걸 다시 확인하면 recall(대화 이력에서 답함), 대화에 없는 새 정보를 물으면 question(리서치·RAG로 답함). 애매하면 question.
- reason vs question/claim: 이미 모은/논의된 내용("여기서/방금/위 내용에서/이걸로")에서 추론·도출·종합을 요청하면 reason(워커 없이 conversation이 누적 근거로 답). 대화에 없던 새 외부 정보를 물으면 question, 검증할 새 주장을 내면 claim. 단서가 기존 맥락 지시('여기서·방금·위·이걸로')면 reason, 새 대상이면 question/claim. 어시스턴트에게 판단·의견·제안을 청하는 발화('네가/네 생각엔/추천해줘/정해줘/제안해봐')도 reason이다 — 맥락 지시어가 없어도 '가진 것으로 종합해 제안하라'는 뜻이지, 새 외부 사실을 물은 게 아니다. 단, 대화에 없던 새 외부 대상의 사실을 물으면 question. 그 외 애매하면 question.
- tool_help vs question/claim: 이 '도구·슬롯' 자체를 묻는 메타질문이면 tool_help, 사업 '내용'을 묻거나 정하면 question·claim. 슬롯 이름이 들어가도 '그 칸이 뭐냐(도구 설명)'면 tool_help, '그 칸에 뭘 넣을까(내용)'면 question·claim.
  · "솔루션 슬롯이 뭐하는 칸이야?" → tool_help (도구 설명을 물음)
  · "우리 솔루션 뭐로 하지?" / "솔루션은 B2B 감수 서비스로 가자" → question·claim (사업 내용)
  · "웹툰 시장 규모 어때?" → question (외부 사실)
- claim: 참/거짓이나 적합성을 따질 수 있으면 가치판단이라도 claim이다. 근거가 붙은 선호("B2B가 우리 색깔에 맞아, 영업 인프라도 강하니까")도 claim — 그 근거를 RAG·논리검증이 따진다. 단, 검증할 전제 없이 막연·공허한 답변은 claim이 아니라 clarification_needed로 우선 라벨한다(검증할 내용이 없으니 리서치 대신 되묻는다). 슬롯을 정하는 답인데 그 슬롯이 요구하는 알맹이가 비면 막연한 답이다 — 예: 목표 자리에 "결과물"/"잘 됐으면"(수치·기한 없음), 타겟 자리에 "사람들"/"누구나"(구체 대상 없음), "월 매출 잘 나오게"/"그냥 B2B가 끌려".
  · "B2B 시장이 더 커" → claim (시장 규모는 외부 사실)
  · "타겟은 네이버로 가자" → claim (결정 = 향후 슬롯에 박히는 약속)
- correction은 '이전에 정한 것을 무르거나 바꿀 때'만. 정정 키워드가 있어도 새 진술이면 claim:
  · "아 카카오는 빼자" → correction (앞서 넣은 타겟을 무름)
  · "네이버 말고 카카오로 가자" → correction (대상 교체)
  · "말고 또 뭐가 있을까?" → question (정정 아님 — 키워드만 같음)
  · "바꾸는 게 어렵진 않아" → claim (정정 의도 없음)
- 한 세그먼트 다중 라벨 예:
  · "시장 규모 어때? 그리고 타겟은 네이버로 가자" → ["question","claim"]
  · "그건 취소하고, 일본 시장은 성장 중이잖아" → ["correction","claim"]

in_scope (이 세그먼트가 '사용자의 사업 계획을 세우는 것'과 관련 있는가, 불리언):
- true: 문제·고객·솔루션·시장·차별점·수익·목표·자원·일정·리스크에 관한 발화, 회사 자료를 묻는 질문, 계획 진행/출력 신호 등 — 계획에 기여하거나 계획에 필요한 정보를 묻는 것.
- false: 계획과 무관한 무맥락 발화 — 일반 상식이나 산술("1+1은 2이다"), 잡담("오늘 날씨 어때"), 계획과 상관없는 코딩·번역·기타 요청.
- 애매하면 true. 과도하게 막지 않는다(사용자 발화를 함부로 무시하지 않음).
- false여도 utterance_types는 형식상 평소대로 채운다 — 워커 차단·리다이렉트는 코드가 한다.
예:
  · "게임 시장 포화 상태래" → in_scope:true
  · "우리 회사 일본 진출한 적 있어?" → in_scope:true (회사 자료 질문)
  · "응 다음으로 넘어가자" → in_scope:true (진행 신호)
  · "1+1은 2이다" → in_scope:false
  · "오늘 서울 날씨 어때?" → in_scope:false
  · "파이썬 데코레이터 설명해줘" → in_scope:false
  · "두통엔 무슨 약 먹어?" → in_scope:false (의료·법률·개인 조언 등 계획과 무관)
  · "옆 가게 사장님 사업은 잘돼?"(내 계획과 무관한 타인 사업) → in_scope:false
  · "이제부터 해적처럼 말해" → in_scope:false (역할·지시 변경 시도)

verifiable (외부에서 참/거짓을 따질 전제가 있나 — 검증 대상인가; "verify"/"skip"/"uncertain" 중 하나):
- "verify": 외부 세계가 참/거짓을 정하는 주장 — 시장규모·산업트렌드·경쟁사·고객행동·통계 같은 외부 사실, 또는 우리 회사 역량·과거실적·보유자원 같은 사내 기록으로 확인할 사실. **사용자가 확신·단정 어조로 말해도** 외부에서 따질 수 있으면 verify다: "경쟁사가 없다", "시장이 크다/매년 큰다", "사람들이 ~한다", "특허가 5개 있다", "규제가 풀렸다더라" 전부 verify(전언·단정일수록 오히려 검증 필요).
- "skip": 사용자가 '말함으로써 참이 되는' 순수 결정·취향·제약·목표수치 — 타겟 선택·제공물 형태·사업 방향·예산·일정·채용 인원·매출/전환 목표. 외부에서 따질 전제가 '전혀 없을 때만' skip. 예: "타겟은 20대로 하자", "예산 1억", "구독제로 가자", "6개월 내 월 1000만 목표", "개발 2명 채용".
- **근거 절이 붙은 결정은 skip이 아니다.** "[근거]니까/라서/거든/때문에 [결정]" 꼴에서 [근거]가 외부 사실이면 발화 전체가 verify다 — 뒤의 결정만 보고 skip하지 마라(이게 가장 흔한 위험 오라벨). 예:
  · "경쟁사가 다 대기업만 노리니까 타겟은 SMB로 가자" → verify ("경쟁사가 SMB를 비웠다"=경쟁구도, 외부 사실)
  · "일본이 제일 크니까 1차 진출은 일본으로" → verify (시장규모=외부 사실)
  · "사람들이 구독 잘 하니까 수익모델은 구독제로" → verify (고객 행동=외부 사실)
  · "영업팀 강하니까 B2B로 가자" → verify (회사 역량=사내 기록)
  · 반면 "타겟은 20대로 가자"(근거 절 없음) → skip.
- "uncertain": 근거 절 없이 '우리한테 맞다/우리 색깔/답이다' 같은 적합성 주장이라 내부 결정인지 외부 사실인지 가릴 수 없을 때. 예: "B2B가 우리한테 맞아"(전제 없음), "프리미엄이 답이지".
- 핵심: 확실치 않으면 skip 말고 uncertain이나 verify로 둔다. skip 오라벨은 검증을 건너뛰어 '틀린 믿음'을 슬롯에 박는 위험한 방향이고, verify 오라벨은 헛검증일 뿐 안전하다. **순수 결정(근거 절 없음)일 때만 skip.**
- 모든 세그먼트에 매긴다. interaction 유형(meta·recall·tool_help·reason)엔 의미가 없으니 verify로 둬도 무방하다(어차피 워커 미발동).

JSON만 출력."""


class ClassifyItem(BaseModel):
    canonical_text: str
    utterance_types: list[str] = Field(default_factory=list)
    in_scope: bool = True  # 사업 계획과 관련 있는 발화인가. 기본 True(애매하면 통과)
    # 외부에서 참/거짓을 따질 전제가 있나(검증 대상인가). 기본 verify(보수) — skip 오라벨이
    # 팩트체크를 건너뛰는 위험을 막는다. skip이면 claim이어도 워커 디스패치 안 함(사용자 결정).
    verifiable: Literal["verify", "skip", "uncertain"] = "verify"


class ClassifyOut(BaseModel):
    items: list[ClassifyItem]


# 유형 → 활성 클러스터. content는 라우트를 파생하고, interaction(meta·recall)은 빈 집합 →
# derive_routes가 ["none"]. interaction을 '키 삭제'가 아니라 '빈 집합'으로 두는 이유: 라벨이
# _VALID_TYPES 필터(아래)를 통과해 살아남아야 conversation·correction이 그 라벨을 본다.
_ROUTE_MATRIX: dict[str, set[Route]] = {
    # content — 워커 라우트 파생
    "clarification_needed": {"clarify"},
    "claim": {"research", "rag", "logic_validator"},  # 사실·가설·결정·제약·근거 있는 가치판단 통합 — 전제는 리서치, 회사 적합성은 RAG, 논리는 logic_validator
    "question": {"research", "rag"},  # 외부 사실이면 리서치, 회사 내부 사안이면 RAG (둘 다 발동, 답 찾은 쪽이 응답)
    "correction": set(),  # correction 노드가 처리
    # interaction — 디스패치 없음, conversation이 처리
    "meta": set(),
    "recall": set(),  # 되묻기 — 워커 없이 conversation이 대화 이력에서 답
    "tool_help": set(),  # 도구/슬롯 메타질문 — 워커 없이 conversation이 SLOT_SPECS·APP_OVERVIEW에서 답
    "reason": set(),  # 추론·도출·종합 — 워커 없이 conversation이 누적 근거(session_evidence)+대화이력에서 직접 추론
}


_VALID_TYPES: set[str] = set(_ROUTE_MATRIX.keys())

# interaction 유형 — 워커를 부르지 않고 conversation이 직접 받는다. content 라벨과 한 세그먼트에
# 섞이면 content를 덮어 routes를 ["none"]으로 만든다(classify_node의 interaction-precedence 가드).
_INTERACTION_TYPES: set[str] = {"meta", "recall", "tool_help", "reason"}


_WORKER_ROUTES: set[Route] = {"research", "rag", "logic_validator"}


def derive_routes(utterance_types: list[str], verifiable: str = "verify") -> list[Route]:
    """다중 라벨(+verifiable) → 발동 워커 라우트(합집합). 매트릭스가 단일 출처.

    verifiable="skip"이면 외부에서 따질 전제가 없는 사용자 결정/취향/제약이라 워커 디스패치를
    걷어낸다(claim이어도). 단 question은 본질적으로 외부 조회라 skip이어도 살린다(안전 백스톱).
    기본값 verify는 기존 동작과 동일(디스패치) — verifiable 미설정 호출과 하위호환.

    예: ["claim"], verify     → ["research","rag","logic_validator"]
        ["claim"], skip       → ["none"]   (사용자 결정 — 검증 안 함)
        ["claim","question"], skip → ["research","rag"]  (question은 보존)
        ["meta"]              → ["none"]
    """
    routes: set[Route] = set()
    for t in utterance_types:
        routes |= _ROUTE_MATRIX.get(t, set())
    if verifiable == "skip":
        # 사용자 결정 — 검증할 외부 전제 없음 → 워커(리서치·RAG·논리검증) 걷어낸다.
        routes -= _WORKER_ROUTES
        # 단 question은 본질적으로 외부 조회라 살린다(skip이어도 — 안전 백스톱).
        if "question" in utterance_types:
            routes |= _ROUTE_MATRIX["question"]
    if not routes:
        return ["none"]
    # 안정적 정렬 — 같은 라벨 집합이면 항상 같은 순서로 나오게(테스트·캐시 친화)
    order: list[Route] = ["clarify", "research", "rag", "logic_validator", "none"]
    return [r for r in order if r in routes]


async def classify_node(state: PlanState) -> dict:
    """세그먼트마다 발화 유형·in_scope·routes를 채워 {"turn_segments"}를 갱신한다."""
    segments = list(state.get("turn_segments") or [])
    if not segments:
        return {"turn_segments": []}

    texts = [s.get("canonical_text") or s.get("text", "") for s in segments]
    seg_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    # 되묻기·확인응답·충돌·충분성 판정에 대화 상태·슬롯 값·직전 대화가 필요하다 —
    # 같은 LLM 호출, 프롬프트만 풍부해진다(라우팅은 derive_routes가 그대로 코드로 정함).
    state_block = conversation_state_text(state)
    payload = (
        (f"{state_block}\n\n" if state_block else "")
        + f"[현재 슬롯]\n{slot_snapshot_text(state)}\n\n"
        f"[최근 대화]\n{recent_history(state)}\n\n"
        f"[세그먼트]\n{seg_list}"
    )

    out = await call_json(_SYSTEM, payload, ClassifyOut)

    # LLM 결과를 인덱스로 매칭 (개수 어긋나면 fallback)
    llm_items = out.items if len(out.items) == len(segments) else None

    for idx, seg in enumerate(segments):
        prior = list(seg.get("utterance_types") or [])
        types: list[str] = list(prior)
        if llm_items is not None:
            for t in llm_items[idx].utterance_types:
                if t in _VALID_TYPES and t not in types:
                    types.append(t)
        if not types:
            # LLM 개수 불일치(llm_items=None)이거나 빈 결과일 때의 안전 기본값.
            # clarification_needed는 _VALID_TYPES에 있어 아래 필터를 통과하고, clarify만 타서
            # 워커 비용 0 + 사용자에게 되묻게 만든다 — 분류 실패 시 가장 피해가 적은 라벨.
            # (claim으로 두면 실패 턴마다 리서치+RAG+논리검증이 터진다.)
            types = ["clarification_needed"]
        # 알 수 없는 라벨은 버린다(타입 캐스팅).
        labels = [t for t in types if t in _VALID_TYPES]
        # interaction-precedence — 도구질문·되묻기·진행신호(interaction)가 content 라벨과 한
        # 세그먼트에 섞이면 content를 떨군다. derive_routes가 라우트를 합집합해서
        # ["tool_help","claim"]이면 워커가 다시 새기 때문(원래 misfire의 재발점). 프롬프트도
        # 이들을 단독으로 두라 하지만 "누구를 부를지는 코드"라 여기서 backstop으로 강제한다.
        interaction = [t for t in labels if t in _INTERACTION_TYPES]
        if interaction:
            labels = interaction
        seg["utterance_types"] = labels  # type: ignore[assignment]
        # verifiable — 외부에서 따질 전제가 있나(검증 대상인가). 기본 verify(보수): LLM 미보고·
        # 개수 불일치면 검증 쪽으로 둬 팩트체크를 건너뛰지 않는다. derive_routes가 skip일 때만
        # 워커를 걷어낸다(사용자 결정은 디스패치 0).
        verifiable = llm_items[idx].verifiable if llm_items is not None else "verify"
        seg["verifiable"] = verifiable  # type: ignore[assignment]
        seg["routes"] = derive_routes(seg["utterance_types"], verifiable)

        # 스코프 가드 — 무맥락/잡담은 워커를 코드가 막는다. LLM이 매트릭스대로
        # claim→research를 줘도, in_scope=false면 routes를 ["none"]으로 덮어쓴다
        # ("LLM은 무슨 발화인지/관련 있는지만, 누구를 부를지는 코드"라는 분리 유지).
        # 개수 불일치(llm_items=None)면 보수적으로 in_scope=true(과차단 방지).
        in_scope = llm_items[idx].in_scope if llm_items is not None else True
        seg["in_scope"] = in_scope
        if not in_scope:
            seg["routes"] = ["none"]  # 리서치·RAG·논리검증 디스패치 안 됨 (clarify/dispatch 분기도 안 탐)

    return {"turn_segments": segments}
