"""슬롯 10개(필수 3 + 선택 7) + 턴 처리 상태 (LangGraph PlanState)."""
from __future__ import annotations

from typing import Literal, TypedDict

from common.schema.labels import SourceLabel


# 슬롯 질문(조사) 순서 = 사업계획을 자연스럽게 전개하는 순서.
# 문제 → 고객 → 무엇을(solution) → 시장/경쟁(market) → 그래서 차별점(advantage) →
# 수익모델(revenue) → 목표수치(goal) → 자원(resources) → 일정(milestones) → 리스크(risks).
# 이 튜플 순서가 곧 conversation_node가 "다음 빈칸"을 고르는 기본 질문 순서다(중간에
# 사용자가 다른 슬롯을 말하면 그건 채워지고, 다음 턴엔 남은 첫 빈칸을 묻는다).
# 예시는 웹툰 감수 사업.
ALL_SLOTS: tuple[str, ...] = (
    "problem",     # 문제: "웹툰 신작 공개 후 3~5일 내 성 감수성 논란 1건+, 30%가 휴재로"  [필수]
    "target",      # 타겟: "네이버·카카오 콘텐츠 운영팀(5~10명), 의사결정자 콘텐츠본부장급"  [필수]
    "solution",    # 솔루션(결정형): "B2B 감수 서비스" / "AI 자동 검수 툴"
    "market",      # 시장 근거(데이터형): "국내 웹툰 시장 규모·경쟁사" — 리서치가 채움
    "advantage",   # 차별점·경쟁우위(파생형): "외주 감수 대비 실시간·1/5 비용" — solution+market 뒤라야 나옴
    "revenue",     # 수익 모델(결정형): "월 구독 SaaS" / "건당 컨설팅 피" — 후보 제시
    "goal",        # 목표(필수): "6개월 유료 3개사·월 1,500만, 미달 시 재검토" — 솔루션·수익모델 뒤라야 현실적 숫자  [필수]
    "resources",   # 필요 리소스(파생형): "감수 인력 2명·예산 1억"
    "milestones",  # 마일스톤(파생형): "3개월 PoC → 6개월 첫 계약" — 추론 도출
    "risks",       # 리스크(파생형): "내부 감수팀 보유 시 니즈 약함"
)

# 출력 게이트 필수 3 — '셋 다 차야 출력'이라는 멤버십(기획서 3장). 질문 순서와 무관하다:
# goal은 질문은 늦게(7번째) 받지만 출력 전엔 반드시 차 있어야 한다(gate가 강제).
REQUIRED_SLOTS: tuple[str, ...] = ("problem", "target", "goal")
# 선택 = 나머지. 자연 질문 순서를 유지하려고 ALL_SLOTS에서 거른다(비어도 [미정]로 출력 가능).
OPTIONAL_SLOTS: tuple[str, ...] = tuple(s for s in ALL_SLOTS if s not in REQUIRED_SLOTS)


# 발화 유형 6종 — 매 턴 세그먼트마다 라벨링(다중 가능). 괄호는 발동 워커.
UtteranceType = Literal[
    "clarification_needed",  # 모호/추상 → 명확화. 예: "웹툰 감수성으로 사업하고 싶어"
    "claim",                 # 검증 가능한 내용 발화(사실·가설·결정·제약) → 리서치+RAG+비평. 예: "게임 시장 포화 상태래" / "일본에서 통할 거 같아" / "타겟은 네이버로 가자" / "예산 1억, 6개월"
    "opinion",               # 주관 선호 → RAG+비평. 예: "우리 색깔엔 B2B가 더 맞아"
    "correction",            # 정정·취소 → 슬롯 덮어쓰기. 예: "아 카카오는 빼자"
    "question",              # 사용자 정보 요청 → 리서치(외부)·RAG(내부). 예: "웹툰 시장 규모가 어떻게 돼?"
    "meta",                  # 단순응답·진행 신호. 예: "응 다음", "여기까지 뽑아줘"
]


# claim 세그먼트의 세부 유형 — 리서치(VerificationRequest.claim_type)가 검증 강도/쿼리 분해에 사용.
# claim 라벨이 붙은 세그먼트에만 의미가 있고, 그 외 유형엔 None.
ClaimType = Literal[
    "fact",                # 검증 가능한 외부 사실. 예: "게임 시장 포화"
    "hypothesis_premise",  # 가설의 검증 가능한 전제. 예: "일본 웹툰 시장 성장 추세"(가설 자체는 비평 몫)
    "decision_context",    # 결정의 배경 사실. 예: "네이버·카카오 콘텐츠 운영 조직 현황"
    "market_fill",         # Type 3 자동 채움의 시장 근거. 더 엄격한 출처 기준 적용
]


# 세그먼트가 발동시킬 워커. classify의 _ROUTE_MATRIX가 발화 유형→라우트로 변환.
Route = Literal[
    "research",  # 웹 리서치 — 외부 사실 검증
    "rag",       # 회사 문서 RAG — 내부 정합성
    "critic",    # 비평 — 추론 점검 + 정합성 판단 (구 inference)
    "clarify",   # 명확화 — 워커 호출 없이 다음 턴까지 보류
    "none",      # 스킵 (meta/correction 등)
]


class Slot(TypedDict, total=False):
    # 예: {"value": "B2B 감수 서비스", "source_label": USER, "status": "filled"}
    value: str | None                                          # 채워진 값, 비면 None
    source_label: SourceLabel                                  # 출처(user/research/...)
    status: Literal["empty", "needs_clarification", "filled"]  # 모호한 한 줄이면 needs_clarification


class Segment(TypedDict, total=False):
    # 한 사용자 발화에서 잘라낸 의미 단위. 예: "일본에서 통할 거 같아"
    text: str                            # 원문 조각 그대로
    canonical_text: str                  # 맥락 복원된 자기충족 문장: "웹툰 IP가 일본 시장에서 통할 것이다"
    utterance_types: list[UtteranceType] # 다중 라벨: ["claim"] (주장이면서 질문이면 ["claim","question"])
    claim_type: ClaimType | None         # "claim" 유형일 때만 세부 분류, 그 외 None. 리서치로 전달.
    target_slot: str | None              # 들어갈 슬롯(있으면): "target"
    routes: list[Route]                  # 발동 워커: ["research","rag","critic"]
    # 0=correction, 1=clarification, 2=dispatch(claim/question), 3=opinion/meta
    priority: int


class Correction(TypedDict):
    # 정정 이벤트 한 건. 예: 5번째 턴에 솔루션을 바꿈
    slot: str             # "solution"
    previous: str | None  # "B2B 감수 서비스"
    new: str | None       # "AI 자동 검수 툴" (clear면 None)
    turn: int             # 5


class ValidationReport(TypedDict, total=False):
    # 워커 한 번의 결과. 예: 리서치가 "게임 시장 포화" 주장을 검증
    subject: str                                                       # "게임 시장이 포화 상태다"
    findings: list[str]                                                # ["2024년 모바일 게임 신규 출시 -12%", ...]
    sources: list[str]                                                 # ["https://...", "업계 리포트 X"]
    agreement: Literal["confirms", "contradicts", "partial", "unknown"]# 사용자 주장과의 일치도
    cluster: Literal["research", "rag", "critic"]                      # 어느 워커가 냈는지


class Message(TypedDict):
    # 대화 한 줄. 예: {"role": "user", "content": "타겟은 네이버", "turn": 2}
    role: Literal["user", "assistant"]
    content: str
    turn: int


def _empty_slot() -> Slot:
    return {"value": None, "source_label": SourceLabel.EMPTY, "status": "empty"}


def initial_state() -> "PlanState":
    return {
        "session_id": "",
        "turn": 0,
        "user_input": "",
        "messages": [],
        "turn_segments": [],
        "slots": {name: _empty_slot() for name in ALL_SLOTS},
        "correction_log": [],
        "validation_reports": [],
        "pending_clarifications": [],
        "pending_question": "",
        "output_request": None,
    }


class PlanState(TypedDict, total=False):
    session_id: str
    turn: int
    user_input: str

    messages: list[Message]
    turn_segments: list[Segment]
    slots: dict[str, Slot]
    correction_log: list[Correction]
    validation_reports: list[ValidationReport]

    pending_clarifications: list[str]
    pending_question: str
    # 출력 요청 분기 결과 (8장 Type 0/1/2/3)
    output_request: Literal["type0", "type1", "type2", "type3"] | None
