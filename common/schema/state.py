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
    "solution",    # 솔루션: "B2B 감수 서비스" / "AI 자동 검수 툴"
    "market",      # 시장 근거: "국내 웹툰 시장 규모·경쟁사" — 리서치가 근거를 회수
    "advantage",   # 차별점·경쟁우위: "외주 감수 대비 실시간·1/5 비용" — solution+market 뒤라야 나옴
    "revenue",     # 수익 모델: "월 구독 SaaS" / "건당 컨설팅 피"
    "goal",        # 목표(필수): "6개월 유료 3개사·월 1,500만, 미달 시 재검토" — 솔루션·수익모델 뒤라야 현실적 숫자  [필수]
    "resources",   # 필요 리소스: "감수 인력 2명·예산 1억"
    "milestones",  # 마일스톤: "3개월 PoC → 6개월 첫 계약"
    "risks",       # 리스크: "내부 감수팀 보유 시 니즈 약함"
)

# 출력 게이트 필수 3 — '셋 다 차야 출력'이라는 멤버십(기획서 3장). 질문 순서와 무관하다:
# goal은 질문은 늦게(7번째) 받지만 출력 전엔 반드시 차 있어야 한다(gate가 강제).
REQUIRED_SLOTS: tuple[str, ...] = ("problem", "target", "goal")
# 선택 = 나머지. 자연 질문 순서를 유지하려고 ALL_SLOTS에서 거른다(비어도 [미정]로 출력 가능).
OPTIONAL_SLOTS: tuple[str, ...] = tuple(s for s in ALL_SLOTS if s not in REQUIRED_SLOTS)


# --- 슬롯 정의(단일 원천) ---------------------------------------------------
# 슬롯 선택은 segment(target_slot_hint)·fill(extract_slot_fills)·correction 세 군데서
# 일어난다. 정의가 갈리면 같은 내용이 다른 슬롯에 박히므로, 정의·경계를 여기 한 곳에
# 모으고 slot_guide_text()로 렌더해 세 프롬프트가 모두 같은 문구를 임베드한다.
# boundary = "이건 여기 NOT 저기" — 헷갈리는 이웃 슬롯과의 경계(=fill의 애매도 판정 근거).
class SlotSpec(TypedDict):
    title: str       # 표시 이름 — "솔루션"
    definition: str  # 무엇이 들어가나
    boundary: str    # 헷갈리는 이웃 슬롯과의 경계
    question: str    # 대화에서 이 슬롯을 물을 때의 톤(질문 예시)


SLOT_SPECS: dict[str, SlotSpec] = {
    "problem": {
        "title": "문제 정의",
        "definition": "해결하려는 핵심 문제 — 누가·어떤 상황에서·무엇 때문에·어떤 손실을 보는지",
        "boundary": "고객이 겪는 고통·손실만. 시장 규모·경쟁 데이터는 market, 그 돈 낼 사람은 target.",
        "question": "어떤 문제예요? — 누가 · 어떤 상황에서 · 무엇 때문에 · 어떤 손실을 보는지까지 얘기해주면 좋아요.",
    },
    "target": {
        "title": "타겟 / 고객",
        "definition": "돈을 낼 사람·조직 — 회사·부서·직책·규모·접촉 경로",
        "boundary": "'누가 사는가'만. 그들이 겪는 고통은 problem.",
        "question": "타겟이 누구예요? — '어느 회사'가 아니라 그 안에서 계약서에 도장 찍는 사람·부서·규모·접촉 경로까지.",
    },
    "solution": {
        "title": "솔루션",
        "definition": "제공할 것의 형태 — 서비스/제품/플랫폼/툴 중 무엇을 만드는가",
        "boundary": "'무엇을 만드나'만. 과금 방식은 revenue, 왜 더 나은지는 advantage.",
        "question": "솔루션 형태는 어떻게 가져갈 거예요? (서비스 / 제품 / 플랫폼 중에)",
    },
    "market": {
        "title": "시장 근거",
        "definition": "시장 규모·성장 추세·경쟁사 존재 등 검증 가능한 외부 사실·데이터",
        "boundary": "외부 '데이터'만. 그 경쟁사 대비 우리 우위는 advantage, 고객 고통은 problem.",
        "question": "시장 규모나 경쟁사 쪽은 짚어둔 데이터 있어요? 없으면 제가 찾아볼게요.",
    },
    "advantage": {
        "title": "차별점 / 경쟁우위",
        "definition": "기존 대안·경쟁사 대비 우리가 이기는 이유·지속 우위(해자)",
        "boundary": "'왜 우리가 이기나'만. 무엇을 만드나는 solution, 경쟁사 데이터 자체는 market, 위협은 risks.",
        "question": "기존 대안이나 경쟁사 대비 우리만의 차별점·이기는 이유는 뭐예요?",
    },
    "revenue": {
        "title": "수익 모델",
        "definition": "어떻게 돈을 버는가 — 구독/건당/라이선싱 등 과금 방식",
        "boundary": "과금 '방식'만. 무엇을 파나는 solution, 목표 매출 수치는 goal.",
        "question": "수익 모델 — 구독, 건당, 라이선싱 중 어떤 쪽 그림이에요?",
    },
    "goal": {
        "title": "목표 수치",
        "definition": "목표 수치 — 언제까지 얼마, 그리고 실패 임계값(어디까지 안 되면 접나/방향 트나)",
        "boundary": "측정 가능한 '목표 숫자+기한+실패선'만. 과금 방식 자체는 revenue.",
        "question": "목표 수치는요? — 언제까지 얼마, 그리고 어디까지 안 되면 접거나 방향을 트는지 실패 임계값도 같이.",
    },
    "resources": {
        "title": "필요 리소스",
        "definition": "확보해야 할 역량 — 인력·예산 규모",
        "boundary": "'무엇이 얼마나 필요한가(역량)'만. 시간순 단계·일정은 milestones.",
        "question": "필요한 인력·예산 규모는 어떻게 보세요?",
    },
    "milestones": {
        "title": "마일스톤 / 일정",
        "definition": "시간순 단계·일정 — 언제까지 어느 단계까지",
        "boundary": "'언제 무엇을(시간축)'만. 필요한 인력·예산 규모는 resources.",
        "question": "마일스톤 — 언제까지 어느 단계까지 가야 한다고 보세요?",
    },
    "risks": {
        "title": "리스크 / 대응",
        "definition": "계획을 위협하는 요인과 대응",
        "boundary": "'위협과 대응'만. 지속 우위(해자)는 advantage.",
        "question": "걱정되는 리스크부터 하나 짚어주실래요?",
    },
}

# ALL_SLOTS와 SLOT_SPECS가 어긋나면 곧장 터뜨린다(정의 누락·오타 방지).
assert set(SLOT_SPECS) == set(ALL_SLOTS), "SLOT_SPECS와 ALL_SLOTS 불일치"


def slot_title(slot: str) -> str:
    spec = SLOT_SPECS.get(slot)
    return spec["title"] if spec else slot


def slot_guide_text() -> str:
    """슬롯 정의+경계를 한 블록으로 렌더 — segment/fill/correction 프롬프트에 임베드.

    질문 순서(ALL_SLOTS)대로 한 줄씩:
      - solution (솔루션): <정의> | 경계: <경계 규칙>
    """
    return "\n".join(
        f"- {name} ({SLOT_SPECS[name]['title']}): "
        f"{SLOT_SPECS[name]['definition']} | 경계: {SLOT_SPECS[name]['boundary']}"
        for name in ALL_SLOTS
    )


# 발화 유형 6종 — 매 턴 세그먼트마다 라벨링(다중 가능). 괄호는 발동 워커.
UtteranceType = Literal[
    "clarification_needed",  # 모호/추상 → 명확화. 예: "웹툰 감수성으로 사업하고 싶어"
    "claim",                 # 검증 가능한 내용 발화(사실·가설·결정·제약) → 리서치+RAG+비평. 예: "게임 시장 포화 상태래" / "일본에서 통할 거 같아" / "타겟은 네이버로 가자" / "예산 1억, 6개월"
    "opinion",               # 주관 선호 → RAG+비평. 예: "우리 색깔엔 B2B가 더 맞아"
    "correction",            # 정정·취소 → 슬롯 덮어쓰기. 예: "아 카카오는 빼자"
    "question",              # 사용자 정보 요청 → 리서치(외부)·RAG(내부). 예: "웹툰 시장 규모가 어떻게 돼?"
    "meta",                  # 단순응답·진행 신호. 예: "응 다음", "여기까지 뽑아줘"
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
    in_scope: bool                       # 사업 계획과 관련 있는 발화인가. False면(무맥락 사실·잡담·무관 요청) classify가 routes=["none"]로 막고 integrator가 부드럽게 리다이렉트. 기본 True(애매하면 통과 — 과차단 방지).
    target_slot: str | None              # 들어갈 슬롯(있으면): "target"
    routes: list[Route]                  # 발동 워커: ["research","rag","critic"]. 처리 우선순위·분기는 routes/utterance_types에서 직접 파생(별도 priority 필드 없음).


class Correction(TypedDict):
    # 정정 이벤트 한 건. 예: 5번째 턴에 솔루션을 바꿈
    slot: str             # "solution"
    previous: str | None  # "B2B 감수 서비스"
    new: str | None       # "AI 자동 검수 툴" (clear면 None)
    turn: int             # 5


class PendingConfirmation(TypedDict, total=False):
    # 애매한 슬롯 주입을 사용자에게 확인받으려 보류한 한 건(턴을 넘어 영속).
    # extract_slot_fills가 confidence='ambiguous'로 본 값을 슬롯 대신 여기 쌓고,
    # conversation이 confirm_slot으로 묻고, 다음 턴 confirm_resolve가 해소한다.
    value: str                  # 채우려던 값
    proposed_slot: str          # fill이 1순위로 고른 슬롯
    candidate_slots: list[str]  # [proposed, *alt_slots] — 사용자에게 제시할 후보
    source_text: str            # 근거가 된 세그먼트 canonical_text(질문 문구용)
    reason: str                 # 왜 애매한지(짧게)
    attempts: int               # 재질문 횟수 — 2회 이상 미응답이면 proposed로 자동 확정


class ValidationReport(TypedDict, total=False):
    # 워커 한 번의 결과. 예: 리서치가 "게임 시장 포화" 주장을 검증
    subject: str                                                       # "게임 시장이 포화 상태다"
    findings: list[str]                                                # ["2024년 모바일 게임 신규 출시 -12%", ...]
    sources: list[str]                                                 # ["https://...", "업계 리포트 X"]
    agreement: Literal["confirms", "contradicts", "partial", "unknown"]# 사용자 주장과의 일치도
    cluster: Literal["research", "rag", "critic"]                      # 어느 워커가 냈는지


class VerificationRequest(TypedDict, total=False):
    # 리서치 클러스터 입력. dispatch가 세그먼트당 1건 구성해 run_research에 넘긴다.
    claim: str                       # 검증 대상 (사실 주장 또는 가설의 전제)
    utterance_label: UtteranceType   # 원본 발화 라벨 (참고용)
    slot_context: dict               # 관련 슬롯 발췌 — 분해기가 검증 방식을 정하는 단서
    freshness_max_days: int          # 검색 recency 힌트 (작을수록 최신 우선; 캐시 TTL 아님)
    session_id: str
    turn_id: int


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
        "turn_validation_reports": [],
        "pending_clarifications": [],
        "pending_question": "",
        "output_request": None,
        "pending_confirmations": [],
    }


class PlanState(TypedDict, total=False):
    session_id: str
    turn: int
    user_input: str

    messages: list[Message]
    turn_segments: list[Segment]
    slots: dict[str, Slot]
    correction_log: list[Correction]
    # 이번 턴 dispatch가 낸 리포트만 — 매 턴 리셋. 대화 에이전트의 결과 보고와
    # SSE 에이전트 활동 표시에 쓴다.
    turn_validation_reports: list[ValidationReport]

    pending_clarifications: list[str]
    pending_question: str
    # 출력 요청 분기 결과 (8장 Type 0/1/2)
    output_request: Literal["type0", "type1", "type2"] | None
    # 애매해서 주입 보류된 확인 큐 — 한 번에 하나씩 confirm_slot으로 묻는다. 턴 넘어 영속.
    pending_confirmations: list[PendingConfirmation]
