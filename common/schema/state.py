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

# 출력 필수 3 — '셋 다 차야 출력'이라는 멤버십(기획서 3장). 질문 순서와 무관하다:
# goal은 질문은 늦게(7번째) 받지만 출력 전엔 반드시 차 있어야 한다(plan 라우트의
# required_missing 검사 + 프론트 '계획서 생성' 버튼 비활성으로 강제).
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
        "definition": "해결하려는 핵심 문제 — 누가, 어떤 상황에서, 무엇 때문에, 어떤 손실을 보는지",
        "boundary": "고객이 겪는 고통과 손실만. 시장 규모와 경쟁 데이터는 market, 그 돈 낼 사람은 target.",
        "question": "어떤 문제예요? — 누가 · 어떤 상황에서 · 무엇 때문에 · 어떤 손실을 보는지까지 얘기해주면 좋아요.",
    },
    "target": {
        "title": "타겟 / 고객",
        "definition": "돈을 낼 사람과 조직 — 회사·부서·직책·규모·접촉 경로",
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
        "definition": "시장 규모·성장 추세·경쟁사 존재 등 검증 가능한 외부 사실과 데이터",
        "boundary": "외부 '데이터'만. 그 경쟁사 대비 우리 우위는 advantage, 고객 고통은 problem.",
        "question": "시장 규모나 경쟁사 쪽은 짚어둔 데이터 있어요? 없으면 제가 찾아볼게요.",
    },
    "advantage": {
        "title": "차별점 / 경쟁우위",
        "definition": "기존 대안과 경쟁사 대비 우리가 이기는 이유와 지속 우위(해자)",
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
        "definition": "확보해야 할 역량 — 인력과 예산 규모",
        "boundary": "'무엇이 얼마나 필요한가(역량)'만. 시간순 단계와 일정은 milestones.",
        "question": "필요한 인력·예산 규모는 어떻게 보세요?",
    },
    "milestones": {
        "title": "마일스톤 / 일정",
        "definition": "시간순 단계와 일정 — 언제까지 어느 단계까지",
        "boundary": "'언제 무엇을(시간축)'만. 필요한 인력과 예산 규모는 resources.",
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


# 도구 자체를 설명하는 한 문단 — tool_help(사용법·능력 질문)에 conversation이 답할 때 쓴다.
# 슬롯별 설명은 SLOT_SPECS가 단일 원천이라 여기 중복하지 않는다(tool_help_text가 둘을 합쳐 렌더).
APP_OVERVIEW: str = (
    "이건 대화로 사업 계획을 함께 세워가는 도구다. 사업 아이디어를 말하면 계획을 "
    "문제·타겟·솔루션·시장·차별점·수익모델·목표·리소스·일정·리스크 10개 항목(슬롯)으로 나눠 "
    "하나씩 채워간다. 말한 내용은 웹 리서치로 외부 사실을 확인하고, 회사 문서로 내부 정합성을 "
    "따지고, 그 근거가 주장을 뒷받침하는지 논리 검증까지 거친다. 필수 항목(문제·타겟·목표)이 "
    "차면 계획서를 뽑을 수 있다."
)


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


def tool_help_text() -> str:
    """tool_help(도구/슬롯 메타질문) 응답용 '참고 자료' 한 덩이 — 도구 전체 설명 + 슬롯 정의 전부.

    어느 슬롯을·얼마나 답할지(특정 1개 / 여럿 / 전체 / 도구 개요)는 코드가 가르지 않는다 —
    conversation이 사용자 질문(subject)에 맞춰 이 재료에서 필요한 만큼 골라 답한다. 슬롯
    텍스트는 SLOT_SPECS 단일 원천에서만 가져온다(설명이 갈리지 않게).
    """
    per_slot = "\n".join(
        f"- {SLOT_SPECS[name]['title']}: {SLOT_SPECS[name]['definition']} "
        f"(경계: {SLOT_SPECS[name]['boundary']})"
        for name in ALL_SLOTS
    )
    return f"{APP_OVERVIEW}\n\n슬롯(항목) 정의:\n{per_slot}"


def recent_history(state: PlanState, n: int = 10) -> str:
    """최근 messages n개를 프롬프트용 한 덩이로 렌더 — segment·classify·conversation 공용.

    되묻기(recall) 판정과 대화 이력 기반 응답에 쓴다(화자·턴·내용 한 줄씩).
    """
    messages = state.get("messages") or []
    tail = messages[-n:]
    if not tail:
        return "[이전 대화 없음]"
    return "\n".join(f"[{m['role']} t{m['turn']}] {m['content']}" for m in tail)


def required_missing(state: PlanState) -> list[str]:
    """필수 슬롯 중 출력을 막는 것들. status=='filled'이 아니면 미달.
    needs_clarification(모호한 한 줄 답변)도 막는다 — 값이 들어있어도 통과 불가.
    계획서 생성(plan 라우트)이 출력 직전 이 결과로 거절(Type 0)을 판정한다."""
    slots = state.get("slots") or {}
    return [s for s in REQUIRED_SLOTS if (slots.get(s) or {}).get("status") != "filled"]


def optional_missing(state: PlanState) -> list[str]:
    """선택 슬롯 중 값이 빈 것들. 예: ["market","revenue"]. 계획서의 '조기 출력' 표기에 쓰인다."""
    slots = state.get("slots") or {}
    return [s for s in OPTIONAL_SLOTS if not (slots.get(s) or {}).get("value")]


# 발화 유형 — content는 매트릭스로 워커 라우트를 파생하고, interaction(meta·recall)은
# 디스패치 없이 conversation이 직접 받는다(derive_routes가 ["none"]). 다중 라벨 가능.
UtteranceType = Literal[
    # content — 워커 라우트 파생
    "clarification_needed",  # 모호/추상 → 명확화. 예: "웹툰 감수성으로 사업하고 싶어"
    "claim",                 # 검증 가능한 내용 발화(사실·가설·결정·제약·근거 있는 가치판단) → 리서치+RAG+논리검증. 예: "게임 시장 포화 상태래" / "타겟은 네이버로 가자" / "우리 색깔엔 B2B가 맞아(영업 인프라 강함)"
    "correction",            # 정정·취소 → 슬롯 덮어쓰기(correction_node). 예: "아 카카오는 빼자"
    "question",              # 새 정보 요청 → 리서치(외부)·RAG(내부). 예: "웹툰 시장 규모가 어떻게 돼?"
    # interaction — 디스패치 없음, conversation이 처리
    "meta",                  # 단순응답·진행 신호. 예: "응 다음", "그래 그거"
    "recall",                # 되묻기 — 직전 대화를 다시 묻거나 확인 → conversation이 대화 이력에서 답. 예: "아까 일본 된다며?"
    "tool_help",             # 도구/슬롯/사용법 메타질문 → conversation이 SLOT_SPECS·APP_OVERVIEW에서 답(워커 없음). 예: "솔루션 슬롯이 뭐야?", "넌 뭐 할 수 있어?"
]


# 세그먼트가 발동시킬 워커. classify의 _ROUTE_MATRIX가 발화 유형→라우트로 변환.
Route = Literal[
    "research",  # 웹 리서치 — 외부 사실 검증
    "rag",       # 회사 문서 RAG — 내부 정합성
    "logic_validator",  # 논리 검증 — claim ↔ 사내 근거(RAG)의 논리적 지지 판정 (구 critic)
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
    in_scope: bool                       # 사업 계획과 관련 있는 발화인가. False면(무맥락 사실·잡담·무관 요청) classify가 routes=["none"]로 막고 conversation이 redirect intent로 부드럽게 되돌린다. 기본 True(애매하면 통과 — 과차단 방지).
    target_slot: str | None              # 들어갈 슬롯(있으면): "target"
    routes: list[Route]                  # 발동 워커: ["research","rag","logic_validator"]. 처리 우선순위·분기는 routes/utterance_types에서 직접 파생(별도 priority 필드 없음).


class Correction(TypedDict):
    # 정정 이벤트 한 건. 예: 5번째 턴에 솔루션을 바꿈
    slot: str             # "solution"
    previous: str | None  # "B2B 감수 서비스"
    new: str | None       # "AI 자동 검수 툴" (clear면 None)
    turn: int             # 5


class PendingConfirmation(TypedDict, total=False):
    # 슬롯 주입을 사용자에게 확인받으려 보류한 한 건(턴을 넘어 영속).
    # extract_slot_fills가 슬롯 대신 여기 쌓고, conversation이 confirm_slot으로 묻고,
    # 다음 턴 confirm_resolve가 해소한다. confirm_kind로 두 경우를 가른다:
    #   "slot"  — 값은 결정됐는데 어느 슬롯인지 애매 → 미응답 2회면 proposed로 자동 확정.
    #   "commit"— 결정 자체가 미확정(탐색) → 미응답이면 드롭(결정 안 한 건 안 채운다).
    value: str                  # 채우려던 값
    proposed_slot: str          # fill이 1순위로 고른 슬롯
    candidate_slots: list[str]  # [proposed, *alt_slots] — 사용자에게 제시할 후보(commit이면 1개)
    source_text: str            # 근거가 된 세그먼트 canonical_text(질문 문구용)
    reason: str                 # 왜 애매한지(짧게)
    attempts: int               # 재질문 횟수 — slot kind는 2회 이상 미응답이면 proposed로 자동 확정
    confirm_kind: Literal["slot", "commit"]  # 확인 종류(기본 slot, 하위호환)


class Citation(TypedDict, total=False):
    # 근거 1건의 구조화 출처 — research/RAG를 한 타입으로 표현한다. 그동안 sources:list[str]로
    # 납작하게 버려지던 제목·인용문·페이지·관련도·접근일을 보존해 계획서가 출처를 자세히 인용한다.
    # 예(research): {"cluster":"research","title":"콘진원 2024 백서","url":"https://...",
    #               "snippet":"매출 1.8조","score":0.82,"score_kind":"relevance","accessed_at":"2026-06-03"}
    # 예(rag):      {"cluster":"rag","title":"영업역량.pdf","source_file":"영업역량.pdf","page":"7",
    #               "folder":"report","snippet":"B2B 영업망 12개사","score_kind":"none"}
    cluster: Literal["research", "rag", "logic_validator"]
    title: str          # research=출처 제목 / rag=파일명
    url: str            # research만 (rag는 "")
    snippet: str        # research=근거 한 줄 / rag=highlight 또는 raw_source 발췌
    source_file: str    # rag만
    page: str           # rag만 (RagExtractorResult.source_page)
    folder: str         # rag만 (paper/report/proposal/etc)
    score: float        # research=relevance(0~1) / 없으면 0
    score_kind: Literal["relevance", "similarity_pct", "none"]  # score 해석 단위(none이면 표시 생략)
    accessed_at: str    # 수집 시점 ISO 날짜
    raw_source: str     # rag만 — 선별된 원문 청크 전체(UI '원문 보기'용). snippet은 짧은 하이라이트.
                        # planner는 안 읽으므로(계획서/프롬프트 불변) 비용은 세션 메모리뿐.


class ValidationReport(TypedDict, total=False):
    # 워커 한 번의 결과. 예: 리서치가 "게임 시장 포화" 주장을 검증
    subject: str                                                       # "게임 시장이 포화 상태다"
    findings: list[str]                                                # ["2024년 모바일 게임 신규 출시 -12%", ...]
    sources: list[str]                                                 # ["https://...", "업계 리포트 X"] — SSE/프론트 계약(유지)
    agreement: Literal["confirms", "contradicts", "partial", "unknown"]# 사용자 주장과의 일치도
    cluster: Literal["research", "rag", "logic_validator"]             # 어느 워커가 냈는지
    citations: list[Citation]                                          # 구조화 출처(부가). 없으면 빈 리스트로 본다.


class EvidenceRecord(TypedDict, total=False):
    # 세션 전체에 누적되는 근거 1건 — ValidationReport에 "어느 슬롯을 지지하는지"를 더한 형태.
    # turn_validation_reports는 매 턴 리셋되므로, 계획서가 세션 동안 모은 근거를 모두 인용하려면
    # 이렇게 영속 누적분(session_evidence)으로 따로 쌓는다(run_turn이 적재·중복제거).
    subject: str
    cluster: Literal["research", "rag", "logic_validator"]
    findings: list[str]
    agreement: Literal["confirms", "contradicts", "partial", "unknown"]
    citations: list[Citation]
    target_slot: str | None   # 이 근거가 지지하는 슬롯(없으면 끝 출처목록에만 실림)
    turn: int                 # 적재된 턴(디버그/정렬용)


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
        "pending_confirmations": [],
        "last_asked_slot": None,
        "turn_evidence": [],
        "session_evidence": [],
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
    # 이번 턴 dispatch가 낸 근거를 슬롯 연결정보와 함께 — 매 턴 리셋. run_turn이 끝에서
    # session_evidence로 합친다(turn_validation_reports는 SSE 계약 유지 위해 그대로 둔다).
    turn_evidence: list[EvidenceRecord]
    # 세션 전체에 누적된 근거 — 턴을 넘어 영속(중복 제거). 계획서(planner)가 출처를 인용하는 원천.
    session_evidence: list[EvidenceRecord]

    pending_clarifications: list[str]
    pending_question: str
    # 애매해서 주입 보류된 확인 큐 — 한 번에 하나씩 confirm_slot으로 묻는다. 턴 넘어 영속.
    pending_confirmations: list[PendingConfirmation]
    # 어시스턴트가 직전에 ask_slot으로 물은 슬롯(턴 넘어 영속). fill이 "직전 질문에 직접 답"
    # (kind=decision 기준 (b))을 결정론으로 잡는 근거 — 그 슬롯에 대한 답이면 짧은 명사구라도 결정.
    last_asked_slot: str | None
