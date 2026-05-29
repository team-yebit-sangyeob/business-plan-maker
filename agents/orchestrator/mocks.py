"""빌트인 데모 mock 핸들러 (BPM_LLM_MODE=mock).

키 없이도 그럴듯한 흐름을 볼 수 있게, 각 LLM 프롬프트(첫 줄을 key로)에 대해
가벼운 휴리스틱 응답을 만든다. **데모/오프라인 전용** — 실제 판단은 live 모드.

핸들러는 `(system, user) -> dict` 시그니처. 반환 dict는 호출부의 pydantic
스키마로 검증된다. 테스트가 `register_mock`으로 같은 key를 등록하면 그쪽이 우선.
"""
from __future__ import annotations

import re
from typing import Callable


# --- 공통 파서 -------------------------------------------------------------

def _section(payload: str, header: str) -> str:
    """'[header]' 다음부터 다음 '[' 섹션 전까지의 본문을 반환."""
    lines = payload.splitlines()
    out: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if stripped == f"[{header}]":
            capturing = True
            continue
        if capturing and stripped.startswith("[") and stripped.endswith("]"):
            break
        if capturing:
            out.append(line)
    return "\n".join(out).strip()


def _numbered_lines(payload: str) -> list[str]:
    """'1. ...', '2. ...' 형태의 라인들을 본문만 떼어 리스트로."""
    items: list[str] = []
    for line in payload.splitlines():
        m = re.match(r"\s*\d+\.\s*(.+)", line)
        if m:
            items.append(m.group(1).strip())
    return items


# --- 키워드 휴리스틱 -------------------------------------------------------

_CORRECTION_KW = ("말고", "빼자", "아니", "사실은", "다시 생각", "바꾸자", "취소")
_META_KW = ("뽑아", "출력", "생성", "그만", "다음", "정리해", "만들어")
_CLARIFY_KW = ("추상", "모호", "잘 모르", "애매", "어떻게 해야")
_QUESTION_KW = ("어떻게 돼", "얼마", "몇 ", "어디", "뭐야", "어때", "있어?", "되나")


def _is_question(text: str) -> bool:
    return text.strip().endswith("?") or any(k in text for k in _QUESTION_KW)


# claim 휴리스틱 — 사실·가설·결정·제약을 모두 claim 단일 라벨로 본다.
_FACT_KW = ("포화", "시장", "추세", "통계", "점유", "규모", "성장 중")
_HYPOTHESIS_KW = ("통할", "될 거", "거 같", "가능성", "예상", "듯")
_DECISION_KW = ("정했", "하자", "가자", "결정", "타겟은", "으로 간다")
_CONSTRAINT_KW = ("예산", "개월", "억", "만원", "%", "인력", "명까지")
_CLAIM_KW = _FACT_KW + _HYPOTHESIS_KW + _DECISION_KW + _CONSTRAINT_KW


# 위 _FACT_KW/_HYPOTHESIS_KW/_DECISION_KW는 claim 라벨 판정용 키워드 묶음일 뿐 —
# 주장의 세부 분류(어떻게 검증할지)는 리서치 쿼리 분해기 몫이라 오케는 따지지 않는다.


def _utterance_label(text: str) -> str:
    if any(k in text for k in _META_KW):
        return "meta"
    if any(k in text for k in _CORRECTION_KW):
        return "correction"
    if _is_question(text):
        return "question"
    if any(k in text for k in _CLAIM_KW):
        return "claim"
    return "opinion"


# 스코프 휴리스틱(데모 전용) — 계획과 무관한 무맥락/잡담만 false. 애매하면 true.
_OFFTOPIC_KW = ("날씨", "몇 시", "점심 뭐", "운세", "로또", "번역해줘", "데코레이터")
_ARITH_RE = re.compile(r"\d+\s*[+\-*/×÷]\s*\d+")


def _in_scope(text: str) -> bool:
    if _ARITH_RE.search(text):
        return False
    if any(k in text for k in _OFFTOPIC_KW):
        return False
    return True


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    segs = [p.strip(" .") for p in parts if p.strip(" .")]
    return segs[:4] if segs else ([text.strip()] if text.strip() else [])


# --- 핸들러 ----------------------------------------------------------------

def _segment(system: str, user: str) -> dict:
    utterance = _section(user, "이번 턴 사용자 발화") or user.strip()
    segs = _split_sentences(utterance)
    items = []
    for s in segs:
        hints: list[str] = []
        if any(k in s for k in _CORRECTION_KW):
            hints = ["correction"]
        elif any(k in s for k in _CLARIFY_KW):
            hints = ["clarification"]
        elif _is_question(s):
            hints = ["question"]
        elif any(k in s for k in _META_KW):
            hints = ["meta"]
        items.append(
            {
                "text": s,
                "canonical_text": s,
                "target_slot_hint": None,
                "hints": hints,
            }
        )
    return {"segments": items}


def _classify(system: str, user: str) -> dict:
    lines = _numbered_lines(user)
    items = []
    for t in lines:
        label = _utterance_label(t)
        items.append({"canonical_text": t, "utterance_types": [label], "in_scope": _in_scope(t)})
    return {"items": items}


def _correction(system: str, user: str) -> dict:
    # 데모에서는 슬롯 자동 매칭이 위험 — 비워두고 흐름만 유지.
    return {"actions": []}


def _slot_fill(system: str, user: str) -> dict:
    empty_block = _section(user, "비어있는 슬롯")
    empty_slots = [s.strip() for s in empty_block.replace("\n", ",").split(",") if s.strip()]
    segs = _numbered_lines(_section(user, "세그먼트"))
    if not empty_slots or not segs:
        return {"fills": []}
    first = re.sub(r"^\([^)]*\)\s*", "", segs[0]).strip()  # "(labels) text" → text
    if not first:
        return {"fills": []}
    return {"fills": [{"slot": empty_slots[0], "value": first}]}


def _intent(system: str, user: str) -> dict:
    wants_output = any(k in user for k in ("뽑아", "출력", "생성", "만들어", "정리해", "그만"))
    return {"wants_output": wants_output}


_SLOT_Q = {
    "problem": "어떤 문제를 풀려는 거예요? 누가·어떤 상황에서·뭐 때문에 손해 보는지 한 번 풀어주세요.",
    "target": "그 돈을 낼 사람이 구체적으로 누구예요? 회사 안에서 도장 찍는 사람·규모까지요.",
    "goal": "목표 수치는요? 언제까지 얼마, 그리고 안 되면 접는 기준까지 같이 잡아볼까요?",
    "solution": "솔루션은 서비스·제품·플랫폼 중 어떤 형태로 가져갈 생각이에요?",
    "advantage": "기존 대안·경쟁사 대비 우리만의 차별점은 뭐예요?",
    "market": "시장 규모나 경쟁사 데이터는 짚어둔 게 있어요? 없으면 제가 찾아볼게요.",
    "revenue": "수익 모델은 구독·건당·라이선싱 중 어느 쪽 그림이에요?",
    "milestones": "마일스톤 — 언제까지 어느 단계까지 가야 한다고 보세요?",
    "risks": "가장 걱정되는 리스크부터 하나 짚어주실래요?",
    "resources": "필요한 인력·예산 규모는 어느 정도로 보세요?",
}


def _conversation(system: str, user: str) -> dict:
    mode_m = re.search(r"\[모드\]\s*(\S+)", user)
    slot_m = re.search(r"\[채울 슬롯\]\s*(\S+)", user)
    mode = mode_m.group(1) if mode_m else ""
    slot = slot_m.group(1) if slot_m else ""

    if mode == "type0_reject":
        missing = _section(user, "필수 부족") or "필수 항목"
        # "[필수 부족] a, b" 한 줄짜리라 section이 비면 정규식으로 보강
        if not missing or missing == "없음":
            mm = re.search(r"\[필수 부족\]\s*(.+)", user)
            missing = mm.group(1).strip() if mm else "필수 항목"
        return {
            "question": (
                f"지금은 {missing} 쪽이 비어 있어서 계획서로 뽑기엔 일러요. "
                "이것만 채우면 바로 출력할 수 있어요."
            )
        }

    return {"question": _SLOT_Q.get(slot, "다음으로 어떤 항목을 채워볼까요?")}


# key = system prompt 첫 줄
DEFAULT_MOCK_HANDLERS: dict[str, Callable[[str, str], dict]] = {
    "오케스트레이터 세그멘테이션": _segment,
    "오케스트레이터 다중라벨 분류": _classify,
    "오케스트레이터 정정 해소": _correction,
    "오케스트레이터 슬롯 채움": _slot_fill,
    "오케스트레이터 출력 의도 판정": _intent,
    "대화 에이전트": _conversation,
}
