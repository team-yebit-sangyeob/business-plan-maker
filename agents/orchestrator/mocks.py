"""빌트인 데모 mock 핸들러 (BPM_LLM_MODE=mock).

키 없이도 그럴듯한 흐름을 볼 수 있게, 각 LLM 프롬프트(첫 줄을 key로)에 대해
가벼운 휴리스틱 응답을 만든다. **데모/오프라인 전용** — 실제 판단은 live 모드.

핸들러는 `(system, user) -> dict` 시그니처. 반환 dict는 호출부의 pydantic
스키마로 검증된다. 테스트가 `register_mock`으로 같은 key를 등록하면 그쪽이 우선.
"""
from __future__ import annotations

import json
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


# 데모용 애매 케이스 — 경계 키워드면 confidence=ambiguous로 내보내 확인 흐름을 시연.
_AMBIG_DEMO = [
    ("검수", "solution", ["advantage"], "솔루션 형태이자 차별점으로 읽힘"),
    ("구독", "revenue", ["solution"], "수익 방식이자 솔루션 형태로 읽힘"),
]


def _slot_fill(system: str, user: str) -> dict:
    empty_block = _section(user, "비어있는 슬롯")
    empty_slots = [s.strip() for s in empty_block.replace("\n", ",").split(",") if s.strip()]
    segs = _numbered_lines(_section(user, "세그먼트"))
    if not empty_slots or not segs:
        return {"fills": []}
    # "(labels) [힌트:슬롯] text" → text (앞쪽 (...)·[...] 접두를 모두 제거)
    first = re.sub(r"^((\([^)]*\)|\[[^\]]*\])\s*)+", "", segs[0]).strip()
    if not first:
        return {"fills": []}
    for kw, slot, alts, reason in _AMBIG_DEMO:
        if kw in first:
            return {
                "fills": [
                    {
                        "slot": slot,
                        "value": first,
                        "confidence": "ambiguous",
                        "alt_slots": alts,
                        "reason": reason,
                    }
                ]
            }
    return {"fills": [{"slot": empty_slots[0], "value": first}]}


def _confirm_resolve(system: str, user: str) -> dict:
    """확인 해소 데모 — 발화에 후보 슬롯(키/제목)이 있으면 pick, 정정 키워드면 reject."""
    utterance = _section(user, "이번 사용자 발화") or ""
    pairs = re.findall(r"([a-z_]+)\(([^)]+)\)", _section(user, "후보"))
    if any(k in utterance for k in _CORRECTION_KW):
        return {"decision": "reject", "slot": None}
    for slot, title in pairs:
        if slot in utterance or title in utterance:
            return {"decision": "pick", "slot": slot}
    if any(k in utterance for k in ("응", "맞아", "그래", "그걸로", "네", "좋아")):
        return {"decision": "pick", "slot": pairs[0][0] if pairs else None}
    return {"decision": "unclear", "slot": None}


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


def _rag(system: str, user: str) -> dict:
    """RAG 시뮬레이션 데모 응답 — 사내 자료에서 나왔을 법한 근거를 결정론으로 생성."""
    subject = _section(user, "발화 주제") or user.strip()
    head = subject[:36]
    return {
        "findings": [
            f"사내 자료 기준 '{head}' 관련 전담 조직·예산 배정 근거는 제한적입니다(추정).",
            "2026 로드맵은 기존 B2C 라이브 운영 강화에 무게 — 신규 방향과 부분적으로 어긋남.",
            "유사 신사업 선례상 초기 6개월 인력 4~5명 규모가 일반적.",
        ],
        "sources": ["company_roadmap_2026.md", "org_chart.pdf", "newbiz_precedent.xlsx"],
        "agreement": "partial",
    }


def _critic(system: str, user: str) -> dict:
    """비평 시뮬레이션 데모 응답 — 추론 비약 + (근거 있으면) 정합성 충돌을 결정론으로."""
    utterance = _section(user, "발화") or user.strip()
    head = utterance[:34]
    has_research = "[리서치 근거]" in user
    has_rag = "[회사(RAG) 근거]" in user
    findings = [
        f"전제에서 결론('{head}…')으로의 비약 가능성 — 핵심 변수 1개 이상이 명시되지 않았습니다.",
    ]
    if has_research:
        findings.append("리서치 근거와 사용자 전제 사이에 부분 충돌 — 수용도·범위 조건 재확인 권장.")
    if has_rag:
        findings.append("회사 로드맵·조직 현황과 방향이 일부 어긋남 — 분기/보류 옵션 검토 권장.")
    return {
        "findings": findings,
        "sources": [],
        "agreement": "partial" if (has_research or has_rag) else "unknown",
    }


def _first(items, default: str = "") -> str:
    for it in items or []:
        if str(it).strip():
            return str(it).strip()
    return default


def _render_intent(intent: dict) -> str:
    """단일 intent → 한 문장(데모용 결정론 렌더)."""
    t = intent.get("type")
    if t == "acknowledge":
        slot = intent.get("slot") or "그 항목"
        new = intent.get("new")
        return f"{slot}은(는) '{new}'로 반영했어요." if new else f"{slot}은(는) 비워뒀어요."
    if t == "redirect":
        return "그건 지금 짜는 사업 계획과는 좀 떨어진 얘기라 그쪽은 넘어갈게요."
    if t == "report_findings":
        fact = _first(intent.get("research")) or _first(intent.get("rag"))
        crit = _first(intent.get("critic"))
        parts = []
        if fact:
            parts.append(f"찾아보니 {fact}")
        if crit:
            parts.append(f"다만 {crit}")
        return " ".join(parts) or "관련 근거를 살펴봤어요."
    if t == "answer_question":
        ans = _first(intent.get("research")) or _first(intent.get("rag"))
        return f"확인해보니 {ans}" if ans else "관련해서 찾아보고 있어요."
    if t == "clarify":
        text = intent.get("text") or "그 부분"
        return f"먼저 '{text}' 이 부분만 조금 더 풀어줄래요?"
    if t == "reject_output":
        missing = ", ".join(intent.get("missing_required") or []) or "필수 항목"
        return f"지금 뽑기엔 {missing} 쪽이 비어 있어요. 그것만 채우면 바로 출력 가능해요."
    if t == "deliver_plan":
        if intent.get("output_type") == "type2":
            return "필수는 다 찼으니 지금도 뽑을 수 있어요(빈 항목은 [미정]으로 들어가요)."
        return "필요한 항목이 다 찼어요. '계획서 생성'으로 뽑아볼까요?"
    if t == "confirm_slot":
        cands = " / ".join(c.get("title", "") for c in (intent.get("candidates") or []))
        val = intent.get("value") or "그 내용"
        return f"방금 '{val}' 말씀하신 거, {cands or '어느 슬롯'} 중 어디에 넣을까요?"
    if t == "ask_slot":
        return intent.get("example") or _SLOT_Q.get(
            intent.get("slot", ""), "다음으로 어떤 항목을 채워볼까요?"
        )
    return ""


def _conversation(system: str, user: str) -> dict:
    try:
        data = json.loads(user)
        intents = data.get("intents") or []
    except (json.JSONDecodeError, AttributeError):
        intents = []
    parts = [p for p in (_render_intent(i) for i in intents) if p]
    return {"message": "\n".join(parts) or "다음으로 어떤 항목을 채워볼까요?"}


# key = system prompt 첫 줄
DEFAULT_MOCK_HANDLERS: dict[str, Callable[[str, str], dict]] = {
    "오케스트레이터 세그멘테이션": _segment,
    "오케스트레이터 다중라벨 분류": _classify,
    "오케스트레이터 정정 해소": _correction,
    "오케스트레이터 슬롯 채움": _slot_fill,
    "오케스트레이터 확인 해소": _confirm_resolve,
    "오케스트레이터 출력 의도 판정": _intent,
    "대화 에이전트": _conversation,
    "RAG 시뮬레이션": _rag,
    "비평 시뮬레이션": _critic,
}
