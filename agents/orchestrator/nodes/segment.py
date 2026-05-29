"""세그멘테이션 — 긴 발화를 의미 단위로 분해 + 맥락 복원 (Fig.0 첫 단계).

이전 턴 messages와 현재 슬롯 스냅샷을 같이 LLM에 넣어서, 각 조각이
자기충족 문장(canonical_text)으로 다시 쓰이게 한다. 다운스트림 클러스터의
쿼리 분해기가 그 문장만 받아도 검색 쿼리를 만들 수 있어야 함.

여기서는 분류를 끝내지 않는다 — hints로 신호가 뚜렷한 4종(correction/clarification/
question/meta)만 미리 박고, 나머지 본분류(claim/opinion)는 classify_node가 한다.

worked example
--------------
이전 맥락: 사용자가 앞서 "웹툰 IP 신사업"을 언급함.
이번 턴 user_input:
    "게임 시장 포화고, 일본에서 통할 거 같아. 근데 '신사업'이 좀 추상적이긴 해."
→ segments (3개):
    1. text="게임 시장 포화고"            canonical="한국 게임 시장이 포화 상태다"           hints=[]
    2. text="일본에서 통할 거 같아"        canonical="웹툰 IP가 일본 시장에서 통할 것이다"     hints=[]
    3. text="'신사업'이 추상적이긴 해"     canonical="'신사업'이라는 방향이 아직 추상적이다"   hints=["clarification"]
  (1·2의 본분류는 classify가 claim으로 채움. 3은 여기서 clarification_needed로 박음.)
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from common.schema import PlanState, Segment
from common.schema.state import ALL_SLOTS
from agents.orchestrator.llm import call_json


_SYSTEM = """오케스트레이터 세그멘테이션
너는 사업 계획 대화의 분해기다. 사용자 한 턴 발화를 의미 단위로 쪼개되, 각 조각은 앞뒤 맥락이 사라져도 단독으로 검색 쿼리·검증 주장으로 쓸 수 있어야 한다.

규칙:
1. text: 원문에서 잘라낸 그대로의 조각.
2. canonical_text: 앞 턴의 주체·대상·전제를 복원한 자기충족 한국어 한 문장. 대명사/지시어/생략된 주어를 모두 채워 넣는다.
3. target_slot_hint: 아래 10개 슬롯 중 하나 또는 null.
   problem, target, solution, market, advantage, revenue, goal, resources, milestones, risks
4. hints: 다음 중 해당하는 것만 배열로 — "correction"(아니/말고/빼자/사실은 등), "meta"(다음/그만/뽑아 등), "clarification"(모호/추상), "question"(사용자가 물어봄).

맥락 복원 핵심: [현재 슬롯]·[최근 대화]를 근거로 대명사·지시어("그거","거기","그쪽")·생략된 주어/대상을 모두 채운다. 한 발화에 여러 의미 단위가 있으면 쪼개고, 단일하면 1개만 낸다.

예시 (이전 맥락: 사용자가 "웹툰 IP 신사업", 타겟 "네이버·카카오"를 언급한 상태):
입력: "카카오는 빼고, 통할 거 같아."
출력:
  1. text="카카오는 빼고"   canonical_text="타겟에서 카카오를 뺀다"   target_slot_hint="target"  hints=["correction"]
  2. text="통할 거 같아"     canonical_text="웹툰 IP가 일본 시장에서 통할 것이다"  target_slot_hint="market"  hints=[]

입력: "그거 시장 규모는 어떻게 돼?"
출력:
  1. text="그거 시장 규모는 어떻게 돼?"  canonical_text="웹툰 IP 신사업의 시장 규모는 어느 정도인가?"  target_slot_hint="market"  hints=["question"]

입력: "음 신사업이라기엔 좀 막연하네"
출력:
  1. text="신사업이라기엔 좀 막연하네"  canonical_text="'웹툰 IP 신사업'이라는 방향이 아직 막연하다"  target_slot_hint=null  hints=["clarification"]

JSON만 출력. 다른 텍스트 금지."""


class SegmentItem(BaseModel):
    text: str
    canonical_text: str
    target_slot_hint: Optional[str] = None
    hints: list[str] = Field(default_factory=list)


class SegmentOut(BaseModel):
    segments: list[SegmentItem]


def _slot_snapshot(state: PlanState) -> str:
    slots = state.get("slots") or {}
    lines = []
    for name in ALL_SLOTS:
        v = (slots.get(name) or {}).get("value")
        lines.append(f"- {name}: {v if v else '[비어있음]'}")
    return "\n".join(lines)


def _recent_history(state: PlanState, n: int = 6) -> str:
    messages = state.get("messages") or []
    tail = messages[-n:]
    if not tail:
        return "[이전 대화 없음]"
    return "\n".join(f"[{m['role']} t{m['turn']}] {m['content']}" for m in tail)


async def segment_node(state: PlanState) -> dict:
    user_input = state.get("user_input", "")
    if not user_input.strip():
        return {"turn_segments": []}

    prompt = (
        f"[현재 슬롯]\n{_slot_snapshot(state)}\n\n"
        f"[최근 대화]\n{_recent_history(state)}\n\n"
        f"[이번 턴 사용자 발화]\n{user_input}"
    )
    out = await call_json(_SYSTEM, prompt, SegmentOut)

    valid_slots = set(ALL_SLOTS)
    segments: list[Segment] = []
    for item in out.segments:
        slot_hint = item.target_slot_hint if item.target_slot_hint in valid_slots else None
        seg: Segment = {
            "text": item.text,
            "canonical_text": item.canonical_text or item.text,
            "utterance_types": [],
            "target_slot": slot_hint,
            "routes": [],
        }
        # hints로 신호가 뚜렷한 4종만 미리 박는다. 본분류·라우팅은 classify가 한다.
        if "correction" in item.hints:
            seg["utterance_types"] = ["correction"]
        elif "clarification" in item.hints:
            seg["utterance_types"] = ["clarification_needed"]
        elif "question" in item.hints:
            seg["utterance_types"] = ["question"]
        elif "meta" in item.hints:
            seg["utterance_types"] = ["meta"]
        segments.append(seg)

    if not segments:
        segments.append(
            {
                "text": user_input,
                "canonical_text": user_input,
                "utterance_types": [],
                "target_slot": None,
                "routes": [],
            }
        )
    return {"turn_segments": segments}
