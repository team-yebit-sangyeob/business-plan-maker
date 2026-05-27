"""세그멘테이션 — 긴 발화를 의미 단위로 분해 + 맥락 복원 (Fig.0).

이전 턴 messages와 현재 슬롯 스냅샷을 같이 LLM에 넣어서, 각 조각이
자기충족 문장(canonical_text)으로 다시 쓰이게 한다. 다운스트림 클러스터의
쿼리 분해기가 그 문장만 받아도 검색 쿼리를 만들 수 있어야 함.
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
3. target_slot_hint: 아래 9개 슬롯 중 하나 또는 null.
   problem, target, goal, solution, market, revenue, milestones, risks, resources
4. hints: 다음 중 해당하는 것만 배열로 — "correction"(아니/말고/빼자/사실은 등), "meta"(다음/그만/뽑아 등), "clarification"(모호/추상), "question"(사용자가 물어봄).

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
            "priority": 3,
        }
        # hints는 classify가 참고할 수 있도록 임시로 routes에 안 들어가는 마커로 보존
        if "correction" in item.hints:
            seg["utterance_types"] = ["correction"]
            seg["priority"] = 0
        elif "clarification" in item.hints:
            seg["utterance_types"] = ["clarification_needed"]
            seg["priority"] = 1
        elif "meta" in item.hints:
            seg["utterance_types"] = ["meta"]
            seg["priority"] = 3
        segments.append(seg)

    if not segments:
        segments.append(
            {
                "text": user_input,
                "canonical_text": user_input,
                "utterance_types": [],
                "target_slot": None,
                "routes": [],
                "priority": 3,
            }
        )
    return {"turn_segments": segments}
