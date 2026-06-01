"""애매한 슬롯 주입 확인 해소 — pending_confirmations 한 건을 사용자 답으로 정리.

extract_slot_fills가 '어느 슬롯인지 애매'하다고 본 값은 슬롯에 쓰지 않고
pending_confirmations에 쌓인다. 그러면 conversation이 confirm_slot으로
"이거 [솔루션]일까요 [차별점]일까요?"를 묻고, 다음 턴 이 노드가 사용자의 답을
해석해 확정한다. 그래프 맨 앞(START 직후)에서 돈다.

decision:
- pick    → 사용자가 후보 중 하나를 고르거나 긍정 → 그 슬롯에 value 주입, 큐에서 제거.
- reject  → 넣지 말라(아니/빼) → 큐에서 제거(슬롯은 빈 채).
- unclear → 그 질문과 무관한 다른 얘기 → attempts++; 한도(2회) 넘으면 제안 슬롯으로
            자동 확정 후 제거(무한 재질문 방지), 아니면 유지(다음 턴 재질문).

pending이 비어 있으면 no-op이라 일반 턴엔 영향이 없다. 해소 후에도 파이프라인은
계속 흐른다(같은 발화에 추가 정보가 있으면 segment 이하가 정상 처리). 방금 채운
슬롯은 더 이상 empty가 아니라 fill이 다시 건드리지 않는다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from common.schema import PlanState
from common.schema.labels import SourceLabel
from common.schema.state import slot_title
from agents.orchestrator.llm import call_json


_MAX_ATTEMPTS = 2  # 미응답 재질문 한도 — 넘으면 제안(proposed) 슬롯으로 자동 확정


_CONFIRM_SYSTEM = """오케스트레이터 확인 해소
직전 턴에 시스템이 "이 값을 어느 슬롯에 넣을지" 사용자에게 물어봤다. 이번 사용자 발화가 그 질문에 대한 답인지 보고 결정한다.

- decision="pick": 사용자가 후보 슬롯 중 하나를 고르거나 긍정("응","맞아","그걸로")했다 → slot=고른 슬롯(긍정이면 제안 슬롯).
- decision="reject": 넣지 말라거나 부정("아니","빼","둘 다 아냐")했다 → slot=null.
- decision="unclear": 그 질문과 무관한 다른 얘기를 한다 → slot=null.

slot은 반드시 [후보] 중 하나여야 한다. JSON만 출력."""


class ConfirmOut(BaseModel):
    decision: Literal["pick", "reject", "unclear"]
    slot: Optional[str] = None


async def confirm_resolve_node(state: PlanState) -> dict:
    pending = list(state.get("pending_confirmations") or [])
    if not pending:
        return {}

    item = dict(pending[0])
    rest = pending[1:]
    candidates = [c for c in (item.get("candidate_slots") or []) if c]
    proposed = item.get("proposed_slot") or (candidates[0] if candidates else None)
    if not proposed:
        return {"pending_confirmations": rest}  # 망가진 항목 — 버린다

    cand_line = ", ".join(f"{c}({slot_title(c)})" for c in candidates)
    payload = (
        f"[확인하려던 값]\n{item.get('value', '')}\n\n"
        f"[후보]\n{cand_line}\n"
        f"(제안: {proposed}({slot_title(proposed)}))\n\n"
        f"[이번 사용자 발화]\n{state.get('user_input', '')}"
    )
    out = await call_json(_CONFIRM_SYSTEM, payload, ConfirmOut)

    slots = dict(state.get("slots") or {})

    def _fill(slot: str) -> None:
        # 이미 다른 경로로 찬 슬롯이면 덮어쓰지 않는다(정정 노드 몫).
        if not (slots.get(slot) or {}).get("value"):
            slots[slot] = {
                "value": item.get("value", ""),
                "source_label": SourceLabel.USER,
                "status": "filled",
            }

    if out.decision == "pick":
        _fill(out.slot if out.slot in candidates else proposed)
        return {"slots": slots, "pending_confirmations": rest}

    if out.decision == "reject":
        return {"pending_confirmations": rest}

    # unclear — 답을 안 했다. 재질문 한도를 넘으면 제안 슬롯으로 자동 확정.
    attempts = int(item.get("attempts", 0)) + 1
    if attempts >= _MAX_ATTEMPTS:
        _fill(proposed)
        return {"slots": slots, "pending_confirmations": rest}
    item["attempts"] = attempts
    return {"pending_confirmations": [item, *rest]}
