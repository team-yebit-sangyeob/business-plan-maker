"""애매한 슬롯 주입 확인 해소 — pending_confirmations 한 건을 사용자 답으로 정리.

extract_slot_fills가 '어느 슬롯인지 애매'하다고 본 값은 슬롯에 쓰지 않고
pending_confirmations에 쌓인다. 그러면 conversation이 confirm_slot으로
"이거 [솔루션]일까요 [차별점]일까요?"를 묻고, 다음 턴 이 노드가 사용자의 답을
해석해 확정한다. 그래프 맨 앞(START 직후)에서 돈다.

decision(제안에 대한 답의 태도):
- accept   → 제안대로 넣으라는 긍정·명령("이대로 넣어"/"넣어"/"응") → 제안 슬롯 주입(replace면 덮어쓰기).
- pick     → 후보 둘 이상에서 하나 선택("솔루션에 넣어") → 그 슬롯 주입.
- reject   → 넣지 말라(아니/빼) → 큐에서 제거(슬롯 그대로).
- revise   → 값을 고쳐서 넣으라("앞부분만 빼고") → 스테일 값 버리고 파이프라인이 고친 값 재추출(주입 안 함).
- unrelated→ 그 질문과 무관한 다른 얘기 → confirm_kind로 가른다:
            · "commit"(결정 미확정)·"replace"(교체 확인) → 드롭(결정/교체 안 한 건 안 건드린다).
            · "slot"(값은 결정, 칸만 모름) → attempts++; 한도(2회) 넘으면 제안 슬롯으로 자동 확정.

순수 확인 답(accept/pick/reject + 추가 내용 없음)이면 confirmation_consumed=True를 내보내
graph가 segment 이하를 건너뛴다 — 열린 제안에 대한 답을 새 리서치 주장으로 재처리하지 않는다.
추가 내용이 섞였거나(has_additional_content) revise/unrelated면 파이프라인이 계속 흐른다(그 내용을
segment 이하가 정상 처리). pending이 비어 있으면 no-op이라 일반 턴엔 영향이 없다.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from common.schema import PlanState, Correction
from common.schema.labels import SourceLabel
from common.schema.state import slot_title
from agents.orchestrator.llm import call_json


_MAX_ATTEMPTS = 2  # 미응답 재질문 한도 — 넘으면 제안(proposed) 슬롯으로 자동 확정


_CONFIRM_SYSTEM = """오케스트레이터 확인 해소
직전 턴에 시스템이 "이 값을 슬롯에 넣을지 / 어느 슬롯에 넣을지 / 기존 값과 바꿀지" 사용자에게 물어봤다. 이번 사용자 발화가 그 질문에 대한 답인지, 답이면 어떤 태도인지 본다. 표현이 명령형이어도("그대로 넣어", "넣어", "넣으라고") 그 제안에 대한 긍정이면 accept다 — 새 리서치 주문이 아니다. 아래 보기는 태도를 가르는 예시지 정답 키워드가 아니다. 발화의 의미로 판단한다.

- decision="accept": 제안대로 넣으라고 긍정·명령한다("응", "맞아", "그걸로", "이대로 넣어", "그대로 넣어줘", "넣어", "넣자") → slot=제안 슬롯.
- decision="pick": 후보가 둘 이상일 때 그중 하나를 고른다("솔루션에 넣어", "차별점 쪽") → slot=고른 슬롯.
- decision="reject": 넣지 말라거나 부정한다("아니", "빼", "됐어", "넣지마", "둘 다 아냐") → slot=null.
- decision="revise": 넣되 값을 고쳐서 넣으라고 한다("앞부분만 빼고", "좀 줄여서 넣어") → slot=null(고친 값은 뒤 단계가 다시 뽑는다).
- decision="unrelated": 그 질문에 답하지 않고 다른 얘기를 한다("타겟은 20대야", "시장 규모 어때?") → slot=null.

has_additional_content: 이 발화에 위 확인 답 '말고' 새로 채울 내용이나 질문이 더 섞여 있으면 true(예: "응 넣고, 타겟은 20대야"). 순수 확인 답뿐이면 false.

slot은 반드시 [후보] 중 하나여야 한다. JSON만 출력."""


class ConfirmOut(BaseModel):
    decision: Literal["accept", "pick", "reject", "revise", "unrelated"]
    slot: Optional[str] = None
    has_additional_content: bool = False


async def confirm_resolve_node(state: PlanState) -> dict:
    """보류된 확인 1건을 사용자 답으로 해소한다 → {"slots"·"correction_log"(주입/교체 시),
    "pending_confirmations", "confirmation_consumed"(순수 확인 답일 때)}(pending 없으면 {})."""
    pending = list(state.get("pending_confirmations") or [])
    if not pending:
        return {}

    item = dict(pending[0])
    rest = pending[1:]
    candidates = [c for c in (item.get("candidate_slots") or []) if c]
    proposed = item.get("proposed_slot") or (candidates[0] if candidates else None)
    if not proposed:
        return {"pending_confirmations": rest}  # 망가진 항목 — 버린다

    kind = item.get("confirm_kind", "slot")
    cand_line = ", ".join(f"{c}({slot_title(c)})" for c in candidates)
    asked = {
        "replace": f"기존 {proposed}({slot_title(proposed)}) 값을 이 값으로 바꿀지",
        "commit": f"이 값을 {proposed}({slot_title(proposed)}) 슬롯에 넣을지",
    }.get(kind, f"이 값을 어느 슬롯({cand_line})에 넣을지")
    payload = (
        f"[확인하려던 값]\n{item.get('value', '')}\n\n"
        + (f"[기존 값]\n{item.get('previous_value', '')}\n\n" if kind == "replace" else "")
        + f"[후보]\n{cand_line}\n(제안: {proposed}({slot_title(proposed)}))\n\n"
        f"[직전에 물은 것]\n{asked}\n\n"
        f"[이번 사용자 발화]\n{state.get('user_input', '')}"
    )
    out = await call_json(_CONFIRM_SYSTEM, payload, ConfirmOut)

    slots = dict(state.get("slots") or {})
    log = list(state.get("correction_log") or [])
    turn = state.get("turn", 0)

    def _fill(slot: str, *, force: bool = False) -> None:
        existing = (slots.get(slot) or {}).get("value")
        # 이미 다른 경로로 찬 슬롯은 덮지 않는다(정정 노드 몫). replace 확인만 force로 덮어쓴다.
        if existing and not force:
            return
        if force and existing:
            log.append(
                Correction(slot=slot, previous=existing, new=item.get("value", ""), turn=turn)
            )
        slots[slot] = {
            "value": item.get("value", ""),
            "source_label": SourceLabel.USER,
            "status": "filled",
        }

    # 순수 확인 답(추가 내용 없음)이면 소비 — graph가 segment 이하를 건너뛴다.
    consumed = not bool(out.has_additional_content)

    if out.decision in ("accept", "pick"):
        target = out.slot if (out.decision == "pick" and out.slot in candidates) else proposed
        _fill(target, force=(kind == "replace"))
        return {
            "slots": slots,
            "correction_log": log,
            "pending_confirmations": rest,
            "confirmation_consumed": consumed,
        }

    if out.decision == "reject":
        return {"pending_confirmations": rest, "confirmation_consumed": consumed}

    if out.decision == "revise":
        # 값을 고쳐 넣으라는 답 — 스테일 값은 버리고, 고친 내용은 파이프라인이 다시 뽑는다(fall through).
        return {"pending_confirmations": rest}

    # unrelated — 그 질문에 답하지 않고 다른 얘기를 했다(파이프라인이 그 내용을 처리, fall through).
    # commit(결정 미확정)·replace(교체 확인)는 미응답이면 채우지/바꾸지 않고 드롭한다.
    if kind in ("commit", "replace"):
        return {"pending_confirmations": rest}
    # slot(값은 결정, 칸만 모름) → 재질문 한도를 넘으면 제안 슬롯으로 자동 확정.
    attempts = int(item.get("attempts", 0)) + 1
    if attempts >= _MAX_ATTEMPTS:
        _fill(proposed)
        return {"slots": slots, "correction_log": log, "pending_confirmations": rest}
    item["attempts"] = attempts
    return {"pending_confirmations": [item, *rest]}
