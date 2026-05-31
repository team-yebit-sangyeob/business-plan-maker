"""정정 처리 + 슬롯 채움 — LLM 기반 (기획서 Case E).

correction_node: utterance_types에 correction 포함 세그먼트만 모아 LLM에 넘김.
  현재 슬롯·correction_log·최근 messages를 보고 "어느 슬롯의 어떤 값을 어떻게"
  바꿀지 판단. action은 replace(교체)/clear(비움)/ignore(모호하면 건너뜀).

  예: 슬롯 solution="B2B 감수 서비스" 상태에서
      "아 그냥 감수 말고 AI 자동 검수 툴로 바꾸자"
      → action: replace solution = "AI 자동 검수 툴"
      → correction_log += {slot:"solution", previous:"B2B 감수 서비스",
                           new:"AI 자동 검수 툴", turn:5}
      (target·goal 등 다른 슬롯은 그대로 — 부분 롤백)

  예: 슬롯 target="네이버·카카오" 상태에서 "카카오는 빼자"
      → action: replace target = "네이버" (또는 맥락상 clear)

extract_slot_fills_node: 정정 이후 단계에서, claim/opinion 세그먼트 중
  '비어 있는' 슬롯에 명백히 들어맞는 값만 골라 채움(이미 찬 슬롯은 안 건드림).

  예: 빈 슬롯 [target, goal] + 세그먼트 "(claim) 타겟은 네이버 콘텐츠 운영팀"
      → fills: target = "네이버 콘텐츠 운영팀" (source=user). goal은 근거 없으면 그대로 빔.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from common.schema import PlanState, Correction, Slot
from common.schema.labels import SourceLabel
from common.schema.state import ALL_SLOTS
from agents.orchestrator.llm import call_json


# --- correction ------------------------------------------------------------

_CORR_SYSTEM = """오케스트레이터 정정 해소
사용자가 정정 신호를 낸 세그먼트 목록과 현재 슬롯 상태를 보고, 어떤 슬롯을 어떻게 갱신할지 결정한다.

- action="clear": 슬롯을 비움 (특정 값을 빼는 경우).
- action="replace": 슬롯 값을 new_value로 교체.
- action="ignore": 슬롯 매칭이 모호하면 건너뜀.

slot은 다음 중 하나: problem, target, solution, market, advantage, revenue, goal, resources, milestones, risks.

JSON만 출력."""


class CorrectionAction(BaseModel):
    slot: str
    action: str  # "clear" | "replace" | "ignore"
    new_value: Optional[str] = None
    previous_hint: Optional[str] = None


class CorrectionOut(BaseModel):
    actions: list[CorrectionAction] = Field(default_factory=list)


def _slot_snapshot_lines(state: PlanState) -> str:
    slots = state.get("slots") or {}
    return "\n".join(
        f"- {name}: {(slots.get(name) or {}).get('value') or '[비어있음]'}"
        for name in ALL_SLOTS
    )


async def correction_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    targets = [s for s in segments if "correction" in (s.get("utterance_types") or [])]
    if not targets:
        return {}

    slots = dict(state.get("slots") or {})
    log = list(state.get("correction_log") or [])
    turn = state.get("turn", 0)

    user_payload = (
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        "[정정 세그먼트]\n"
        + "\n".join(
            f"{i+1}. {s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(targets)
        )
    )
    out = await call_json(_CORR_SYSTEM, user_payload, CorrectionOut)

    valid = set(ALL_SLOTS)
    for action in out.actions:
        if action.slot not in valid:
            continue
        existing = slots.get(action.slot) or {}
        previous = existing.get("value")
        if action.action == "clear":
            slots[action.slot] = {
                "value": None,
                "source_label": SourceLabel.EMPTY,
                "status": "empty",
            }
            log.append(Correction(slot=action.slot, previous=previous, new=None, turn=turn))
        elif action.action == "replace":
            new_val = action.new_value or ""
            if not new_val.strip():
                continue
            slots[action.slot] = {
                "value": new_val,
                "source_label": SourceLabel.USER,
                "status": "filled",
            }
            log.append(
                Correction(slot=action.slot, previous=previous, new=new_val, turn=turn)
            )
        # ignore는 패스

    # TODO(재검증 보류): 기획서 5장 매트릭스는 정정 시 리서치·RAG '재발동'을 요구하나,
    # 현재는 슬롯 덮어쓰기만 하고 교체된 새 값에 대한 재검증은 하지 않는다(워커 stub 단계).
    # 실 워커 연결 시 replace된 슬롯의 새 value를 dispatch subject로 재투입할 것.

    # 정정으로 처리한 세그먼트에 target_slot 표시
    for seg in targets:
        if not seg.get("target_slot") and out.actions:
            seg["target_slot"] = out.actions[0].slot if out.actions[0].slot in valid else None

    return {"slots": slots, "correction_log": log, "turn_segments": segments}


# --- slot fills ------------------------------------------------------------

_FILL_SYSTEM = """오케스트레이터 슬롯 채움
사업 계획 슬롯 10개와 비어있는 항목을 보고, 사용자 세그먼트에서 채울 수 있는 값을 추출한다.

슬롯 의미 (질문 순서):
- problem: 해결하려는 문제 — 누가/어떤 상황에서/무엇 때문에/어떤 손실
- target: 타겟 고객 — 회사/부서/직책/규모
- solution: 솔루션 형태 (서비스/제품/플랫폼/툴)
- market: 시장 규모·경쟁 데이터
- advantage: 차별점·경쟁우위 — 기존 대안/경쟁사 대비 우리가 이기는 이유
- revenue: 수익 모델 (구독/건당/라이선싱 등)
- goal: 목표 수치 — 언제까지 얼마, 실패 임계값
- resources: 인력·예산 규모
- milestones: 일정 단계
- risks: 리스크

이미 채워진 슬롯은 건드리지 마라(정정 노드가 처리함). 세그먼트 내용이 슬롯에 명백히 들어맞을 때만 추출.

JSON만 출력."""


class FillItem(BaseModel):
    slot: str
    value: str


class FillOut(BaseModel):
    fills: list[FillItem] = Field(default_factory=list)


async def extract_slot_fills_node(state: PlanState) -> dict:
    segments = state.get("turn_segments") or []
    candidates = [
        s
        for s in segments
        if any(
            t in (s.get("utterance_types") or [])
            for t in ("claim", "opinion")
        )
    ]
    if not candidates:
        return {}

    slots = dict(state.get("slots") or {})
    empty_slots = [
        name for name in ALL_SLOTS if not (slots.get(name) or {}).get("value")
    ]
    if not empty_slots:
        return {}

    user_payload = (
        "[현재 슬롯]\n" + _slot_snapshot_lines(state) + "\n\n"
        f"[비어있는 슬롯]\n{', '.join(empty_slots)}\n\n"
        "[세그먼트]\n"
        + "\n".join(
            f"{i+1}. ({','.join(s.get('utterance_types') or [])}) "
            f"{s.get('canonical_text') or s.get('text','')}"
            for i, s in enumerate(candidates)
        )
    )
    out = await call_json(_FILL_SYSTEM, user_payload, FillOut)

    empty_set = set(empty_slots)
    for fill in out.fills:
        if fill.slot not in empty_set:
            continue
        value = (fill.value or "").strip()
        if not value:
            continue
        slots[fill.slot] = {
            "value": value,
            "source_label": SourceLabel.USER,
            "status": "filled",
        }
        empty_set.discard(fill.slot)
        # 세그먼트에 target_slot 표시
        for seg in candidates:
            if seg.get("target_slot") is None:
                seg["target_slot"] = fill.slot
                break

    return {"slots": slots, "turn_segments": segments}


# 하위 호환 — 기존 라우터가 import할 수 있는 동기 함수 자리(deprecated)
def collect_slot_fills(state: PlanState) -> dict:
    """deprecated — extract_slot_fills_node로 대체됨."""
    return {"slots": dict(state.get("slots") or {})}
