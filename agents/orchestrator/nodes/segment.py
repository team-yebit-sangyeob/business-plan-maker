"""세그멘테이션 — 긴 발화를 의미 단위로 나누고 맥락을 복원한다 (Fig.0 첫 단계).

이전 턴 messages와 현재 슬롯 스냅샷을 함께 LLM에 넣어, 각 조각을 그 문장만 봐도
뜻이 통하는 문장(canonical_text)으로 다시 쓰게 한다. 후속 클러스터의 쿼리
분해기가 그 문장만 받아도 검색 쿼리를 만들 수 있어야 한다.

분류(utterance_types)는 classify_node가 전담한다.

예시
----
이전 맥락: 사용자가 앞서 "웹툰 IP 신사업"을 언급함.
이번 턴 user_input:
    "게임 시장 포화고, 일본에서 통할 거 같아. 근데 '신사업'이 좀 추상적이긴 해."
→ segments (3개):
    1. text="게임 시장 포화고"            canonical="한국 게임 시장이 포화 상태다"
    2. text="일본에서 통할 거 같아"        canonical="웹툰 IP가 일본 시장에서 통할 것이다"
    3. text="'신사업'이 추상적이긴 해"     canonical="'신사업'이라는 방향이 아직 추상적이다"
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from common.schema import PlanState, Segment
from common.schema.state import ALL_SLOTS, recent_history, slot_guide_text
from agents.orchestrator.llm import call_json


_SYSTEM = (
    """오케스트레이터 세그멘테이션
너는 사업 계획 대화에서 사용자의 한 턴 발화를 의미 단위로 나눈다. 나눈 각 조각은 앞뒤 맥락이 사라져도 그것만으로 검색 쿼리나 검증할 주장으로 쓸 수 있어야 한다.

규칙:
1. text: 원문에서 그대로 잘라낸 조각.
2. canonical_text: 앞선 턴의 주어와 대상, 전제를 되살려 그 문장만 봐도 뜻이 통하는 한국어 한 문장으로 다시 쓴다. 대명사와 지시어, 생략된 주어를 채우되, 복원하는 주어·대상은 사업 도메인(회사·제품·시장·타깃·수치 등)으로 한정한다.
   - 확정된 고유명사·수치를 먼저 되살린다: 대화에서 확정된 회사명, 제품, 시장, 지역, 숫자(예산·기간·비율)는 추상적인 표현 대신 그 확정값으로 되살린다. (예: "거기 시장"→앞서 나온 "일본 시장", "그 예산"→앞서 정한 "1억")
   - 여러 턴에 걸친 지시어를 추적한다: 가리키는 대상이 직전 턴이 아니라 몇 턴 전에 나왔어도 [최근 대화]에서 찾아 잇는다. 가장 최근에 확정된 값을 우선한다(정정이 있었으면 정정 후 값).
   - 슬롯이 비어 있어도 주어를 되살린다: [현재 슬롯]이 비어 있더라도 [최근 대화]에 주어나 대상이 나왔다면 그것으로 채운다. 슬롯과 대화가 어긋나면 더 최근 발화를 따른다.
   - 없는 맥락은 지어내지 않는다: [현재 슬롯]에도 [최근 대화]에도 단서가 없으면 원문 표현을 그대로 둔다.
   - 대화 참여자는 주어로 넣지 않는다: '너/어시스턴트/AI/사용자/우리/내가' 같은 대화 속 화자·청자는 생략돼 있어도 주어로 만들어 넣지 않는다.
   - 되묻기·확인 발화: 사용자가 직전 대화 내용을 되묻거나 확인하는 발화("아까 ~라고 했잖아?", "~라매", "방금 뭐랬지?")는 '누가 말했는가'를 지어내지 말고, 되묻는 내용(검증 가능한 주장이나 질문)만 복원한다.
   - 도구·슬롯 메타질문: 사업 내용이 아니라 이 도구나 슬롯 자체를 묻는 발화("솔루션 슬롯이 뭐야?", "슬롯이 뭔데?", "이거 어떻게 써?", "넌 뭐 할 수 있어?")는 사업 주장으로 고쳐쓰지 않는다. 해당 슬롯 값이 이미 차 있어도 그 값으로 바꾸지 말고 '그 슬롯이 무엇인가'를 묻는 원래 의미를 그대로 둔다. 이때 target_slot_hint는 null(슬롯 식별은 뒤 단계가 한다).
3. target_slot_hint: 아래 슬롯 중 하나, 또는 null. 각 슬롯의 정의와 경계를 따른다(경계가 헷갈리면 억지로 고르지 말고 null로 두면 fill이 정한다).
"""
    + slot_guide_text()
    + """

맥락 복원의 핵심: [현재 슬롯]과 [최근 대화]를 근거로 대명사와 지시어("그거", "거기", "그쪽"), 생략된 주어·대상을 채운다(단 사업 도메인에 한정 — 대화 화자 '어시스턴트/사용자'는 주어로 넣지 않는다). 한 발화에 의미 단위가 여럿이면 나누고, 하나뿐이면 1개만 낸다.
정정·취소하는 절과 거기에 이어지는 독립된 새 주장은 서로 다른 의미 단위다 — 한 발화에 같이 와도 나눈다. 취소 절은 취소 절대로, 새 주장은 그 문장만으로 검증할 수 있게 분리한다(합치면 "취소" 같은 정정 표현이 새 주장에 섞여 뒤 단계 검색·검증을 흐린다).

예시 (이전 맥락: 사용자가 "웹툰 IP 신사업", 타겟 "네이버·카카오"를 언급한 상태):
입력: "카카오는 빼고, 통할 거 같아."
출력:
  1. text="카카오는 빼고"   canonical_text="타겟에서 카카오를 뺀다"   target_slot_hint="target"
  2. text="통할 거 같아"     canonical_text="웹툰 IP가 일본 시장에서 통할 것이다"  target_slot_hint="market"

입력: "그건 취소하고, 일본 시장은 성장 중이잖아"
출력:
  1. text="그건 취소하고"       canonical_text="직전 결정을 취소한다"        target_slot_hint=null
  2. text="일본 시장은 성장 중이잖아"  canonical_text="일본 웹툰 시장이 성장 중이다"  target_slot_hint="market"

입력: "그거 시장 규모는 어떻게 돼?"
출력:
  1. text="그거 시장 규모는 어떻게 돼?"  canonical_text="웹툰 IP 신사업의 시장 규모는 어느 정도인가?"  target_slot_hint="market"

입력: "음 신사업이라기엔 좀 막연하네"
출력:
  1. text="신사업이라기엔 좀 막연하네"  canonical_text="'웹툰 IP 신사업'이라는 방향이 아직 막연하다"  target_slot_hint=null

입력: "어? 아까 일본에서 통한다고 하지 않았어?"
출력:
  1. text="어? 아까 일본에서 통한다고 하지 않았어?"  canonical_text="웹툰 IP가 일본 시장에서 통하는가?"  target_slot_hint="market"

JSON만 출력. 다른 텍스트 금지."""
)


class SegmentItem(BaseModel):
    text: str
    canonical_text: str
    target_slot_hint: Optional[str] = None


class SegmentOut(BaseModel):
    segments: list[SegmentItem]


def _slot_snapshot(state: PlanState) -> str:
    slots = state.get("slots") or {}
    lines = []
    for name in ALL_SLOTS:
        v = (slots.get(name) or {}).get("value")
        lines.append(f"- {name}: {v if v else '[비어있음]'}")
    return "\n".join(lines)


async def segment_node(state: PlanState) -> dict:
    """발화를 의미 단위로 나누고 맥락을 복원해 {"turn_segments"}를 갱신한다."""
    user_input = state.get("user_input", "")
    if not user_input.strip():
        return {"turn_segments": []}

    prompt = (
        f"[현재 슬롯]\n{_slot_snapshot(state)}\n\n"
        f"[최근 대화]\n{recent_history(state)}\n\n"
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
