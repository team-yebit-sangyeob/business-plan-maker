"""서술 레이어 — 항목별 설명 문단·핵심 요약을 LLM이 쓴다(call_json 1회, 배치).

하이브리드의 'LLM' 쪽. 다만 근거에 갇힌다: 입력은 슬롯 값과 그 슬롯에 연결된 finding 텍스트
뿐이고(출처 URL·번호는 주지 않는다), 모델은 그 안의 사실만으로 문단을 쓴다. 인용 번호는 코드가
붙이므로 모델이 [n]을 써도 compose 단계에서 떼고 결정론 마커로 갈아 끼운다 — 그래서 서술이
환각해도 출처 정확성은 깨지지 않는다.

배치 1회로 전 섹션을 한꺼번에 생성 — 톤 일관·비용·단일 grounding 계약을 위해.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from common.schema import ALL_SLOTS, EvidenceRecord
from common.schema.state import slot_title


class SectionNarrative(BaseModel):
    slot: str
    prose: str = ""


class NarrativeOut(BaseModel):
    summary: str = ""                                          # 핵심 요약 1문단
    sections: list[SectionNarrative] = Field(default_factory=list)  # 슬롯별 1문단


_SYSTEM = """계획서 본문 작성기
사업 계획서의 핵심 요약과 각 항목 설명 문단을 쓴다. 주어진 슬롯 값과 그 슬롯에 연결된 근거(evidence)만 근거로 삼는다.

[엄수]
- 주어진 value와 evidence 안의 사실만 쓴다. 새 사실·숫자·회사명·URL·출처를 지어내지 않는다.
- 출처 번호([1] 같은 표기)는 쓰지 않는다 — 인용 번호는 코드가 붙인다.
- value가 "[미정]"인 슬롯은 "아직 정해지지 않았다"는 취지의 한 문장만 쓰고 내용을 지어내지 않는다.
- 문체는 사업 계획서 평서체(존댓말 금지). 슬롯당 2~4문장, 핵심 요약은 4~6문장.
- 핵심 요약(summary)은 문제·타겟·목표를 중심으로 사업 전체를 압축한다.

입력 JSON의 slots 각 항목마다 sections에 {slot, prose}를 만들고, 전체를 summary로 요약한다.
JSON만 출력."""


def _findings_by_slot(records: list[EvidenceRecord]) -> dict[str, list[str]]:
    """슬롯 → 그 슬롯에 연결된 finding 텍스트 모음(중복 제거). 서술 grounding 입력."""
    out: dict[str, list[str]] = {}
    for rec in records or []:
        slot = rec.get("target_slot")
        if not slot:
            continue
        for f in (rec.get("findings") or []):
            f = (f or "").strip()
            if f and f not in out.setdefault(slot, []):
                out[slot].append(f)
    return out


async def write_narratives(
    slots: dict, records: list[EvidenceRecord], missing: list[str]
) -> NarrativeOut:
    """슬롯+근거 → 섹션별 서술. 키 없거나 호출 실패 시 호출자(compose)가 폴백 처리."""
    # call_json은 함수 안에서 import — planner를 leaf로 유지(모듈 로드 시 orchestrator 그래프
    # 전체를 끌어오지 않게). api_server는 어차피 orchestrator를 따로 import한다.
    from agents.orchestrator.llm import call_json

    fbs = _findings_by_slot(records)
    payload_slots = []
    for name in ALL_SLOTS:
        slot = slots.get(name) or {}
        value = (slot.get("value") or "").strip()
        payload_slots.append(
            {
                "slot": name,
                "title": slot_title(name),
                "value": value or "[미정]",
                "evidence": fbs.get(name, []),
            }
        )
    user = json.dumps(
        {"slots": payload_slots, "missing": missing}, ensure_ascii=False, indent=2
    )
    return await call_json(_SYSTEM, user, NarrativeOut)
