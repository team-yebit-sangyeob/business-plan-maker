"""계획서 작성 에이전트 — stub. 슬롯 → 마크다운, 출처 라벨 인라인 (12장 Block 1 자리)."""
from __future__ import annotations

from datetime import datetime

from common.schema import PlanState
from common.schema.labels import SourceLabel
from common.schema.state import REQUIRED_SLOTS, OPTIONAL_SLOTS


_SLOT_TITLES = {
    "problem": "문제 정의",
    "target": "타겟 / 고객",
    "goal": "목표 수치",
    "solution": "솔루션",
    "market": "시장 근거",
    "revenue": "수익 모델",
    "milestones": "마일스톤 / 일정",
    "risks": "리스크 / 대응",
    "resources": "필요 리소스",
}


def _render_slot(name: str, slot: dict) -> str:
    title = _SLOT_TITLES[name]
    value = slot.get("value") or "[미정]"
    label = SourceLabel(slot.get("source_label", SourceLabel.EMPTY)).korean
    return f"### {title}\n\n> {label}\n\n{value}\n"


def compose_markdown(state: PlanState) -> str:
    slots = state.get("slots") or {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# 사업 계획서 (초안)",
        "",
        f"_생성: {timestamp}_",
        "",
        "## 필수 항목",
        "",
    ]
    for name in REQUIRED_SLOTS:
        lines.append(_render_slot(name, slots.get(name, {})))
    lines.extend(["", "## 보강 항목", ""])
    for name in OPTIONAL_SLOTS:
        lines.append(_render_slot(name, slots.get(name, {})))

    return "\n".join(lines)
