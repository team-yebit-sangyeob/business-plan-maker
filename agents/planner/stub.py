"""계획서 작성 에이전트 — stub. 슬롯 → 마크다운, 출처 라벨 인라인 (12장 Block 1 자리)."""
from __future__ import annotations

from datetime import datetime

from common.schema import PlanState
from common.schema.labels import SourceLabel
from common.schema.state import REQUIRED_SLOTS, OPTIONAL_SLOTS


# 질문 순서(ALL_SLOTS)대로 — 렌더는 REQUIRED/OPTIONAL 두 섹션으로 묶지만 제목 매핑은 동일
_SLOT_TITLES = {
    "problem": "문제 정의",
    "target": "타겟 / 고객",
    "solution": "솔루션",
    "market": "시장 근거",
    "advantage": "차별점 / 경쟁우위",
    "revenue": "수익 모델",
    "goal": "목표 수치",
    "resources": "필요 리소스",
    "milestones": "마일스톤 / 일정",
    "risks": "리스크 / 대응",
}


def _render_slot(name: str, slot: dict) -> str:
    """슬롯 1개 → 마크다운 섹션. 출처 라벨을 인용구로 인라인(두 달 뒤 출처 구분용).

    예: ("market", {"value":"국내 웹툰 1.8조", "source_label": RESEARCH}) →
        "### 시장 근거\n\n> 리서치 결과\n\n국내 웹툰 1.8조\n"
    빈 슬롯이면 value="[미정]", 라벨="[미정]"(EMPTY).
    """
    title = _SLOT_TITLES[name]
    value = slot.get("value") or "[미정]"
    label = SourceLabel(slot.get("source_label", SourceLabel.EMPTY)).korean
    return f"### {title}\n\n> {label}\n\n{value}\n"


def compose_markdown(state: PlanState) -> str:
    """슬롯 10개 → 계획서 마크다운(필수 3 + 보강 7 두 섹션). 결정론(LLM 0).

    출력 골격:
        # 사업 계획서 (초안)
        _생성: 2026-05-29 14:30_
        ## 필수 항목
        ### 문제 정의 / > 사용자 입력 / ...
        ## 보강 항목
        ### 솔루션 / > 후보 선택 / ...   (빈 항목은 [미정])
    """
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
