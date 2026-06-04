"""골격 — 결정론(LLM 0). 일관된 번호형 섹션 구조로 슬롯·서술·각주를 조립한다.

LLM이 만든 섹션별 서술(narratives_by_slot)과 출처 엔진(citations)의 결과를 받아 최종 마크다운을
찍는다. 같은 입력이면 (생성 시각만 빼고) 항상 같은 출력 — 구조·번호·출처는 전부 결정론.
슬롯 제목은 SLOT_SPECS(단일 원천)의 slot_title()을 쓴다(stub의 _SLOT_TITLES 중복 제거).
"""
from __future__ import annotations

from common.schema import ALL_SLOTS, Correction, PlanState
from common.schema.state import slot_title
from agents.planner.citations import (
    Footnotes,
    inline_markers,
    render_references,
    strip_markers,
)


# 10개 슬롯을 사업계획 전개 순서대로 5개 챕터로 묶는다. (질문 순서 ALL_SLOTS와 결이 같다.)
SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("문제와 고객", ("problem", "target")),
    ("솔루션과 시장", ("solution", "market")),
    ("차별점과 수익", ("advantage", "revenue")),
    ("목표와 실행 계획", ("goal", "resources", "milestones")),
    ("리스크와 대응", ("risks",)),
)

# SECTIONS가 모든 슬롯을 정확히 한 번씩 덮는지 — 어긋나면 곧장 터뜨린다(슬롯 추가 시 누락 방지).
_covered = tuple(s for _, names in SECTIONS for s in names)
assert set(_covered) == set(ALL_SLOTS) and len(_covered) == len(ALL_SLOTS), (
    "SECTIONS와 ALL_SLOTS 슬롯 집합 불일치"
)


def _fallback_summary(slots: dict) -> str:
    """서술 LLM 폴백 시 핵심 요약 자리 — 필수 슬롯 값으로 결정론 한 줄."""
    parts = []
    for name, label in (("problem", "문제"), ("target", "타겟"), ("goal", "목표")):
        value = ((slots.get(name) or {}).get("value") or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return " / ".join(parts) if parts else "핵심 요약 생략."


def _slot_lines(
    name: str, slots: dict, narratives_by_slot: dict[str, str], foot: Footnotes
) -> list[str]:
    """슬롯 1개 → 마크다운 줄들. 값 + (근거에 갇힌)서술 + 코드가 붙인 각주 마커."""
    slot = slots.get(name) or {}
    value = ((slot.get("value") or "").strip()) or "[미정]"
    prose = strip_markers(narratives_by_slot.get(name, ""))  # LLM이 쓴 [n]은 떼고 코드 번호로 교체
    markers = inline_markers(name, foot)

    value_line = value
    body = ""
    if prose:
        body = prose + (f" {markers}" if markers else "")
    elif markers:
        value_line = f"{value} {markers}"  # 서술이 없으면 값 줄 끝에 마커

    lines = [f"### {slot_title(name)}", "", value_line]
    if body:
        lines += ["", body]
    return lines


def _render_corrections(log: list[Correction]) -> str:
    lines = []
    for c in log:
        prev = (c.get("previous") or "").strip() or "(비어있음)"
        new = (c.get("new") or "").strip() or "(비움)"
        lines.append(f'- t{c.get("turn", "?")} {slot_title(c.get("slot", ""))}: "{prev}" → "{new}"')
    return "\n".join(lines)


def render(
    *,
    slots: dict,
    narratives_by_slot: dict[str, str],
    summary: str,
    foot: Footnotes,
    correction_log: list[Correction],
    early: bool,
    generated_at: str,
) -> str:
    """전체 계획서 마크다운(결정론). 선택 슬롯이 비어 조기 출력이면 버전에 '(조기 출력)' 표기."""
    version = "v1 (조기 출력)" if early else "v1"

    out: list[str] = [
        "# 사업 계획서",
        "",
        f"_생성: {generated_at}_ · _버전: {version}_",
        "",
        "## 핵심 요약",
        "",
        strip_markers(summary) or _fallback_summary(slots),
        "",
    ]
    for idx, (sec_title, names) in enumerate(SECTIONS, start=1):
        out += [f"## {idx}. {sec_title}", ""]
        for name in names:
            out += _slot_lines(name, slots, narratives_by_slot, foot) + [""]

    out += ["## 근거 및 출처", "", render_references(foot), ""]

    if correction_log:
        out += ["## 정정 이력", "", _render_corrections(correction_log)]

    return "\n".join(out).rstrip() + "\n"
