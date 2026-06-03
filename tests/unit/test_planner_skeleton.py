"""골격(skeleton) — 일관 구조·type2 [미정]·빈 근거·정정 이력·슬롯 커버리지."""
from common.schema import ALL_SLOTS
from common.schema.state import slot_title
from agents.planner.citations import build_footnotes
from agents.planner.skeleton import SECTIONS, render


def _slots(filled: dict):
    out = {n: {"value": None, "source_label": "empty", "status": "empty"} for n in ALL_SLOTS}
    for n, v in filled.items():
        out[n] = {"value": v, "source_label": "user", "status": "filled"}
    return out


def _render(slots, *, records=None, output_request="type1", corrections=None):
    foot = build_footnotes(records or [])
    return render(
        slots=slots, narratives_by_slot={}, summary="", foot=foot,
        correction_log=corrections or [], output_request=output_request,
        generated_at="2026-06-03 14:30",
    )


def test_sections_cover_all_slots_exactly_once():
    covered = [s for _, names in SECTIONS for s in names]
    assert sorted(covered) == sorted(ALL_SLOTS)
    assert len(covered) == len(ALL_SLOTS)


def test_structure_has_all_slot_titles_and_numbered_sections():
    md = _render(_slots({n: f"값-{n}" for n in ALL_SLOTS}))
    assert md.startswith("# 사업 계획서")
    assert "## 핵심 요약" in md
    for idx in range(1, len(SECTIONS) + 1):
        assert f"## {idx}. " in md
    for n in ALL_SLOTS:
        assert f"### {slot_title(n)}" in md
    assert "## 근거 및 출처" in md


def test_type2_marks_early_and_shows_undecided():
    md = _render(_slots({"problem": "p", "target": "t", "goal": "g"}), output_request="type2")
    assert "(조기 출력)" in md
    assert "[미정]" in md  # 빈 선택 슬롯


def test_type1_full_no_early_marker():
    md = _render(_slots({n: f"v{n}" for n in ALL_SLOTS}), output_request="type1")
    assert "(조기 출력)" not in md
    assert "_버전: v1_" in md


def test_empty_evidence_reference_fallback_no_crash():
    md = _render(_slots({"problem": "p", "target": "t", "goal": "g"}), records=[])
    assert "근거 없음 — 사용자 입력 기반." in md


def test_correction_history_only_when_present():
    md_none = _render(_slots({"problem": "p", "target": "t", "goal": "g"}))
    assert "## 정정 이력" not in md_none
    md_some = _render(
        _slots({"problem": "p", "target": "t", "goal": "g"}),
        corrections=[{"slot": "solution", "previous": "외주", "new": "B2B", "turn": 4}],
    )
    assert "## 정정 이력" in md_some
    assert '- t4 솔루션: "외주" → "B2B"' in md_some
