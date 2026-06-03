"""출처 엔진(citations) — 번호 결정론·중복제거·cluster별 렌더·마커 정리.

이 레이어가 보고서 '정확성'을 책임진다(LLM 미경유). 번호·출처가 입력에 대해 항상 같아야 한다.
"""
from agents.planner.citations import (
    build_footnotes,
    inline_markers,
    render_references,
    strip_markers,
)


def _research(subject, slot, url, *, turn=1, score=0.8):
    return {
        "subject": subject, "cluster": "research", "findings": ["f"], "agreement": "confirms",
        "target_slot": slot, "turn": turn,
        "citations": [{"cluster": "research", "title": "콘진원 2024 백서", "url": url,
                       "snippet": "매출 1.8조", "score": score, "score_kind": "relevance",
                       "accessed_at": "2026-06-03"}],
    }


def _rag(subject, slot, file, page, *, turn=1):
    return {
        "subject": subject, "cluster": "rag", "findings": ["f"], "agreement": "unknown",
        "target_slot": slot, "turn": turn,
        "citations": [{"cluster": "rag", "title": file, "source_file": file, "page": page,
                       "folder": "report", "snippet": "영업망 12개사", "score_kind": "none"}],
    }


def test_numbering_is_slot_then_cluster_order():
    # market(슬롯 순서 앞) → [1], advantage → [2].
    records = [_rag("B2B", "advantage", "영업.pdf", "7"), _research("시장", "market", "https://k/x")]
    foot = build_footnotes(records)
    assert foot.slot_to_numbers["market"] == [1]
    assert foot.slot_to_numbers["advantage"] == [2]


def test_dedup_same_url_across_records_one_number():
    records = [
        _research("시장", "market", "https://k/x", turn=1),
        _research("시장", "market", "https://k/x", turn=5),  # 같은 url → 1번호
    ]
    foot = build_footnotes(records)
    assert len(foot.numbered) == 1
    assert foot.slot_to_numbers["market"] == [1]


def test_numbering_deterministic():
    records = [_research("시장", "market", "https://k/x"), _rag("B2B", "advantage", "영업.pdf", "7")]
    assert build_footnotes(records).numbered == build_footnotes(records).numbered


def test_reference_line_formats():
    foot = build_footnotes([_research("시장", "market", "https://k/x"), _rag("B2B", "advantage", "영업.pdf", "7")])
    refs = render_references(foot)
    assert '[1] (리서치) 콘진원 2024 백서 — "매출 1.8조" (접근일 2026-06-03, 관련도 0.80) https://k/x' in refs
    assert '[2] (사내문서) 영업.pdf p.7 (report) — "영업망 12개사"' in refs
    assert "유사도" not in refs  # score_kind=none이면 유사도 표기 없음


def test_empty_records_reference_fallback():
    foot = build_footnotes([])
    assert render_references(foot) == "근거 없음 — 사용자 입력 기반."
    assert inline_markers("market", foot) == ""


def test_strip_markers_removes_and_cleans():
    assert strip_markers("성장세다 [99].") == "성장세다."
    assert strip_markers("A [1] 그리고 B [2] 끝") == "A 그리고 B 끝"
    assert strip_markers("") == ""


def test_inline_marker_cap_four():
    # 한 슬롯에 출처 6개 → 인라인은 상한 4개(끝 목록엔 전부).
    records = [_research(f"c{i}", "market", f"https://k/{i}", score=1.0 - i * 0.1) for i in range(6)]
    foot = build_footnotes(records)
    assert len(foot.slot_to_numbers["market"]) == 4
    assert len(foot.numbered) == 6
