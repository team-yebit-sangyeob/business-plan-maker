"""출처 엔진 — 결정론(LLM 0). 누적 근거(EvidenceRecord)에서 각주 번호를 매기고 렌더한다.

이 모듈은 보고서의 '정확성'을 책임진다. LLM은 본문 서술만 쓰고, 번호·URL·페이지 같은
출처 정보는 전부 여기서 코드가 EvidenceRecord.citations 값 그대로 찍는다 — 그래서 모델이
환각해도 출처는 위·변조되지 않는다(하이브리드 보장).

번호 부여는 결정론:
  슬롯 순서(ALL_SLOTS) → 슬롯 내 cluster 순(research, rag, logic_validator) → 턴 → 등장 순.
동일 출처(키: url 있으면 url, 없으면 (파일,페이지,스니펫[:60]))는 한 번호로 합친다(중복 제거).
같은 번호가 여러 슬롯 아래 인라인으로 나타날 수 있다(한 출처가 여러 주장을 지지하는 경우).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from common.schema import ALL_SLOTS, Citation, EvidenceRecord


# 슬롯 내에서 cluster 표시 우선순위(작을수록 먼저 번호 받음).
_CLUSTER_ORDER = {"research": 0, "rag": 1, "logic_validator": 2}
# 한 슬롯 본문에 다는 인라인 마커 상한(전부는 끝 출처목록에 실린다).
_MAX_INLINE = 4
_MARKER_RE = re.compile(r"\[\d+\]")


@dataclass
class Footnotes:
    slot_to_numbers: dict[str, list[int]] = field(default_factory=dict)  # 슬롯 → 인라인 번호(상한 4)
    numbered: list[tuple[int, Citation]] = field(default_factory=list)   # 끝 목록(전체, 번호 오름차순)
    unlinked_numbers: list[int] = field(default_factory=list)            # 슬롯 미연결 근거 번호


def _dedup_key(c: Citation) -> tuple:
    """동일 출처 판정 키. url이 있으면 url 하나로, 없으면 사내문서 식별자 묶음."""
    url = (c.get("url") or "").strip()
    if url:
        return ("url", url)
    return (
        "doc",
        (c.get("source_file") or "").strip(),
        (c.get("page") or "").strip(),
        (c.get("snippet") or "").strip()[:60],
    )


def _score(c: Citation) -> float:
    return float(c.get("score") or 0.0)


def _slot_rank(slot: str | None) -> int:
    """ALL_SLOTS 순서. 슬롯 없으면(미연결) 맨 뒤로."""
    try:
        return ALL_SLOTS.index(slot)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return len(ALL_SLOTS)


def build_footnotes(records: list[EvidenceRecord]) -> Footnotes:
    """누적 근거 → 각주 번호 맵. 같은 입력이면 항상 같은 번호(결정론)."""
    # 안정 정렬: (슬롯순, cluster순, 턴, 원래 등장순)
    ordered = sorted(
        enumerate(records or []),
        key=lambda it: (
            _slot_rank(it[1].get("target_slot")),
            _CLUSTER_ORDER.get(it[1].get("cluster", ""), 9),
            it[1].get("turn", 0),
            it[0],
        ),
    )

    num_by_key: dict[tuple, int] = {}
    cite_by_num: dict[int, Citation] = {}
    slots_by_num: dict[int, set[str]] = {}
    next_num = 1
    for _, rec in ordered:
        slot = rec.get("target_slot")
        for c in (rec.get("citations") or []):
            key = _dedup_key(c)
            n = num_by_key.get(key)
            if n is None:
                n = next_num
                num_by_key[key] = n
                cite_by_num[n] = c
                slots_by_num[n] = set()
                next_num += 1
            if slot:
                slots_by_num[n].add(slot)

    # 슬롯 → 인라인 번호: 점수 desc로 상한 4개 추린 뒤 번호 오름차순으로 표시.
    slot_to_numbers: dict[str, list[int]] = {}
    for n, slots in slots_by_num.items():
        for s in slots:
            slot_to_numbers.setdefault(s, []).append(n)
    for s, nums in slot_to_numbers.items():
        top = sorted(nums, key=lambda n: (-_score(cite_by_num[n]), n))[:_MAX_INLINE]
        slot_to_numbers[s] = sorted(top)

    numbered = [(n, cite_by_num[n]) for n in sorted(cite_by_num)]
    unlinked = sorted(n for n, slots in slots_by_num.items() if not slots)
    return Footnotes(
        slot_to_numbers=slot_to_numbers, numbered=numbered, unlinked_numbers=unlinked
    )


def strip_markers(text: str) -> str:
    """LLM이 본문에 끼워 넣었을지 모르는 [숫자] 표기를 제거 — 인라인 번호는 코드가 다시 붙인다.

    제거 후 남는 군더더기 공백(구두점 앞 공백·이중 공백)도 정리해 '성장세다 .' 같은 흔적을 없앤다.
    """
    s = _MARKER_RE.sub("", text or "")
    s = re.sub(r"\s+([.,!?;:)\]」』”])", r"\1", s)  # 구두점 앞 공백 제거
    s = re.sub(r"[ \t]{2,}", " ", s)                # 이중 공백 축소
    return s.strip()


def inline_markers(slot: str, foot: Footnotes) -> str:
    """슬롯에 달 인라인 각주 마커 문자열. 예: '[1][3]' (없으면 '')."""
    return "".join(f"[{n}]" for n in (foot.slot_to_numbers.get(slot) or []))


def _reference_line(n: int, c: Citation) -> str:
    """근거 1건을 끝 출처목록 한 줄로. cluster별 상세 포맷(평탄화 전 정보 복원)."""
    cluster = c.get("cluster")
    snippet = (c.get("snippet") or "").strip()
    snip = f' — "{snippet}"' if snippet else ""

    if cluster == "research":
        title = (c.get("title") or c.get("url") or "출처 미상").strip()
        meta_bits: list[str] = []
        accessed = (c.get("accessed_at") or "").strip()
        if accessed:
            meta_bits.append(f"접근일 {accessed}")
        if c.get("score_kind") == "relevance":
            meta_bits.append(f"관련도 {_score(c):.2f}")
        meta = f" ({', '.join(meta_bits)})" if meta_bits else ""
        url = (c.get("url") or "").strip()
        url_part = f" {url}" if url else ""
        return f"[{n}] (리서치) {title}{snip}{meta}{url_part}"

    if cluster == "rag":
        fname = (c.get("source_file") or c.get("title") or "출처 미상").strip()
        page = (c.get("page") or "").strip()
        folder = (c.get("folder") or "").strip()
        page_part = f" p.{page}" if page else ""
        folder_part = f" ({folder})" if folder else ""
        sim = ""
        if c.get("score_kind") == "similarity_pct":
            sim = f" (유사도 {_score(c):.0f}%)"
        return f"[{n}] (사내문서) {fname}{page_part}{folder_part}{snip}{sim}"

    # 알 수 없는 cluster — 가진 정보로 최소 렌더.
    label = (c.get("title") or c.get("url") or snippet or "출처").strip()
    return f"[{n}] {label}"


def render_references(foot: Footnotes) -> str:
    """'## 근거 및 출처' 본문. 근거가 없으면 사용자 입력 기반임을 명시.

    항목 사이를 빈 줄(\\n\\n)로 띄워 각 출처가 별도 문단으로 렌더되게 한다 — 한 항목이 여러
    줄로 접혀도 다음 항목과 시각적으로 구분된다(렌더러가 nl2br 없이도 문단으로 분리).
    """
    if not foot.numbered:
        return "근거 없음 — 사용자 입력 기반."
    return "\n\n".join(_reference_line(n, c) for n, c in foot.numbered)
