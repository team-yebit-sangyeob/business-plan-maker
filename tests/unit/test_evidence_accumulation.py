"""근거 세션 누적(run_turn._merge_session_evidence) — 적재·중복제거·슬롯 백필.

turn_validation_reports는 매 턴 리셋되므로, 계획서가 세션 전체 근거를 인용하려면 별도 누적분
(session_evidence)이 턴을 넘어 살아남아야 한다. 그 합치기 로직만 순수 함수로 검증(그래프/LLM 불필요).
"""
from agents.orchestrator.graph import _merge_session_evidence


def _rec(subject, cluster, *, slot=None, turn=1, cite_url="u"):
    return {
        "subject": subject,
        "cluster": cluster,
        "findings": [f"{subject} finding"],
        "agreement": "confirms",
        "citations": [{"cluster": cluster, "url": cite_url}],
        "target_slot": slot,
        "turn": turn,
    }


def test_accumulates_across_turns():
    prev = [_rec("시장 1.8조", "research", slot="market", turn=1)]
    turn2 = [_rec("B2B 적합", "rag", slot="advantage", turn=2)]
    merged = _merge_session_evidence(prev, turn2, segments=[])
    keys = {(m["subject"], m["cluster"]) for m in merged}
    assert keys == {("시장 1.8조", "research"), ("B2B 적합", "rag")}


def test_dedup_same_subject_cluster_last_write_wins():
    # 같은 claim을 다음 턴 재검증 → 최신 1건만 남고 값은 갱신된다.
    prev = [_rec("시장 1.8조", "research", slot="market", turn=1, cite_url="old")]
    again = [_rec("시장 1.8조", "research", slot="market", turn=4, cite_url="new")]
    merged = _merge_session_evidence(prev, again, segments=[])
    assert len(merged) == 1
    assert merged[0]["turn"] == 4
    assert merged[0]["citations"][0]["url"] == "new"


def test_backfill_target_slot_from_segments():
    # dispatch 시점엔 슬롯 힌트가 없을 수 있다(None). extract_fills가 정한 슬롯을 세그먼트에서 백필.
    turn_ev = [_rec("타겟은 네이버", "research", slot=None, turn=2)]
    segments = [{"canonical_text": "타겟은 네이버", "target_slot": "target"}]
    merged = _merge_session_evidence([], turn_ev, segments)
    assert merged[0]["target_slot"] == "target"


def test_backfill_keeps_dispatch_hint_when_no_segment_match():
    turn_ev = [_rec("시장 근거", "research", slot="market", turn=2)]
    merged = _merge_session_evidence([], turn_ev, segments=[])
    assert merged[0]["target_slot"] == "market"
