"""dispatch._evidence_record — ValidationReport + 슬롯 연결 → EvidenceRecord 매핑."""
from agents.orchestrator.nodes.dispatch import _evidence_record


def test_evidence_record_carries_slot_turn_and_citations():
    report = {
        "subject": "시장 1.8조",
        "cluster": "research",
        "findings": ["1.8조"],
        "agreement": "confirms",
        "sources": ["https://k/x"],
        "citations": [{"cluster": "research", "url": "https://k/x"}],
    }
    rec = _evidence_record(report, "market", 3)
    assert rec["target_slot"] == "market"
    assert rec["turn"] == 3
    assert rec["cluster"] == "research"
    assert rec["citations"] == report["citations"]
    assert rec["findings"] == ["1.8조"]


def test_evidence_record_handles_missing_slot_and_citations():
    report = {"subject": "x", "cluster": "rag", "findings": [], "agreement": "unknown"}
    rec = _evidence_record(report, None, 1)
    assert rec["target_slot"] is None
    assert rec["citations"] == []
