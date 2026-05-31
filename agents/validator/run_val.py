"""
run_val.py

agents/validator/rag_output_*.json 을 인풋으로 받아 ValidatorAgent 를 실행하고
결과를 data/output/ 에 JSON 으로 저장하는 실행 스크립트.

실행 방법:
    python run_val.py

출력:
    data/output/validator_output_{timestamp}.json
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 이 파일은 agents/validator/ 에 위치한다.
# 프로젝트 루트는 두 단계 위(agents/validator → agents → project root).
VALIDATOR_DIR = Path(__file__).parent          # agents/validator/
PROJECT_ROOT = VALIDATOR_DIR.parent.parent     # 프로젝트 루트
sys.path.insert(0, str(PROJECT_ROOT))

from agents.rag.rag_extractor import RagExtractorResult
from agents.validator.validator import run_validator

# ─── 경로 설정 ──────────────────────────────────────────────────────────────────

# 인풋 파일: 이 스크립트와 같은 폴더(agents/validator/)의 rag_output JSON
INPUT_FILE = Path(
    os.getenv(
        "VALIDATOR_INPUT",
        str(VALIDATOR_DIR / "rag_output_20260530_224423.json"),
    )
)

# 아웃풋 디렉토리: 프로젝트 루트 기준 data/output/
OUTPUT_DIR = Path(os.getenv("VALIDATOR_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 터미널 출력 상세 여부
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"


# ─── 헬퍼: RAG JSON → RagExtractorResult ────────────────────────────────────────

def build_rag_result(qk: str, claim_entry: dict) -> RagExtractorResult:
    """
    rag_output JSON 의 claims 항목 하나를 RagExtractorResult TypedDict 로 변환한다.

    [필드 매핑]
    qk             ← step1.input  (원문 입력 텍스트)
    claim          ← claims[i].agents.claim_extractor.claim
    keywords       ← claims[i].agents.claim_extractor.keywords
    folders_searched ← claims[i].agents.folder_router.folders_selected
    highlight      ← claims[i].highlight
    highlight_reason ← claims[i].agents.search_highlight.highlight_reason
    keyword_used   ← claims[i].agents.search_highlight.keyword_used
    folder_searched ← claims[i].agents.search_highlight.folder_searched
    source_file    ← claims[i].source_file
    source_page    ← claims[i].source_page
    raw_source     ← claims[i].agents.search_highlight.raw_source
    """
    agents = claim_entry.get("agents", {})
    claim_extractor = agents.get("claim_extractor", {})
    folder_router = agents.get("folder_router", {})
    search_highlight = agents.get("search_highlight", {})

    return RagExtractorResult(
        qk=qk,
        claim=claim_extractor.get("claim", ""),
        keywords=claim_extractor.get("keywords", []),
        folders_searched=folder_router.get("folders_selected", []),
        highlight=claim_entry.get("highlight", ""),
        highlight_reason=search_highlight.get("highlight_reason", ""),
        keyword_used=search_highlight.get("keyword_used", ""),
        folder_searched=search_highlight.get("folder_searched", ""),
        source_file=claim_entry.get("source_file", ""),
        source_page=str(claim_entry.get("source_page", "")),
        raw_source=search_highlight.get("raw_source", ""),
    )


# ─── 메인 ────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  ValidatorAgent 실행 스크립트")
    print("=" * 60)

    # ── 인풋 파일 로드 ──────────────────────────────────────────
    if not INPUT_FILE.exists():
        print(f"[오류] 인풋 파일을 찾을 수 없습니다: {INPUT_FILE}")
        sys.exit(1)

    print(f"\n[1/3] 인풋 파일 로드 중...")
    print(f"  경로: {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        rag_data = json.load(f)

    qk: str = rag_data.get("step1", {}).get("input", "")
    claims: list = rag_data.get("step2", {}).get("claims", [])

    print(f"  원문(qk): {qk[:60]}{'...' if len(qk) > 60 else ''}")
    print(f"  처리할 claim 수: {len(claims)}개")

    # ── ValidatorAgent 실행 ────────────────────────────────────
    print(f"\n[2/3] ValidatorAgent 실행 중...")
    print("-" * 60)

    results = []
    counts = {"supports": 0, "contradicts": 0, "insufficient": 0, "unrelated": 0, "skipped": 0}

    for i, claim_entry in enumerate(claims, start=1):
        claim_id = claim_entry.get("claim_id", f"c{i}")
        status = claim_entry.get("status", "")

        print(f"\n  [{i}/{len(claims)}] claim_id={claim_id}  (RAG status: {status})")

        # RAG 단계에서 실패한 claim 은 건너뛴다.
        if status != "success":
            print(f"  → 건너뜀 (RAG status가 success 가 아님)")
            counts["skipped"] += 1
            results.append({
                "claim_id": claim_id,
                "rag_status": status,
                "validator_status": "skipped",
                "verdict": None,
                "confidence": None,
                "reasoning": "RAG 단계 실패로 검증 불가",
                "evidence_used": [],
                "additional_searches": 0,
            })
            continue

        # RagExtractorResult 로 변환 후 ValidatorAgent 실행
        rag_result = build_rag_result(qk, claim_entry)
        validator_result, meta = run_validator(rag_result, verbose=VERBOSE)

        verdict = validator_result["verdict"]
        counts[verdict] = counts.get(verdict, 0) + 1

        print(f"  → verdict: {verdict}  |  confidence: {validator_result['confidence']}")
        print(f"  → reasoning: {validator_result['reasoning']}")

        results.append({
            "claim_id": claim_id,
            "rag_status": status,
            "validator_status": "completed",
            "claim": rag_result["claim"],
            "highlight": rag_result["highlight"],
            "source_file": rag_result["source_file"],
            "source_page": rag_result["source_page"],
            "verdict": verdict,
            "confidence": validator_result["confidence"],
            "reasoning": validator_result["reasoning"],
            "evidence_used": validator_result["evidence_used"],
            "additional_searches": validator_result["additional_searches"],
        })

    # ── 아웃풋 저장 ────────────────────────────────────────────
    print("\n" + "-" * 60)
    print(f"[3/3] 결과 저장 중...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"validator_output_{timestamp}.json"

    output = {
        "pipeline": "Validator Pipeline",
        "input_file": str(INPUT_FILE),
        "run_at": timestamp,
        "original_qk": qk,
        "summary": {
            "total": len(claims),
            "supports": counts["supports"],
            "contradicts": counts["contradicts"],
            "insufficient": counts["insufficient"],
            "unrelated": counts["unrelated"],
            "skipped": counts["skipped"],
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  저장 완료: {output_path}")

    # ── 최종 요약 출력 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  실행 완료 요약")
    print("=" * 60)
    print(f"  전체 claim 수 : {len(claims)}")
    print(f"  supports      : {counts['supports']}")
    print(f"  contradicts   : {counts['contradicts']}")
    print(f"  insufficient  : {counts['insufficient']}")
    print(f"  unrelated     : {counts['unrelated']}")
    print(f"  건너뜀        : {counts['skipped']}")
    print(f"\n  아웃풋 파일   : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
