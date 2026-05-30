"""
run_rag_val.py

claim 추출 → 사내 문서 검색 + 하이라이트 → 검증(validator) 까지
전체 파이프라인을 한 번에 실행하는 스크립트.

[파이프라인 순서]
  Step 1: ClaimExtractionAgent  (claim_router_validation.py)
          자연어 입력에서 claim, keywords, topic 추출
  Step 2: RAG Pipeline          (rag_extractor.py)
          FolderRouterAgent + SearchHighlightAgent
          사내 벡터DB에서 근거 청크 검색 후 highlight 생성
  Step 3: ValidatorAgent        (validator.py)
          highlight + raw_source 가 claim 을 논리적으로 지지하는지 평가
  Step 4: 최종 요약 저장

[인풋 모드]
  - 인라인 텍스트: INPUT_TEXT 에 문장 직접 작성 → 바로 실행
  - JSON 파일 모드: INPUT_TEXT 를 비우면 INPUT_PATH 의 JSON 파일 로드
                   파일 구조: {"claims": [{"claim_id": str, "claim": str, ...}]}

[아웃풋]
  중간 RAG 결과는 별도 저장 없음.
  Step 1~3 의 모든 기록을 포함한 최종 JSON 하나만 data/output/ 에 저장.

실행 방법:
    python run_rag_val.py

환경변수:
    RAG_VAL_INPUT_PATH  → JSON 파일 모드에서 사용할 입력 파일 경로
    RAG_VAL_OUTPUT_DIR  → 아웃풋 저장 폴더 (기본: data/output/)
    MAX_CLAIMS          → 처리할 최대 claim 수 (0 이면 전체, JSON 파일 모드 한정)
    VERBOSE             → 에이전트 상세 로그 출력 여부 (기본: true)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ─── 경로 설정 ──────────────────────────────────────────────────────────────────
# 이 파일은 agents/validator/ 에 위치한다.
# 프로젝트 루트는 두 단계 위 (validator → agents → root).
VALIDATOR_DIR = Path(__file__).parent          # agents/validator/
PROJECT_ROOT  = VALIDATOR_DIR.parent.parent    # 프로젝트 루트

# 프로젝트 루트를 sys.path 에 추가해 agents 패키지를 import 가능하게 한다.
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(str(PROJECT_ROOT / ".env"))

from agents.rag.claim_router_validation import run_router_pipeline
from agents.rag.rag_extractor import (
    FallbackRequired,
    run_rag_extractor,
)
from agents.validator.validator import run_validator

# ─── 인라인 인풋 ────────────────────────────────────────────────────────────────
# 검증할 문장을 여기에 직접 작성하세요.
# 비워두면 INPUT_PATH 의 JSON 파일을 로드합니다.
INPUT_TEXT = "현실적으로 지금 웹툰이 만들어지는 상황을 보면 작가들이 선정성 여부를 일일히 파악하는게 어려운게 사실이야."

# ─── 환경변수 기반 설정 ─────────────────────────────────────────────────────────

# JSON 파일 모드에서 사용할 입력 파일 경로
INPUT_PATH = Path(
    os.getenv(
        "RAG_VAL_INPUT_PATH",
        str(PROJECT_ROOT / "data" / "input" / "3rd_claim_extractor_output.json"),
    )
)

# 최종 아웃풋 저장 경로
OUTPUT_DIR = Path(
    os.getenv("RAG_VAL_OUTPUT_DIR", str(PROJECT_ROOT / "data" / "output"))
)

# JSON 파일 모드에서 처리할 최대 claim 수 (0 이면 전체)
MAX_CLAIMS: int = int(os.getenv("MAX_CLAIMS", "0"))

# 에이전트 실행 중 상세 로그 출력 여부
VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"


# ─── 헬퍼: JSON 파일 로드 ───────────────────────────────────────────────────────

def load_input_file(path: Path) -> dict:
    """
    claim_extractor_output.json 을 로드하고 기본 구조를 검증한다.

    파일이 없거나 'claims' 필드가 없으면 오류 출력 후 종료한다.

    :param path: 입력 JSON 파일 경로
    :return: 파싱된 dict
    """
    if not path.exists():
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "claims" not in data:
        print("[오류] 입력 JSON 에 'claims' 필드가 없습니다.")
        sys.exit(1)

    return data


# ─── 헬퍼: 인라인 텍스트 → claim 리스트 변환 ───────────────────────────────────

def build_claims_from_text(text: str) -> tuple[list, dict]:
    """
    단일 문장 입력을 받아 Step 1(ClaimExtractionAgent)을 실행하고
    claim 1개짜리 리스트와 step1 출력 dict 를 반환한다.

    run_router_pipeline 을 호출해 LangGraph 기반 claim 추출 파이프라인을 실행한다.

    :param text: 검증할 원본 자연어 문장
    :return: (claims 리스트, step1_output dict)
    """
    print("[Step 1] ClaimExtractionAgent — claim 추출 중...")
    result = run_router_pipeline(text)
    print()

    claim_result = result.get("claim_result") or {}

    step1_output = {
        "claim": claim_result.get("claim") or "",
        "normalized_claim": claim_result.get("normalized_claim") or "",
        "topic": claim_result.get("topic", ""),
        "keywords": claim_result.get("keywords") or [],
        "search_ready": claim_result.get("search_ready", True),
    }

    claims = [{
        "claim_id": "c1",
        "original_text": text,
        "claim": claim_result.get("claim") or text,
        "normalized_claim": claim_result.get("normalized_claim") or text,
        "topic": claim_result.get("topic", ""),
        "keywords": claim_result.get("keywords") or [],
        "search_ready": claim_result.get("search_ready", True),
        "error": result.get("error"),
    }]

    return claims, step1_output


# ─── 메인 ────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    전체 파이프라인(Step 1 → Step 2 → Step 3)을 실행하고 결과를 저장한다.

    [처리 순서]
    1. 인풋 준비 (인라인 텍스트 or JSON 파일)
    2. Step 1: claim 추출 (인라인 텍스트 모드 한정; JSON 모드는 이미 추출된 claim 사용)
    3. Step 2: RAG — 사내 문서 검색 + 하이라이트 생성
    4. Step 3: Validator — 하이라이트가 claim 을 논리적으로 지지하는지 평가
    5. 모든 기록을 단일 JSON 으로 저장
    """
    input_mode = "인라인 텍스트" if INPUT_TEXT.strip() else "JSON 파일"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  RAG + Validator 통합 파이프라인 시작")
    print(f"  입력 모드: {input_mode}  |  VERBOSE={VERBOSE}")
    print("=" * 60)

    # ── 인풋 준비 ───────────────────────────────────────────────────────────────
    step1_output: dict = {}
    step1_input_text: str = ""

    if INPUT_TEXT.strip():
        # 인라인 텍스트 모드
        step1_input_text = INPUT_TEXT.strip()
        print(f"\n  입력 문장: {step1_input_text}")
        print()
        claims, step1_output = build_claims_from_text(step1_input_text)
    else:
        # JSON 파일 모드: 이미 추출된 claim 목록을 파일에서 로드
        print(f"\n[Step 1] 입력 파일 로드 중...")
        print(f"  경로: {INPUT_PATH}")
        data = load_input_file(INPUT_PATH)
        claims = data["claims"]
        step1_input_text = data.get("input", {}).get("qw", "")
        step1_output = {}   # JSON 모드에서는 claim 이 이미 추출되어 있으므로 step1 별도 없음

        if MAX_CLAIMS > 0:
            claims = claims[:MAX_CLAIMS]

        print(f"  원문(qw): {step1_input_text[:60]}{'...' if len(step1_input_text) > 60 else ''}")
        print(f"  처리할 claim 수: {len(claims)}개\n")

    # ── Step 2: RAG 파이프라인 ──────────────────────────────────────────────────
    print("[Step 2] RAG — 사내 문서 검색 및 하이라이트 생성")
    print("-" * 60)

    step2_claims = []    # 최종 아웃풋에 저장할 claim별 결과 누적
    debug_traces = []    # 디버그용: claim별 OpenAI SDK response 원본 누적

    for i, rc in enumerate(claims, start=1):
        claim_id = rc.get("claim_id", f"c{i}")

        # claim 이 있으면 그대로 사용, 없으면 원문으로 대체
        qk          = rc.get("claim") or rc.get("normalized_claim") or rc.get("original_text", "")
        pre_claim   = rc.get("normalized_claim") or rc.get("claim") or None
        pre_keywords = rc.get("keywords") or None
        # 사전 추출된 claim/keywords 가 있으면 rag_extractor 내부 Agent 1 을 건너뜀
        skip_agent1 = bool(pre_claim) and bool(pre_keywords)

        print(f"\n  [{i}/{len(claims)}] {claim_id} — RAG 검색 시작...")

        claim_entry: dict = {"claim_id": claim_id}

        try:
            rag_result, agent_trace, agent_steps = run_rag_extractor(
                qk=qk,
                claim=pre_claim if skip_agent1 else None,
                keywords=pre_keywords if skip_agent1 else None,
                verbose=VERBOSE,
            )

            # agent1 trace 가 빈 경우(skip) 사전 추출 trace 가 있으면 주입
            if not agent_trace.get("agent1_claim_extractor") and rc.get("_agent1_trace"):
                agent_trace["agent1_claim_extractor"] = rc["_agent1_trace"]

            claim_entry["status"] = "success"
            claim_entry["agents"] = {
                "claim_extractor":  agent_steps["claim_extractor"],
                "folder_router":    agent_steps["folder_router"],
                "search_highlight": agent_steps["search_highlight"],
            }
            claim_entry["highlight"]   = rag_result["highlight"]
            claim_entry["source_file"] = rag_result["source_file"]
            claim_entry["source_page"] = rag_result["source_page"]

            debug_traces.append({"claim_id": claim_id, "stage": "rag", "agent_trace": agent_trace})
            print(f"  → RAG 완료: {rag_result['highlight'][:60]}...")

        except FallbackRequired as e:
            # 사내 문서에서 유효한 근거를 찾지 못한 경우
            claim_entry["status"]       = "fallback"
            claim_entry["fallback_reason"]  = str(e)
            claim_entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "stage": "rag", "agent_trace": None, "fallback_reason": str(e)})
            print(f"  → RAG fallback: {e}")
            step2_claims.append(claim_entry)
            continue   # validator 는 RAG 성공 시에만 실행

        except Exception as e:
            claim_entry["status"]       = "error"
            claim_entry["error"]            = str(e)
            claim_entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "stage": "rag", "agent_trace": None, "error": str(e)})
            print(f"  → RAG 오류: {e}")
            step2_claims.append(claim_entry)
            continue   # validator 는 RAG 성공 시에만 실행

        # ── Step 3: ValidatorAgent ───────────────────────────────────────────────
        # RAG 가 성공한 claim 에 대해서만 검증을 실행한다.
        # rag_result(RagExtractorResult) 를 그대로 run_validator 에 전달한다.
        print(f"  [{i}/{len(claims)}] {claim_id} — ValidatorAgent 실행 중...")

        validator_result, validator_meta = run_validator(rag_result, verbose=VERBOSE)

        claim_entry["validator"] = {
            "verdict":            validator_result["verdict"],
            "confidence":         validator_result["confidence"],
            "reasoning":          validator_result["reasoning"],
            "evidence_used":      validator_result["evidence_used"],
            "additional_searches": validator_result["additional_searches"],
        }

        debug_traces.append({
            "claim_id":               claim_id,
            "stage":                  "validator",
            "validator_response_trace": validator_meta["validator_response_trace"],
            "validator_turn_log":       validator_meta["validator_turn_log"],
        })

        print(f"  → verdict: {validator_result['verdict']}  |  confidence: {validator_result['confidence']}")
        print(f"  → reasoning: {validator_result['reasoning']}")

        step2_claims.append(claim_entry)

    # ── 집계 ────────────────────────────────────────────────────────────────────
    rag_success   = sum(1 for r in step2_claims if r.get("status") == "success")
    rag_fallback  = sum(1 for r in step2_claims if r.get("status") == "fallback")
    rag_error     = sum(1 for r in step2_claims if r.get("status") == "error")

    verdict_counts: dict = {"supports": 0, "contradicts": 0, "insufficient": 0, "unrelated": 0}
    for r in step2_claims:
        v = r.get("validator", {}).get("verdict")
        if v in verdict_counts:
            verdict_counts[v] += 1

    # ── 최종 아웃풋 저장 ─────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("[Step 4] 결과 저장 중...")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"rag_val_output_{timestamp}.json"
    debug_path  = OUTPUT_DIR / f"rag_val_debug_{timestamp}.json"

    output_payload = {
        "pipeline":   "RAG + Validator Pipeline",
        "input_mode": input_mode,
        "step1": {
            "label": "claim 추출",
            "agent": "claim_router_validation / ClaimExtractionAgent",
            "role":  "자연어 입력에서 검색용 claim, normalized_claim, 키워드, 주제어 추출",
            "input": step1_input_text,
            "output": step1_output,
        },
        "step2": {
            "label":  "사내 문서 검색 및 하이라이트 생성",
            "agent":  "FolderRouterAgent + SearchHighlightAgent",
            "claims": [
                {k: v for k, v in r.items() if k != "validator"}
                for r in step2_claims
            ],
        },
        "step3": {
            "label":  "claim 검증",
            "agent":  "ValidatorAgent",
            "claims": [
                {
                    "claim_id":   r["claim_id"],
                    "status": r.get("status"),
                    **r.get("validator", {"verdict": None, "confidence": None,
                                          "reasoning": "RAG 실패로 검증 불가",
                                          "evidence_used": [], "additional_searches": 0}),
                }
                for r in step2_claims
            ],
        },
        "step4": {
            "label": "최종 요약",
            "rag": {
                "total":    len(step2_claims),
                "success":  rag_success,
                "fallback": rag_fallback,
                "error":    rag_error,
            },
            "validator": {
                "supports":     verdict_counts["supports"],
                "contradicts":  verdict_counts["contradicts"],
                "insufficient": verdict_counts["insufficient"],
                "unrelated":    verdict_counts["unrelated"],
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # 디버그 파일: RAG + validator 의 OpenAI SDK response 원본 누적
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_traces, f, ensure_ascii=False, indent=2)

    print(f"  결과 저장:  {output_path}")
    print(f"  디버그 저장: {debug_path}")

    # ── 최종 요약 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  실행 완료 요약")
    print("=" * 60)
    print(f"  전체 claim       : {len(step2_claims)}")
    print(f"  RAG 성공         : {rag_success}")
    print(f"  RAG fallback     : {rag_fallback}")
    print(f"  RAG 오류         : {rag_error}")
    print()
    print(f"  supports         : {verdict_counts['supports']}")
    print(f"  contradicts      : {verdict_counts['contradicts']}")
    print(f"  insufficient     : {verdict_counts['insufficient']}")
    print(f"  unrelated        : {verdict_counts['unrelated']}")
    print()
    print(f"  아웃풋 파일      : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
