"""
rag_main.py

실행 진입점: python rag_main.py

[입력 모드]
  1) 인라인 텍스트 모드 (기본): 아래 INPUT_TEXT에 문장을 직접 작성하여 실행
  2) JSON 파일 모드: INPUT_TEXT를 비워두면 INPUT_PATH의 JSON 파일을 로드

[디버그 모드 (.env 또는 환경변수)]
  DEBUG_SKIP_ROUTER=true  → decide_search_route를 건너뛰고 바로 rag_extractor로 전달
  MAX_CLAIMS=N            → 처음 N개 claim만 처리 (JSON 파일 모드 한정)
  VERBOSE=false           → 에이전트 상세 로그 숨김
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# experiments 루트의 .env 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 인라인 입력 ────────────────────────────────────────────────────────────────
# 검증할 문장을 여기에 직접 작성하세요.
# 비워두면 아래 INPUT_PATH의 JSON 파일을 로드합니다.
INPUT_TEXT = "현실적으로 지금 웹툰이 만들어지는 상황을 보면 작가들이 선정성 여부를 일일히 파악하는게 어려운게 사실이야."


# ─── 공통 경로 설정 ─────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent.parent.parent

# INPUT_TEXT가 비어 있을 때 로드할 JSON 파일 경로 (환경변수로 재지정 가능)
INPUT_PATH = Path(
    os.getenv("RAG_INPUT_PATH", str(_BASE_DIR / "data" / "input" / "3rd_claim_extractor_output.json"))
)

# 출력 폴더 (환경변수로 재지정 가능)
OUTPUT_DIR = Path(os.getenv("RAG_OUTPUT_DIR", str(_BASE_DIR / "data" / "output")))

# 처리할 최대 claim 수 (0이면 전체 처리)
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "0"))

# 에이전트 실행 중 상세 로그 출력 여부
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# ─── 디버그 설정 ────────────────────────────────────────────────────────────────
# claim_router_validation을 건너뛰고 rag_extractor로 직행할 때 True로 변경.
# .env/환경변수 DEBUG_SKIP_ROUTER=true 설정으로도 동일하게 활성화된다.
DEBUG_SKIP_ROUTER: bool = True  # ← 직접 켜고 끄려면 True/False로 변경
DEBUG_SKIP_ROUTER = DEBUG_SKIP_ROUTER or os.getenv("DEBUG_SKIP_ROUTER", "false").lower() == "true"


# ─── 모듈 임포트 ────────────────────────────────────────────────────────────────
from claim_router_validation import run_router_pipeline, run_router_pipeline_for_claims, extract_claim_from_text
from rag_extractor import run_rag_extractor, FallbackRequired


# ─── 입력 로드 함수 ─────────────────────────────────────────────────────────────

def load_input(path: Path) -> dict:
    """
    claim_extractor_output.json을 로드하고 기본 구조를 검증한다.

    :param path: 입력 JSON 파일 경로
    :return: 파싱된 dict
    :raises SystemExit: 파일이 없거나 구조가 맞지 않으면 오류 출력 후 종료
    """
    if not path.exists():
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    # claims 필드 필수 확인
    if "claims" not in data:
        print(f"[오류] 입력 JSON에 'claims' 필드가 없습니다.")
        sys.exit(1)

    return data


# ─── 결과 저장 함수 ─────────────────────────────────────────────────────────────

def save_results(results: list, output_dir: Path, timestamp: str) -> Path:
    """
    처리 결과를 JSON 파일로 저장한다.

    :param results: 저장할 결과 리스트
    :param output_dir: 출력 폴더 경로
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :return: 저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rag_output_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_path


def save_debug_trace(traces: list, output_dir: Path, timestamp: str) -> Path:
    """
    각 claim에 대한 OpenAI SDK response 원본을 디버그용 JSON 파일로 저장한다.
    에이전트별(agent1/2/3) response 목록과 claim_id가 함께 저장된다.

    :param traces: [{"claim_id": str, "agent_trace": {agent1: [...], ...}}, ...]
    :param output_dir: 출력 폴더 경로
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :return: 저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / f"rag_debug_{timestamp}.json"
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    return debug_path


# ─── 메인 실행 ──────────────────────────────────────────────────────────────────

def _build_routed_claims_from_text(text: str) -> list:
    """
    단일 문장 입력을 claim 1개짜리 routed_claims 리스트로 변환한다.

    DEBUG_SKIP_ROUTER=true이면 라우터를 건너뛰고 route=internal로 고정한다.
    그렇지 않으면 run_router_pipeline으로 claim 추출과 라우팅을 모두 수행한다.
    """
    if DEBUG_SKIP_ROUTER:
        # claim 추출은 실행하되 검색 경로 결정(라우팅)은 건너뜀
        print("[Step 1] claim 추출 중... (라우팅은 DEBUG_SKIP_ROUTER로 건너뜀)")
        claim_result, claim_trace = extract_claim_from_text(text)
        print()
        return [{
            "claim_id": "c1",
            "claim": claim_result.claim or text,
            "normalized_claim": claim_result.normalized_claim or text,
            "topic": claim_result.topic,
            "keywords": claim_result.keywords,
            "search_ready": claim_result.search_ready,
            "route_decision": {"route": "internal", "reason": "DEBUG_SKIP_ROUTER"},
            "_agent1_trace": [claim_trace],
        }]

    print("[Step 1] claim 추출 및 검색 경로 라우팅 중...")
    result = run_router_pipeline(text)
    print()

    claim_result = result.get("claim_result") or {}
    route_decision = result.get("route_decision")

    return [{
        "claim_id": "c1",
        "claim": claim_result.get("claim", text),
        "normalized_claim": claim_result.get("normalized_claim", text),
        "topic": claim_result.get("topic", ""),
        "keywords": claim_result.get("keywords", []),
        "search_ready": claim_result.get("search_ready", True),
        "route_decision": route_decision,
        "error": result.get("error"),
    }]


def main() -> None:
    """
    전체 파이프라인을 실행한다.

    처리 순서:
      1. 입력 준비 (인라인 텍스트 or JSON 파일)
      2. 검색 경로 라우팅 (DEBUG_SKIP_ROUTER=true이면 건너뜀)
      3. internal/both 판정된 claim마다 사내 문서 검색 + 하이라이트
      4. fallback 발생 시 웹 에이전트 이관 안내 출력
      5. 결과를 JSON 파일로 저장
    """
    input_mode = "인라인 텍스트" if INPUT_TEXT.strip() else "JSON 파일"

    print("=" * 60)
    print("RAG Main Pipeline 시작")
    print(f"입력 모드: {input_mode}")
    print(f"디버그 모드: DEBUG_SKIP_ROUTER={DEBUG_SKIP_ROUTER} | VERBOSE={VERBOSE}")
    print("=" * 60)

    # ── 1) 입력 준비 ────────────────────────────────────────────────────────────
    if INPUT_TEXT.strip():
        # 인라인 텍스트 모드: 파일 없이 바로 실행
        print(f"입력 문장: {INPUT_TEXT.strip()}")
        print()
        routed_claims = _build_routed_claims_from_text(INPUT_TEXT.strip())
    else:
        # JSON 파일 모드: claim_extractor_output.json 로드
        print(f"입력 파일: {INPUT_PATH}")
        data = load_input(INPUT_PATH)
        claims = data["claims"]
        qw = data.get("input", {}).get("qw", "")

        if MAX_CLAIMS > 0:
            claims = claims[:MAX_CLAIMS]

        print(f"qw: {qw}")
        print(f"처리할 claim 수: {len(claims)}")
        if DEBUG_SKIP_ROUTER:
            print("[디버그] DEBUG_SKIP_ROUTER=true → 라우터를 건너뛰고 전체 claim을 rag_extractor로 직행합니다.")
        print()

        if DEBUG_SKIP_ROUTER:
            # 모든 claim을 route=internal로 고정하여 라우터 LLM 호출 없이 바로 RAG 단계로 진입
            routed_claims = [
                {**c, "route_decision": {"route": "internal", "reason": "DEBUG_SKIP_ROUTER"}}
                for c in claims
            ]
        else:
            print("[Step 1] 검색 경로 라우팅 중...")
            routed_claims = run_router_pipeline_for_claims(claims)
            print()

    # ── 2) 라우팅 결과 요약 출력 ────────────────────────────────────────────────
    print("[라우팅 결과 요약]")
    print("-" * 40)
    for rc in routed_claims:
        rd = rc.get("route_decision")
        if rd:
            print(f"  {rc['claim_id']}: route={rd['route']} | {rd['reason'][:50]}...")
        else:
            reason = rc.get("skip_reason") or rc.get("error") or "unknown"
            print(f"  {rc['claim_id']}: 라우팅 없음 ({reason})")
    print()

    # ── 4) 사내 문서 검색 + 하이라이트 생성 ─────────────────────────────────────
    print("[Step 2] 사내 문서 검색 및 하이라이트 생성 중...")
    print("-" * 40)

    final_results = []
    debug_traces = []  # 디버그용: claim별 OpenAI SDK response 원본 누적
    for rc in routed_claims:
        claim_id = rc.get("claim_id", "?")
        rd = rc.get("route_decision")

        entry: dict = {
            "claim_id": claim_id,
            "claim": rc.get("claim", ""),
            "normalized_claim": rc.get("normalized_claim", ""),
            "topic": rc.get("topic", ""),
            "route": rd["route"] if rd else None,
        }

        if not rd or rd["route"] not in ("internal", "both"):
            # web only: 사내 문서 검색 불필요, 웹 에이전트로 처리
            entry["status"] = "web_only"
            entry["highlight"] = None
            entry["web_agent_needed"] = True
            print(f"  {claim_id}: [web] 웹 에이전트로 이관됩니다.")
            final_results.append(entry)
            continue

        # internal 또는 both: 사내 문서 검색 실행
        print(f"\n  {claim_id}: [internal] 사내 문서 검색 시작...")
        qk = rc.get("claim", rc.get("normalized_claim", ""))

        # 디버그 모드이고 사전 추출된 keywords가 있으면 Agent 1(ClaimExtractor)을 건너뜀
        pre_claim = rc.get("normalized_claim") or rc.get("claim") or None
        pre_keywords = rc.get("keywords") or None
        skip_agent1 = DEBUG_SKIP_ROUTER and bool(pre_claim) and bool(pre_keywords)

        try:
            result, agent_trace = run_rag_extractor(
                qk=qk,
                claim=pre_claim if skip_agent1 else None,
                keywords=pre_keywords if skip_agent1 else None,
                verbose=VERBOSE,
            )
            entry["status"] = "success"
            entry["claim"] = result["claim"]  # run_rag_extractor에서 실제 추출된 claim으로 갱신
            entry["highlight"] = result["highlight"]
            entry["highlight_reason"] = result["highlight_reason"]
            entry["keyword_used"] = result["keyword_used"]
            entry["folder_searched"] = result["folder_searched"]
            entry["source_file"] = result["source_file"]
            entry["source_page"] = result["source_page"]
            entry["raw_source"] = result["raw_source"]
            entry["web_agent_needed"] = rd["route"] == "both"

            # agent1 trace가 비어 있을 때(skip_agent1=True였던 경우)만 사전 추출 trace를 주입
            if not agent_trace.get("agent1_claim_extractor") and rc.get("_agent1_trace"):
                agent_trace["agent1_claim_extractor"] = rc["_agent1_trace"]

            # 디버그 trace 누적: claim_id와 에이전트별 response 원본을 함께 저장
            debug_traces.append({"claim_id": claim_id, "agent_trace": agent_trace})

            print(f"  {claim_id}: [완료] {result['highlight'][:60]}...")
            if rd["route"] == "both":
                # both인 경우 사내 문서 검색 후 웹 검색도 추가로 필요
                print(f"  {claim_id}: [both] 웹 에이전트 추가 검색도 필요합니다.")

        except FallbackRequired as e:
            # 사내 문서에서 찾지 못한 경우 웹 에이전트로 이관
            entry["status"] = "fallback"
            entry["highlight"] = None
            entry["fallback_reason"] = str(e)
            entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "agent_trace": None, "fallback_reason": str(e)})
            print(f"  {claim_id}: [fallback] 웹 에이전트로 이관 → {e}")

        except Exception as e:
            # 예기치 못한 오류
            entry["status"] = "error"
            entry["highlight"] = None
            entry["error"] = str(e)
            entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "agent_trace": None, "error": str(e)})
            print(f"  {claim_id}: [오류] {e}")

        final_results.append(entry)

    # ── 5) 결과 저장 ─────────────────────────────────────────────────────────────
    print()
    print("[Step 3] 결과 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_results(final_results, OUTPUT_DIR, timestamp)
    debug_path = save_debug_trace(debug_traces, OUTPUT_DIR, timestamp)
    print(f"결과 저장:  {output_path}")
    print(f"디버그 저장: {debug_path}")

    # ── 6) 최종 요약 출력 ────────────────────────────────────────────────────────
    success_count = sum(1 for r in final_results if r.get("status") == "success")
    fallback_count = sum(1 for r in final_results if r.get("status") == "fallback")
    web_only_count = sum(1 for r in final_results if r.get("status") == "web_only")
    error_count = sum(1 for r in final_results if r.get("status") == "error")

    print()
    print("=" * 60)
    print("최종 요약")
    print(f"  전체 claim: {len(final_results)}개")
    print(f"  성공 (highlight 생성): {success_count}개")
    print(f"  fallback (웹 이관):    {fallback_count}개")
    print(f"  web only:              {web_only_count}개")
    print(f"  오류:                  {error_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()
