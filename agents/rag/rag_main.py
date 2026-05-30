"""
rag_main.py

실행 진입점: python rag_main.py

[입력 모드]
  1) 인라인 텍스트 모드 (기본): 아래 INPUT_TEXT에 문장을 직접 작성하여 실행
  2) JSON 파일 모드: INPUT_TEXT를 비워두면 INPUT_PATH의 JSON 파일을 로드

[환경변수]
  MAX_CLAIMS=N   → 처음 N개 claim만 처리 (JSON 파일 모드 한정)
  VERBOSE=false  → 에이전트 상세 로그 숨김

검색 경로(web/internal) 결정은 상위 오케스트레이터에서 처리한다.
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


# ─── 모듈 임포트 ────────────────────────────────────────────────────────────────
from claim_router_validation import run_router_pipeline, extract_claim_from_text
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

def save_results(payload: dict, output_dir: Path, timestamp: str) -> Path:
    """
    파이프라인 전체 결과를 터미널 출력 순서(step1/step2/step3)와 동일한 구조의 JSON으로 저장한다.

    :param payload: step1/step2/step3 키를 포함한 전체 결과 dict
    :param output_dir: 출력 폴더 경로
    :param timestamp: 파일명에 사용할 타임스탬프 문자열
    :return: 저장된 파일 경로
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rag_output_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
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

def _build_claims_from_text(text: str) -> tuple[list, dict]:
    """
    단일 문장 입력을 claim 1개짜리 리스트로 변환한다.
    run_router_pipeline으로 claim 추출을 수행한다.

    :return: (claims 리스트, step1 출력 dict)
    """
    print("[Step 1] claim 추출 중...")
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


def main() -> None:
    """
    전체 파이프라인을 실행한다.

    처리 순서:
      1. 입력 준비 (인라인 텍스트 or JSON 파일)
      2. claim 추출 (인라인 텍스트 모드 한정)
      3. 사내 문서 검색 + 하이라이트
      4. fallback 발생 시 웹 에이전트 이관 안내 출력
      5. 결과를 JSON 파일로 저장
    """
    input_mode = "인라인 텍스트" if INPUT_TEXT.strip() else "JSON 파일"

    print("=" * 60)
    print("RAG Main Pipeline 시작")
    print(f"입력 모드: {input_mode} | VERBOSE={VERBOSE}")
    print("=" * 60)

    # ── 1) 입력 준비 ────────────────────────────────────────────────────────────
    step1_output: dict = {}

    if INPUT_TEXT.strip():
        # 인라인 텍스트 모드: 파일 없이 바로 실행
        print(f"입력 문장: {INPUT_TEXT.strip()}")
        print()
        claims, step1_output = _build_claims_from_text(INPUT_TEXT.strip())
        step1_input_text = INPUT_TEXT.strip()
    else:
        # JSON 파일 모드: claim_extractor_output.json 로드
        print(f"입력 파일: {INPUT_PATH}")
        data = load_input(INPUT_PATH)
        claims = data["claims"]
        qw = data.get("input", {}).get("qw", "")
        step1_input_text = qw

        if MAX_CLAIMS > 0:
            claims = claims[:MAX_CLAIMS]

        print(f"qw: {qw}")
        print(f"처리할 claim 수: {len(claims)}")
        print()

    # ── 2) 사내 문서 검색 + 하이라이트 생성 ─────────────────────────────────────
    print("[Step 2] 사내 문서 검색 및 하이라이트 생성 중...")
    print("-" * 40)

    step2_claims = []
    debug_traces = []  # 디버그용: claim별 OpenAI SDK response 원본 누적
    for rc in claims:
        claim_id = rc.get("claim_id", "?")

        qk = rc.get("claim") or rc.get("normalized_claim") or rc.get("original_text", "")
        pre_claim = rc.get("normalized_claim") or rc.get("claim") or None
        pre_keywords = rc.get("keywords") or None
        # 사전 추출된 claim/keywords가 있으면 rag_extractor 내부의 Agent 1을 건너뜀
        skip_agent1 = bool(pre_claim) and bool(pre_keywords)

        print(f"\n  {claim_id}: 사내 문서 검색 시작...")

        claim_entry: dict = {"claim_id": claim_id}

        try:
            result, agent_trace, agent_steps = run_rag_extractor(
                qk=qk,
                claim=pre_claim if skip_agent1 else None,
                keywords=pre_keywords if skip_agent1 else None,
                verbose=VERBOSE,
            )

            # agent1 trace가 비어 있을 때(skip_agent1=True) 사전 추출 trace 주입
            if not agent_trace.get("agent1_claim_extractor") and rc.get("_agent1_trace"):
                agent_trace["agent1_claim_extractor"] = rc["_agent1_trace"]

            claim_entry["status"] = "success"
            claim_entry["agents"] = {
                "claim_extractor": agent_steps["claim_extractor"],
                "folder_router": agent_steps["folder_router"],
                "search_highlight": agent_steps["search_highlight"],
            }
            claim_entry["highlight"] = result["highlight"]
            claim_entry["source_file"] = result["source_file"]
            claim_entry["source_page"] = result["source_page"]

            debug_traces.append({"claim_id": claim_id, "agent_trace": agent_trace})
            print(f"  {claim_id}: [완료] {result['highlight'][:60]}...")

        except FallbackRequired as e:
            claim_entry["status"] = "fallback"
            claim_entry["fallback_reason"] = str(e)
            claim_entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "agent_trace": None, "fallback_reason": str(e)})
            print(f"  {claim_id}: [fallback] 웹 에이전트로 이관 → {e}")

        except Exception as e:
            claim_entry["status"] = "error"
            claim_entry["error"] = str(e)
            claim_entry["web_agent_needed"] = True
            debug_traces.append({"claim_id": claim_id, "agent_trace": None, "error": str(e)})
            print(f"  {claim_id}: [오류] {e}")

        step2_claims.append(claim_entry)

    # ── 3) 결과 저장 ─────────────────────────────────────────────────────────────
    success_count = sum(1 for r in step2_claims if r.get("status") == "success")
    fallback_count = sum(1 for r in step2_claims if r.get("status") == "fallback")
    error_count = sum(1 for r in step2_claims if r.get("status") == "error")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_payload = {
        "pipeline": "RAG Main Pipeline",
        "input_mode": input_mode,
        "step1": {
            "label": "claim 추출",
            "agent": "claim_router_validation / ClaimExtractionAgent",
            "role": "자연어 입력에서 검색용 claim, normalized_claim, 키워드, 주제어 추출",
            "input": step1_input_text,
            "output": step1_output,
        },
        "step2": {
            "label": "사내 문서 검색 및 하이라이트 생성",
            "claims": step2_claims,
        },
        "step3": {
            "label": "최종 요약",
            "total": len(step2_claims),
            "success": success_count,
            "fallback": fallback_count,
            "error": error_count,
        },
    }

    print()
    print("[Step 3] 결과 저장 중...")
    output_path = save_results(output_payload, OUTPUT_DIR, timestamp)
    debug_path = save_debug_trace(debug_traces, OUTPUT_DIR, timestamp)
    print(f"결과 저장:  {output_path}")
    print(f"디버그 저장: {debug_path}")

    # ── 4) 최종 요약 출력 ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("최종 요약")
    print(f"  전체 claim: {len(step2_claims)}개")
    print(f"  성공 (highlight 생성): {success_count}개")
    print(f"  fallback (웹 이관):    {fallback_count}개")
    print(f"  오류:                  {error_count}개")
    print("=" * 60)


if __name__ == "__main__":
    main()
