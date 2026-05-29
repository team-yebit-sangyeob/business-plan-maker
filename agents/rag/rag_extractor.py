"""
rag_extractor.py

사내 문서 벡터 DB에서 claim에 대한 근거를 찾아 하이라이트를 생성하는 에이전트 모듈.

파이프라인:
  1) ClaimExtractorAgent  : qk(원문)에서 claim + 키워드 추출
  2) FolderRouterAgent    : read_directory_map 툴로 적합한 폴더 선택
  3) SearchHighlightAgent : search_vector_db 툴로 청크 검색 + 하이라이트 작성
  fallback               : 제한 횟수 내에 유효한 청크를 못 찾으면 FallbackRequired 발생

참고 파일: exp_openai_sdk.ipynb (폴더 서치 + 하이라이트 - 에이전트 & 툴 구현)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# experiments 루트의 .env 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# ─── 공통 설정 ─────────────────────────────────────────────────────────────────
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# agents/rag 기준 두 단계 위가 experiments 루트
_BASE_DIR = Path(__file__).parent.parent.parent

# 벡터DB 루트 폴더 (환경변수로 재지정 가능)
DOCS_BASE_PATH: Path = Path(os.getenv("DOCS_BASE_PATH", str(_BASE_DIR / "infra" / "vector_db" / "documents")))

# 폴더 구조·신뢰도 계층 정보 파일 경로
DIRECTORY_MAP_PATH: Path = Path(
    os.getenv("DIRECTORY_MAP_PATH", str(_BASE_DIR / "infra" / "vector_db" / "documents_info.md"))
)

# SearchHighlightAgent의 최대 도구 호출 횟수 (fallback 기준)
MAX_AGENT_TURNS: int = int(os.getenv("MAX_AGENT_TURNS", "12"))

# 검색 결과에서 제외할 최소 청크 길이 (자)
MIN_CHUNK_LEN: int = int(os.getenv("MIN_CHUNK_LEN", "50"))

# 벡터DB 폴더 목록 (신뢰도 계층 순서)
ALL_FOLDERS: List[str] = ["paper", "report", "proposal", "etc"]


# ─── 결과 타입 정의 ─────────────────────────────────────────────────────────────

class RagExtractorResult(TypedDict):
    """run_rag_extractor의 최종 반환 타입"""
    qk: str                     # 입력 원문
    claim: str                  # 추출된 claim
    keywords: List[str]         # 검색 키워드
    folders_searched: List[str] # 실제 검색한 폴더 목록
    highlight: str              # 생성된 하이라이트 (100자 내외)
    highlight_reason: str       # 해당 청크를 선택한 이유
    keyword_used: str           # 최종적으로 사용한 키워드
    folder_searched: str        # 최종적으로 사용한 폴더
    source_file: str            # 출처 파일명
    source_page: str            # 출처 페이지
    raw_source: str             # 선별된 원문 청크 전체


class FallbackRequired(Exception):
    """
    사내 문서에서 유효한 청크를 찾지 못했을 때 발생하는 예외.
    호출자(rag_main.py)에서 캐치하여 웹 에이전트로 이관한다.
    """
    pass


# ─── 툴 구현 ────────────────────────────────────────────────────────────────────

def _tool_read_directory_map() -> str:
    """
    벡터DB의 폴더 구조와 신뢰도 계층 정보를 담은 directory_map 파일을 읽어 반환한다.
    FolderRouterAgent가 폴더를 선택하기 전에 반드시 호출해야 한다.
    """
    if not DIRECTORY_MAP_PATH.exists():
        # 파일이 없는 경우 기본 폴더 목록 안내로 대체
        return (
            "directory_map.txt 파일을 찾을 수 없습니다.\n"
            f"사용 가능한 기본 폴더: {', '.join(ALL_FOLDERS)}\n"
            "신뢰도 계층: paper > report > proposal > etc"
        )
    return DIRECTORY_MAP_PATH.read_text(encoding="utf-8")


def _tool_search_vector_db(folder: str, keyword: str, min_len: int = MIN_CHUNK_LEN) -> dict:
    """
    지정 폴더의 Chroma 벡터DB에서 keyword로 유사도 검색을 실행한다.
    min_len 미만 청크는 노이즈로 간주해 자동 제외한다.

    :param folder: 검색할 폴더명 (paper/report/proposal/etc)
    :param keyword: 검색 키워드
    :param min_len: 유효 청크의 최소 글자 수
    :return: {folder, keyword, valid_count, skipped_count, results, error(선택)}
    """
    db_path = str(DOCS_BASE_PATH / folder / "chroma_db")

    if not Path(db_path).exists():
        # DB 폴더가 없으면 fallback 신호를 포함한 결과 반환
        return {
            "folder": folder,
            "keyword": keyword,
            "valid_count": 0,
            "skipped_count": 0,
            "results": [],
            "error": "db_not_found",
        }

    # Chroma 벡터DB에서 유사도 검색 (상위 10개)
    vs = Chroma(
        persist_directory=db_path,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="langchain",
    )
    raw = vs.similarity_search_with_score(keyword, k=10)

    valid, skipped = [], 0
    for doc, score in raw:
        content = doc.page_content.strip()
        if len(content) < min_len:
            # 너무 짧은 청크는 의미 있는 근거가 되기 어려우므로 제외
            skipped += 1
            continue
        valid.append({
            "content": content[:800],  # 토큰 절약을 위해 800자로 자름
            "content_len": len(content),
            "score": round(float(score), 4),
            "file_name": doc.metadata.get("file_name", ""),
            "page_label": doc.metadata.get("page_label", ""),
            "folder": folder,
        })

    return {
        "folder": folder,
        "keyword": keyword,
        "valid_count": len(valid),
        "skipped_count": skipped,
        "results": valid,
    }


def _dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """
    에이전트가 요청한 툴 이름과 인수를 받아 실제 함수를 실행하고 결과 문자열을 반환한다.
    """
    if tool_name == "read_directory_map":
        return _tool_read_directory_map()
    if tool_name == "search_vector_db":
        result = _tool_search_vector_db(**tool_args)
        return json.dumps(result, ensure_ascii=False, indent=2)
    raise ValueError(f"알 수 없는 툴: {tool_name}")


# ─── 툴 스키마 ──────────────────────────────────────────────────────────────────

FOLDER_ROUTER_TOOLS = [
    {
        "type": "function",
        "name": "read_directory_map",
        "description": "벡터DB 폴더 구조와 각 폴더 설명, 신뢰도 계층(paper>report>proposal>etc)을 읽어옵니다.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }
]

SEARCH_HIGHLIGHT_TOOLS = [
    {
        "type": "function",
        "name": "search_vector_db",
        "description": (
            "지정 폴더의 Chroma DB에서 keyword로 청크를 검색합니다. "
            f"{MIN_CHUNK_LEN}자 미만 청크는 자동 제외됩니다. "
            "valid_count=0이거나 결과가 부족하면 다른 keyword나 folder로 재호출하세요."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "검색할 폴더명",
                    "enum": ALL_FOLDERS,
                },
                "keyword": {
                    "type": "string",
                    "description": "벡터DB에서 검색할 핵심 키워드",
                },
                "min_len": {
                    "type": "integer",
                    "description": f"유효 청크 최소 글자 수 (기본값: {MIN_CHUNK_LEN})",
                },
            },
            "required": ["folder", "keyword"],
            "additionalProperties": False,
        },
    }
]


# ─── Response 직렬화 헬퍼 ──────────────────────────────────────────────────────

def _serialize_response(response: Any) -> dict:
    """
    OpenAI SDK Response 객체를 JSON 저장 가능한 dict로 변환한다.
    model_dump()를 우선 시도하고, 실패하면 문자열 변환으로 폴백한다.
    """
    try:
        return response.model_dump()
    except Exception:
        return {"_raw": str(response)}


# ─── Agent 실행 헬퍼 ────────────────────────────────────────────────────────────

def _run_agent(
    system: str,
    tools: List[dict],
    user_msg: str,
    max_turns: int = MAX_AGENT_TURNS,
    verbose: bool = True,
) -> Tuple[str, List[dict]]:
    """
    OpenAI Responses API를 사용해 툴 호출 루프를 실행한다.
    previous_response_id로 멀티턴 대화 문맥을 유지한다.

    :param system: 에이전트 시스템 프롬프트
    :param tools: 에이전트에게 제공할 툴 스키마 목록
    :param user_msg: 최초 사용자 메시지
    :param max_turns: 최대 도구 호출 횟수 (초과 시 FallbackRequired 발생)
    :param verbose: True이면 각 턴의 툴 호출 정보를 출력
    :return: (최종 텍스트 응답, 각 턴의 직렬화된 response 목록)
    :raises FallbackRequired: max_turns 초과 시
    """
    previous_response_id: Optional[str] = None
    current_input = [{"role": "user", "content": user_msg}]
    # 각 턴에서 받은 response를 직렬화하여 누적 (디버그용)
    response_trace: List[dict] = []

    for turn in range(1, max_turns + 1):
        kwargs: Dict[str, Any] = {
            "model": MODEL,
            "instructions": system,
            "input": current_input,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            # 이전 응답 ID를 전달해 대화 문맥을 이어나감
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
        previous_response_id = response.id
        response_trace.append(_serialize_response(response))

        if verbose:
            print(f"  [turn {turn}] status={response.status}")

        # 툴 호출 처리: 에이전트가 tool_call을 요청하면 결과를 다음 턴 input으로 전달
        tool_outputs = []
        for item in response.output:
            if item.type == "function_call":
                args = json.loads(item.arguments)
                if verbose:
                    if item.name == "search_vector_db":
                        print(f"    → {item.name}(folder={args.get('folder')}, keyword={args.get('keyword')})")
                    else:
                        print(f"    → {item.name}()")
                result = _dispatch_tool(item.name, args)
                if verbose and item.name == "search_vector_db":
                    r = json.loads(result)
                    print(f"      valid={r['valid_count']}, skipped={r['skipped_count']}")
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": result,
                })

        # 툴 호출이 없으면 에이전트가 최종 응답을 완료한 것
        if not tool_outputs:
            return response.output_text, response_trace

        current_input = tool_outputs

    # 제한 횟수 안에 완료하지 못한 경우 웹 에이전트로 이관
    raise FallbackRequired(f"max_turns={max_turns} 초과: 사내 문서에서 적합한 청크를 찾지 못했습니다.")


# ─── Agent 1: ClaimExtractorAgent ──────────────────────────────────────────────

_CLAIM_EXTRACTOR_SYSTEM = """당신은 claim 추출 전문가입니다.

입력된 qk(원문 주장/지식)를 분석하여 다음을 수행하세요.

1. 검색에 적합하게 정제된 핵심 claim을 작성하세요 (원문의 핵심 주장을 명확한 서술문으로)
2. claim을 잘 반영하는 한국어 검색 키워드 3개를 추출하세요 (구체적이고 검색에 효과적인 단어)

반드시 아래 JSON 형식으로만 출력하세요:
{
  "claim": "정제된 핵심 주장 문장",
  "keywords": ["키워드1", "키워드2", "키워드3"]
}"""


def run_claim_extractor_agent(
    qk: str, verbose: bool = True
) -> Tuple[Dict[str, Any], List[dict]]:
    """
    ClaimExtractorAgent: qk(원문)에서 claim과 검색 키워드를 추출한다.
    툴을 사용하지 않으므로 단일 호출로 완료된다.

    :param qk: 원문 주장/지식 문자열
    :param verbose: True이면 실행 정보 출력
    :return: ({"claim": str, "keywords": List[str]}, [직렬화된 response])
    :raises ValueError: JSON 파싱 실패 시
    """
    if verbose:
        print("[ClaimExtractorAgent] claim 추출 중...")

    response = client.responses.create(
        model=MODEL,
        instructions=_CLAIM_EXTRACTOR_SYSTEM,
        input=[{"role": "user", "content": f"qk:\n{qk}"}],
    )
    # 단일 호출이므로 response를 리스트로 감싸 다른 에이전트와 형식을 통일
    response_trace = [_serialize_response(response)]

    text = response.output_text.strip()

    # JSON 블록이 코드 펜스로 감싸져 있을 경우 제거
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()

    result = json.loads(text)
    if verbose:
        print(f"  claim: {result.get('claim', '')[:60]}...")
        print(f"  keywords: {result.get('keywords', [])}")
    return result, response_trace


# ─── Agent 2: FolderRouterAgent ─────────────────────────────────────────────────

_FOLDER_ROUTER_SYSTEM = """당신은 RAG 검색 폴더 라우터입니다.

입력된 claim과 keywords를 분석하여 어느 폴더에서 검색할지 결정하세요.
반드시 read_directory_map 툴을 호출하여 폴더 구조와 신뢰도 계층을 확인한 뒤 판단하세요.

[결정 기준]
- 문서 유형과 claim의 성격을 매칭하세요 (학술 주장이면 paper, 통계/시장데이터면 report 등)
- 신뢰도 계층(paper > report > proposal > etc)을 고려하여 우선순위를 정하세요
- 관련성 없는 폴더는 제외하고, 1~4개 폴더를 우선순위 순으로 반환하세요

반드시 아래 JSON 형식으로만 출력하세요:
{
  "folders": ["folder1", "folder2"],
  "reason": "폴더 선택 이유 (간단히)"
}"""


def run_folder_router_agent(
    claim: str,
    keywords: List[str],
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[dict]]:
    """
    FolderRouterAgent: claim과 키워드를 기반으로 검색할 폴더 목록을 결정한다.
    read_directory_map 툴을 사용하여 실제 폴더 구조를 확인한 뒤 판단한다.

    :param claim: 추출된 핵심 claim 문장
    :param keywords: 검색 키워드 목록
    :param verbose: True이면 실행 정보 출력
    :return: ({"folders": List[str], "reason": str}, [직렬화된 response 목록])
    :raises FallbackRequired: max_turns 초과 시
    """
    if verbose:
        print("[FolderRouterAgent] 검색 폴더 결정 중...")

    user_msg = (
        f"claim: {claim}\n"
        f"keywords: {', '.join(keywords)}"
    )
    text, response_trace = _run_agent(
        system=_FOLDER_ROUTER_SYSTEM,
        tools=FOLDER_ROUTER_TOOLS,
        user_msg=user_msg,
        verbose=verbose,
    )

    # JSON 블록 파싱
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()

    result = json.loads(text)
    # 폴더 목록이 비어 있으면 전체 폴더를 기본값으로 사용
    if not result.get("folders"):
        result["folders"] = ALL_FOLDERS
    if verbose:
        print(f"  folders: {result.get('folders', [])}")
    return result, response_trace


# ─── Agent 3: SearchHighlightAgent ─────────────────────────────────────────────

_SEARCH_HIGHLIGHT_SYSTEM = f"""당신은 RAG 검색 및 하이라이트 생성 에이전트입니다.

입력된 claim, keywords, folders를 기반으로 아래 절차를 수행하세요.

[절차]
1. 제공된 folders 순서대로, keywords를 활용해 search_vector_db 툴을 호출하세요
2. 총 {MAX_AGENT_TURNS}회 이내로 검색하세요 (폴더×키워드 조합)
3. valid_count가 0이거나 결과가 부족하면 다른 키워드 또는 다음 폴더로 재시도하세요
4. 수집된 모든 청크 중 claim과 가장 관련성이 높은 청크 1개를 선별하세요
5. 선별된 청크를 근거로 100자 내외의 highlight를 작성하세요
6. 왜 그 청크를 선택했는지 이유(highlight_reason)도 함께 작성하세요

[최종 출력 - 반드시 JSON 형식]
{{
  "claim": "입력된 claim",
  "keyword_used": "최종 사용한 키워드",
  "folder_searched": "검색한 폴더명",
  "highlight": "100자 내외 하이라이트",
  "highlight_reason": "이 청크를 선택한 이유",
  "source_file": "출처 파일명",
  "source_page": "출처 페이지",
  "raw_source": "선별된 원문 청크 전체"
}}"""


def run_search_highlight_agent(
    claim: str,
    keywords: List[str],
    folders: List[str],
    verbose: bool = True,
) -> Tuple[Dict[str, Any], List[dict]]:
    """
    SearchHighlightAgent: 지정 폴더에서 claim에 대한 근거 청크를 찾아 하이라이트를 생성한다.
    처음에는 주어진 키워드와 폴더를 사용하고, 결과가 부족하면 직접 키워드와 폴더를 바꿔 재시도한다.

    :param claim: 추출된 핵심 claim 문장
    :param keywords: 검색 키워드 목록
    :param folders: FolderRouterAgent가 선택한 폴더 목록 (우선순위 순)
    :param verbose: True이면 실행 정보 출력
    :return: ({highlight 결과}, [직렬화된 response 목록])
    :raises FallbackRequired: max_turns 초과 또는 유효한 청크 없음
    """
    if verbose:
        print("[SearchHighlightAgent] 청크 검색 및 하이라이트 생성 중...")

    user_msg = (
        f"claim: {claim}\n"
        f"keywords: {', '.join(keywords)}\n"
        f"folders: {', '.join(folders)}"
    )
    text, response_trace = _run_agent(
        system=_SEARCH_HIGHLIGHT_SYSTEM,
        tools=SEARCH_HIGHLIGHT_TOOLS,
        user_msg=user_msg,
        verbose=verbose,
    )

    # JSON 블록 파싱
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.startswith("```")
        ).strip()

    result = json.loads(text)

    # 유효한 하이라이트가 없으면 fallback 처리
    if not result.get("highlight"):
        raise FallbackRequired("SearchHighlightAgent가 유효한 하이라이트를 생성하지 못했습니다.")

    return result, response_trace


# ─── 오케스트레이터: run_rag_extractor ─────────────────────────────────────────

def run_rag_extractor(
    qk: str,
    claim: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[RagExtractorResult, dict]:
    """
    qk(원문)를 입력받아 사내 벡터DB에서 근거를 찾고 하이라이트를 생성하는 전체 파이프라인을 실행한다.

    이 함수는 앞단에서 사내 문서 검색으로 판별된 claim에 대해서만 호출한다고 가정한다.
    유효한 근거를 찾지 못하면 FallbackRequired를 발생시켜 호출자가 웹 에이전트로 이관하도록 한다.

    :param qk: 검증할 원문 주장/지식 문자열
    :param claim: 이미 추출된 claim 문장 (None이면 Agent 1이 qk에서 직접 추출)
    :param keywords: 이미 추출된 검색 키워드 목록 (None 또는 빈 리스트이면 Agent 1이 추출)
    :param verbose: True이면 각 에이전트의 실행 정보 출력
    :return: (RagExtractorResult, agent_trace)
             agent_trace = {
               "agent1_claim_extractor": [response, ...],  # 건너뛴 경우 빈 리스트
               "agent2_folder_router":   [response, ...],
               "agent3_search_highlight":[response, ...],
             }
    :raises FallbackRequired: 적합한 청크를 찾지 못한 경우
    """
    # Step 1: ClaimExtractorAgent - claim + keywords 추출
    # claim과 keywords가 모두 제공된 경우(디버그 등) Agent 1을 건너뛴다
    if claim and keywords:
        if verbose:
            print("[ClaimExtractorAgent] 건너뜀 (사전 추출된 claim/keywords 사용)")
            print(f"  claim: {claim[:60]}...")
            print(f"  keywords: {keywords}")
        trace_agent1: List[dict] = []
    else:
        claim_data, trace_agent1 = run_claim_extractor_agent(qk, verbose=verbose)
        claim = claim_data["claim"]
        keywords = claim_data.get("keywords", [])

    # Step 2: FolderRouterAgent - 검색할 폴더 결정
    folder_data, trace_agent2 = run_folder_router_agent(claim, keywords, verbose=verbose)
    folders = folder_data.get("folders", ALL_FOLDERS)

    # Step 3: SearchHighlightAgent - 벡터DB 검색 + 하이라이트 생성
    # 선정된 폴더에서 먼저 검색하고, 부족하면 미사용 폴더도 자동으로 탐색
    unused_folders = [f for f in ALL_FOLDERS if f not in folders]
    all_folders_ordered = folders + unused_folders

    highlight_data, trace_agent3 = run_search_highlight_agent(
        claim=claim,
        keywords=keywords,
        folders=all_folders_ordered,
        verbose=verbose,
    )

    result = RagExtractorResult(
        qk=qk,
        claim=claim,
        keywords=keywords,
        folders_searched=all_folders_ordered,
        highlight=highlight_data.get("highlight", ""),
        highlight_reason=highlight_data.get("highlight_reason", ""),
        keyword_used=highlight_data.get("keyword_used", ""),
        folder_searched=highlight_data.get("folder_searched", ""),
        source_file=highlight_data.get("source_file", ""),
        source_page=highlight_data.get("source_page", ""),
        raw_source=highlight_data.get("raw_source", ""),
    )

    # 에이전트별 OpenAI SDK response 원본을 에이전트명으로 분류하여 반환
    agent_trace = {
        "agent1_claim_extractor": trace_agent1,
        "agent2_folder_router": trace_agent2,
        "agent3_search_highlight": trace_agent3,
    }

    return result, agent_trace
