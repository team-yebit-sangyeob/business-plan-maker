from __future__ import annotations

"""
documents/ 하위 폴더별로 PDF를 읽어 Chroma 벡터 DB를 생성한다.
  - chroma_db/ 존재
  - build_manifest.json 존재
  - manifest의 PDF 목록(파일명/mtime/size), embedding_model, chunk 설정이 현재와 동일
  => 동일하면 재생성하지 않고 SKIP
"""

import json
import os
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL_DEFAULT = "text-embedding-3-small"
CHUNK_CONFIG_DEFAULT = {"chunk_size": 1000, "chunk_overlap": 200}
CHROMA_DIRNAME = "chroma_db"
MANIFEST_FILENAME = "build_manifest.json"
KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class PdfLoadFailure:
    pdf_path: Path
    error_type: str
    error_message: str
    traceback: str


def _repo_root_from_this_file() -> Path:
    # infra/vector_db/src/build_chroma_db.py 기준으로 repo 루트는 parents[3]
    return Path(__file__).resolve().parents[3]


def _vector_db_root_from_this_file() -> Path:
    # infra/vector_db/src/build_chroma_db.py 기준으로 vector_db 루트는 parents[1]
    return Path(__file__).resolve().parents[1]


def load_openai_api_key(env_path: Path | None = None) -> str:
    """
    `.env`에서 `OPENAI_API_KEY`를 로드한다.
    vector_db 디렉터리에서 `../../.env`를 로드하며, 이는 repo 루트의 `.env`를 의미한다.
    """
    if env_path is None:
        env_path = _repo_root_from_this_file() / ".env"
    _load_env_file(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY not found after loading env file: {env_path}")
    return api_key


def _load_env_file(env_path: Path) -> None:
    """
    간단한 `.env` 로더 (python-dotenv 없이 동작).
    - `KEY=VALUE` 형태만 지원
    - 이미 설정된 환경 변수는 덮어쓰지 않음
    """
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")  # 단순 따옴표 제거
        if not key:
            continue
        if os.getenv(key) is None:
            os.environ[key] = value


def get_pdf_files(folder: Path) -> list[Path]:
    # 지정 폴더 바로 아래의 PDF만 대상으로 한다(재귀 탐색 X).
    if not folder.exists() or not folder.is_dir():
        return []
    pdfs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    pdfs.sort(key=lambda p: p.name.lower())
    return pdfs


def _pdf_file_fingerprint(pdf_path: Path) -> dict[str, Any]:
    # 최신 여부 비교를 위해 파일의 수정시간/크기를 함께 기록한다.
    stat = pdf_path.stat()
    mtime_dt_kst = datetime.fromtimestamp(stat.st_mtime, tz=KST)

    # manifest에는 documents/ 이하 상대경로만 기록한다. (예: etc/foo.pdf)
    relative_path = _relative_path_under_documents(pdf_path)
    return {
        "file_name": pdf_path.name,
        "relative_path": relative_path,
        "size_bytes": stat.st_size,
        "mtime_ns": int(stat.st_mtime_ns),
        "mtime_kst": mtime_dt_kst.isoformat(),
    }


def _relative_path_under_documents(pdf_path: Path) -> str:
    """
    pdf_path가 documents/ 하위에 있을 때, documents/ 기준 상대경로를 반환한다.
    예) .../documents/etc/a.pdf -> etc/a.pdf
    """
    resolved = pdf_path.resolve()

    # 1) 경로 상의 'documents' 디렉터리를 찾아 상대경로 계산
    for parent in resolved.parents:
        if parent.name.lower() == "documents":
            try:
                return resolved.relative_to(parent).as_posix()
            except Exception:
                break

    # 2) 스크립트 기준의 vector_db/documents로도 시도
    docs_root = (_vector_db_root_from_this_file() / "documents").resolve()
    try:
        return resolved.relative_to(docs_root).as_posix()
    except Exception:
        # 마지막 방어: 파일명만
        return resolved.name


def build_manifest(
    pdf_files: list[Path],
    embedding_model: str,
    chunk_config: dict[str, Any],
    *,
    failures: list[PdfLoadFailure] | None = None,
) -> dict[str, Any]:
    # 폴더별 DB 최신 여부 판단과, 생성 결과 추적을 위한 manifest를 만든다.
    pdf_entries = [_pdf_file_fingerprint(p) for p in pdf_files]
    pdf_entries.sort(key=lambda e: str(e["file_name"]).lower())

    manifest: dict[str, Any] = {
        "version": 1,
        "built_at_kst": datetime.now(KST).isoformat(),
        "embedding_model": embedding_model,
        "chunk_config": {
            "splitter": "RecursiveCharacterTextSplitter",
            "chunk_size": int(chunk_config["chunk_size"]),
            "chunk_overlap": int(chunk_config["chunk_overlap"]),
        },
        "pdf_files": pdf_entries,
        "failures": [
            {
                "file_name": f.pdf_path.name,
                "relative_path": f.pdf_path.as_posix(),
                "error_type": f.error_type,
                "error_message": f.error_message,
            }
            for f in (failures or [])
        ],
    }
    return manifest


def _manifest_for_freshness(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """
    최신(fresh) 비교만을 위해 manifest를 정규화한다.
    - `built_at_*`는 비교에서 제외
    """
    normalized: dict[str, Any] = dict(manifest)
    normalized.pop("built_at_utc", None)  # 과거 버전 호환
    normalized.pop("built_at_kst", None)
    normalized.pop("failures", None)

    # 요구사항 기준: PDF 파일명/수정시간/크기만 비교 대상으로 삼는다.
    pdf_files = normalized.get("pdf_files") or []
    compact_entries: list[dict[str, Any]] = []
    for e in pdf_files:
        if not isinstance(e, dict):
            continue
        compact_entries.append(
            {
                "file_name": e.get("file_name"),
                "size_bytes": e.get("size_bytes"),
                "mtime_ns": e.get("mtime_ns"),
            }
        )
    compact_entries.sort(key=lambda x: str(x.get("file_name") or "").lower())
    normalized["pdf_files"] = compact_entries
    return normalized


def is_vector_db_fresh(folder: Path, manifest: dict[str, Any]) -> bool:
    # 디스크에 있는 build_manifest.json과 현재 계산한 manifest가 동일하면 fresh로 간주한다.
    chroma_dir = folder / CHROMA_DIRNAME
    manifest_path = chroma_dir / MANIFEST_FILENAME
    if not chroma_dir.exists() or not chroma_dir.is_dir():
        return False
    if not manifest_path.exists() or not manifest_path.is_file():
        return False

    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    # If previous run had failures, treat as stale so we can retry.
    if existing.get("failures"):
        return False

    return _manifest_for_freshness(existing) == _manifest_for_freshness(manifest)


def load_and_split_pdfs(pdf_files: list[Path], folder_name: str) -> tuple[list[Document], list[PdfLoadFailure]]:
    # PDF를 로드하고, 페이지 단위를 chunk로 쪼갠 뒤 metadata를 보강한다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(CHUNK_CONFIG_DEFAULT["chunk_size"]),
        chunk_overlap=int(CHUNK_CONFIG_DEFAULT["chunk_overlap"]),
    )

    all_docs: list[Document] = []
    failures: list[PdfLoadFailure] = []
    chunk_id = 0

    for pdf_path in pdf_files:
        try:
            # temp_vb.ipynb 검증 흐름: PyPDFLoader -> load() -> split_documents()
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            splits = splitter.split_documents(pages)

            for d in splits:
                # 최소 요구 metadata를 강제로 채워 넣는다.
                source = d.metadata.get("source") or str(pdf_path)
                page = d.metadata.get("page")
                d.metadata = {
                    **d.metadata,
                    "source": source,
                    "file_name": pdf_path.name,
                    "folder": folder_name,
                    "page": page,
                    "chunk_id": chunk_id,
                    "embedding_model": EMBEDDING_MODEL_DEFAULT,
                }
                chunk_id += 1
                all_docs.append(d)
        except Exception as e:
            # 일부 PDF가 실패해도 전체 작업은 계속 진행한다.
            failures.append(
                PdfLoadFailure(
                    pdf_path=pdf_path,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback=traceback.format_exc(),
                )
            )
            continue

    return all_docs, failures


def _safe_rmtree(path: Path) -> None:
    # 폴더 단위 재생성을 위해 기존 chroma_db/를 안전하게 제거한다.
    if not path.exists():
        return
    shutil.rmtree(path)


def build_chroma_for_folder(folder: Path, force: bool = False) -> dict[str, Any]:
    """
    documents/ 아래의 단일 하위 폴더에 대해 Chroma DB를 생성(또는 스킵)한다.
    - `<folder>/chroma_db/`를 생성
    - `<folder>/chroma_db/build_manifest.json`를 기록
    """
    folder = folder.resolve()
    folder_name = folder.name

    # 요구사항: API 키가 없으면 즉시 중단(폴더 단위 호출에서도 동일)
    load_openai_api_key()

    print(f"\n=== FOLDER START: {folder_name} ===")
    print(f"- path: {folder}")

    pdf_files = get_pdf_files(folder)
    if not pdf_files:
        print(f"- pdf_count: 0 -> EMPTY (skip)")
        return {"folder": str(folder), "status": "EMPTY", "pdf_count": 0}

    embedding_model = EMBEDDING_MODEL_DEFAULT
    chunk_config = dict(CHUNK_CONFIG_DEFAULT)

    chroma_dir = folder / CHROMA_DIRNAME
    manifest = build_manifest(pdf_files, embedding_model, chunk_config)

    if not force and is_vector_db_fresh(folder, manifest):
        print(f"- pdf_count: {len(pdf_files)}")
        print(f"- status: SKIP (fresh)")
        return {"folder": str(folder), "status": "SKIP", "pdf_count": len(pdf_files)}

    if chroma_dir.exists():
        print(f"- pdf_count: {len(pdf_files)}")
        print(f"- status: REBUILD (stale)")
        _safe_rmtree(chroma_dir)
    else:
        print(f"- pdf_count: {len(pdf_files)}")
        print(f"- status: BUILD")

    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings(model=embedding_model)

    print(f"- embedding_model: {embedding_model}")
    print(f"- chunk_config: size={chunk_config['chunk_size']}, overlap={chunk_config['chunk_overlap']}")
    print("- loading PDFs and splitting...")
    documents, failures = load_and_split_pdfs(pdf_files, folder_name)
    if not documents:
        print("- doc_chunks: 0 -> FAILED (no documents)")
        # Nothing to embed; keep chroma_dir clean.
        _safe_rmtree(chroma_dir)
        return {
            "folder": str(folder),
            "status": "FAILED",
            "pdf_count": len(pdf_files),
            "doc_chunks": 0,
            "failures": [f.pdf_path.name for f in failures],
        }

    print(f"- doc_chunks: {len(documents)}")
    print("- building chroma DB (embedding + persist)...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
    )

    # manifest 기록: 실패가 있으면 함께 저장하고, 다음 실행에서는 fresh로 보지 않도록 처리
    manifest = build_manifest(pdf_files, embedding_model, chunk_config, failures=failures)
    (chroma_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"- manifest_written: {chroma_dir / MANIFEST_FILENAME}")

    result: dict[str, Any] = {
        "folder": str(folder),
        "status": "OK" if not failures else "PARTIAL",
        "pdf_count": len(pdf_files),
        "doc_chunks": len(documents),
        "chroma_dir": str(chroma_dir),
        "failure_count": len(failures),
    }
    if failures:
        print(f"- failure_count: {len(failures)} (partial)")
        result["failures"] = [f.pdf_path.name for f in failures]
    else:
        print("- failure_count: 0")

    print(f"=== FOLDER END: {folder_name} ({result['status']}) ===")
    return result


def build_all_chroma(document_root: Path = Path("documents"), force: bool = False) -> list[dict[str, Any]]:
    """
    documents/ 아래의 각 하위 폴더를 순회하며 폴더별 Chroma DB를 생성한다.
    """
    # 요구사항: API 키가 없으면 즉시 중단(전체 실행 시작 시점에 체크)
    load_openai_api_key()

    doc_root = document_root
    if not doc_root.is_absolute():
        # 작업 디렉터리가 어디든 안정적으로 동작하도록 vector_db/documents 기준으로 해석
        candidate = _vector_db_root_from_this_file() / doc_root
        doc_root = candidate

    if not doc_root.exists():
        raise FileNotFoundError(f"document_root not found: {doc_root}")

    results: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    folders = sorted([p for p in doc_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    print(f"DOCUMENT ROOT: {doc_root}")
    print(f"FOLDER COUNT: {len(folders)}")
    print(f"FORCE: {force}\n")

    for idx, folder in enumerate(folders, start=1):
        print(f"\n[{idx}/{len(folders)}] processing: {folder.name}")
        try:
            res = build_chroma_for_folder(folder, force=force)
            results.append(res)
            if res.get("status") in {"FAILED", "PARTIAL"} and res.get("failures"):
                all_failures.append({"folder": str(folder), "files": res["failures"]})
        except Exception as e:
            print(f"- status: ERROR ({type(e).__name__})")
            results.append(
                {
                    "folder": str(folder),
                    "status": "ERROR",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            all_failures.append({"folder": str(folder), "files": ["<folder_error>"]})
            continue

    # 터미널에서 한눈에 볼 수 있는 요약 출력
    status_counts: dict[str, int] = {}
    for r in results:
        status = str(r.get("status", "UNKNOWN"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nSummary:")
    for k in sorted(status_counts.keys()):
        print(f"- {k}: {status_counts[k]}")

    if all_failures:
        print("\nFailures:")
        for f in all_failures:
            print(f"- {f['folder']}: {', '.join(f['files'])}")

    return results


if __name__ == "__main__":
    # vector_db 루트/ repo 루트 어디서 실행해도 infra/vector_db/documents로 해석된다.
    build_all_chroma(force=bool(os.getenv("FORCE_REBUILD")))
