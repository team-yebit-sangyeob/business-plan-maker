"""회사 RAG 클러스터 — 사내 벡터DB 근거 회수(retrieval).

실 파이프라인은 rag_extractor(claim추출→폴더라우팅→검색+하이라이트). worker가 그 결과를
오케스트레이터 seam(ValidationReport)으로 어댑트한다. 논리 검증(지지 여부 판정)은
logic_validator 워커가 후속으로 수행한다.
"""
from agents.rag.worker import run_rag_check

__all__ = ["run_rag_check"]
