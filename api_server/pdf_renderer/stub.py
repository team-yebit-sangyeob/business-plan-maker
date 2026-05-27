"""PDF 렌더 — stub. 실 파이프라인(WeasyPrint·markdown-it·PyMuPDF)은 후순위 (12장 Block 2)."""
from __future__ import annotations

from dataclasses import dataclass


# 최소한의 유효한 PDF 한 페이지 (placeholder).
_MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n"
    b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>endobj\n"
    b"xref\n0 4\n"
    b"0000000000 65535 f \n"
    b"0000000010 00000 n \n"
    b"0000000053 00000 n \n"
    b"0000000102 00000 n \n"
    b"trailer<< /Size 4 /Root 1 0 R >>\n"
    b"startxref\n160\n%%EOF\n"
)


@dataclass
class RenderedPdf:
    pdf_bytes: bytes
    pages: int


def render_pdf(markdown: str) -> RenderedPdf:
    return RenderedPdf(pdf_bytes=_MINIMAL_PDF, pages=1)
