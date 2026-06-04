"""PDF 렌더 — 실 파이프라인 (12장 Block 2). markdown → HTML → WeasyPrint → PDF.

계획서 마크다운(agents/planner.compose_markdown 산출)을 사람이 읽는 PDF로 굳힌다:
  1) python-markdown 으로 MD → HTML 본문(표·들여쓰기 목록 포함)
  2) 번들 Noto Sans KR(@font-face)로 한글 글리프를 보장하는 HTML 템플릿에 본문 삽입
  3) WeasyPrint 로 HTML+CSS → PDF, document.pages 로 실제 페이지 수 산출

폰트를 시스템에 의존하지 않고 번들(fonts/)에서 직접 읽어 환경 간 결과를 일정하게 만든다.
"""
from __future__ import annotations

from pathlib import Path

import markdown as md
from weasyprint import HTML

from api_server.pdf_renderer.stub import RenderedPdf

_FONTS = Path(__file__).resolve().parent / "fonts"
_REGULAR = (_FONTS / "NotoSansKR-Regular.ttf").as_uri()
_BOLD = (_FONTS / "NotoSansKR-Bold.ttf").as_uri()

# 본문 스타일 — 번들 한글 폰트 등록 + 사업 계획서 위계(제목/섹션/표/출처)에 맞춘 기본 타이포.
_CSS = f"""
@font-face {{
  font-family: 'Noto Sans KR';
  font-weight: 400;
  src: url('{_REGULAR}');
}}
@font-face {{
  font-family: 'Noto Sans KR';
  font-weight: 700;
  src: url('{_BOLD}');
}}
@page {{
  size: A4;
  margin: 22mm 20mm;
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: 'Noto Sans KR', sans-serif;
  font-size: 10.5pt;
  line-height: 1.65;
  color: #1a1a1a;
}}
h1 {{ font-size: 21pt; margin: 0 0 4pt; }}
h2 {{
  font-size: 14pt;
  margin: 22pt 0 8pt;
  padding-bottom: 4pt;
  border-bottom: 1px solid #ccc;
}}
h3 {{ font-size: 11.5pt; margin: 14pt 0 2pt; color: #333; }}
p {{ margin: 4pt 0; }}
em {{ color: #666; font-style: normal; font-size: 9pt; }}
ul, ol {{ margin: 4pt 0; padding-left: 18pt; }}
li {{ margin: 2pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
th, td {{ border: 1px solid #ccc; padding: 4pt 6pt; text-align: left; }}
a {{ color: #1a1a1a; }}
"""


def render_pdf(markdown: str) -> RenderedPdf:
    """계획서 마크다운 → 한글 PDF. document.pages 로 실제 페이지 수를 센다."""
    body = md.markdown(markdown, extensions=["extra", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>{body}</body></html>"
    document = HTML(string=html).render()
    pdf_bytes = document.write_pdf()
    return RenderedPdf(pdf_bytes=pdf_bytes, pages=len(document.pages))
