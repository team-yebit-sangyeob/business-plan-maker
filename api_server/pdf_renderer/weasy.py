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

# 본문 스타일 — 미니멀 모노크롬 보고서. 번들 한글 폰트 + 흑/회 1포인트 위계, 강조 컬러 없음.
# 디자인 토큰: 본문 #2a2a2a / 강조선·헤딩 #111 / 보조·푸터 #999 / 옅은 규칙선 #e2e2e2.
# 페이지 하단(@page 마진 박스)에 문서명(좌)·페이지번호(우)를 모든 장에 자동으로 찍는다.
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
  margin: 24mm 20mm 20mm;
  @bottom-left {{
    content: "사업 계획서";
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 8pt;
    color: #999;
  }}
  @bottom-right {{
    content: counter(page) " / " counter(pages);
    font-family: 'Noto Sans KR', sans-serif;
    font-size: 8pt;
    color: #999;
  }}
}}
* {{ box-sizing: border-box; }}
body {{
  font-family: 'Noto Sans KR', sans-serif;
  font-size: 10.5pt;
  line-height: 1.7;
  color: #2a2a2a;
}}

/* 헤더 밴드 — 제목 + 메타 한 줄, 아래 굵은 흑선으로 본문과 분리 */
h1 {{
  font-size: 24pt;
  font-weight: 700;
  letter-spacing: -0.3pt;
  color: #111;
  margin: 0 0 5pt;
}}
h1 + p {{                      /* 생성·버전 메타 줄 */
  margin: 0;
  padding-bottom: 14pt;
  border-bottom: 2px solid #111;
}}

/* 챕터(및 요약·출처·정정이력) — 흑선 밑줄, 위 여백 넉넉히, 페이지 끝 고아 방지 */
h2 {{
  font-size: 13pt;
  font-weight: 700;
  letter-spacing: -0.2pt;
  color: #111;
  margin: 26pt 0 10pt;
  padding-bottom: 5pt;
  border-bottom: 1.5px solid #111;
  break-after: avoid;
}}

/* 슬롯 제목 */
h3 {{
  font-size: 10.5pt;
  font-weight: 700;
  color: #111;
  margin: 13pt 0 3pt;
  break-after: avoid;
}}
/* 슬롯 핵심 값 — 제목 바로 다음 문단. 좌측 흑선 마커로 본문 서술과 구분 */
h3 + p {{
  margin: 3pt 0 4pt;
  padding-left: 9pt;
  border-left: 2.5px solid #111;
  font-weight: 500;
  color: #111;
  break-inside: avoid;
}}

p {{ margin: 4pt 0; }}
em {{ color: #999; font-style: normal; font-size: 9pt; letter-spacing: 0.2pt; }}
ul, ol {{ margin: 5pt 0; padding-left: 18pt; }}
li {{ margin: 2pt 0; }}

/* 표 — 가로 규칙선만(모던 보고서 톤). 헤더는 흑선으로 강조 */
table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 9.5pt; }}
th, td {{ padding: 6pt 8pt; text-align: left; border-bottom: 1px solid #e2e2e2; }}
th {{ border-bottom: 1.5px solid #111; font-weight: 700; }}

a {{ color: #111; }}
"""


def render_pdf(markdown: str) -> RenderedPdf:
    """계획서 마크다운 → 한글 PDF. document.pages 로 실제 페이지 수를 센다."""
    body = md.markdown(markdown, extensions=["extra", "sane_lists"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{_CSS}</style></head><body>{body}</body></html>"
    document = HTML(string=html).render()
    pdf_bytes = document.write_pdf()
    return RenderedPdf(pdf_bytes=pdf_bytes, pages=len(document.pages))
