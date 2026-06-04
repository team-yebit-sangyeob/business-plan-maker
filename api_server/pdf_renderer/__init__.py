"""PDF 렌더 진입점 — 실 파이프라인(weasy) 우선, 시스템 의존성 부재 시 stub 폴백.

WeasyPrint 는 pango/cairo/gdk-pixbuf 등 시스템 라이브러리를 요구한다. 미설치 환경에서도
서버 자체는 떠야 하므로, import 가 실패하면 빈 PDF stub 으로 폴백하고 경고를 남긴다
(이 경우 다운로드 PDF 는 빈 페이지가 된다 — 로그로 원인을 명확히 한다).
"""
import logging

logger = logging.getLogger(__name__)

try:
    from api_server.pdf_renderer.weasy import render_pdf
except Exception as exc:  # 시스템 라이브러리 부재 등 — 서버 기동은 막지 않는다.
    logger.warning(
        "WeasyPrint 로드 실패 → 빈 PDF stub 으로 폴백(다운로드 PDF가 빈 페이지가 됩니다). "
        "해결: brew install pango gdk-pixbuf 후 재기동. 원인: %s",
        exc,
    )
    from api_server.pdf_renderer.stub import render_pdf

__all__ = ["render_pdf"]
