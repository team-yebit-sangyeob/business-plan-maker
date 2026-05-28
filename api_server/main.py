from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# .env 로드 — uvicorn은 자동 로드하지 않으므로 진입점에서 명시적으로.
# python-dotenv 미설치 시엔 조용히 건너뜀(이미 export된 env로 동작).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from api_server.routes import chat, plan, session


def create_app() -> FastAPI:
    app = FastAPI(title="business-plan-maker / api_server")

    # Vite dev 서버 포트들 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(session.router)
    app.include_router(chat.router)
    app.include_router(plan.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
