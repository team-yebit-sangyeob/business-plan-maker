"""POST /plan — 명시적 트리거 (12장). async REST: planner(하이브리드) → pdf_renderer stub.

planner가 서술 1회를 위해 LLM(call_json, async)을 부르므로 라우트도 async다 — sync 라우트에서
asyncio.run을 중첩하는 함정을 피한다. compose_markdown은 LLM이 죽어도 결정론 골격으로 폴백한다.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from fastapi import APIRouter, Body, HTTPException, Response

from agents.planner import compose_markdown
from common.schema.state import required_missing, optional_missing
from api_server.pdf_renderer import render_pdf
from api_server.session_store import get_store

router = APIRouter()


class PlanRequest(TypedDict):
    session_id: str


@router.post("/plan")
async def create_plan(req: Annotated[PlanRequest, Body()]):
    """필수 슬롯 게이트 통과 시 계획서를 합성·저장하고 메타와 다운로드 URL을 돌려준다(미달이면 Type 0 거절)."""
    store = get_store()
    state = store.get(req["session_id"])
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")

    missing_req = required_missing(state)
    if missing_req:
        # Type 0 (8장)
        raise HTTPException(
            status_code=400,
            detail={
                "code": "required_slots_missing",
                "missing": missing_req,
                "message": "필수 슬롯 미달 — 출력 거절",
            },
        )

    markdown = await compose_markdown(state)
    rendered = render_pdf(markdown)

    plan_id = uuid.uuid4().hex[:10]
    empty_opt = len(optional_missing(state))

    meta = {
        "plan_id": plan_id,
        "title": "계획서 v1",
        "pages": rendered.pages,
        "empty_slots": empty_opt,
        "created_at": datetime.now().isoformat(),
        "pdf_bytes": rendered.pdf_bytes,
        "markdown": markdown,
    }
    store.save_pdf(plan_id, meta)

    return {
        "plan_id": plan_id,
        "title": meta["title"],
        "pages": rendered.pages,
        "empty_slots": empty_opt,
        "download_url": f"/plan/{plan_id}/download",
        "created_at": meta["created_at"],
    }


@router.get("/plan/{plan_id}/download")
def download_plan(plan_id: str):
    """저장된 계획서를 PDF 응답으로 내려준다(없으면 404)."""
    meta = get_store().get_pdf(plan_id)
    if not meta:
        raise HTTPException(status_code=404, detail="plan not found")
    return Response(
        content=meta["pdf_bytes"],
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="plan_{plan_id}.pdf"',
        },
    )
