"""POST /plan — 명시적 트리거 (12장). 동기 REST: planner stub → pdf_renderer stub."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from fastapi import APIRouter, Body, HTTPException, Response

from agents.planner import compose_markdown
from agents.orchestrator.nodes.gate import required_missing, optional_missing
from api_server.pdf_renderer import render_pdf
from api_server.session_store import get_store

router = APIRouter()


class PlanRequest(TypedDict):
    session_id: str


@router.post("/plan")
def create_plan(req: Annotated[PlanRequest, Body()]):
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

    markdown = compose_markdown(state)
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
