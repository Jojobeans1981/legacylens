"""Feedback route."""

from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.post("/feedback")
async def feedback_endpoint(request: Request):
    from models import FeedbackRequest
    from db import log_feedback
    from pydantic import ValidationError

    body = await request.json()
    try:
        fb = FeedbackRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    log_feedback(query=fb.query, mode=fb.mode, feedback=fb.feedback, comment=fb.comment)
    return {"status": "ok"}
