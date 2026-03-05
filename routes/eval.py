"""Evaluation API routes."""

import asyncio
import json

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from middleware import require_api_key

router = APIRouter(prefix="/eval")


@router.post("/seed")
async def eval_seed(request: Request):
    require_api_key(request)
    from db import seed_ground_truth
    count = seed_ground_truth()
    return {"seeded": count}


@router.get("/ground-truth")
async def eval_ground_truth():
    from db import get_ground_truth
    return JSONResponse(content=get_ground_truth())


@router.post("/run")
async def eval_run(request: Request):
    require_api_key(request)
    from retrieval import run_evaluation
    from ingest import connect_pinecone
    try:
        if not getattr(request.app.state, "index", None):
            request.app.state.index = connect_pinecone()
            request.app.state.index_connected = True
        index = request.app.state.index
        result = await asyncio.to_thread(run_evaluation, index)
        return JSONResponse(content=result)
    except Exception as e:
        from db import log_error
        log_error("/eval/run", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
async def eval_results():
    from db import get_eval_results
    results = get_eval_results()
    for r in results:
        if r.get("details"):
            try:
                r["details"] = json.loads(r["details"])
            except (json.JSONDecodeError, TypeError):
                pass
    return JSONResponse(content=results)
