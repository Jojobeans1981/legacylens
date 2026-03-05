"""GRIMOIRE - FastAPI application for RAG-powered BLAS codebase querying."""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request

load_dotenv()

from config import SOURCE_DIRS as DEFAULT_SOURCE_DIRS
from middleware import require_api_key
from routes.pages import router as pages_router
from routes.query import router as query_router
from routes.explorer import router as explorer_router
from routes.dashboard import router as dashboard_router
from routes.ingest import router as ingest_router
from routes.eval import router as eval_router
from routes.feedback import router as feedback_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal startup - heavy init deferred to first request."""
    app.state.index = None
    app.state.index_connected = False
    print("GRIMOIRE starting (lazy mode)...")
    yield


app = FastAPI(title="GRIMOIRE", version="2.0.0", lifespan=lifespan)

app.include_router(pages_router)
app.include_router(query_router)
app.include_router(explorer_router)
app.include_router(dashboard_router)
app.include_router(ingest_router)
app.include_router(eval_router)
app.include_router(feedback_router)


@app.get("/health")
async def health():
    from models import HealthResponse
    return HealthResponse(
        status="ok",
        model_loaded=True,
        index_connected=getattr(app.state, "index_connected", False),
    )


@app.get("/debug/env")
async def debug_env(request: Request):
    require_api_key(request)
    resolved = DEFAULT_SOURCE_DIRS.split(",")
    dir_status = {}
    for d in resolved:
        d = d.strip()
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.endswith(('.f', '.f90'))])
            dir_status[d] = f"exists ({count} Fortran files)"
        else:
            dir_status[d] = "does not exist"
    return {
        "resolved_dirs": resolved,
        "dir_status": dir_status,
    }
