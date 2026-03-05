"""HTML page routes."""

import os

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")


@router.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@router.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    with open(os.path.join(STATIC_DIR, "dashboard.html"), "r") as f:
        return HTMLResponse(content=f.read())


@router.get("/explorer", response_class=HTMLResponse)
async def serve_explorer():
    with open(os.path.join(STATIC_DIR, "explorer.html"), "r") as f:
        return HTMLResponse(content=f.read())


@router.get("/callgraph", response_class=HTMLResponse)
async def serve_callgraph():
    with open(os.path.join(STATIC_DIR, "callgraph.html"), "r") as f:
        return HTMLResponse(content=f.read())


@router.get("/source", response_class=HTMLResponse)
async def serve_source():
    with open(os.path.join(STATIC_DIR, "source.html"), "r") as f:
        return HTMLResponse(content=f.read())
