"""LegacyLens — FastAPI application for RAG-powered BLAS codebase querying."""

import json
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _ensure_blas_source():
    """Download BLAS source if not present on disk."""
    source_dir = os.getenv("BLAS_SOURCE_DIR", "/data/blas_source")
    if os.path.isdir(source_dir):
        has_fortran = any(
            f.endswith('.f') for f in os.listdir(source_dir)
            if os.path.isfile(os.path.join(source_dir, f))
        )
        has_subdirs = any(
            os.path.isdir(os.path.join(source_dir, d)) for d in os.listdir(source_dir)
            if not d.startswith('.')
        )
        if has_fortran or has_subdirs:
            print(f"BLAS source found at {source_dir}")
            return
    print(f"BLAS source not found at {source_dir}, downloading...")
    os.makedirs(source_dir, exist_ok=True)
    import subprocess
    subprocess.run(
        ["bash", "-c",
         f"curl -sL https://www.netlib.org/blas/blas.tgz | tar xz -C {source_dir}"],
        check=True
    )
    print("BLAS source downloaded.")


def _get_index():
    """Lazy-connect to Pinecone on first use."""
    if not getattr(app.state, "index", None):
        from ingest import connect_pinecone
        app.state.index = connect_pinecone()
        app.state.index_connected = True
        print("Pinecone connected (lazy init).")
    return app.state.index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Minimal startup — heavy init deferred to first request."""
    app.state.index = None
    app.state.index_connected = False
    print("LegacyLens starting (lazy mode)...")
    yield


app = FastAPI(title="LegacyLens", version="1.0.0", lifespan=lifespan)


# ─── HTML Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    with open(os.path.join(STATIC_DIR, "dashboard.html"), "r") as f:
        return HTMLResponse(content=f.read())


# ─── Health Check ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from models import HealthResponse
    return HealthResponse(
        status="ok",
        model_loaded=True,
        index_connected=getattr(app.state, "index_connected", False),
    )


# ─── Ingestion ──────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: Request):
    from ingest import run_ingestion
    from db import log_error

    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    _ensure_blas_source()
    source_dir = os.getenv("BLAS_SOURCE_DIR", "/data/blas_source")
    if not os.path.isdir(source_dir):
        raise HTTPException(status_code=400,
                            detail=f"Source directory not found: {source_dir}")

    try:
        index = _get_index()
        result = run_ingestion(source_dir=source_dir, index=index)
        return result.model_dump()
    except Exception as e:
        log_error("/ingest", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ─── Query Routes ───────────────────────────────────────────────────────────

async def _handle_query(query_request, mode: str):
    """Common handler for all query modes. Returns streaming SSE response."""
    from retrieval import retrieve
    from llm import generate_answer
    from db import log_query, log_error

    start_time = time.time()

    try:
        index = _get_index()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pinecone not ready: {e}")

    # Retrieval
    try:
        retrieval_result = retrieve(query=query_request.query, index=index)
    except Exception as e:
        log_error(f"/{mode}", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    top_score = retrieval_result.chunks[0].score if retrieval_result.chunks else 0.0
    chunks_retrieved = len(retrieval_result.chunks)

    retrieval_meta = {
        "type": "retrieval",
        "found": retrieval_result.found,
        "chunks": [c.model_dump() for c in retrieval_result.chunks],
    }

    async def event_stream():
        answer_text = ""
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        yield f"data: {json.dumps(retrieval_meta)}\n\n"

        try:
            async for chunk in generate_answer(
                query=query_request.query,
                context=retrieval_result.context,
                mode=mode,
            ):
                if chunk.startswith("\x00"):
                    meta = json.loads(chunk[1:])
                    input_tokens = meta["input_tokens"]
                    output_tokens = meta["output_tokens"]
                    cost_usd = meta["cost_usd"]
                else:
                    answer_text += chunk
                    yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        except Exception as e:
            log_error(f"/{mode}", type(e).__name__, str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        latency_ms = int((time.time() - start_time) * 1000)

        yield f"data: {json.dumps({'type': 'done', 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'cost_usd': cost_usd, 'latency_ms': latency_ms})}\n\n"

        try:
            log_query(
                query=query_request.query,
                mode=mode,
                chunks_retrieved=chunks_retrieved,
                top_score=top_score,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                answer_preview=answer_text[:200],
            )
        except Exception:
            pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/query")
async def query_endpoint(request: Request):
    from models import QueryRequest
    body = await request.json()
    return await _handle_query(QueryRequest(**body), "query")


@app.post("/explain")
async def explain_endpoint(request: Request):
    from models import QueryRequest
    body = await request.json()
    return await _handle_query(QueryRequest(**body), "explain")


@app.post("/docgen")
async def docgen_endpoint(request: Request):
    from models import QueryRequest
    body = await request.json()
    return await _handle_query(QueryRequest(**body), "docgen")


@app.post("/translate")
async def translate_endpoint(request: Request):
    from models import QueryRequest
    body = await request.json()
    return await _handle_query(QueryRequest(**body), "translate")


@app.post("/patterns")
async def patterns_endpoint(request: Request):
    from models import QueryRequest
    body = await request.json()
    return await _handle_query(QueryRequest(**body), "patterns")


# ─── Dashboard API ──────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    from db import get_stats, log_error
    try:
        stats = get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        log_error("/api/stats", type(e).__name__, str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/recent-queries")
async def api_recent_queries():
    from db import get_connection
    try:
        conn = get_connection()
        rows = conn.execute(
            """SELECT timestamp, query, mode, latency_ms, cost_usd, top_score,
                      chunks_retrieved, answer_preview
               FROM query_log ORDER BY id DESC LIMIT 20"""
        ).fetchall()
        conn.close()
        return JSONResponse(content=[dict(row) for row in rows])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/errors")
async def api_errors():
    from db import get_connection
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT timestamp, endpoint, error_type, error_message FROM error_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return JSONResponse(content=[dict(row) for row in rows])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
