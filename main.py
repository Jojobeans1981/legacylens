"""LegacyLens — FastAPI application for RAG-powered BLAS codebase querying."""

import json
import os
import time
import traceback
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from models import QueryRequest, HealthResponse
from ingest import run_ingestion, connect_pinecone
from retrieval import retrieve
from llm import generate_answer
from db import log_query, log_error, get_stats, get_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to Pinecone at startup."""
    print("Loading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
        app.state.model = SentenceTransformer('all-MiniLM-L6-v2')
        app.state.model_loaded = True
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        app.state.model = None
        app.state.model_loaded = False

    print("Connecting to Pinecone...")
    try:
        app.state.index = connect_pinecone()
        app.state.index_connected = True
        print("Pinecone connected.")
    except Exception as e:
        print(f"Warning: Could not connect to Pinecone: {e}")
        app.state.index = None
        app.state.index_connected = False

    yield


app = FastAPI(title="LegacyLens", version="1.0.0", lifespan=lifespan)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


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
    return HealthResponse(
        status="ok",
        model_loaded=getattr(app.state, "model_loaded", False),
        index_connected=getattr(app.state, "index_connected", False),
    )


# ─── Ingestion ──────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: Request):
    # Simple API key protection
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    source_dir = os.getenv("BLAS_SOURCE_DIR", "/data/blas_source")
    if not os.path.isdir(source_dir):
        raise HTTPException(status_code=400,
                            detail=f"Source directory not found: {source_dir}")

    try:
        result = run_ingestion(
            source_dir=source_dir,
            model=app.state.model,
            index=app.state.index,
        )
        return result.model_dump()
    except Exception as e:
        log_error("/ingest", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ─── Query Routes ───────────────────────────────────────────────────────────

async def _handle_query(request: QueryRequest, mode: str):
    """Common handler for all query modes. Returns streaming SSE response."""
    start_time = time.time()

    if not app.state.model or not app.state.index:
        raise HTTPException(status_code=503,
                            detail="Model or index not ready. Run /ingest first.")

    # Retrieval
    try:
        retrieval_result = retrieve(
            query=request.query,
            model=app.state.model,
            index=app.state.index,
        )
    except Exception as e:
        log_error(f"/{mode}", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    top_score = retrieval_result.chunks[0].score if retrieval_result.chunks else 0.0
    chunks_retrieved = len(retrieval_result.chunks)

    # Prepare retrieval metadata to send before streaming
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

        # Send retrieval metadata first
        yield f"data: {json.dumps(retrieval_meta)}\n\n"

        # Stream LLM answer
        try:
            async for chunk in generate_answer(
                query=request.query,
                context=retrieval_result.context,
                mode=mode,
            ):
                if chunk.startswith("\x00"):
                    # Metadata chunk
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

        # Send final metadata
        yield f"data: {json.dumps({'type': 'done', 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'cost_usd': cost_usd, 'latency_ms': latency_ms})}\n\n"

        # Log to database
        try:
            log_query(
                query=request.query,
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
async def query_endpoint(request: QueryRequest):
    return await _handle_query(request, "query")


@app.post("/explain")
async def explain_endpoint(request: QueryRequest):
    return await _handle_query(request, "explain")


@app.post("/docgen")
async def docgen_endpoint(request: QueryRequest):
    return await _handle_query(request, "docgen")


@app.post("/translate")
async def translate_endpoint(request: QueryRequest):
    return await _handle_query(request, "translate")


@app.post("/patterns")
async def patterns_endpoint(request: QueryRequest):
    return await _handle_query(request, "patterns")


# ─── Dashboard API ──────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    try:
        stats = get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        log_error("/api/stats", type(e).__name__, str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/recent-queries")
async def api_recent_queries():
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
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT timestamp, endpoint, error_type, error_message FROM error_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return JSONResponse(content=[dict(row) for row in rows])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
