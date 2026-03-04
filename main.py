"""GRIMOIRE — FastAPI application for RAG-powered BLAS codebase querying."""

import asyncio
import collections
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

load_dotenv()

from config import (
    CACHE_MAX_SIZE, CACHE_TTL, SOURCE_DIRS as DEFAULT_SOURCE_DIRS,
    RATE_LIMIT_RPM, MAX_QUERY_LENGTH, MAX_CONVERSATION_TURNS,
    MAX_CALL_GRAPH_DEPTH, MAX_SEARCH_LENGTH, MAX_ROUTINE_LENGTH,
    ALLOWED_LIBRARIES,
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_query_cache: dict[str, dict] = {}

# ─── Rate Limiting ───────────────────────────────────────────────────────────
_rate_limits: dict[str, collections.deque] = {}


RATE_LIMIT_WINDOW = 60  # seconds

def _check_rate_limit(client_ip: str) -> bool:
    """Sliding window rate limiter. Returns True if request is allowed."""
    now = time.time()
    window = _rate_limits.setdefault(client_ip, collections.deque())
    while window and window[0] < now - RATE_LIMIT_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT_RPM:
        return False
    window.append(now)
    return True


def _require_api_key(request: Request):
    """Validate API key from X-API-Key header."""
    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _cache_key(mode: str, query: str) -> str:
    return hashlib.sha256(f"{mode}:{query}".encode()).hexdigest()


def _cache_get(mode: str, query: str) -> dict | None:
    key = _cache_key(mode, query)
    entry = _query_cache.get(key)
    if not entry:
        return None
    if (time.time() - entry["timestamp"]) < CACHE_TTL:
        return entry
    _query_cache.pop(key, None)
    return None


def _cache_put(mode: str, query: str, chunks: list, answer: str, meta: dict):
    if len(_query_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_query_cache, key=lambda k: _query_cache[k]["timestamp"])
        del _query_cache[oldest_key]
    key = _cache_key(mode, query)
    _query_cache[key] = {
        "chunks": chunks,
        "answer": answer,
        "meta": meta,
        "timestamp": time.time(),
    }


def _dir_has_fortran(path: str) -> bool:
    """Check if a directory has Fortran files or subdirectories."""
    if not os.path.isdir(path):
        return False
    has_fortran = any(
        f.endswith('.f') or f.endswith('.f90') for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    has_subdirs = any(
        os.path.isdir(os.path.join(path, d)) for d in os.listdir(path)
        if not d.startswith('.')
    )
    return has_fortran or has_subdirs


def _ensure_sources():
    """Download BLAS and LAPACK source if not present on disk."""
    import subprocess

    source_dirs = DEFAULT_SOURCE_DIRS.split(",")

    for source_dir in source_dirs:
        source_dir = source_dir.strip()
        if _dir_has_fortran(source_dir):
            print(f"Source found at {source_dir}")
            continue

        os.makedirs(source_dir, exist_ok=True)

        if "scalapack" in source_dir.lower():
            print(f"ScaLAPACK source not found at {source_dir}, downloading...")
            subprocess.run(
                ["bash", "-c",
                 f"curl -sL https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.0.tar.gz"
                 f" | tar xz --strip-components=1 -C {source_dir} scalapack-2.2.0/SRC scalapack-2.2.0/PBLAS scalapack-2.2.0/TOOLS scalapack-2.2.0/BLACS"],
                check=True
            )
            print("ScaLAPACK source downloaded.")
        elif "lapack" in source_dir.lower():
            print(f"LAPACK source not found at {source_dir}, downloading...")
            subprocess.run(
                ["bash", "-c",
                 f"curl -sL https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.0.tar.gz"
                 f" | tar xz --strip-components=2 -C {source_dir} lapack-3.12.0/SRC"],
                check=True
            )
            print("LAPACK source downloaded.")
        else:
            print(f"BLAS source not found at {source_dir}, downloading...")
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
    print("GRIMOIRE starting (lazy mode)...")
    yield


app = FastAPI(title="GRIMOIRE", version="2.0.0", lifespan=lifespan)


# ─── HTML Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    with open(os.path.join(STATIC_DIR, "dashboard.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/explorer", response_class=HTMLResponse)
async def serve_explorer():
    with open(os.path.join(STATIC_DIR, "explorer.html"), "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/callgraph", response_class=HTMLResponse)
async def serve_callgraph():
    with open(os.path.join(STATIC_DIR, "callgraph.html"), "r") as f:
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


@app.get("/debug/env")
async def debug_env(request: Request):
    _require_api_key(request)
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


@app.get("/api/cache-stats")
async def cache_stats():
    return {
        "size": len(_query_cache),
        "max_size": CACHE_MAX_SIZE,
        "ttl_seconds": CACHE_TTL,
    }


@app.get("/api/routines")
async def api_routines(library: str = None, search: str = None):
    from db import get_routines
    if library and library not in ALLOWED_LIBRARIES:
        raise HTTPException(status_code=422, detail=f"Invalid library. Must be one of: {', '.join(ALLOWED_LIBRARIES)}")
    if search and len(search) > MAX_SEARCH_LENGTH:
        raise HTTPException(status_code=422, detail=f"Search query too long (max {MAX_SEARCH_LENGTH} chars)")
    try:
        routines = get_routines(library=library, search=search)
        return JSONResponse(content=routines)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/routine-detail")
async def api_routine_detail(name: str = None):
    from db import get_routine_detail
    if not name or len(name) > MAX_ROUTINE_LENGTH:
        raise HTTPException(status_code=422, detail="Invalid routine name")
    detail = get_routine_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Routine '{name}' not found")
    return JSONResponse(content=detail)


@app.get("/api/compare")
async def api_compare(routine1: str = None, routine2: str = None):
    from db import get_routine_detail
    if not routine1 or not routine2:
        raise HTTPException(status_code=422, detail="Both routine1 and routine2 are required")
    if len(routine1) > MAX_ROUTINE_LENGTH or len(routine2) > MAX_ROUTINE_LENGTH:
        raise HTTPException(status_code=422, detail=f"Routine name too long (max {MAX_ROUTINE_LENGTH} chars)")
    detail1 = get_routine_detail(routine1)
    detail2 = get_routine_detail(routine2)
    if not detail1:
        raise HTTPException(status_code=404, detail=f"Routine '{routine1}' not found")
    if not detail2:
        raise HTTPException(status_code=404, detail=f"Routine '{routine2}' not found")
    return JSONResponse(content={"routine1": detail1, "routine2": detail2})


@app.get("/api/dead-code")
async def api_dead_code():
    from db import get_dead_code
    try:
        results = get_dead_code()
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/call-graph")
async def api_call_graph(routine: str = None, depth: int = 2):
    from db import get_call_graph
    if routine and len(routine) > MAX_ROUTINE_LENGTH:
        raise HTTPException(status_code=422, detail=f"Routine name too long (max {MAX_ROUTINE_LENGTH} chars)")
    if depth < 1 or depth > MAX_CALL_GRAPH_DEPTH:
        raise HTTPException(status_code=422, detail=f"Depth must be between 1 and {MAX_CALL_GRAPH_DEPTH}")
    try:
        graph = get_call_graph(routine=routine, depth=depth)
        return JSONResponse(content=graph)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── Ingestion ──────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: Request):
    from ingest import run_ingestion
    from db import log_error

    _require_api_key(request)

    try:
        _ensure_sources()
    except Exception as e:
        print(f"Warning: source download failed: {e}")

    source_dirs_raw = DEFAULT_SOURCE_DIRS.split(",")
    source_dirs = [d.strip() for d in source_dirs_raw if os.path.isdir(d.strip())]
    print(f"Resolved dirs: {source_dirs}")
    if not source_dirs:
        raise HTTPException(status_code=400,
                            detail="No valid source directories found")

    try:
        index = _get_index()
        result = run_ingestion(source_dirs=source_dirs, index=index)
        _query_cache.clear()
        from retrieval import invalidate_bm25_cache
        invalidate_bm25_cache()
        resp = result.model_dump()
        resp["source_dirs_used"] = source_dirs
        return resp
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
    has_history = bool(getattr(query_request, "conversation_history", None))

    # ── Cache hit path (skip for follow-up queries) ──
    cached = None if has_history else _cache_get(mode, query_request.query)
    if cached:
        async def cached_stream():
            yield f"data: {json.dumps({'type': 'retrieval', 'found': bool(cached['chunks']), 'chunks': cached['chunks']})}\n\n"
            yield f"data: {json.dumps({'type': 'text', 'content': cached['answer']})}\n\n"
            latency_ms = int((time.time() - start_time) * 1000)
            done_meta = {**cached["meta"], "latency_ms": latency_ms, "cache_hit": True}
            yield f"data: {json.dumps({'type': 'done', **done_meta})}\n\n"

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    # ── Cache miss path ──
    try:
        index = _get_index()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pinecone not ready: {e}")

    # Retrieval (run in thread pool to avoid blocking the event loop)
    try:
        retrieval_result = await asyncio.to_thread(retrieve, query=query_request.query, index=index)
    except Exception as e:
        log_error(f"/{mode}", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    top_score = retrieval_result.chunks[0].score if retrieval_result.chunks else 0.0
    chunks_retrieved = len(retrieval_result.chunks)
    chunks_data = [c.model_dump() for c in retrieval_result.chunks]

    retrieval_meta = {
        "type": "retrieval",
        "found": retrieval_result.found,
        "chunks": chunks_data,
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
                conversation_history=getattr(query_request, "conversation_history", None),
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

        yield f"data: {json.dumps({'type': 'done', 'input_tokens': input_tokens, 'output_tokens': output_tokens, 'cost_usd': cost_usd, 'latency_ms': latency_ms, 'cache_hit': False})}\n\n"

        # Store in cache
        _cache_put(mode, query_request.query, chunks_data, answer_text, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })

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
        except Exception as e:
            print(f"Warning: query log failed: {e}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


async def _parse_query_request(request: Request):
    """Parse, validate, and rate-limit query request."""
    from models import QueryRequest
    from pydantic import ValidationError

    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_RPM} requests/min)")

    body = await request.json()
    try:
        return QueryRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@app.post("/query")
async def query_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "query")


@app.post("/explain")
async def explain_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "explain")


@app.post("/docgen")
async def docgen_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "docgen")


@app.post("/translate")
async def translate_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "translate")


@app.post("/patterns")
async def patterns_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "patterns")


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


# ─── Feedback API ────────────────────────────────────────────────────────────

@app.post("/feedback")
async def feedback_endpoint(request: Request):
    """Record user feedback (thumbs up/down)."""
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


@app.get("/api/feedback-stats")
async def api_feedback_stats():
    """Get aggregated feedback statistics."""
    from db import get_feedback_stats
    try:
        stats = get_feedback_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── Evaluation API ──────────────────────────────────────────────────────────

@app.post("/eval/seed")
async def eval_seed(request: Request):
    """Seed ground truth test cases."""
    _require_api_key(request)
    from db import seed_ground_truth
    count = seed_ground_truth()
    return {"seeded": count}


@app.get("/eval/ground-truth")
async def eval_ground_truth():
    """View all ground truth test cases."""
    from db import get_ground_truth
    return JSONResponse(content=get_ground_truth())


@app.post("/eval/run")
async def eval_run(request: Request):
    """Run evaluation against ground truth."""
    _require_api_key(request)
    from retrieval import run_evaluation
    try:
        index = _get_index()
        result = await asyncio.to_thread(run_evaluation, index)
        return JSONResponse(content=result)
    except Exception as e:
        from db import log_error
        log_error("/eval/run", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/eval/results")
async def eval_results():
    """View past evaluation results."""
    from db import get_eval_results
    results = get_eval_results()
    for r in results:
        if r.get("details"):
            try:
                r["details"] = json.loads(r["details"])
            except (json.JSONDecodeError, TypeError):
                pass
    return JSONResponse(content=results)
