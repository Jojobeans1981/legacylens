"""LegacyLens — FastAPI application for RAG-powered BLAS codebase querying."""

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ─── Query Cache ─────────────────────────────────────────────────────────────
CACHE_MAX_SIZE = 500
CACHE_TTL = 3600  # 1 hour
_query_cache: dict[str, dict] = {}


def _cache_key(mode: str, query: str) -> str:
    return hashlib.sha256(f"{mode}:{query}".encode()).hexdigest()


def _cache_get(mode: str, query: str) -> dict | None:
    key = _cache_key(mode, query)
    entry = _query_cache.get(key)
    if entry and (time.time() - entry["timestamp"]) < CACHE_TTL:
        return entry
    if entry:
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

    source_dirs = os.getenv("SOURCE_DIRS", "./data/blas_source,./data/lapack_source,./data/scalapack_source").split(",")

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
async def debug_env():
    import subprocess
    resolved = os.getenv("SOURCE_DIRS", "./data/blas_source,./data/lapack_source,./data/scalapack_source").split(",")
    dir_status = {}
    for d in resolved:
        d = d.strip()
        if os.path.isdir(d):
            count = len([f for f in os.listdir(d) if f.endswith(('.f', '.f90'))])
            dir_status[d] = f"exists ({count} Fortran files)"
        else:
            dir_status[d] = "does not exist"
    return {
        "SOURCE_DIRS_env": os.getenv("SOURCE_DIRS", "NOT SET"),
        "BLAS_SOURCE_DIR_env": os.getenv("BLAS_SOURCE_DIR", "NOT SET"),
        "resolved_dirs": resolved,
        "dir_status": dir_status,
        "all_env_keys": sorted([k for k in os.environ.keys() if not k.startswith("_")]),
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
    try:
        routines = get_routines(library=library, search=search)
        return JSONResponse(content=routines)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/call-graph")
async def api_call_graph(routine: str = None, depth: int = 2):
    from db import get_call_graph
    try:
        graph = get_call_graph(routine=routine, depth=min(depth, 4))
        return JSONResponse(content=graph)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ─── Ingestion ──────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(request: Request):
    from ingest import run_ingestion
    from db import log_error

    api_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        _ensure_sources()
    except Exception as e:
        print(f"Warning: source download failed: {e}")

    source_dirs_raw = os.getenv("SOURCE_DIRS", "./data/blas_source,./data/lapack_source,./data/scalapack_source").split(",")
    source_dirs = [d.strip() for d in source_dirs_raw if os.path.isdir(d.strip())]
    print(f"Resolved dirs: {source_dirs}")
    if not source_dirs:
        raise HTTPException(status_code=400,
                            detail="No valid source directories found")

    try:
        index = _get_index()
        result = run_ingestion(source_dirs=source_dirs, index=index)
        _query_cache.clear()
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

    # Retrieval
    try:
        retrieval_result = retrieve(query=query_request.query, index=index)
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
