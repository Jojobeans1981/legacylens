"""Query endpoints (query, explain, docgen, translate, patterns)."""

import asyncio
import hashlib
import json
import time

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from config import CACHE_MAX_SIZE, CACHE_TTL

router = APIRouter()

# ─── Query Cache ─────────────────────────────────────────────────────────────
_query_cache: dict[str, dict] = {}


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


def get_cache_stats() -> dict:
    return {
        "size": len(_query_cache),
        "max_size": CACHE_MAX_SIZE,
        "ttl_seconds": CACHE_TTL,
    }


def clear_cache():
    _query_cache.clear()


# ─── Query Handler ───────────────────────────────────────────────────────────

async def _handle_query(query_request, mode: str, app):
    """Common handler for all query modes. Returns streaming SSE response."""
    from retrieval import retrieve
    from llm import generate_answer
    from db import log_query, log_error

    start_time = time.time()
    has_history = bool(getattr(query_request, "conversation_history", None))

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

    from ingest import connect_pinecone
    if not getattr(app.state, "index", None):
        try:
            app.state.index = connect_pinecone()
            app.state.index_connected = True
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Pinecone not ready: {e}")

    index = app.state.index

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
        "strategy": retrieval_result.strategy,
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
    from middleware import check_rate_limit
    from config import RATE_LIMIT_RPM

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded ({RATE_LIMIT_RPM} requests/min)")

    body = await request.json()
    try:
        return QueryRequest(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@router.post("/query")
async def query_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "query", request.app)


@router.post("/explain")
async def explain_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "explain", request.app)


@router.post("/docgen")
async def docgen_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "docgen", request.app)


@router.post("/translate")
async def translate_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "translate", request.app)


@router.post("/patterns")
async def patterns_endpoint(request: Request):
    qr = await _parse_query_request(request)
    return await _handle_query(qr, "patterns", request.app)
