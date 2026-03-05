"""Dashboard API routes (stats, recent queries, errors, feedback stats, cache stats)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")


@router.get("/stats")
async def api_stats():
    from db import get_stats, log_error
    try:
        stats = get_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        log_error("/api/stats", type(e).__name__, str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/recent-queries")
async def api_recent_queries():
    from db import get_recent_queries
    try:
        return JSONResponse(content=get_recent_queries())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/errors")
async def api_errors():
    from db import get_recent_errors
    try:
        return JSONResponse(content=get_recent_errors())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/cache-stats")
async def cache_stats():
    from routes.query import get_cache_stats
    return get_cache_stats()


@router.get("/feedback-stats")
async def api_feedback_stats():
    from db import get_feedback_stats
    try:
        stats = get_feedback_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
