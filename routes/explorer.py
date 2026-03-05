"""Explorer API routes (routines, compare, dead-code, call-graph)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from config import (
    ALLOWED_LIBRARIES, MAX_SEARCH_LENGTH, MAX_ROUTINE_LENGTH, MAX_CALL_GRAPH_DEPTH,
)

router = APIRouter(prefix="/api")


@router.get("/routines")
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


@router.get("/routine-detail")
async def api_routine_detail(name: str = None):
    from db import get_routine_detail
    if not name or len(name) > MAX_ROUTINE_LENGTH:
        raise HTTPException(status_code=422, detail="Invalid routine name")
    detail = get_routine_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Routine '{name}' not found")
    return JSONResponse(content=detail)


@router.get("/compare")
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


@router.get("/dead-code")
async def api_dead_code():
    from db import get_dead_code
    try:
        results = get_dead_code()
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/call-graph")
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
