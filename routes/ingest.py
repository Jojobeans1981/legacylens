"""Ingestion route and source download helpers."""

import os
import subprocess

from fastapi import APIRouter, Request, HTTPException

from config import SOURCE_DIRS as DEFAULT_SOURCE_DIRS
from middleware import require_api_key

router = APIRouter()


def _dir_has_fortran(path: str) -> bool:
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


@router.post("/ingest")
async def ingest(request: Request):
    from ingest import run_ingestion, connect_pinecone
    from db import log_error

    require_api_key(request)

    try:
        _ensure_sources()
    except Exception as e:
        print(f"Warning: source download failed: {e}")

    source_dirs_raw = DEFAULT_SOURCE_DIRS.split(",")
    source_dirs = [d.strip() for d in source_dirs_raw if os.path.isdir(d.strip())]
    print(f"Resolved dirs: {source_dirs}")
    if not source_dirs:
        raise HTTPException(status_code=400, detail="No valid source directories found")

    try:
        if not getattr(request.app.state, "index", None):
            request.app.state.index = connect_pinecone()
            request.app.state.index_connected = True
        index = request.app.state.index
        result = run_ingestion(source_dirs=source_dirs, index=index)
        from routes.query import clear_cache
        from retrieval import invalidate_bm25_cache
        clear_cache()
        invalidate_bm25_cache()
        resp = result.model_dump()
        resp["source_dirs_used"] = source_dirs
        return resp
    except Exception as e:
        log_error("/ingest", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail=str(e))
