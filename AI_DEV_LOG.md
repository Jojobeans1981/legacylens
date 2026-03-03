# LegacyLens AI Dev Log

## 2026-03-02 17:30 Step 1: Project Scaffold + Dev Log Init
**What I did:** Created the full project directory structure and all placeholder files for LegacyLens — a production-ready RAG system that makes the BLAS Fortran codebase queryable via natural language.

**Decisions made:**
- Stack is locked per spec: FastAPI + Pinecone + sentence-transformers (all-MiniLM-L6-v2) + Claude Sonnet via Anthropic API
- Custom RAG pipeline (no LangChain/LlamaIndex) for full control over chunking, retrieval, and prompt assembly
- SQLite for observability logging (lightweight, no external DB dependency)
- Single-file HTML frontends (index.html + dashboard.html) served by FastAPI — no build step
- Render for deployment with persistent disk at /data for SQLite + BLAS source

**Problems encountered:** None yet — clean scaffold.

**How I solved them:** N/A

**Next step:** Step 2 — Download and inspect BLAS source code from netlib.org

## 2026-03-02 17:35 Step 2: Download and Inspect BLAS Source
**What I did:** Downloaded BLAS 3.12.0 from netlib.org/blas/blas.tgz and extracted to data/blas_source/BLAS-3.12.0/.

**Findings:**
- 155 total .f files in archive (including 12 macOS `._` resource fork files to skip)
- 143 real Fortran source files
- 71,368 total lines of code
- All files are in a single flat directory (no subdirectories)
- Files follow BLAS naming convention: prefix (s/d/c/z for precision) + operation name
- Each file typically contains one subroutine or function

**Problems encountered:** Archive contained macOS `._` resource fork files that are binary junk. Updated discover_fortran_files() to filter them out.

**How I solved them:** Added `if not f.name.startswith('._')` filter to file discovery.

**Next step:** Step 3 — Build and test the Fortran chunker

## 2026-03-02 17:38 Step 3: Fortran Chunker
**What I did:** Built chunker.py with regex-based Fortran parsing. Tested across all 143 BLAS source files.

**Decisions made:**
- Used regex to detect SUBROUTINE/FUNCTION boundaries including type-prefixed functions (e.g., `COMPLEX*16 FUNCTION`)
- Include preceding comment blocks with each routine (BLAS files have extensive documentation comments)
- 500-line split threshold with 64-line overlap for large routines (none needed — max BLAS routine is ~400 lines)

**Results:**
- 143 files → 143 chunks (1:1 mapping, expected since each BLAS file has one routine)
- Chunk types: 124 subroutines, 19 functions
- Zero fallback chunks (all routines properly detected)

**Problems encountered:** Initial regex didn't match `COMPLEX*16 FUNCTION ZDOTC` — the `*16` qualifier wasn't handled.

**How I solved them:** Extended regex to include `COMPLEX(?:\*\d+)?` pattern. Fixed 2 fallback chunks (zdotc.f, zdotu.f) → now properly parsed as functions.

**Next step:** Steps 4-8 — Core Python modules already written, need to test end-to-end

## 2026-03-02 17:40 Steps 4-11: Core Pipeline + UI + Config
**What I did:** Wrote all core Python modules and frontend files:
- `ingest.py` — Full ingestion pipeline with Pinecone upsert batching
- `retrieval.py` — RAG retrieval with score threshold filtering (0.35)
- `llm.py` — Claude Sonnet streaming wrapper with 5 mode-specific system prompts
- `db.py` — SQLite observability logging with full dashboard query support
- `models.py` — All Pydantic models
- `main.py` — FastAPI app with all routes, lifespan startup, SSE streaming
- `static/index.html` — Query UI with mode selector, streaming display, syntax highlighting
- `static/dashboard.html` — Auto-refreshing observability dashboard with histograms
- `render.yaml` — Render deployment config with persistent disk
- `requirements.txt` — All dependencies locked
- `.env.example` — Environment variable template

**Decisions made:**
- API key protection for /ingest uses ANTHROPIC_API_KEY as a shared secret (simple, adequate for this app)
- SSE streaming for query responses — sends retrieval metadata first, then streams answer text, then final cost/latency metadata
- Score threshold of 0.35 for retrieval filtering — balances recall vs noise
- Highlight.js from CDN for Fortran syntax highlighting in results
- Dashboard auto-refreshes every 10s via setInterval polling /api/stats

**Next step:** Test locally, then prepare for deployment

## 2026-03-02 18:10 Local Testing & Bug Fixes
**What I did:** Ran comprehensive local tests of all components.

**Problems encountered:**
1. `pinecone-client` package renamed to `pinecone` — the old package now raises an Exception on import telling you to switch.
2. Server process crashed on startup because the Pinecone import error was not caught at import time (it happens at module level, before lifespan handler).

**How I solved them:**
1. Updated `requirements.txt` from `pinecone-client` to `pinecone`. Uninstalled old, installed new (`pinecone==8.1.0`).
2. Verified lifespan handler catches the `UnauthorizedException` from Pinecone when using invalid keys — server starts gracefully with `index_connected: false`.

**Local test results (with fake API keys):**
- `GET /health` → `{"status":"ok","model_loaded":true,"index_connected":false}` ✅
- `GET /` → Serves index.html correctly ✅
- `GET /dashboard` → Serves dashboard.html correctly ✅
- `GET /api/stats` → Returns empty but valid JSON stats ✅
- Model loads in ~5s, embeddings in 18.9s for all 143 chunks ✅
- Chunker: 143/143 files parsed, 0 fallbacks ✅

**Next step:** Deploy to Render with real API keys, run ingestion, test all 8 queries
