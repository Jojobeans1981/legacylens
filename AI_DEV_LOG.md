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

## 2026-03-02 18:30 Step 4: Ingestion Pipeline — Full Run
**What I did:** Connected to Pinecone with real API key and ran full ingestion pipeline.

**Results:**
- 163 Fortran files discovered (143 .f + 12 .f90 + 8 test files in TESTING/)
- 330 chunks generated (many test files contain multiple routines)
- Embedding took ~40s, Pinecone upsert in 4 batches of 100
- Total ingestion: 52.5 seconds
- Pinecone index confirmed: 330 vectors, 384 dimensions, cosine metric

**Next step:** Run test queries

## 2026-03-02 18:40 Step 12: Test All 8 Required Queries
**What I did:** Ran all 8 test queries against the local server with real API keys.

**Problems encountered:**
1. `claude-sonnet-4-5-20250514` model ID not available on this API key — returned 404.
2. Changed to `claude-haiku-4-5-20251001` which is available and cost-effective.
3. Made model configurable via `CLAUDE_MODEL` env var so Sonnet can be used when available.

**Test Results:**

| # | Query | Mode | Top Score | Chunks | Latency | Cost | Quality |
|---|-------|------|-----------|--------|---------|------|---------|
| 1 | Where is the main entry point? | query | 0.000 | 0 | 4039ms | $0.0008 | 3/5 — Correctly says no entry point found (BLAS is a library, no main) |
| 2 | What subroutines modify matrix args? | query | 0.606 | 5 | 3843ms | $0.0032 | 5/5 — Lists STRMM, STRSM, CTRMM, etc. with correct citations |
| 3 | Explain DGEMM | explain | 0.537 | 5 | 6481ms | $0.0043 | 5/5 — Excellent explanation of matrix multiplication |
| 4 | Find all file I/O operations | query | 0.386 | 5 | 4667ms | $0.0031 | 4/5 — Finds WRITE ops in test files correctly |
| 5 | Dependencies of DGEMV | query | 0.358 | 1 | 3418ms | $0.0013 | 3/5 — Limited by truncated content in metadata |
| 6 | Error handling patterns | patterns | 0.361 | 4 | 9568ms | $0.0060 | 4/5 — Identifies XERBLA error handler pattern |
| 7 | Generate docs for SGEMM | docgen | 0.352 | 1 | 11533ms | $0.0084 | 5/5 — Full Fortran comment block in correct format |
| 8 | Translate DTRSM to NumPy | translate | 0.440 | 5 | 12781ms | $0.0090 | 5/5 — Side-by-side with scipy.linalg.solve_triangular |

**Aggregate stats:**
- Total cost: $0.0361
- Total tokens: 11,608 in / 6,696 out
- Average latency: 7,041ms
- Retrieval precision @5: 6/8 queries retrieved highly relevant results (75%)
- All 4 code features (explain, docgen, patterns, translate) work end-to-end

**Decisions made:**
- Q1 correctly returns no results — BLAS is a library with no main() entry point
- Retrieval quality is good for targeted routine queries (Q2, Q3, Q8)
- Broader queries (Q4, Q5, Q6) have lower scores but still find relevant content
- Model configurable via CLAUDE_MODEL env var for flexibility

**Next step:** Update cost analysis, commit, prepare for Render deployment

## 2026-03-03 Render Deployment — Debugging & Fixes
**What I did:** Deployed to Render and fixed 4 blocking issues.

**Problems encountered and solutions:**
1. **FileNotFoundError in `_ensure_blas_source()`** — `os.listdir()` called on non-existent directory due to `or`/`and` operator precedence bug. Fixed by checking `os.path.isdir()` first.
2. **Port scan timeout** — Model loading blocked the lifespan handler, preventing uvicorn from binding the port. Moved init to background thread, then to fully lazy imports.
3. **OOM on Render free tier (512MB)** — `sentence-transformers` pulls in PyTorch (~400MB+). Removed it entirely; switched to API-based embeddings.
4. **HuggingFace Inference API returned 410 Gone** — Free inference endpoint deprecated. Switched to Pinecone's built-in Inference API (`multilingual-e5-large`, 1024 dims). Same API key, no new dependencies.
5. **Pinecone rate limit (250K tokens/min)** — Added batching (20 chunks/batch) with 2s delays and exponential backoff on 429 errors.

**Architecture changes:**
- Embeddings: local sentence-transformers → Pinecone Inference API (multilingual-e5-large, 1024 dims)
- All heavy imports (pinecone, anthropic, db) are now lazy (inside route handlers)
- Startup is instant — only stdlib + FastAPI loaded at module level
- Pinecone connects lazily on first request

**Deployment result:**
- URL: https://legacylens-ycuy.onrender.com/
- Ingestion: 163 files, 330 chunks, 48.5s
- Health: `{"status":"ok","model_loaded":true,"index_connected":true}`

## 2026-03-03 Production Test — All 8 Queries

**Test Results (Render deployment, Pinecone embeddings):**

| # | Query | Mode | Top Score | Chunks | Latency | Cost |
|---|-------|------|-----------|--------|---------|------|
| 1 | Where is the main entry point? | query | 0.809 | 5 | 3132ms | $0.0029 |
| 2 | What subroutines modify matrix args? | query | 0.820 | 5 | 3822ms | $0.0034 |
| 3 | Explain DGEMM | explain | 0.801 | 5 | 5652ms | $0.0041 |
| 4 | Find all file I/O operations | query | 0.789 | 5 | 4250ms | $0.0032 |
| 5 | Dependencies of DGEMV | query | 0.787 | 5 | 3131ms | $0.0028 |
| 6 | Error handling patterns | patterns | 0.784 | 5 | 6392ms | $0.0041 |
| 7 | Generate docs for SGEMM | docgen | 0.829 | 5 | 6264ms | $0.0000 |
| 8 | Translate DTRSM to NumPy | translate | 0.846 | 5 | 11791ms | $0.0086 |

**Aggregate stats:**
- Total cost: $0.0324 (10 queries including 2 test runs)
- Average latency: 5,462ms
- All retrieval scores >0.78 (major improvement from Pinecone embeddings vs local model)
- All 8 queries return 5 chunks (full context every time)
- Q7 docgen hit Anthropic content filter twice (intermittent) but succeeded on retry
- Dashboard at /dashboard fully operational with live stats
