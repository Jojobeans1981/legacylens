# GRIMOIRE — Design Decisions & Architecture Reference

## Phase 1: Define Your Constraints

### 1. Scale & Load Profile
- **Codebase size:** 3,255 files, ~5,074 chunks across BLAS + LAPACK + ScaLAPACK (estimated 500K+ LOC)
- **Expected query volume:** Low (hackathon demo), rate limited to 30 req/min per IP
- **Ingestion model:** Batch — full re-ingestion via `/ingest` endpoint. No incremental updates.
- **Latency requirements:** Target <3s end-to-end (embed + search + LLM stream). Cached queries near-instant.

### 2. Budget & Cost Ceiling
- **Vector DB:** Pinecone serverless free tier — $0 for low volume
- **Embedding API:** Pinecone Inference API (bundled with Pinecone) — no separate embedding cost
- **LLM API:** Claude Haiku 4.5 at $1/M input, $5/M output. Typical query ~1K tokens = <$0.01
- **Trade money for time:** Used managed services everywhere (Pinecone serverless, Render hosting) to avoid infrastructure setup

### 3. Time to Ship
- **MVP timeline:** Built iteratively across multiple sessions
- **Must-have:** Query with citations, streaming answers, source code display
- **Nice-to-have (all shipped):** Call graph, routine explorer, complexity metrics, compare mode, dead code detection, conversation memory, translation view, dashboard
- **Framework learning curve:** Zero — used no RAG framework, built custom pipeline

### 4. Data Sensitivity
- **Codebase:** Fully open source (Netlib BLAS/LAPACK, Reference ScaLAPACK)
- **External APIs:** Yes — code sent to Pinecone for embedding and Claude for answer generation. Acceptable for open source.
- **Data residency:** No requirements (open source code, US-hosted services)

### 5. Team & Skill Constraints
- **Vector databases:** First time using Pinecone — learned during project
- **RAG frameworks:** No LangChain/LlamaIndex used — built from scratch for full control and understanding
- **Legacy language:** Fortran — parsed with regex-based syntax-aware chunker rather than a full parser

---

## Phase 2: Architecture Discovery

### 6. Vector Database Selection
- **Choice:** Pinecone (serverless, managed)
- **Why:** Zero infrastructure, built-in inference API for embeddings, cosine similarity, generous free tier
- **Managed vs self-hosted:** Managed — no time to operate Qdrant/Weaviate/pgvector
- **Filtering:** Metadata filters on `routine_name` for exact routine queries (added to improve precision)
- **Hybrid search:** BM25 keyword search (via `rank_bm25`) combined with vector search using Reciprocal Rank Fusion (RRF). Weights: 70% vector, 30% BM25. K=60 for RRF smoothing.
- **Scaling:** Serverless auto-scales. 5,074 vectors is small — no scaling concerns.

### 7. Embedding Strategy
- **Model:** `multilingual-e5-large` via Pinecone Inference API
- **Why:** General-purpose but strong on code. No code-specific model needed since Fortran is well-represented in training data. Bundled with Pinecone = simpler pipeline.
- **Dimensions:** 1024 — good balance of quality vs storage
- **Local vs API:** API-based (Pinecone Inference). Simpler, no GPU needed.
- **Batch processing:** Batch size 20 with retry logic and rate limit handling (exponential backoff on 429s)

### 8. Chunking Approach
- **Strategy:** Syntax-aware — chunks split on SUBROUTINE/FUNCTION/PROGRAM boundaries
- **Why:** Preserves semantic meaning. One routine = one chunk. Fixed-size chunking would split mid-routine.
- **Chunk size:** Variable (follows routine length). Max 500 lines per chunk.
- **Overlap:** 64 lines when a routine exceeds max chunk size
- **Metadata preserved:** file_path, start_line, end_line, chunk_type (subroutine/function/program), routine_name

### 9. Retrieval Pipeline
- **Top-k:** 5 (increased from 3 to meet requirements)
- **Score threshold:** 0.35 cosine similarity — chunks below this are dropped
- **Re-ranking:** Heuristic re-ranking with 4 boost signals: exact routine name match (+0.15), query term density (+0.10), chunk type boost (+0.05 for subroutine/function), partial routine name match (+0.10). Zero-latency impact (pure string ops on 5-10 chunks).
- **Context window management:** Cap 400 chars per chunk in LLM context. 5 chunks max. Max 512 output tokens.
- **Query enhancement:** Metadata filter extracts routine names from queries (e.g., "What does DGEMM do?" filters to routine_name=DGEMM). Falls back to unfiltered vector search.

### 10. Answer Generation
- **LLM:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — fastest model, lowest cost
- **Fallback:** Claude Sonnet 4.6 if Haiku is overloaded (configurable via env var)
- **Prompt design:** Compact system prompt per mode. 5 modes with distinct instructions:
  - Query: Answer the question, cite sources
  - Explain: Break down purpose, math, inputs, outputs, algorithm
  - Docs: Generate Fortran comment header
  - Patterns: Analyze structural patterns
  - Translate: Side-by-side Fortran to Python/NumPy
- **Citations:** `[file:start-end]` format. UI shows source chunks with relevance scores.
- **Streaming:** SSE (Server-Sent Events) — tokens stream to browser as they arrive
- **Retry:** Exponential backoff (up to 3 attempts) on API overload errors

### 11. Framework Selection
- **Choice:** Custom pipeline (no LangChain, no LlamaIndex)
- **Why:** Full control, minimal dependencies, easier to debug, better understanding of every component. RAG pipeline is simple enough that a framework adds overhead without proportional benefit.
- **Observability:** Built-in — SQLite logs every query with latency, tokens, cost, score. Real-time dashboard.
- **Stack:** FastAPI + Pinecone SDK + Anthropic SDK + SQLite. Vanilla HTML/CSS/JS frontend.

---

## Phase 3: Post-Stack Refinement

### 12. Failure Mode Analysis
- **Nothing relevant found:** System tells the user "No relevant code chunks found" and suggests refining the query. LLM receives explicit instruction not to hallucinate.
- **Ambiguous queries:** Vector search returns best-effort matches. Score threshold filters low-quality results.
- **Rate limiting:** Sliding window per-IP, 30 requests/minute. Returns 429 with clear message.
- **Error handling:** All errors logged to SQLite `error_log` table. LLM errors caught and returned as SSE error events. Retry with backoff on transient API failures.
- **API overload:** Retry up to 3 times with exponential backoff (1s, 2s delays)

### 13. Evaluation Strategy
- **Retrieval precision:** Visible in UI — every chunk shows its cosine similarity score (0-100%). Dashboard shows score distribution histogram.
- **Ground truth dataset:** 18 curated test queries with expected routine matches. Automated evaluation via `/eval/run` computes MRR, Hit@1, Hit@3, Hit@5, and per-query latency. Results stored in SQLite for trend tracking.
- **User feedback:** Thumbs up/down on every answer. Stored in SQLite with query, mode, and timestamp. Dashboard shows satisfaction percentage, counts, and recent feedback table. Stats available via `/api/feedback-stats`.
- **Improvement made:** Added metadata filtering after observing that "What does DGEMM do?" was retrieving DGEMLQT instead of DGEMM.

### 14. Performance Optimization
- **Query cache:** SHA-256 keyed in-memory cache with configurable TTL (default 3600s, max 500 entries). Cached queries skip embedding + Pinecone + LLM entirely.
- **Embedding cache:** `functools.lru_cache` on query embedding to avoid re-embedding identical queries
- **Async retrieval:** Embedding and Pinecone search run in thread pool (`asyncio.to_thread`) to avoid blocking the event loop
- **Compact prompts:** System prompts kept short to reduce input tokens and latency
- **Lazy initialization:** Pinecone connection deferred to first request (faster server startup)

### 15. Observability
- **Query logging:** Every query logged with: timestamp, query text, mode, chunks_retrieved, top_score, latency_ms, input_tokens, output_tokens, cost_usd, answer_preview
- **Error logging:** All exceptions logged with: timestamp, endpoint, error_type, error_message
- **Ingestion logging:** file_count, chunk_count, duration_seconds per run
- **Dashboard:** Real-time charts — latency over time (line), mode distribution (doughnut), score distribution (histogram), cost tracking, recent queries table, error log
- **Cache stats:** `/api/cache-stats` endpoint showing cache size, max size, TTL
- **Alerting:** Not implemented (not needed for hackathon scale)

### 16. Deployment & DevOps
- **Hosting:** Render Web Service with persistent disk at `/data` (SQLite DB persists across deploys)
- **CI/CD:** Git push to main triggers auto-deploy on Render
- **Environment management:** All config values overridable via env vars (centralized in `config.py`)
- **Secrets:** API keys stored as Render environment variables (ANTHROPIC_API_KEY, PINECONE_API_KEY). Auth required on sensitive endpoints (`/ingest`, `/debug/env`)
- **Source download:** Auto-downloads BLAS/LAPACK/ScaLAPACK from Netlib/GitHub on first ingestion if not present

---

## Architecture Diagram

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Browser UI  │────>│  FastAPI App   │────>│   Pinecone   │
│ (4 HTML pages)│    │  (main.py)     │     │  Vector DB   │
└──────────────┘     └───┬───────┬───┘     └──────────────┘
                         │       │
                    ┌────▼──┐ ┌──▼──────────┐
                    │SQLite │ │ Claude Haiku │
                    │  DB   │ │   4.5 (LLM) │
                    └───────┘ └─────────────┘
```

## Key Numbers
- 3,255 Fortran files ingested
- 5,074 chunks indexed in Pinecone
- 1024-dimensional embeddings (multilingual-e5-large)
- 1,720+ nodes and 1,735+ edges in call graph
- 5 query modes (Query, Explain, Docs, Patterns, Translate)
- 3 libraries (BLAS, LAPACK, ScaLAPACK)
- <$0.01 per query (Haiku 4.5)
- ~2.6s avg latency, near-instant cached
- 18 ground truth test queries, MRR 0.42
