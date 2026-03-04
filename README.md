# GRIMOIRE

**G**enerative **R**etrieval **I**ntelligence for **M**apping **O**ld **I**mperative **R**outine **E**ngineering

A production-ready RAG system that makes BLAS, LAPACK, and ScaLAPACK Fortran codebases queryable via natural language.

**Live demo:** https://legacylens-ycuy.onrender.com/
**Dashboard:** https://legacylens-ycuy.onrender.com/dashboard

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Browser UI  │────▶│  FastAPI App   │────▶│   Pinecone   │
│ (index.html) │     │  (main.py)     │     │  Vector DB   │
└──────────────┘     └───┬───────┬───┘     └──────────────┘
                         │       │
                    ┌────▼──┐ ┌──▼──────────┐
                    │SQLite │ │ Claude Haiku │
                    │  DB   │ │   (LLM)     │
                    └───────┘ └─────────────┘
```

### Pipeline Flow

1. **Ingestion**: Fortran `.f`/`.f90` files → syntax-aware chunker → Pinecone Inference embeddings → Pinecone upsert → routine index + call graph
2. **Query**: Natural language → embed → Pinecone cosine search → context assembly → Claude Haiku streaming response
3. **Observability**: All queries, costs, and errors logged to SQLite → real-time dashboard with charts

## Stack

| Layer | Technology |
|-------|-----------|
| Codebases | BLAS + LAPACK + ScaLAPACK (3,200+ files) |
| Vector DB | Pinecone (serverless, cosine similarity) |
| Embeddings | Pinecone Inference API (`multilingual-e5-large`, 1024-dim) |
| Answer LLM | Claude Haiku 4.5 via Anthropic API (streaming SSE) |
| Backend | Python 3.11+ / FastAPI / uvicorn |
| Frontend | Vanilla HTML/CSS/JS with marked.js, highlight.js, Chart.js, D3.js |
| Database | SQLite (observability + routine index + call graph) |
| Deployment | Render Web Service with persistent disk |
| Config | Centralized `config.py` — all values env-overridable |

## Features

- **5 Query Modes**: Query, Explain, Docs, Patterns, Translate — all with streaming SSE
- **Routine Explorer**: Searchable/filterable table with complexity metrics (LOC, variables, calls, nesting depth)
- **Complexity Metrics**: Per-routine LOC, variable count, call count, and nesting depth computed during ingestion
- **Compare Mode**: Side-by-side comparison of any two routines — metrics, callers, callees
- **Dead Code Detection**: Identifies routines with no incoming calls across BLAS/LAPACK/ScaLAPACK
- **Call Graph**: Interactive D3.js force-directed graph of routine call dependencies
- **Translation View**: Side-by-side Fortran → Python/NumPy comparison
- **Conversation Memory**: Multi-turn follow-up queries with context
- **Dashboard**: Real-time charts (latency, mode distribution, cost tracking)
- **Query Cache**: SHA-256 keyed in-memory cache with configurable TTL
- **Input Validation**: Pydantic field validators with length limits
- **Rate Limiting**: Sliding window per-IP rate limiter on query endpoints
- **Auth**: API key required on sensitive endpoints (`/ingest`, `/debug/env`)

## Local Setup

```bash
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run
uvicorn main:app --reload --port 8000

# Ingest source (one-time)
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: YOUR_ANTHROPIC_API_KEY"
```

Visit `http://localhost:8000` for the query UI and `/dashboard` for observability.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Query UI with onboarding + examples |
| GET | `/dashboard` | Observability dashboard with charts |
| GET | `/explorer` | Searchable routine browser |
| GET | `/callgraph` | Interactive call graph visualization |
| GET | `/health` | Health check |
| GET | `/docs` | Auto-generated API documentation |
| POST | `/ingest` | Run ingestion pipeline (API key required) |
| POST | `/query` | RAG query (SSE streaming, rate limited) |
| POST | `/explain` | Code explanation (SSE, rate limited) |
| POST | `/docgen` | Documentation generation (SSE, rate limited) |
| POST | `/patterns` | Pattern analysis (SSE, rate limited) |
| POST | `/translate` | Fortran→Python translation (SSE, rate limited) |
| GET | `/api/stats` | Dashboard metrics JSON |
| GET | `/api/routines` | Routine index with search/filter + complexity metrics |
| GET | `/api/routine-detail` | Full routine detail with callers/callees |
| GET | `/api/compare` | Side-by-side routine comparison |
| GET | `/api/dead-code` | Unreferenced routines (dead code detection) |
| GET | `/api/call-graph` | Call graph data with depth traversal |
| GET | `/api/cache-stats` | Cache hit/miss statistics |
| GET | `/debug/env` | Debug endpoint (API key required) |

## Project Structure

```
grimoire/
├── main.py           # FastAPI app with all routes + rate limiting
├── config.py         # Centralized configuration (env-overridable)
├── ingest.py         # Multi-source ingestion pipeline
├── retrieval.py      # RAG retrieval with score filtering
├── chunker.py        # Fortran syntax-aware chunking
├── embed.py          # Pinecone Inference API embeddings
├── llm.py            # Claude API wrapper with streaming
├── db.py             # SQLite observability + routine index + call graph
├── models.py         # Pydantic models with input validation
├── static/
│   ├── index.html    # Query UI with onboarding + conversation
│   ├── dashboard.html # Charts + metrics dashboard
│   ├── explorer.html  # Routine browser
│   └── callgraph.html # D3.js call graph
├── render.yaml       # Render deployment config
├── requirements.txt
├── AI_DEV_LOG.md     # Development log
└── AI_COST_ANALYSIS.md # Cost tracking
```
