# LegacyLens

A production-ready RAG (Retrieval-Augmented Generation) system that makes the BLAS (Basic Linear Algebra Subprograms) Fortran codebase queryable via natural language.

## Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Browser UI  │────▶│  FastAPI App   │────▶│   Pinecone   │
│ (index.html) │     │  (main.py)     │     │  Vector DB   │
└──────────────┘     └───┬───────┬───┘     └──────────────┘
                         │       │
                    ┌────▼──┐ ┌──▼─────────┐
                    │SQLite │ │Claude Sonnet│
                    │  DB   │ │   (LLM)    │
                    └───────┘ └────────────┘
```

### Pipeline Flow

1. **Ingestion**: BLAS `.f` files → Fortran-aware chunker → sentence-transformers embeddings → Pinecone upsert
2. **Query**: Natural language → embed → Pinecone similarity search → context assembly → Claude Sonnet streaming response
3. **Observability**: All queries, costs, and errors logged to SQLite → real-time dashboard

## Stack

| Layer | Technology |
|-------|-----------|
| Codebase | BLAS 3.12.0 Fortran (143 files, 71K LOC) |
| Vector DB | Pinecone (serverless, cosine similarity) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (384-dim, local) |
| Answer LLM | Claude Sonnet via Anthropic API |
| Backend | Python 3.11+ / FastAPI / uvicorn |
| Frontend | Vanilla HTML/CSS/JS (no build step) |
| Database | SQLite (observability logging) |
| Deployment | Render Web Service |

## Features

- **Query**: Ask natural language questions about BLAS code
- **Explain**: Get plain-English explanations of Fortran routines
- **Docs**: Auto-generate documentation headers for routines
- **Patterns**: Analyze coding patterns and conventions
- **Translate**: Get Python/NumPy equivalents of Fortran code
- **Dashboard**: Real-time observability with cost tracking

## Local Setup

```bash
# Clone and install
cd legacylens
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Download BLAS source
mkdir -p data/blas_source
curl -sL https://www.netlib.org/blas/blas.tgz | tar xz -C data/blas_source

# Run
uvicorn main:app --reload --port 8000

# Ingest BLAS source (one-time)
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: YOUR_ANTHROPIC_API_KEY"
```

Visit `http://localhost:8000` for the query UI and `http://localhost:8000/dashboard` for observability.

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on Render pointing to your repo
3. Set environment variables: `PINECONE_API_KEY`, `ANTHROPIC_API_KEY`
4. Render will auto-deploy using `render.yaml`
5. After deploy, trigger ingestion via `POST /ingest`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Query UI |
| GET | `/dashboard` | Observability dashboard |
| GET | `/health` | Health check |
| POST | `/ingest` | Run ingestion pipeline |
| POST | `/query` | RAG query (SSE streaming) |
| POST | `/explain` | Code explanation |
| POST | `/docgen` | Documentation generation |
| POST | `/patterns` | Pattern analysis |
| POST | `/translate` | Fortran→Python translation |
| GET | `/api/stats` | Dashboard metrics JSON |
| GET | `/api/recent-queries` | Last 20 queries |
| GET | `/api/errors` | Last 10 errors |

## Project Structure

```
legacylens/
├── main.py           # FastAPI app with all routes
├── ingest.py         # Ingestion pipeline
├── retrieval.py      # RAG retrieval pipeline
├── chunker.py        # Fortran syntax-aware chunking
├── llm.py            # Claude API wrapper with streaming
├── db.py             # SQLite observability logging
├── models.py         # Pydantic models
├── static/
│   ├── index.html    # Query UI
│   └── dashboard.html # Observability dashboard
├── render.yaml       # Render deployment config
├── requirements.txt
├── AI_DEV_LOG.md     # Development log
└── AI_COST_ANALYSIS.md # Cost tracking
```
