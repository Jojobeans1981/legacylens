"""Centralized configuration for GRIMOIRE. All values overridable via env vars."""

import os

# ─── Cache ───────────────────────────────────────────────────────────────────
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "500"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds

# ─── Retrieval ───────────────────────────────────────────────────────────────
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.35"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))

# ─── LLM ─────────────────────────────────────────────────────────────────────
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
INPUT_COST_PER_MTOK = float(os.getenv("INPUT_COST_PER_MTOK", "0.80"))
OUTPUT_COST_PER_MTOK = float(os.getenv("OUTPUT_COST_PER_MTOK", "4.0"))

# ─── Embedding ───────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "multilingual-e5-large")
EMBED_DIMENSION = int(os.getenv("EMBED_DIMENSION", "1024"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "20"))
EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "5"))
EMBED_RATE_LIMIT_DELAY = float(os.getenv("EMBED_RATE_LIMIT_DELAY", "2.0"))

# ─── Chunker ─────────────────────────────────────────────────────────────────
MAX_CHUNK_LINES = int(os.getenv("MAX_CHUNK_LINES", "500"))
OVERLAP_LINES = int(os.getenv("OVERLAP_LINES", "64"))
FALLBACK_CHUNK_LINES = int(os.getenv("FALLBACK_CHUNK_LINES", "400"))

# ─── Ingestion ───────────────────────────────────────────────────────────────
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "./data/legacylens.db")

# ─── Source ──────────────────────────────────────────────────────────────────
SOURCE_DIRS = os.getenv("SOURCE_DIRS", "./data/blas_source,./data/lapack_source,./data/scalapack_source")

# ─── Rate Limiting ───────────────────────────────────────────────────────────
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "30"))

# ─── Validation ──────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 2000
MAX_CONVERSATION_TURNS = 10
MAX_CONVERSATION_CONTENT_LENGTH = 5000
MAX_CALL_GRAPH_DEPTH = 4
MAX_SEARCH_LENGTH = 200
MAX_ROUTINE_LENGTH = 50
ALLOWED_LIBRARIES = {"BLAS", "LAPACK", "ScaLAPACK"}
