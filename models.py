"""Pydantic models for LegacyLens."""

from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # "subroutine" | "function" | "comment_block" | "fallback"
    routine_name: Optional[str] = None


class IngestResult(BaseModel):
    file_count: int
    chunk_count: int
    duration_seconds: float
    files_processed: list[str]


class QueryRequest(BaseModel):
    query: str
    routine: Optional[str] = None
    file: Optional[str] = None
    conversation_history: Optional[list[dict]] = None


class RetrievalChunk(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    routine_name: str
    score: float


class RetrievalResult(BaseModel):
    chunks: list[RetrievalChunk]
    context: str
    found: bool


class StatsResponse(BaseModel):
    total_queries: int
    avg_latency_ms: float
    total_cost_usd: float
    chunks_indexed: int
    files_covered: int
    score_distribution: list[dict]
    recent_queries: list[dict]
    errors: list[dict]
    ingestion_history: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_connected: bool
