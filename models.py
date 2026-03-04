"""Pydantic models for GRIMOIRE."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional

from config import MAX_QUERY_LENGTH, MAX_CONVERSATION_TURNS, MAX_CONVERSATION_CONTENT_LENGTH


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
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    routine: Optional[str] = Field(None, max_length=50)
    file: Optional[str] = Field(None, max_length=500)
    conversation_history: Optional[list[dict]] = None

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("conversation_history")
    @classmethod
    def validate_conversation_history(cls, v):
        if v is None:
            return v
        if len(v) > MAX_CONVERSATION_TURNS:
            v = v[-MAX_CONVERSATION_TURNS:]
        for turn in v:
            if turn.get("role") not in ("user", "assistant"):
                raise ValueError("Invalid role in conversation history")
            content = turn.get("content")
            if not isinstance(content, str) or len(content) > MAX_CONVERSATION_CONTENT_LENGTH:
                raise ValueError("Invalid or oversized content in conversation history")
        return v


class RetrievalChunk(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    routine_name: Optional[str] = ""
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


class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    mode: Optional[str] = "query"
    feedback: str = Field(..., pattern=r"^(up|down)$")
    comment: Optional[str] = Field("", max_length=500)
