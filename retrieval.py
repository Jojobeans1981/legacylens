"""RAG retrieval pipeline: embed query -> search Pinecone -> assemble context."""

import re

from models import RetrievalChunk, RetrievalResult
from embed import embed_query
from config import SCORE_THRESHOLD, DEFAULT_TOP_K

# Pattern to detect routine names in queries (uppercase Fortran identifiers)
_ROUTINE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{2,})\b')


def _extract_routine_name(query: str) -> str | None:
    """Try to extract a specific routine name from the query."""
    matches = _ROUTINE_PATTERN.findall(query.upper())
    # Filter out common English words that match the pattern
    stopwords = {'THE', 'AND', 'FOR', 'HOW', 'WHAT', 'DOES', 'WHY', 'ALL', 'ARE', 'NOT', 'CAN', 'HAS'}
    for m in matches:
        if m not in stopwords and len(m) >= 3:
            return m
    return None


def retrieve(query: str, index,
             top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """Embed query, search Pinecone, filter and assemble context.

    Args:
        query: Natural language query string
        index: Connected Pinecone index
        top_k: Number of results to return after filtering

    Returns:
        RetrievalResult with chunks, assembled context, and found flag
    """
    # Embed query via Pinecone Inference API (cached)
    query_vector = list(embed_query(query))

    # Try exact routine match first
    routine_name = _extract_routine_name(query)
    filter_dict = None
    if routine_name:
        filter_dict = {"routine_name": {"$eq": routine_name}}

    # Query Pinecone — try with filter first, fall back to unfiltered
    results = index.query(
        vector=query_vector,
        top_k=top_k + 3,
        include_metadata=True,
        filter=filter_dict,
    )

    # If filtered search returned no results, retry without filter
    if filter_dict and not results.get("matches"):
        results = index.query(
            vector=query_vector,
            top_k=top_k + 3,
            include_metadata=True,
        )

    # Filter by score threshold and take top_k
    chunks = []
    for match in results.get("matches", []):
        if match["score"] < SCORE_THRESHOLD:
            continue
        meta = match.get("metadata", {})
        chunks.append(RetrievalChunk(
            content=meta.get("content", ""),
            file_path=meta.get("file_path", "unknown"),
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
            chunk_type=meta.get("chunk_type", "unknown"),
            routine_name=meta.get("routine_name", ""),
            score=round(match["score"], 4)
        ))

    chunks = chunks[:top_k]

    if not chunks:
        return RetrievalResult(chunks=[], context="", found=False)

    # Assemble context string (cap per-chunk to reduce input tokens)
    MAX_CONTEXT_PER_CHUNK = 800
    context_parts = []
    for chunk in chunks:
        content = chunk.content[:MAX_CONTEXT_PER_CHUNK]
        context_parts.append(
            f"[Source: {chunk.file_path} lines {chunk.start_line}-{chunk.end_line}]\n"
            f"{content}\n---"
        )
    context = "\n\n".join(context_parts)

    return RetrievalResult(chunks=chunks, context=context, found=True)
