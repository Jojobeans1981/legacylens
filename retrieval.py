"""RAG retrieval pipeline: embed query -> search Pinecone -> assemble context."""

from models import RetrievalChunk, RetrievalResult
from embed import embed_query
from config import SCORE_THRESHOLD, DEFAULT_TOP_K


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
    # Embed query via HF Inference API
    query_vector = embed_query(query)

    # Query Pinecone (fetch extra to allow for filtering)
    results = index.query(
        vector=query_vector,
        top_k=top_k + 3,
        include_metadata=True
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

    # Assemble context string
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk.file_path} lines {chunk.start_line}-{chunk.end_line}]\n"
            f"{chunk.content}\n---"
        )
    context = "\n\n".join(context_parts)

    return RetrievalResult(chunks=chunks, context=context, found=True)
