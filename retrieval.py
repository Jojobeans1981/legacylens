"""RAG retrieval pipeline: embed query -> search Pinecone -> assemble context."""

from models import RetrievalChunk, RetrievalResult

SCORE_THRESHOLD = 0.35
DEFAULT_TOP_K = 5


def retrieve(query: str, model, index,
             top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """Embed query, search Pinecone, filter and assemble context.

    Args:
        query: Natural language query string
        model: Pre-loaded SentenceTransformer model
        index: Connected Pinecone index
        top_k: Number of results to return after filtering

    Returns:
        RetrievalResult with chunks, assembled context, and found flag
    """
    # Embed query
    query_vector = model.encode(query).tolist()

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
