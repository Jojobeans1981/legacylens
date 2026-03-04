"""RAG retrieval pipeline: hybrid search (vector + BM25) with heuristic re-ranking."""

import json
import re
import time as _time

from rank_bm25 import BM25Okapi

from models import RetrievalChunk, RetrievalResult
from embed import embed_query
from config import SCORE_THRESHOLD, DEFAULT_TOP_K, VECTOR_WEIGHT, BM25_WEIGHT

# Pattern to detect routine names in queries (uppercase Fortran identifiers)
_ROUTINE_PATTERN = re.compile(r'\b([A-Z][A-Z0-9]{2,})\b')

# Module-level BM25 index (lazily loaded)
_bm25_index = None
_bm25_chunks = None


def _extract_routine_name(query: str) -> str | None:
    """Try to extract a specific routine name from the query."""
    matches = _ROUTINE_PATTERN.findall(query.upper())
    stopwords = {'THE', 'AND', 'FOR', 'HOW', 'WHAT', 'DOES', 'WHY', 'ALL', 'ARE', 'NOT', 'CAN', 'HAS'}
    for m in matches:
        if m not in stopwords and len(m) >= 3:
            return m
    return None


def _get_bm25_index():
    """Lazily build/return BM25 index from chunk_content table."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    from db import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT chunk_id, routine_name, file_path, chunk_type, start_line, end_line, content FROM chunk_content"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None, None

    _bm25_chunks = [dict(r) for r in rows]
    tokenized = [r["content"].lower().split() for r in _bm25_chunks]
    _bm25_index = BM25Okapi(tokenized)
    return _bm25_index, _bm25_chunks


def invalidate_bm25_cache():
    """Call after re-ingestion to rebuild BM25 index."""
    global _bm25_index, _bm25_chunks
    _bm25_index = None
    _bm25_chunks = None


def _bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Run BM25 keyword search, return scored results."""
    bm25, chunks = _get_bm25_index()
    if bm25 is None or chunks is None:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = [(scores[i], chunks[i]) for i in range(len(chunks))]
    scored.sort(key=lambda x: x[0], reverse=True)

    max_score = scored[0][0] if scored and scored[0][0] > 0 else 1.0
    results = []
    for score, chunk in scored[:top_k]:
        if score <= 0:
            break
        results.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "file_path": chunk["file_path"],
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "chunk_type": chunk["chunk_type"],
            "routine_name": chunk["routine_name"],
            "bm25_score": score / max_score,
        })
    return results


def _rerank_chunks(chunks: list[RetrievalChunk], query: str) -> list[RetrievalChunk]:
    """Heuristic re-ranker: boost chunks based on relevance signals."""
    routine_name = _extract_routine_name(query)
    query_terms = set(query.lower().split())
    stopwords = {'the', 'and', 'for', 'how', 'what', 'does', 'why', 'all', 'are', 'not',
                 'can', 'has', 'is', 'in', 'of', 'to', 'a', 'an', 'it', 'this', 'that', 'with'}
    query_terms -= stopwords

    reranked = []
    for chunk in chunks:
        boost = 0.0

        # 1. Exact routine name match
        if routine_name and chunk.routine_name and chunk.routine_name.upper() == routine_name:
            boost += 0.15

        # 2. Query term density
        if query_terms:
            content_lower = chunk.content.lower()
            matches = sum(1 for t in query_terms if t in content_lower)
            term_density = matches / len(query_terms)
            boost += term_density * 0.10

        # 3. Chunk type boost
        type_boosts = {"subroutine": 0.05, "function": 0.05, "comment_block": 0.02, "fallback": 0.0}
        boost += type_boosts.get(chunk.chunk_type, 0.0)

        # 4. Partial routine name match
        if chunk.routine_name:
            rname = chunk.routine_name.upper()
            if any(rname in t.upper() or t.upper() in rname for t in query_terms if len(t) >= 3):
                boost += 0.10

        reranked.append(RetrievalChunk(
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            chunk_type=chunk.chunk_type,
            routine_name=chunk.routine_name,
            score=round(chunk.score + boost, 4),
        ))

    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked


def retrieve(query: str, index,
             top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """Hybrid retrieval: vector search + BM25 keyword search, merged with RRF, then re-ranked."""
    # --- Vector search ---
    query_vector = list(embed_query(query))
    routine_name = _extract_routine_name(query)
    filter_dict = {"routine_name": {"$eq": routine_name}} if routine_name else None

    results = index.query(
        vector=query_vector, top_k=top_k + 5,
        include_metadata=True, filter=filter_dict,
    )
    if filter_dict and not results.get("matches"):
        results = index.query(
            vector=query_vector, top_k=top_k + 5, include_metadata=True,
        )

    # Build vector results keyed by file_path:start_line
    vector_results = {}
    for rank, match in enumerate(results.get("matches", [])):
        if match["score"] < SCORE_THRESHOLD:
            continue
        meta = match.get("metadata", {})
        key = f"{meta.get('file_path', '')}:{meta.get('start_line', 0)}"
        vector_results[key] = {
            "content": meta.get("content", ""),
            "file_path": meta.get("file_path", "unknown"),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "chunk_type": meta.get("chunk_type", "unknown"),
            "routine_name": meta.get("routine_name", ""),
            "vector_score": match["score"],
            "vector_rank": rank,
        }

    # --- BM25 keyword search ---
    bm25_results = {}
    for rank, item in enumerate(_bm25_search(query, top_k=top_k + 5)):
        key = f"{item['file_path']}:{item['start_line']}"
        bm25_results[key] = {**item, "bm25_rank": rank}

    # --- Reciprocal Rank Fusion ---
    K = 60
    all_keys = set(vector_results.keys()) | set(bm25_results.keys())
    fused = []
    for key in all_keys:
        rrf_score = 0.0
        vec = vector_results.get(key)
        bm25 = bm25_results.get(key)

        if vec:
            rrf_score += VECTOR_WEIGHT / (K + vec["vector_rank"])
        if bm25:
            rrf_score += BM25_WEIGHT / (K + bm25["bm25_rank"])

        source = vec or bm25
        fused.append({
            "content": source["content"],
            "file_path": source["file_path"],
            "start_line": source["start_line"],
            "end_line": source["end_line"],
            "chunk_type": source["chunk_type"],
            "routine_name": source["routine_name"],
            "score": round(vec["vector_score"] if vec else bm25.get("bm25_score", 0.0), 4),
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    fused = fused[:top_k]

    # Convert to RetrievalChunk list
    chunks = [RetrievalChunk(
        content=r["content"], file_path=r["file_path"],
        start_line=r["start_line"], end_line=r["end_line"],
        chunk_type=r["chunk_type"], routine_name=r["routine_name"],
        score=r["score"],
    ) for r in fused]

    # Re-rank using heuristics
    chunks = _rerank_chunks(chunks, query)

    if not chunks:
        return RetrievalResult(chunks=[], context="", found=False)

    # Assemble context
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


def run_evaluation(index, top_k: int = DEFAULT_TOP_K) -> dict:
    """Run ground truth evaluation: retrieve for each test query, compute MRR and Hit@k."""
    from db import get_ground_truth, save_eval_result

    test_cases = get_ground_truth()
    if not test_cases:
        return {"error": "No ground truth data. Call /eval/seed first."}

    details = []
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total_latency = 0.0

    for tc in test_cases:
        expected = set(r.strip().upper() for r in tc["expected_routines"].split(","))

        start = _time.time()
        result = retrieve(tc["query"], index, top_k=top_k)
        latency_ms = (_time.time() - start) * 1000
        total_latency += latency_ms

        retrieved_routines = [c.routine_name.upper() for c in result.chunks if c.routine_name]

        first_hit_rank = None
        for i, rname in enumerate(retrieved_routines):
            if rname in expected:
                first_hit_rank = i + 1
                break

        rr = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        reciprocal_ranks.append(rr)

        if first_hit_rank and first_hit_rank <= 1:
            hits_at_1 += 1
        if first_hit_rank and first_hit_rank <= 3:
            hits_at_3 += 1
        if first_hit_rank and first_hit_rank <= 5:
            hits_at_5 += 1

        details.append({
            "query": tc["query"],
            "expected": list(expected),
            "retrieved": retrieved_routines[:5],
            "first_hit_rank": first_hit_rank,
            "reciprocal_rank": round(rr, 4),
            "latency_ms": round(latency_ms, 1),
            "scores": [round(c.score, 4) for c in result.chunks[:5]],
        })

    n = len(test_cases)
    mrr = sum(reciprocal_ranks) / n
    eval_result = {
        "total_queries": n,
        "mrr": round(mrr, 4),
        "hit_at_1": round(hits_at_1 / n, 4),
        "hit_at_3": round(hits_at_3 / n, 4),
        "hit_at_5": round(hits_at_5 / n, 4),
        "avg_latency_ms": round(total_latency / n, 1),
        "details": details,
    }

    save_eval_result(
        run_name=f"eval_{_time.strftime('%Y%m%d_%H%M%S')}",
        total_queries=n,
        mrr=eval_result["mrr"],
        hit_at_1=eval_result["hit_at_1"],
        hit_at_3=eval_result["hit_at_3"],
        hit_at_5=eval_result["hit_at_5"],
        avg_latency_ms=eval_result["avg_latency_ms"],
        details=json.dumps(details),
    )

    return eval_result
