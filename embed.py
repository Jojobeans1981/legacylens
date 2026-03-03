"""Embedding via Pinecone Inference API (no local models needed)."""

import os

_pc = None

EMBED_MODEL = "multilingual-e5-large"
EMBED_DIMENSION = 1024


def _get_client():
    global _pc
    if _pc is None:
        from pinecone import Pinecone
        _pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
    return _pc


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts for indexing. Returns 1024-dim vectors."""
    pc = _get_client()
    results = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=batch,
            parameters={"input_type": "passage", "truncate": "END"}
        )
        results.extend([e.values for e in embeddings])
        if i > 0 and (i // batch_size) % 2 == 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)} chunks...")
    return results


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    pc = _get_client()
    embeddings = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"}
    )
    return embeddings[0].values
