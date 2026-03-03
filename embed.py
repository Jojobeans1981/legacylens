"""Embedding via Pinecone Inference API (no local models needed)."""

import os
import time

_pc = None

EMBED_MODEL = "multilingual-e5-large"
EMBED_DIMENSION = 1024
BATCH_SIZE = 20  # Small batches to stay under rate limits
MAX_RETRIES = 5


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
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                embeddings = pc.inference.embed(
                    model=EMBED_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                results.extend([e.values for e in embeddings])
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Pinecone inference failed after {MAX_RETRIES} retries")
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...")
        # Small delay between batches to avoid rate limits
        if done < len(texts):
            time.sleep(2)
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
