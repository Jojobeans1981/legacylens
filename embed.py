"""Embedding via Pinecone Inference API (no local models needed)."""

import os
import time
from functools import lru_cache

from config import EMBED_MODEL, EMBED_DIMENSION, EMBED_BATCH_SIZE, EMBED_MAX_RETRIES, EMBED_RATE_LIMIT_DELAY

_pc = None


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
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                embeddings = pc.inference.embed(
                    model=EMBED_MODEL,
                    inputs=batch,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                results.extend([e.values for e in embeddings])
                break
            except (ConnectionError, TimeoutError) as e:
                wait = 30 * (attempt + 1)
                print(f"  Connection error, retrying in {wait}s (attempt {attempt + 1}/{EMBED_MAX_RETRIES}): {e}")
                time.sleep(wait)
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{EMBED_MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError(f"Pinecone inference failed after {EMBED_MAX_RETRIES} retries")
        done = min(i + EMBED_BATCH_SIZE, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks...")
        # Small delay between batches to avoid rate limits
        if done < len(texts):
            time.sleep(EMBED_RATE_LIMIT_DELAY)
    return results


@lru_cache(maxsize=256)
def embed_query(text: str) -> tuple[float, ...]:
    """Embed a single query string. Cached to avoid re-embedding repeated queries."""
    pc = _get_client()
    embeddings = pc.inference.embed(
        model=EMBED_MODEL,
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"}
    )
    return tuple(embeddings[0].values)  # tuple for lru_cache hashability
