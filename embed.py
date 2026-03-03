"""Lightweight embedding via HuggingFace Inference API (no local PyTorch)."""

import os
import time

import httpx

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
BATCH_SIZE = 32
MAX_RETRIES = 3


def _get_headers():
    token = os.getenv("HF_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using HF Inference API. Returns 384-dim vectors."""
    headers = _get_headers()
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            response = httpx.post(
                API_URL,
                json={"inputs": batch},
                headers=headers,
                timeout=120.0,
            )
            if response.status_code == 503:
                # Model is loading on HF side, wait and retry
                wait = response.json().get("estimated_time", 10)
                print(f"  HF model loading, waiting {wait:.0f}s...")
                time.sleep(min(wait, 30))
                continue
            response.raise_for_status()
            results.extend(response.json())
            break
        else:
            raise RuntimeError(f"HF Inference API failed after {MAX_RETRIES} retries")
        if (i // BATCH_SIZE) % 3 == 0 and i > 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)} chunks...")
    return results


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
