"""Ingestion pipeline: discover, chunk, embed, and upsert BLAS source to Pinecone."""

import os
import time
from pathlib import Path

from chunker import chunk_fortran_file
from models import Chunk, IngestResult
from db import log_ingestion


def discover_fortran_files(source_dir: str) -> list[Path]:
    """Recursively find all Fortran source files."""
    source_path = Path(source_dir)
    extensions = {'.f', '.f90', '.for', '.F', '.F90'}
    files = []
    for ext in extensions:
        files.extend(source_path.rglob(f'*{ext}'))
    # Deduplicate, filter macOS resource forks, and sort
    files = sorted(f for f in set(files) if not f.name.startswith('._'))
    return files


def connect_pinecone():
    """Connect to Pinecone and return the index."""
    from pinecone import Pinecone, ServerlessSpec

    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legacylens-blas")

    pc = Pinecone(api_key=api_key)

    # Check if index exists, create if not
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,  # all-MiniLM-L6-v2 output dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        # Wait for index to be ready
        import time as _time
        _time.sleep(5)

    return pc.Index(index_name)


def run_ingestion(source_dir: str, model=None, index=None) -> IngestResult:
    """Full ingestion pipeline: discover -> chunk -> embed -> upsert.

    Args:
        source_dir: Path to BLAS source directory
        model: Pre-loaded SentenceTransformer model (or loads one)
        index: Pre-connected Pinecone index (or connects)
    """
    start_time = time.time()

    # Load model if not provided
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Connect to Pinecone if not provided
    if index is None:
        index = connect_pinecone()

    # Discover files
    files = discover_fortran_files(source_dir)
    print(f"Found {len(files)} Fortran files")

    # Chunk all files
    all_chunks: list[Chunk] = []
    files_processed = []
    for filepath in files:
        chunks = chunk_fortran_file(filepath)
        # Update file_path to be relative to source_dir
        source_path = Path(source_dir)
        for chunk in chunks:
            try:
                chunk.file_path = str(filepath.relative_to(source_path))
            except ValueError:
                chunk.file_path = filepath.name
        all_chunks.extend(chunks)
        files_processed.append(str(filepath.relative_to(source_path)
                                   if filepath.is_relative_to(source_path)
                                   else filepath.name))

    print(f"Generated {len(all_chunks)} chunks from {len(files)} files")

    if not all_chunks:
        duration = time.time() - start_time
        log_ingestion(len(files), 0, duration)
        return IngestResult(
            file_count=len(files),
            chunk_count=0,
            duration_seconds=duration,
            files_processed=files_processed
        )

    # Embed all chunks
    print("Embedding chunks...")
    contents = [c.content for c in all_chunks]
    embeddings = model.encode(contents, batch_size=64, show_progress_bar=True)

    # Upsert to Pinecone in batches
    print("Upserting to Pinecone...")
    batch_size = 100
    for batch_start in range(0, len(all_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(all_chunks))
        vectors = []
        for i in range(batch_start, batch_end):
            chunk = all_chunks[i]
            embedding = embeddings[i]
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding.tolist(),
                "metadata": {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "routine_name": chunk.routine_name or "",
                    "content": chunk.content[:1000],
                }
            })
        index.upsert(vectors=vectors)
        print(f"  Upserted batch {batch_start // batch_size + 1}/"
              f"{(len(all_chunks) + batch_size - 1) // batch_size}")

    duration = time.time() - start_time
    print(f"Ingestion complete in {duration:.1f}s")

    # Log to SQLite
    log_ingestion(len(files), len(all_chunks), duration)

    return IngestResult(
        file_count=len(files),
        chunk_count=len(all_chunks),
        duration_seconds=round(duration, 2),
        files_processed=files_processed
    )
