"""Ingestion pipeline: discover, chunk, embed, and upsert BLAS source to Pinecone."""

import os
import re
import time
from pathlib import Path

from chunker import chunk_fortran_file
from models import Chunk, IngestResult
from db import log_ingestion, log_routines, log_call_graph
from embed import embed_texts


def _parse_call_graph(chunks: list[Chunk]) -> list[tuple]:
    """Extract CALL statements from Fortran chunks to build a call graph."""
    call_pattern = re.compile(r'^\s*CALL\s+(\w+)', re.IGNORECASE | re.MULTILINE)
    edges = []
    for chunk in chunks:
        if not chunk.routine_name:
            continue
        caller = chunk.routine_name.upper()
        for match in call_pattern.finditer(chunk.content):
            callee = match.group(1).upper()
            if callee != caller:  # skip self-calls
                # Estimate line number
                line_offset = chunk.content[:match.start()].count('\n')
                edges.append((caller, callee, chunk.file_path, chunk.start_line + line_offset))
    return edges


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
    from embed import EMBED_DIMENSION

    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legacylens-blas")

    pc = Pinecone(api_key=api_key)

    # Check if index exists with correct dimensions
    existing = {idx.name: idx for idx in pc.list_indexes()}
    if index_name in existing:
        if existing[index_name].dimension != EMBED_DIMENSION:
            print(f"Index dimension mismatch ({existing[index_name].dimension} vs {EMBED_DIMENSION}), recreating...")
            pc.delete_index(index_name)
            time.sleep(5)
        else:
            return pc.Index(index_name)

    pc.create_index(
        name=index_name,
        dimension=EMBED_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

    return pc.Index(index_name)


def run_ingestion(source_dirs: list[str], index=None) -> IngestResult:
    """Full ingestion pipeline: discover -> chunk -> embed -> upsert.

    Args:
        source_dirs: List of source directory paths to ingest
        index: Pre-connected Pinecone index (or connects)
    """
    start_time = time.time()

    # Connect to Pinecone if not provided
    if index is None:
        index = connect_pinecone()

    # Discover files from all source directories
    files = []
    for source_dir in source_dirs:
        found = discover_fortran_files(source_dir)
        print(f"Found {len(found)} Fortran files in {source_dir}")
        files.extend(found)
    print(f"Total: {len(files)} Fortran files across {len(source_dirs)} directories")

    # Chunk all files
    all_chunks: list[Chunk] = []
    files_processed = []
    for filepath in files:
        chunks = chunk_fortran_file(filepath)
        # Find which source_dir this file belongs to for relative paths
        rel_path = filepath.name
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            try:
                rel_path = str(filepath.relative_to(source_path))
                break
            except ValueError:
                continue
        for chunk in chunks:
            chunk.file_path = rel_path
        all_chunks.extend(chunks)
        files_processed.append(rel_path)

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

    # Embed all chunks via Pinecone Inference API
    print("Embedding chunks via Pinecone Inference API...")
    contents = [c.content for c in all_chunks]
    embeddings = embed_texts(contents)

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
                "values": embedding if isinstance(embedding, list) else embedding.tolist(),
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

    # Populate routine index and call graph
    print("Building routine index and call graph...")
    log_routines(all_chunks)
    edges = _parse_call_graph(all_chunks)
    log_call_graph(edges)
    print(f"  {len([c for c in all_chunks if c.routine_name])} routines indexed, {len(edges)} call edges found")

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
