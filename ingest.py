"""Ingestion pipeline: discover, chunk, embed, and upsert BLAS source to Pinecone."""

import os
import re
import time
from pathlib import Path

from chunker import chunk_fortran_file
from models import Chunk, IngestResult
from db import log_ingestion, log_routines, log_call_graph
from embed import embed_texts
from config import UPSERT_BATCH_SIZE, PINECONE_CLOUD, PINECONE_REGION, EMBED_DIMENSION


def _is_comment_line(line: str) -> bool:
    """Check if a line is a Fortran comment."""
    if not line:
        return False
    return line[0] in ('C', 'c', '*', '!') or line.strip().startswith('!')


def _max_nesting(lines: list[str]) -> int:
    """Compute maximum IF/DO nesting depth in Fortran code."""
    depth = 0
    max_depth = 0
    for line in lines:
        stripped = line.strip().upper()
        if _is_comment_line(line) or not stripped:
            continue
        # Openers: IF...THEN, DO, DO WHILE
        if re.match(r'.*\bIF\b.*\bTHEN\b', stripped):
            depth += 1
        elif re.match(r'^DO\b', stripped):
            depth += 1
        # Closers: END IF, END DO, ENDDO, ENDIF
        if re.match(r'^END\s*(IF|DO)\b', stripped) or stripped in ('ENDDO', 'ENDIF'):
            depth = max(0, depth - 1)
        max_depth = max(max_depth, depth)
    return max_depth


def _compute_complexity(chunk: Chunk) -> dict:
    """Compute complexity metrics for a routine chunk."""
    lines = chunk.content.split('\n')
    loc = len([l for l in lines if l.strip() and not _is_comment_line(l)])
    call_count = len(re.findall(r'^\s*CALL\s+\w+', chunk.content, re.IGNORECASE | re.MULTILINE))
    var_count = len(re.findall(
        r'^\s*(INTEGER|REAL|DOUBLE\s+PRECISION|COMPLEX(\*\d+)?|LOGICAL|CHARACTER)',
        chunk.content, re.IGNORECASE | re.MULTILINE
    ))
    nesting_depth = _max_nesting(lines)
    return {
        "loc": loc,
        "var_count": var_count,
        "call_count": call_count,
        "nesting_depth": nesting_depth,
    }


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
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
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
    source_dir_map = {}  # file_path -> source_dir for library detection
    for filepath in files:
        chunks = chunk_fortran_file(filepath)
        # Find which source_dir this file belongs to for relative paths
        rel_path = filepath.name
        matched_source_dir = ""
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            try:
                rel_path = str(filepath.relative_to(source_path))
                matched_source_dir = source_dir
                break
            except ValueError:
                continue
        for chunk in chunks:
            chunk.file_path = rel_path
        source_dir_map[rel_path] = matched_source_dir
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
    for batch_start in range(0, len(all_chunks), UPSERT_BATCH_SIZE):
        batch_end = min(batch_start + UPSERT_BATCH_SIZE, len(all_chunks))
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
        print(f"  Upserted batch {batch_start // UPSERT_BATCH_SIZE + 1}/"
              f"{(len(all_chunks) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}")

    # Compute complexity metrics for each routine
    print("Computing complexity metrics...")
    routine_metrics = {}
    for chunk in all_chunks:
        if chunk.routine_name:
            routine_metrics[chunk.routine_name] = _compute_complexity(chunk)

    # Populate routine index and call graph
    print("Building routine index and call graph...")
    log_routines(all_chunks, metrics=routine_metrics, source_dir_map=source_dir_map)
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
