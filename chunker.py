"""Fortran syntax-aware chunking for BLAS source files."""

import re
from pathlib import Path
from models import Chunk
from config import MAX_CHUNK_LINES, OVERLAP_LINES, FALLBACK_CHUNK_LINES

# Regex patterns for Fortran routine boundaries
ROUTINE_START = re.compile(
    r'^\s*(SUBROUTINE|FUNCTION|PROGRAM|'
    r'(?:INTEGER|REAL|DOUBLE\s+PRECISION|COMPLEX(?:\*\d+)?|LOGICAL|CHARACTER)\s+FUNCTION)\s+(\w+)',
    re.IGNORECASE | re.MULTILINE
)
ROUTINE_END = re.compile(
    r'^\s*END\s+(SUBROUTINE|FUNCTION|PROGRAM)\s*(\w*)\s*$|^\s*END\s*$',
    re.IGNORECASE
)

# Comment line patterns (Fortran 77 uses C/c/* in column 1, free-form uses !)
COMMENT_LINE = re.compile(r'^[Cc*!]', re.MULTILINE)


def _is_comment_line(line: str) -> bool:
    """Check if a line is a Fortran comment."""
    if not line:
        return False
    return line[0] in ('C', 'c', '*', '!') or line.strip().startswith('!')


def _extract_preceding_comments(lines: list[str], routine_start_idx: int) -> tuple[list[str], int]:
    """Extract the comment block immediately preceding a routine start.

    Returns (comment_lines, first_comment_line_index).
    """
    comments = []
    idx = routine_start_idx - 1
    while idx >= 0 and (_is_comment_line(lines[idx]) or lines[idx].strip() == ''):
        if _is_comment_line(lines[idx]):
            comments.insert(0, lines[idx])
        elif lines[idx].strip() == '' and comments:
            # Allow blank lines within comment block
            comments.insert(0, lines[idx])
        else:
            break
        idx -= 1

    # Trim leading blank lines from comments
    while comments and comments[0].strip() == '':
        comments.pop(0)
        idx += 1

    first_comment_idx = routine_start_idx - len(comments)
    return comments, first_comment_idx


def _split_large_chunk(lines: list[str], file_path: str, start_line: int,
                       chunk_type: str, routine_name: str | None) -> list[Chunk]:
    """Split a large chunk into smaller pieces with overlap."""
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + MAX_CHUNK_LINES, len(lines))
        chunk_lines = lines[i:end]
        chunks.append(Chunk(
            content='\n'.join(chunk_lines),
            file_path=file_path,
            start_line=start_line + i,
            end_line=start_line + end - 1,
            chunk_type=chunk_type,
            routine_name=routine_name
        ))
        if end >= len(lines):
            break
        i = end - OVERLAP_LINES
    return chunks


def chunk_fortran_file(filepath: str | Path) -> list[Chunk]:
    """Parse a Fortran file and produce semantically meaningful chunks.

    Strategy:
    1. Detect subroutine/function boundaries
    2. Include preceding comment blocks with each routine
    3. Split large routines with overlap
    4. Fall back to fixed-size chunks for files with no routines
    """
    filepath = Path(filepath)

    # Read file, handling encoding issues
    try:
        content = filepath.read_text(encoding='utf-8', errors='replace')
    except UnicodeDecodeError:
        try:
            content = filepath.read_text(encoding='latin-1', errors='replace')
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: cannot read {filepath}: {e}")
            return []

    lines = content.split('\n')
    if not lines or (len(lines) == 1 and not lines[0].strip()):
        return []

    rel_path = filepath.name  # Will be updated to relative path during ingestion

    # Find all routine starts and ends
    routines = []
    i = 0
    while i < len(lines):
        match = ROUTINE_START.match(lines[i])
        if match:
            # Determine routine type and name
            type_str = match.group(1).strip().upper()
            name = match.group(2)

            if 'FUNCTION' in type_str:
                rtype = 'function'
            elif 'SUBROUTINE' in type_str:
                rtype = 'subroutine'
            else:
                rtype = 'subroutine'

            # Find the matching END
            end_idx = None
            for j in range(i + 1, len(lines)):
                end_match = ROUTINE_END.match(lines[j])
                if end_match:
                    # Check if it's a matching end (or generic END)
                    end_type = (end_match.group(1) or '').strip().upper()
                    if not end_type or end_type in type_str.upper() or 'FUNCTION' in end_type or 'SUBROUTINE' in end_type:
                        end_idx = j
                        break

            if end_idx is None:
                end_idx = len(lines) - 1

            # Extract preceding comments
            comment_lines, comment_start = _extract_preceding_comments(lines, i)

            actual_start = comment_start if comment_lines else i
            routines.append({
                'start': actual_start,
                'end': end_idx,
                'type': rtype,
                'name': name,
            })
            i = end_idx + 1
        else:
            i += 1

    # If no routines found, fall back to fixed-size chunks
    if not routines:
        return _fallback_chunks(lines, rel_path)

    chunks = []

    # Process each routine
    for routine in routines:
        routine_lines = lines[routine['start']:routine['end'] + 1]

        if len(routine_lines) > MAX_CHUNK_LINES:
            chunks.extend(_split_large_chunk(
                routine_lines, rel_path,
                routine['start'] + 1,  # 1-indexed
                routine['type'], routine['name']
            ))
        else:
            chunks.append(Chunk(
                content='\n'.join(routine_lines),
                file_path=rel_path,
                start_line=routine['start'] + 1,
                end_line=routine['end'] + 1,
                chunk_type=routine['type'],
                routine_name=routine['name']
            ))

    return chunks


def _fallback_chunks(lines: list[str], file_path: str) -> list[Chunk]:
    """Create fixed-size chunks for files with no detected routines."""
    chunks = []
    i = 0
    while i < len(lines):
        end = min(i + FALLBACK_CHUNK_LINES, len(lines))
        chunk_lines = lines[i:end]
        content = '\n'.join(chunk_lines)
        if content.strip():
            chunks.append(Chunk(
                content=content,
                file_path=file_path,
                start_line=i + 1,
                end_line=end,
                chunk_type='fallback',
                routine_name=None
            ))
        if end >= len(lines):
            break
        i = end - OVERLAP_LINES
    return chunks
