"""Tests for Fortran syntax-aware chunker."""

import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from chunker import chunk_fortran_file


def test_simple_subroutine():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f', delete=False) as f:
        f.write("      SUBROUTINE FOO(X, Y)\n      X = Y\n      END SUBROUTINE FOO\n")
        f.flush()
        try:
            chunks = chunk_fortran_file(f.name)
            assert len(chunks) == 1
            assert chunks[0].chunk_type == "subroutine"
            assert chunks[0].routine_name == "FOO"
        finally:
            os.unlink(f.name)


def test_function_with_type():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f', delete=False) as f:
        f.write("      DOUBLE PRECISION FUNCTION BAR(X)\n      BAR = X * 2.0D0\n      END FUNCTION BAR\n")
        f.flush()
        try:
            chunks = chunk_fortran_file(f.name)
            assert len(chunks) == 1
            assert chunks[0].chunk_type == "function"
            assert chunks[0].routine_name == "BAR"
        finally:
            os.unlink(f.name)


def test_preceding_comments():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f', delete=False) as f:
        f.write("C  This is a comment\nC  Another comment\nC  Third comment\n      SUBROUTINE COMMENTED(A)\n      A = 1\n      END SUBROUTINE COMMENTED\n")
        f.flush()
        try:
            chunks = chunk_fortran_file(f.name)
            assert len(chunks) == 1
            assert "This is a comment" in chunks[0].content
            assert "Another comment" in chunks[0].content
            assert "Third comment" in chunks[0].content
        finally:
            os.unlink(f.name)


def test_fallback_chunking():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f', delete=False) as f:
        f.write("      X = 1\n      Y = 2\n      Z = X + Y\n")
        f.flush()
        try:
            chunks = chunk_fortran_file(f.name)
            assert len(chunks) >= 1
            assert all(c.chunk_type == "fallback" for c in chunks)
        finally:
            os.unlink(f.name)


def test_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.f', delete=False) as f:
        f.write("")
        f.flush()
        try:
            chunks = chunk_fortran_file(f.name)
            assert chunks == []
        finally:
            os.unlink(f.name)
