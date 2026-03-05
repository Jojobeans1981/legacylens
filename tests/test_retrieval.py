"""Tests for retrieval query classification and routine extraction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval import _extract_routine_name, _classify_query


def test_extract_routine_name():
    assert _extract_routine_name("What does DGEMM do?") == "DGEMM"
    assert _extract_routine_name("how do we do it?") is None
    assert _extract_routine_name("SGEMV and DGEMV") == "SGEMV"


def test_classify_query_keyword():
    assert _classify_query("DGEMM") == "keyword"


def test_classify_query_vector():
    assert _classify_query("how does the why for all") == "vector"


def test_classify_query_hybrid():
    assert _classify_query("explain DGEMM algorithm") == "hybrid"
