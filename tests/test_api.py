"""Tests for FastAPI endpoints."""

import os
import sys
import tempfile

# Set env vars before any app imports
_tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
os.environ["DB_PATH"] = _tmp_db.name
os.environ["ADMIN_API_KEY"] = "test-key-12345"
_tmp_db.close()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_debug_env_no_auth():
    resp = client.get("/debug/env")
    assert resp.status_code == 401


def test_feedback_validation():
    resp = client.post("/feedback", json={"query": "", "feedback": "invalid"})
    assert resp.status_code == 422


def test_feedback_valid():
    resp = client.post("/feedback", json={"query": "test", "feedback": "up"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_routines_invalid_library():
    resp = client.get("/api/routines?library=INVALID")
    assert resp.status_code == 422


def test_call_graph_depth_validation():
    resp = client.get("/api/call-graph?depth=10")
    assert resp.status_code == 422


def teardown_module():
    try:
        os.unlink(_tmp_db.name)
    except OSError:
        pass
