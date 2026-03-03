"""SQLite helper for observability logging."""

import sqlite3
import os
from datetime import datetime, timezone


DB_PATH = os.getenv("DB_PATH", "./data/legacylens.db")


def _get_db_path() -> str:
    return os.getenv("DB_PATH", DB_PATH)


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection, creating tables if needed."""
    db_path = _get_db_path()
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            chunks_retrieved INTEGER,
            top_score REAL,
            latency_ms INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            answer_preview TEXT
        );

        CREATE TABLE IF NOT EXISTS ingestion_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_count INTEGER,
            chunk_count INTEGER,
            duration_seconds REAL
        );

        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT,
            error_type TEXT,
            error_message TEXT
        );
    """)
    conn.commit()


def log_query(query: str, mode: str, chunks_retrieved: int, top_score: float,
              latency_ms: int, input_tokens: int, output_tokens: int,
              cost_usd: float, answer_preview: str):
    """Log a query to the database."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO query_log
               (timestamp, query, mode, chunks_retrieved, top_score, latency_ms,
                input_tokens, output_tokens, cost_usd, answer_preview)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                query, mode, chunks_retrieved, top_score, latency_ms,
                input_tokens, output_tokens, cost_usd,
                answer_preview[:500] if answer_preview else ""
            )
        )
        conn.commit()
    finally:
        conn.close()


def log_ingestion(file_count: int, chunk_count: int, duration_seconds: float):
    """Log an ingestion run."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO ingestion_log (timestamp, file_count, chunk_count, duration_seconds)
               VALUES (?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), file_count, chunk_count, duration_seconds)
        )
        conn.commit()
    finally:
        conn.close()


def log_error(endpoint: str, error_type: str, error_message: str):
    """Log an error."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO error_log (timestamp, endpoint, error_type, error_message)
               VALUES (?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), endpoint, error_type, error_message[:2000])
        )
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """Get aggregated stats for the dashboard."""
    conn = get_connection()
    try:
        # Total queries
        total_queries = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

        # Average latency (last 100)
        avg_latency = conn.execute(
            "SELECT AVG(latency_ms) FROM (SELECT latency_ms FROM query_log ORDER BY id DESC LIMIT 100)"
        ).fetchone()[0] or 0.0

        # Total cost
        total_cost = conn.execute("SELECT SUM(cost_usd) FROM query_log").fetchone()[0] or 0.0

        # Score distribution (last 100 queries)
        scores = conn.execute(
            "SELECT top_score FROM query_log WHERE top_score IS NOT NULL ORDER BY id DESC LIMIT 100"
        ).fetchall()
        score_distribution = _compute_score_distribution([row[0] for row in scores])

        # Recent queries
        recent = conn.execute(
            """SELECT timestamp, query, mode, latency_ms, cost_usd, top_score, chunks_retrieved
               FROM query_log ORDER BY id DESC LIMIT 20"""
        ).fetchall()
        recent_queries = [dict(row) for row in recent]

        # Errors
        errors = conn.execute(
            "SELECT timestamp, endpoint, error_type, error_message FROM error_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        error_list = [dict(row) for row in errors]

        # Ingestion history
        ingestions = conn.execute(
            "SELECT timestamp, file_count, chunk_count, duration_seconds FROM ingestion_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        ingestion_history = [dict(row) for row in ingestions]

        # Chunks/files coverage from latest ingestion
        latest_ingestion = conn.execute(
            "SELECT chunk_count, file_count FROM ingestion_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        chunks_indexed = latest_ingestion[0] if latest_ingestion else 0
        files_covered = latest_ingestion[1] if latest_ingestion else 0

        return {
            "total_queries": total_queries,
            "avg_latency_ms": round(avg_latency, 1),
            "total_cost_usd": round(total_cost, 6),
            "chunks_indexed": chunks_indexed,
            "files_covered": files_covered,
            "score_distribution": score_distribution,
            "recent_queries": recent_queries,
            "errors": error_list,
            "ingestion_history": ingestion_history,
        }
    finally:
        conn.close()


def _compute_score_distribution(scores: list[float]) -> list[dict]:
    """Compute histogram buckets for score distribution."""
    buckets = [
        {"range": "0.0-0.2", "min": 0.0, "max": 0.2, "count": 0},
        {"range": "0.2-0.4", "min": 0.2, "max": 0.4, "count": 0},
        {"range": "0.4-0.6", "min": 0.4, "max": 0.6, "count": 0},
        {"range": "0.6-0.8", "min": 0.6, "max": 0.8, "count": 0},
        {"range": "0.8-1.0", "min": 0.8, "max": 1.0, "count": 0},
    ]
    for score in scores:
        for bucket in buckets:
            if bucket["min"] <= score < bucket["max"] or (bucket["max"] == 1.0 and score == 1.0):
                bucket["count"] += 1
                break
    return [{"range": b["range"], "count": b["count"]} for b in buckets]
