"""SQLite helper for observability logging."""

import sqlite3
import os
from datetime import datetime, timezone

from config import DB_PATH


def _get_db_path() -> str:
    return DB_PATH


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

        CREATE TABLE IF NOT EXISTS routine_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            routine_name TEXT,
            file_path TEXT,
            chunk_type TEXT,
            start_line INTEGER,
            end_line INTEGER,
            library TEXT,
            loc INTEGER DEFAULT 0,
            var_count INTEGER DEFAULT 0,
            call_count INTEGER DEFAULT 0,
            nesting_depth INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS call_graph (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            caller TEXT,
            callee TEXT,
            file_path TEXT,
            line_number INTEGER
        );
    """)
    conn.commit()
    # Migrate existing routine_index tables missing new columns
    _migrate_routine_index(conn)


def _migrate_routine_index(conn: sqlite3.Connection):
    """Add complexity columns if they don't exist (migration for existing DBs)."""
    cursor = conn.execute("PRAGMA table_info(routine_index)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col in ("loc", "var_count", "call_count", "nesting_depth"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE routine_index ADD COLUMN {col} INTEGER DEFAULT 0")
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

        # Latency series (last 20 queries, chronological order)
        latency_rows = conn.execute(
            "SELECT timestamp, latency_ms FROM query_log ORDER BY id DESC LIMIT 20"
        ).fetchall()
        latency_series = [{"timestamp": r[0], "latency_ms": r[1]} for r in reversed(latency_rows)]

        # Cost by mode
        cost_rows = conn.execute(
            "SELECT mode, SUM(cost_usd) as total FROM query_log GROUP BY mode"
        ).fetchall()
        cost_by_mode = {r[0]: round(r[1] or 0, 6) for r in cost_rows}

        # Queries by mode
        mode_rows = conn.execute(
            "SELECT mode, COUNT(*) as cnt FROM query_log GROUP BY mode"
        ).fetchall()
        queries_by_mode = {r[0]: r[1] for r in mode_rows}

        # Average score
        avg_score = conn.execute(
            "SELECT AVG(top_score) FROM query_log WHERE top_score IS NOT NULL AND top_score > 0"
        ).fetchone()[0] or 0.0

        # Total tokens
        token_row = conn.execute(
            "SELECT SUM(input_tokens), SUM(output_tokens) FROM query_log"
        ).fetchone()
        total_input_tokens = token_row[0] or 0
        total_output_tokens = token_row[1] or 0

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
            "latency_series": latency_series,
            "cost_by_mode": cost_by_mode,
            "queries_by_mode": queries_by_mode,
            "avg_score": round(avg_score, 3),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }
    finally:
        conn.close()


def _detect_library(file_path: str, source_dir: str = "") -> str:
    """Detect library from file path and/or source directory."""
    lower = (file_path or "").lower()
    src_lower = (source_dir or "").lower()
    combined = f"{src_lower}/{lower}"
    if "scalapack" in combined or "pblas" in combined or "blacs" in combined:
        return "ScaLAPACK"
    if "lapack" in combined:
        return "LAPACK"
    return "BLAS"


def log_routines(chunks: list, metrics: dict = None, source_dir_map: dict = None):
    """Populate routine_index from ingested chunks.

    Args:
        chunks: List of Chunk objects or dicts
        metrics: Optional dict mapping routine_name -> {loc, var_count, call_count, nesting_depth}
        source_dir_map: Optional dict mapping file_path -> source_dir for library detection
    """
    metrics = metrics or {}
    source_dir_map = source_dir_map or {}
    conn = get_connection()
    try:
        conn.execute("DELETE FROM routine_index")
        for c in chunks:
            if isinstance(c, dict):
                name = c.get("routine_name", "")
                fpath = c.get("file_path", "")
                ctype = c.get("chunk_type", "")
                sline = c.get("start_line", 0)
                eline = c.get("end_line", 0)
            else:
                name = c.routine_name
                fpath = c.file_path
                ctype = c.chunk_type
                sline = c.start_line
                eline = c.end_line
            if not name:
                continue
            m = metrics.get(name, {})
            src_dir = source_dir_map.get(fpath, "")
            conn.execute(
                """INSERT INTO routine_index
                   (routine_name, file_path, chunk_type, start_line, end_line, library,
                    loc, var_count, call_count, nesting_depth)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, fpath, ctype, sline, eline, _detect_library(fpath, src_dir),
                 m.get("loc", 0), m.get("var_count", 0),
                 m.get("call_count", 0), m.get("nesting_depth", 0))
            )
        conn.commit()
    finally:
        conn.close()


def get_routines(library: str = None, search: str = None) -> list[dict]:
    """Get routines from the index, optionally filtered."""
    conn = get_connection()
    try:
        query = "SELECT routine_name, file_path, chunk_type, start_line, end_line, library, loc, var_count, call_count, nesting_depth FROM routine_index"
        params = []
        conditions = []
        if library:
            conditions.append("library = ?")
            params.append(library)
        if search:
            conditions.append("routine_name LIKE ?")
            params.append(f"%{search}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY library, routine_name"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_call_graph(edges: list[tuple]):
    """Populate call_graph from parsed CALL statements."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM call_graph")
        for caller, callee, file_path, line_number in edges:
            conn.execute(
                "INSERT INTO call_graph (caller, callee, file_path, line_number) VALUES (?, ?, ?, ?)",
                (caller, callee, file_path, line_number)
            )
        conn.commit()
    finally:
        conn.close()


def get_call_graph(routine: str = None, depth: int = 2) -> dict:
    """Get call graph nodes and edges, optionally centered on a routine."""
    conn = get_connection()
    try:
        if routine:
            routine = routine.upper()
            nodes = set()
            edges = []
            frontier = {routine}
            visited = set()
            for _ in range(depth):
                if not frontier:
                    break
                placeholders = ",".join("?" * len(frontier))
                # Outgoing calls
                rows = conn.execute(
                    f"SELECT caller, callee, file_path, line_number FROM call_graph WHERE UPPER(caller) IN ({placeholders})",
                    list(frontier)
                ).fetchall()
                for r in rows:
                    edges.append({"source": r[0], "target": r[1], "file_path": r[2], "line": r[3]})
                    nodes.add(r[0])
                    nodes.add(r[1])
                # Incoming calls
                rows = conn.execute(
                    f"SELECT caller, callee, file_path, line_number FROM call_graph WHERE UPPER(callee) IN ({placeholders})",
                    list(frontier)
                ).fetchall()
                for r in rows:
                    edges.append({"source": r[0], "target": r[1], "file_path": r[2], "line": r[3]})
                    nodes.add(r[0])
                    nodes.add(r[1])
                visited.update(frontier)
                frontier = nodes - visited
        else:
            rows = conn.execute(
                "SELECT DISTINCT caller, callee, file_path, line_number FROM call_graph LIMIT 500"
            ).fetchall()
            edges = [{"source": r[0], "target": r[1], "file_path": r[2], "line": r[3]} for r in rows]
            nodes = set()
            for e in edges:
                nodes.add(e["source"])
                nodes.add(e["target"])

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in edges:
            key = (e["source"], e["target"])
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        # Get library info for nodes
        node_list = []
        for n in nodes:
            row = conn.execute(
                "SELECT file_path, library FROM routine_index WHERE UPPER(routine_name) = ? LIMIT 1",
                (n.upper(),)
            ).fetchone()
            node_list.append({
                "id": n,
                "library": row["library"] if row else "Unknown",
                "file_path": row["file_path"] if row else "",
            })

        return {"nodes": node_list, "edges": unique_edges}
    finally:
        conn.close()


def get_routine_detail(name: str) -> dict | None:
    """Get full detail for a single routine including callers and callees."""
    conn = get_connection()
    try:
        row = conn.execute(
            """SELECT routine_name, file_path, chunk_type, start_line, end_line, library,
                      loc, var_count, call_count, nesting_depth
               FROM routine_index WHERE UPPER(routine_name) = ? LIMIT 1""",
            (name.upper(),)
        ).fetchone()
        if not row:
            return None
        detail = dict(row)
        # Incoming calls (who calls this routine)
        callers = conn.execute(
            "SELECT DISTINCT caller, file_path, line_number FROM call_graph WHERE UPPER(callee) = ?",
            (name.upper(),)
        ).fetchall()
        detail["callers"] = [{"name": r[0], "file_path": r[1], "line": r[2]} for r in callers]
        # Outgoing calls (what this routine calls)
        callees = conn.execute(
            "SELECT DISTINCT callee, file_path, line_number FROM call_graph WHERE UPPER(caller) = ?",
            (name.upper(),)
        ).fetchall()
        detail["callees"] = [{"name": r[0], "file_path": r[1], "line": r[2]} for r in callees]
        return detail
    finally:
        conn.close()


def get_dead_code() -> list[dict]:
    """Find routines with no incoming calls (potential dead code)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT r.routine_name, r.file_path, r.library, r.loc, r.chunk_type,
                      r.var_count, r.call_count, r.nesting_depth
               FROM routine_index r
               WHERE UPPER(r.routine_name) NOT IN (
                   SELECT DISTINCT UPPER(callee) FROM call_graph
               )
               AND r.chunk_type != 'program'
               ORDER BY r.library, r.routine_name"""
        ).fetchall()
        return [dict(r) for r in rows]
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
