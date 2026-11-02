"""Read-only DuckDB access for the agent.

The database file lives under `.cache/`, which is gitignored. `connect()` builds
it from the committed Parquet files whenever it is absent or older than either
file, then reopens it with `read_only=True`. The read-only flag is the real
control: a write statement fails in the engine, not in a prompt.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

# Anchored at the repository root, so the agent runs from any working directory.
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache"
DB_PATH = CACHE_DIR / "curator.duckdb"
TABLE_SOURCES = {
    "filings": ROOT / "output" / "filings.parquet",
    "holdings": ROOT / "output" / "holdings.parquet",
}

QUERY_TIMEOUT_SECONDS = 10.0


class QueryTimeout(RuntimeError):
    """A query ran longer than QUERY_TIMEOUT_SECONDS and was interrupted."""


def _is_stale() -> bool:
    """True when the database file is missing or older than a Parquet source."""
    if not DB_PATH.exists():
        return True
    db_time = DB_PATH.stat().st_mtime
    return any(p.stat().st_mtime > db_time for p in TABLE_SOURCES.values())


def _build() -> None:
    """Rebuild `.cache/curator.duckdb` from the Parquet files.

    Uses a separate writable connection and closes it before any read-only
    connection opens.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = duckdb.connect(str(DB_PATH))
    try:
        for table, parquet in TABLE_SOURCES.items():
            conn.execute(
                f"CREATE TABLE {table} AS SELECT * FROM read_parquet('{parquet.as_posix()}')"
            )
    finally:
        conn.close()


def connect() -> duckdb.DuckDBPyConnection:
    """Return a connection that cannot write."""
    if _is_stale():
        _build()
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    # Must run after the read-only connection opens. It cannot be re-enabled
    # afterwards, and it blocks read_csv, read_parquet, and every HTTP call.
    conn.execute("SET enable_external_access = false")
    return conn


def query(conn: duckdb.DuckDBPyConnection, sql: str, limit: int = 1000) -> list[tuple]:
    """Run one statement on the read-only connection and return at most `limit` rows.

    A timer thread interrupts the connection after QUERY_TIMEOUT_SECONDS, so a
    runaway query raises instead of hanging. DuckDB has no statement-timeout
    setting.
    """
    state = {"timed_out": False}

    def _interrupt() -> None:
        state["timed_out"] = True
        conn.interrupt()

    timer = threading.Timer(QUERY_TIMEOUT_SECONDS, _interrupt)
    timer.daemon = True
    try:
        timer.start()
        result = conn.execute(sql)
        rows = result.fetchmany(limit)
    except Exception as exc:
        if state["timed_out"]:
            raise QueryTimeout(
                f"query exceeded {QUERY_TIMEOUT_SECONDS:g}s and was interrupted"
            ) from exc
        raise
    finally:
        timer.cancel()
    return [tuple(row) for row in rows]
