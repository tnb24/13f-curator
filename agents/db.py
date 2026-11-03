"""The agent's database: a private, read-only DuckDB over the two Parquet tables.

`Engine` writes a DuckDB file into a new temporary directory. The file holds
two views, `filings` and `holdings`, over output/filings.parquet and
output/holdings.parquet, and no data. A query reads the Parquet columns in
place, so startup copies nothing: the build takes about 80 ms, where a copy
into DuckDB tables takes about 400 ms. Nothing persists between processes, so
no file can go stale and no two processes share one. `close()` deletes the
directory.

Four engine settings hold however a statement reaches the database:

- `read_only=True`: a write statement fails in the engine, not in a prompt.
- `enable_external_access=false`: no file, URL, or extension can be read ...
- `allowed_paths`: ... except the two exact Parquet paths the views read.
- `lock_configuration=true`: no statement can change a setting back.

The allow list permits writes as well as reads, so `read_only` alone does not
stop `COPY` to a listed file. `query` therefore parses each statement with
DuckDB first, and it accepts one SELECT statement only. `cursor()` is for
trusted code in this package. Untrusted SQL goes through `query`.
Each query runs on its own cursor, so two queries share no connection state,
and a timeout interrupts only its own statement.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
TABLE_SOURCES = {
    "filings": ROOT / "output" / "filings.parquet",
    "holdings": ROOT / "output" / "holdings.parquet",
}
QUERY_TIMEOUT_SECONDS = 10.0
ROW_LIMIT = 1000


class QueryTimeout(RuntimeError):
    """A query ran longer than its timeout and was interrupted."""


@dataclass(frozen=True)
class Result:
    """The rows of one query, and the name of each column."""

    columns: list[str]
    rows: list[tuple]


def _literal(text: str) -> str:
    """Quote text as a DuckDB string literal. Doubling a quote is the only escape."""
    return "'" + text.replace("'", "''") + "'"


def _close(connection: duckdb.DuckDBPyConnection, directory: str) -> None:
    connection.close()
    shutil.rmtree(directory, ignore_errors=True)


class Engine:
    """One read-only database for the lifetime of one Agent."""

    def __init__(self, sources: dict[str, Path] | None = None) -> None:
        sources = TABLE_SOURCES if sources is None else sources
        files = {table: Path(path).resolve() for table, path in sources.items()}
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing; run main.py to build the tables")

        directory = tempfile.mkdtemp(prefix="curator-db-")
        path = Path(directory) / "curator.duckdb"
        try:
            build = duckdb.connect(str(path))
            try:
                for table, parquet in files.items():
                    location = _literal(parquet.as_posix())
                    build.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet({location})")
            finally:
                build.close()
            connection = duckdb.connect(str(path), read_only=True)
            # The order matters. The allow list must exist before external
            # access goes off, and the lock must come last: after it, no
            # statement can widen the list or turn external access back on.
            listed = ", ".join(_literal(path.as_posix()) for path in files.values())
            connection.execute(f"SET allowed_paths = [{listed}]")
            connection.execute("SET enable_external_access = false")
            connection.execute("SET lock_configuration = true")
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self._connection = connection
        self._finalizer = weakref.finalize(self, _close, connection, directory)

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """A new cursor for trusted code. Untrusted SQL goes through `query`."""
        return self._connection.cursor()

    def query(
        self,
        sql: str,
        limit: int = ROW_LIMIT,
        timeout: float = QUERY_TIMEOUT_SECONDS,
    ) -> Result:
        """Run one statement on its own cursor and return at most `limit` rows.

        A timer interrupts the cursor after `timeout` seconds, so a runaway
        query raises QueryTimeout instead of hanging. DuckDB has no
        statement-timeout setting.
        """
        cursor = self.cursor()
        timed_out = threading.Event()
        finished = threading.Event()

        def interrupt() -> None:
            timed_out.set()
            # DuckDB clears its interrupt flag when a statement starts, so one
            # interrupt before `execute` has no effect. Interrupt again every
            # 10 ms until the query ends.
            while not finished.is_set():
                cursor.interrupt()
                if finished.wait(0.01):
                    break

        timer = threading.Timer(timeout, interrupt)
        timer.daemon = True
        try:
            timer.start()
            statements = cursor.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                # EXPLAIN ANALYZE runs the statement that it explains, so the
                # engine accepts no EXPLAIN either.
                raise duckdb.PermissionException("read-only query: one SELECT statement only")
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded {timeout:g}s before execution")
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description or []]
            rows = [tuple(row) for row in cursor.fetchmany(limit)]
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded {timeout:g}s and was interrupted")
        except Exception as exc:
            if timed_out.is_set():
                raise QueryTimeout(f"query exceeded {timeout:g}s and was interrupted") from exc
            raise
        finally:
            finished.set()
            timer.cancel()
            timer.join()
            cursor.close()
        return Result(columns=columns, rows=rows)

    def close(self) -> None:
        """Close the database and delete its directory."""
        self._finalizer()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
