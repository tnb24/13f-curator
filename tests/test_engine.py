"""The engine's guarantees, attacked directly, with no validator in front.

The engine has two layers. `query` accepts one SELECT statement only. Behind
it, the DuckDB settings hold for any statement. The `raw` fixture sends
statements past `query`, so each layer has its own tests.

Each test builds an Engine over two tiny Parquet files in tmp_path, so the
module needs neither the real tables nor the network.
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agents import db


@pytest.fixture()
def sources(tmp_path: Path) -> dict[str, Path]:
    data = tmp_path / "data"
    data.mkdir()
    pq.write_table(pa.table({"accession_number": ["a-1", "a-2"], "fund_name": ["X", "Y"]}),
                   data / "filings.parquet")
    pq.write_table(pa.table({"accession_number": ["a-1"] * 3, "value": [1, 2, 3]}),
                   data / "holdings.parquet")
    (tmp_path / "secret.txt").write_text("do not read", encoding="utf-8")
    return {"filings": data / "filings.parquet", "holdings": data / "holdings.parquet"}


@pytest.fixture()
def engine(sources):
    with db.Engine(sources) as built:
        yield built


@pytest.fixture()
def raw(engine):
    """A cursor past `query`: only the DuckDB settings stand in front of the data."""
    cursor = engine.cursor()
    yield cursor
    cursor.close()


def test_a_read_works(engine) -> None:
    result = engine.query("SELECT sum(value) AS total FROM holdings")
    assert (result.columns, result.rows) == (["total"], [(6,)])


SCHEMA_CHANGES = [
    "CREATE TABLE evil (a INTEGER)",
    "CREATE VIEW evil AS SELECT 1",
    "DROP VIEW holdings",
    "CREATE OR REPLACE VIEW holdings AS SELECT 'x' AS accession_number, 0 AS value",
]
ROW_CHANGES = [
    "INSERT INTO holdings VALUES ('x', 1)",
    "UPDATE holdings SET value = 0",
    "DELETE FROM holdings",
]
SETTING_CHANGES = [
    "SET enable_external_access = true",
    "SET allowed_paths = ['/']",
    "SET allowed_directories = ['/']",
    "SET threads = 1",
    "RESET lock_configuration",
]


@pytest.mark.parametrize("statement", SCHEMA_CHANGES + ROW_CHANGES + SETTING_CHANGES)
def test_query_accepts_one_select_statement_only(engine, statement: str) -> None:
    with pytest.raises(duckdb.PermissionException, match="one SELECT statement only"):
        engine.query(statement)


@pytest.mark.parametrize("statement", SCHEMA_CHANGES)
def test_a_schema_change_fails_because_the_database_is_read_only(raw, statement) -> None:
    with pytest.raises(Exception, match=r"read-only|read only"):
        raw.execute(statement)


@pytest.mark.parametrize("statement", ROW_CHANGES)
def test_a_row_change_fails_and_the_data_is_unchanged(engine, raw, statement: str) -> None:
    """The tables are views over Parquet, so DuckDB refuses the statement before
    it reaches the read-only check. Either refusal is a refusal; the data is
    the proof."""
    with pytest.raises(duckdb.Error):
        raw.execute(statement)
    assert engine.query("SELECT count(*), sum(value) FROM holdings").rows == [(3, 6)]


@pytest.mark.parametrize("statement", SETTING_CHANGES)
def test_the_configuration_is_locked(raw, statement: str) -> None:
    with pytest.raises(Exception, match=r"(?i)configuration"):
        raw.execute(statement)


def test_a_file_outside_the_table_folder_is_unreadable(engine, sources, tmp_path) -> None:
    secret = (tmp_path / "secret.txt").as_posix()
    with pytest.raises(Exception, match=r"(?i)permission|disabled"):
        engine.query(f"SELECT * FROM read_text('{secret}')")


def test_the_views_read_the_parquet_in_place(engine, sources) -> None:
    """The engine copies nothing: a view reads the file it names."""
    cursor = engine.cursor()  # trusted code: `query` accepts SELECT only
    plan = cursor.execute("EXPLAIN SELECT * FROM holdings").fetchall()
    cursor.close()
    assert "parquet" in str(plan).lower()


def test_a_slow_query_times_out(engine) -> None:
    with pytest.raises(db.QueryTimeout):
        engine.query(
            "SELECT count(*) FROM range(100000000) a, range(100000000) b WHERE a.range = b.range",
            timeout=0.2,
        )
    # The engine still answers after an interrupt.
    assert engine.query("SELECT count(*) FROM filings").rows == [(2,)]


def test_the_row_limit_is_honoured(engine) -> None:
    assert len(engine.query("SELECT * FROM holdings", limit=2).rows) == 2


def test_close_deletes_the_private_database(sources) -> None:
    engine = db.Engine(sources)
    folder = Path(engine.query("SELECT path FROM duckdb_databases() WHERE database_name = "
                               "current_database()").rows[0][0]).parent
    assert folder.exists()
    engine.close()
    assert not folder.exists()
    engine.close()  # a second close is harmless


def test_a_missing_table_file_fails_fast(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=r"main.py"):
        db.Engine({"filings": tmp_path / "nope.parquet"})


def test_a_listed_file_cannot_be_written_and_its_neighbour_cannot_be_read(tmp_path) -> None:
    """The allow list names files, not folders, and `query` accepts SELECT only."""
    data = tmp_path / "holdings.parquet"
    pq.write_table(pa.table({"accession_number": ["a"], "value": [1]}), data)
    original = data.read_bytes()
    secret = tmp_path / "secret.txt"  # in the same folder as the table
    secret.write_text("synthetic secret", encoding="utf-8")
    target = tmp_path / "created.csv"
    statements = [
        f"SELECT * FROM read_text('{secret.as_posix()}')",
        f"SELECT * FROM read_text('{(tmp_path / 'sub/../secret.txt').as_posix()}')",
        f"SELECT * FROM glob('{tmp_path.as_posix()}/*')",
        f"COPY (SELECT 42) TO '{data.as_posix()}'",
        f"COPY (SELECT 42) TO '{target.as_posix()}'",
        f"SELECT * FROM query('COPY (SELECT 42) TO ''{data.as_posix()}''')",
        f"EXPLAIN ANALYZE COPY (SELECT 42) TO '{data.as_posix()}'",
        "EXPLAIN SELECT 1",
        "CREATE TEMP TABLE evil AS SELECT 1",
        "SELECT 1; SELECT 2",
        "ATTACH ':memory:' AS evil",
        "INSTALL httpfs",
        "LOAD httpfs",
    ]
    with db.Engine({"holdings": data}) as engine:
        for statement in statements:
            with pytest.raises(duckdb.Error):
                engine.query(statement)
        assert engine.query("SELECT * FROM holdings").rows == [("a", 1)]
    assert data.read_bytes() == original
    assert not target.exists()


def test_a_timeout_before_execute_still_stops_the_query(engine, monkeypatch) -> None:
    """DuckDB clears an interrupt when a statement starts, so one early interrupt is lost."""
    cursor = engine.cursor()

    class SlowParse:
        executed = False

        def extract_statements(self, sql):
            time.sleep(0.05)
            return cursor.extract_statements(sql)

        def execute(self, sql):
            self.executed = True
            return cursor.execute(sql)

        def __getattr__(self, name):
            return getattr(cursor, name)

    slow = SlowParse()
    monkeypatch.setattr(engine, "cursor", lambda: slow)
    with pytest.raises(db.QueryTimeout):
        engine.query("SELECT 1", timeout=0.01)
    assert not slow.executed
