"""The read-only guarantee, exercised against the real database.

`agents.db.connect()` builds `.cache/curator.duckdb` from `output/*.parquet`.
These tests need those files, so they are marked `integration` and skip with a
clear reason when the tables are absent, so a fresh clone still runs the
suite green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents import db

ROOT = Path(__file__).resolve().parents[1]
FILINGS = ROOT / "output" / "filings.parquet"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not FILINGS.exists(),
        reason="output/filings.parquet is absent; run main.py to build the tables",
    ),
]

WRITES = [
    "CREATE TABLE evil (a INTEGER)",
    "DROP TABLE holdings",
    "UPDATE holdings SET value = 0",
    "DELETE FROM holdings",
    "INSERT INTO holdings (cusip) VALUES ('x')",
]

EXTERNAL = [
    "SELECT * FROM read_csv('verify.py')",
    "SELECT * FROM read_parquet('output/filings.parquet')",
]


@pytest.fixture(scope="module")
def conn():
    connection = db.connect()
    yield connection
    connection.close()


def test_a_read_works(conn) -> None:
    assert db.query(conn, "SELECT count(*) FROM holdings")[0][0] > 0


@pytest.mark.parametrize("statement", WRITES, ids=[w.split()[0] for w in WRITES])
def test_a_write_fails_in_the_engine(conn, statement: str) -> None:
    """The read-only flag is the guarantee, not the prompt and not the model."""
    with pytest.raises(Exception):
        conn.execute(statement)


@pytest.mark.parametrize("statement", EXTERNAL, ids=["read_csv", "read_parquet"])
def test_external_access_is_disabled(conn, statement: str) -> None:
    with pytest.raises(Exception):
        conn.execute(statement)


def test_the_row_limit_is_honoured(conn) -> None:
    assert len(db.query(conn, "SELECT accession_number FROM holdings", limit=7)) == 7
