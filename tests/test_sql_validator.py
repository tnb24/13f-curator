"""The SQL validator: one row per accepted or rejected statement.

Each rejected row pairs a statement with a fragment its reason must hold.
Asserting the reason matters: a validator that rejects everything for the
wrong reason still passes a test that checks only the exception type.
"""

from __future__ import annotations

import duckdb
import pytest

from agents.validate import SQLRejected, check


@pytest.fixture(scope="module")
def conn():
    connection = duckdb.connect()
    # Two empty tables with the real names, so an accepted statement can run.
    connection.execute(
        "CREATE TABLE holdings (accession_number VARCHAR, cik VARCHAR, cusip VARCHAR,"
        " value BIGINT, put_call VARCHAR, name_of_issuer VARCHAR, report_quarter VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE filings (accession_number VARCHAR, fund_name VARCHAR, report_quarter VARCHAR)"
    )
    yield connection
    connection.close()


REJECTED = [
    ("DROP TABLE holdings", "SELECT or WITH"),
    ("SELECT 1; DROP TABLE holdings", "one statement"),
    ("SELECT * FROM holdings; --", "one statement"),
    ("ATTACH 'evil.db' AS e", "SELECT or WITH"),
    ("COPY holdings TO '/tmp/x.csv'", "SELECT or WITH"),
    ("INSTALL httpfs", "SELECT or WITH"),
    ("WITH x AS (SELECT 1) DELETE FROM holdings", "delete"),
    ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
    ("SELECT accession_number FROM read_text('C:/Windows/win.ini')", "read_text"),
    ("SELECT secret FROM holdings", "unknown column"),
    ("SELECT * FROM users", "unknown table"),
    ("SELECT value FROM holdings", "accession_number"),
    ("SELECT * EXCLUDE (accession_number) FROM holdings", "accession_number"),
    ("SELECT count(accession_number) FROM holdings", "accession_number"),
    ("SELECT accession_number AS answer FROM holdings", "accession_number"),
    ("SELECT (SELECT accession_number FROM holdings LIMIT 1) AS answer FROM filings",
     "accession_number"),
    # A table function is never a relation, whatever its name.
    ("SELECT accession_number FROM holdings, range(10)", "table function 'range'"),
    ("SELECT * FROM duckdb_settings()", "table function"),
    ("SELECT * FROM json_execute_serialized_sql('x')", "table function"),
    ("SELECT * FROM query_table('holdings')", "query_table"),
    ("SELECT accession_number FROM memory.main.holdings", "qualified table name"),
    ("SELECT accession_number FROM other.holdings", "qualified table name"),
    # A set operation needs the accession on both sides.
    ("SELECT accession_number FROM holdings UNION SELECT fund_name FROM filings",
     "accession_number"),
    # A keyword hidden after a comment that holds an apostrophe still counts.
    ("SELECT accession_number FROM holdings -- it's fine\n; DROP TABLE holdings",
     "one statement"),
    ("", "empty"),
]

ACCEPTED = [
    "SELECT accession_number, value FROM holdings LIMIT 5",
    "WITH q AS (SELECT * FROM holdings) SELECT accession_number FROM q",
    "SELECT list(DISTINCT accession_number), sum(value) FROM holdings",
    "SELECT accession_number FROM holdings WHERE upper(put_call) = 'CALL'",
    "SELECT accession_number FROM holdings WHERE name_of_issuer = 'DROP TABLE'",
    "SELECT accession_number FROM holdings WHERE name_of_issuer = 'A;B'",
    "SELECT accession_number FROM main.holdings",
    "SELECT accession_number FROM holdings UNION ALL SELECT accession_number FROM filings",
    "SELECT sum(value) AS total, min(accession_number) AS accession_number FROM holdings "
    "GROUP BY cusip HAVING total > 0",
    "SELECT h.accession_number FROM holdings h JOIN filings f USING (accession_number) "
    "WHERE f.fund_name = 'O''Brien Capital'",
    "-- a leading comment\nSELECT accession_number FROM holdings;",
]


@pytest.mark.parametrize(
    "statement, reason", REJECTED, ids=[r[0][:40] or "empty" for r in REJECTED]
)
def test_the_validator_rejects(conn, statement: str, reason: str) -> None:
    with pytest.raises(SQLRejected) as raised:
        check(statement, conn)
    assert reason.lower() in str(raised.value).lower()


@pytest.mark.parametrize("statement", ACCEPTED, ids=[s[:40] for s in ACCEPTED])
def test_the_validator_accepts_and_the_result_runs(conn, statement: str) -> None:
    conn.execute(check(statement, conn)).fetchall()


def test_a_valid_statement_without_a_limit_gains_one(conn) -> None:
    assert check("SELECT accession_number FROM holdings", conn).endswith("\nLIMIT 1000")
    assert check("SELECT accession_number FROM holdings LIMIT 5", conn).endswith("LIMIT 5")


def test_a_trailing_comment_cannot_swallow_the_limit(conn) -> None:
    """The limit goes on its own line, so `-- note` cannot comment it out."""
    statement = check("SELECT accession_number FROM holdings -- note", conn)
    plan = conn.execute(f"EXPLAIN {statement}").fetchall()
    assert "LIMIT" in str(plan).upper()


def test_a_literal_hides_a_keyword_from_the_scan(conn) -> None:
    statement = "SELECT accession_number FROM holdings WHERE name_of_issuer = 'DROP TABLE'"
    assert "DROP TABLE" in check(statement, conn)


def test_a_quote_in_the_statement_cannot_escape_the_parse_literal(conn) -> None:
    """The statement reaches json_serialize_sql as a quoted literal. A quote in it
    must stay inside that literal, so the parse sees one statement, not two."""
    with pytest.raises(SQLRejected):
        check("SELECT accession_number FROM holdings WHERE name_of_issuer = 'x') -- '", conn)
