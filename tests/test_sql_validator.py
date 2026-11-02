"""The SQL validator: one test per accepted or rejected statement.

The last two accept rows are the point of the module: a forbidden word inside a
string literal must not reject the statement, because a real question filters
on the value CALL.
"""

from __future__ import annotations

import pytest

from agents.validate import SQLRejected, check

# Each row pairs a statement with a fragment its rejection reason must hold.
# Asserting the reason matters: a validator that rejects everything for the
# wrong reason still passes a test that checks only the exception type.
REJECTED = [
    ("DROP TABLE holdings", "SELECT or WITH"),
    ("SELECT 1; DROP TABLE holdings", "one statement"),
    ("SELECT * FROM holdings; --", "one statement"),
    ("ATTACH 'evil.db' AS e", "SELECT or WITH"),
    ("COPY holdings TO '/tmp/x.csv'", "SELECT or WITH"),
    ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
    ("INSTALL httpfs", "SELECT or WITH"),
    ("SELECT secret FROM holdings", "unknown column"),
    ("SELECT * FROM users", "unknown table"),
    ("SELECT value FROM holdings", "accession_number"),
    ("WITH x AS (SELECT 1) DELETE FROM holdings", "delete"),
]

ACCEPTED = [
    "SELECT accession_number, value FROM holdings LIMIT 5",
    "WITH q AS (SELECT * FROM holdings) SELECT accession_number FROM q",
    "SELECT list(DISTINCT accession_number), sum(value) FROM holdings",
    "SELECT accession_number FROM holdings WHERE upper(put_call) = 'CALL'",
    "SELECT accession_number FROM holdings WHERE name_of_issuer = 'DROP TABLE'",
]


@pytest.mark.parametrize("statement, reason", REJECTED, ids=[r[0][:32] for r in REJECTED])
def test_the_validator_rejects(statement: str, reason: str) -> None:
    with pytest.raises(SQLRejected) as raised:
        check(statement)
    assert reason.lower() in str(raised.value).lower()


@pytest.mark.parametrize("statement", ACCEPTED, ids=[s[:32] for s in ACCEPTED])
def test_the_validator_accepts(statement: str) -> None:
    assert check(statement)


def test_a_valid_statement_without_a_limit_gains_one() -> None:
    """Check 7 caps the row count, and leaves an existing limit alone."""
    assert "LIMIT 1000" in check("SELECT accession_number FROM holdings")
    assert check("SELECT accession_number FROM holdings LIMIT 5").endswith("LIMIT 5")


def test_a_literal_hides_a_keyword_from_the_scan() -> None:
    """The reason the two accept rows above exist, stated directly."""
    statement = "SELECT accession_number FROM holdings WHERE name_of_issuer = 'DROP TABLE'"
    assert "DROP TABLE" in check(statement)
