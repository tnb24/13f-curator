"""The SQL validator: one row per accepted or rejected statement.

Each rejected row pairs a statement with a fragment its reason must hold.
Asserting the reason matters: a validator that rejects everything for the
wrong reason still passes a test that checks only the exception type.
"""

from __future__ import annotations

import duckdb
import pytest

from agents.validate import Facts, SQLRejected, check, check_facts, describe_failure, inspect


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


# --- check_facts: does the statement fit the dataset and the question? ----------

FACTS = Facts(
    cusips={"037833100", "02079K305"},
    funds={"Alpha Capital LLC"},
    quarters={"2026Q1", "2026Q2"},
)
_H = "SELECT h.accession_number FROM holdings h JOIN filings f USING (accession_number) WHERE "

FACT_REJECTED = [
    (_H + "h.cusip = '999999999'", "cusip '999999999'"),
    (_H + "'999999999' = h.cusip", "cusip '999999999'"),
    (_H + "h.cusip IN ('037833100', '999999999')", "cusip '999999999'"),
    (_H + "h.cusip NOT IN ('999999999')", "cusip '999999999'"),
    (_H + "f.fund_name = 'Alpha Capital'", "fund_name 'Alpha Capital'"),
    (_H + "f.fund_name IN ('Alpha Capital LLC', 'Beta')", "fund_name 'Beta'"),
    (_H + "h.report_quarter = '2026Q3'", "report_quarter '2026Q3'"),
    (_H + "CASE WHEN h.report_quarter = '2025Q4' THEN 1 END = 1", "report_quarter '2025Q4'"),
    # name_of_issuer: any filter that reads it next to a text literal.
    (_H + "h.name_of_issuer = 'APPLE INC'", "name_of_issuer"),
    (_H + "upper(h.name_of_issuer) LIKE '%APPLE%'", "name_of_issuer"),
    (_H + "h.cusip = '037833100' AND NOT (h.name_of_issuer ILIKE 'apple%')", "name_of_issuer"),
    (_H + "h.cusip IN (SELECT cusip FROM holdings WHERE name_of_issuer = 'X')", "name_of_issuer"),
    ("SELECT sum(value) FILTER (WHERE name_of_issuer = 'X'), min(accession_number) "
     "FROM holdings", "name_of_issuer"),
    ("SELECT min(accession_number), sum(CASE WHEN name_of_issuer LIKE 'A%' THEN 1 END) "
     "FROM holdings", "name_of_issuer"),
]

FACT_ACCEPTED = [
    _H + "h.cusip IN ('037833100') AND h.report_quarter = '2026Q2'",
    _H + "f.fund_name = 'Alpha Capital LLC'",
    # The name is data to read, not a filter.
    "SELECT name_of_issuer, min(accession_number) FROM holdings GROUP BY name_of_issuer",
    _H + "h.cusip = '037833100' AND h.name_of_issuer IS NOT NULL",
    _H + "left(h.cusip, 6) = '037833'",
    _H + "h.name_of_issuer || ' (' || h.cusip || ')' <> h.cusip",
    "SELECT a.accession_number FROM holdings a JOIN holdings b "
    "ON a.name_of_issuer = b.name_of_issuer AND a.cusip = b.cusip",
    # A range or a pattern names no literal that can be invented.
    _H + "h.cusip > '03' AND f.fund_name LIKE 'Alpha%'",
]


FACT_REJECTED_IDS = [r[0][-40:] for r in FACT_REJECTED]


@pytest.mark.parametrize("statement, reason", FACT_REJECTED, ids=FACT_REJECTED_IDS)
def test_the_fact_check_rejects(conn, statement: str, reason: str) -> None:
    tree = inspect(statement, conn).tree
    with pytest.raises(SQLRejected) as raised:
        check_facts(tree, FACTS, set())
    assert reason in str(raised.value)


@pytest.mark.parametrize("statement", FACT_ACCEPTED, ids=[s[-40:] for s in FACT_ACCEPTED])
def test_the_fact_check_accepts(conn, statement: str) -> None:
    check_facts(inspect(statement, conn).tree, FACTS, set())


@pytest.mark.parametrize(
    "sql_where, named, ok",
    [
        ("h.report_quarter = '2026Q2'", {"2026Q2"}, True),
        ("h.report_quarter IN ('2026Q1', '2026Q2')", {"2026Q1", "2026Q2"}, True),
        ("h.report_quarter = '2026Q1'", {"2026Q2"}, False),
        ("h.report_quarter = '2026Q2'", {"2026Q1", "2026Q2"}, False),
        ("h.cusip = '037833100'", {"2026Q2"}, False),
        ("h.cusip = '037833100'", set(), True),
        ("CASE WHEN h.report_quarter = '2026Q1' THEN 1 ELSE 0 END = 1 "
         "AND f.report_quarter = '2026Q2'", {"2026Q1", "2026Q2"}, True),
    ],
)
def test_the_sql_must_filter_on_each_quarter_the_question_names(
    conn, sql_where: str, named: set[str], ok: bool
) -> None:
    tree = inspect(_H + sql_where, conn).tree
    if ok:
        check_facts(tree, FACTS, named)
    else:
        with pytest.raises(SQLRejected, match="report_quarter"):
            check_facts(tree, FACTS, named)


_G = "SELECT {} AS answer, list(DISTINCT h.accession_number) AS accession_number FROM holdings h "
_G += "JOIN filings f USING (accession_number) GROUP BY {}"

GROUP_REJECTED = [
    (_G.format("f.fund_name", "f.fund_name, h.accession_number"), "do not group by accession"),
    (_G.format("f.fund_name", "h.accession_number, f.fund_name"), "do not group by accession"),
    (_G.format("h.accession_number", "1"), "do not group by accession"),
    (_G.format("h.accession_number", "answer"), "do not group by accession"),
    (_G.format("h.cik", "h.cik"), "do not group by cik alone"),
    (_G.format("h.cik", "1"), "do not group by cik alone"),
]


@pytest.mark.parametrize("statement, reason", GROUP_REJECTED, ids=[str(i) for i in range(6)])
def test_a_group_by_must_not_split_a_fund_by_filing(conn, statement: str, reason: str) -> None:
    tree = inspect(statement, conn).tree
    with pytest.raises(SQLRejected, match=reason):
        check_facts(tree, FACTS, set())


@pytest.mark.parametrize(
    "group", ["f.fund_name", "1", "h.cik, f.fund_name", "ALL", "f.fund_name, 1"]
)
def test_a_group_by_fund_name_passes(conn, group: str) -> None:
    check_facts(inspect(_G.format("f.fund_name", group), conn).tree, FACTS, set())


def test_a_column_beside_an_aggregate_gets_its_fix_in_the_repair_message(conn) -> None:
    statement = ("SELECT f.fund_name AS answer, list(DISTINCT f.accession_number) "
                 "AS accession_number FROM filings f")
    with pytest.raises(duckdb.BinderException) as raised:
        inspect(statement, conn)
    message = describe_failure(raised.value)
    assert message.startswith("BinderException: ")
    assert 'column "fund_name" must appear in the GROUP BY' in message
    assert "add fund_name to GROUP BY" in message and "any_value(fund_name)" in message
    assert "\n" not in message
    # A list has one row per group, so any_value would collapse it: the fix is GROUP BY.
    listed = describe_failure(raised.value, "list")
    assert "add fund_name to GROUP BY" in listed and "collapse it to one row" in listed
    assert "wrap fund_name in an aggregate" not in listed
    # An error with no known fix keeps the first line only.
    assert describe_failure(SQLRejected("bad\nsecond line")) == "SQLRejected: bad"


def test_a_cusip_must_come_from_the_user_message_when_offered_is_set(conn) -> None:
    tree = inspect(_H + "h.cusip IN ('037833100', '02079K305')", conn).tree
    check_facts(tree, FACTS, set())  # offered is None: only the dataset counts
    offered = Facts(FACTS.cusips, FACTS.funds, FACTS.quarters, offered={"037833100"})
    with pytest.raises(SQLRejected, match="cusip '02079K305' is not in the user message"):
        check_facts(tree, offered, set())


def test_check_returns_what_inspect_returns(conn) -> None:
    statement = "SELECT accession_number FROM holdings"
    assert check(statement, conn) == inspect(statement, conn).sql
