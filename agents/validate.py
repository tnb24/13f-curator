"""Static validation for model-written SQL.

The read-only connection in `agents.db` is the guarantee. This module reduces
noise: it rejects statements that are not a single select over the two known
tables before the database ever sees them.

Checks run in order:
1. One statement only (strip one trailing semicolon, reject any other).
2. Must start with SELECT or WITH, after whitespace and leading comments.
3. No forbidden keyword outside string literals.
4. No file- or environment-reading function name outside string literals.
5. DuckDB must parse it, it must be one statement, and every table and column
   name in the parse tree must be known.
6. The projection must include accession_number.
7. A limit of 1000 rows is appended when the statement has none.

String literals are blanked before checks 3 and 4, so
`WHERE upper(put_call) = 'CALL'` passes even though "call" is forbidden as a
keyword.
"""

from __future__ import annotations

import functools
import json
import re

import duckdb

class SQLRejected(Exception):
    """The statement failed validation. The message holds the reason."""


ALLOWED_TABLES = {"filings", "holdings"}

FILINGS_COLUMNS = [
    "accession_number", "cik", "fund_name", "filing_manager", "form_type",
    "report_period", "report_quarter", "filing_date", "is_amendment",
    "amendment_no", "amendment_type", "report_type", "form_13f_file_number",
    "crd_number", "sec_file_number", "other_included_managers_count",
    "table_entry_total", "table_value_total",
]

HOLDINGS_COLUMNS = [
    "accession_number", "cik", "report_quarter", "name_of_issuer",
    "title_of_class", "cusip", "figi", "value", "ssh_prnamt",
    "ssh_prnamt_type", "put_call", "investment_discretion", "other_manager",
    "voting_sole", "voting_shared", "voting_none",
]

ALLOWED_COLUMNS = set(FILINGS_COLUMNS) | set(HOLDINGS_COLUMNS)

_FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "create", "alter", "truncate",
    "attach", "detach", "copy", "export", "import", "install", "load",
    "pragma", "set", "call", "execute",
}

_FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_parquet", "read_json", "glob", "getenv", "shell",
    "system",
}

_ROW_LIMIT = 1000


def _strip_one_trailing_semicolon(sql: str) -> str:
    stripped = sql.rstrip()
    if stripped.endswith(";"):
        stripped = stripped[:-1]
    return stripped


def _blank_literals_and_comments(sql: str) -> str:
    """Blank out string literals and comments, preserving length.

    Handles the single-quote form with the doubled-quote (`''`) escape. Comments
    are blanked so that an apostrophe inside a comment cannot open a fake
    literal, which would hide real keywords from check 3.
    """
    chars = list(sql)
    i = 0
    n = len(chars)
    in_string = False
    while i < n:
        if in_string:
            if chars[i] == "'":
                if i + 1 < n and chars[i + 1] == "'":
                    chars[i] = " "
                    chars[i + 1] = " "
                    i += 2
                    continue
                in_string = False
            chars[i] = " "
            i += 1
            continue
        if chars[i] == "-" and i + 1 < n and chars[i + 1] == "-":
            while i < n and chars[i] != "\n":
                chars[i] = " "
                i += 1
            continue
        if chars[i] == "/" and i + 1 < n and chars[i + 1] == "*":
            end = sql.find("*/", i + 2)
            end = n if end == -1 else end + 2
            for j in range(i, end):
                if chars[j] != "\n":
                    chars[j] = " "
            i = end
            continue
        if chars[i] == "'":
            in_string = True
            chars[i] = " "
            i += 1
            continue
        i += 1
    return "".join(chars)


def _skip_leading_comments(sql: str) -> str:
    stripped = sql.lstrip()
    while True:
        if stripped.startswith("--"):
            newline = stripped.find("\n")
            stripped = "" if newline == -1 else stripped[newline + 1:].lstrip()
            continue
        if stripped.startswith("/*"):
            end = stripped.find("*/")
            stripped = "" if end == -1 else stripped[end + 2:].lstrip()
            continue
        return stripped


def _reject_keyword(word: str, text: str, kind: str) -> None:
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        raise SQLRejected(f"forbidden {kind} {word!r} at position {match.start()}")


@functools.lru_cache(maxsize=1)
def _parser_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(":memory:")


def _parse(sql: str) -> dict:
    conn = _parser_connection()
    raw = conn.execute(
        "SELECT json_serialize_sql(CAST(? AS VARCHAR))", [sql]
    ).fetchone()[0]
    if not raw:
        raise SQLRejected("DuckDB returned no parse tree")
    tree = json.loads(raw)
    if tree.get("error"):
        raise SQLRejected(
            f"DuckDB failed to parse the statement: {tree.get('error_message', 'unknown error')}"
        )
    statements = tree.get("statements") or []
    if len(statements) != 1:
        raise SQLRejected(f"expected exactly one statement, found {len(statements)}")
    return statements[0]


def _walk(node: object):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _cte_names(statement: dict) -> set[str]:
    names: set[str] = set()
    for node in _walk(statement):
        cte_map = node.get("cte_map")
        if isinstance(cte_map, dict):
            for entry in cte_map.get("map") or []:
                key = entry.get("key")
                if isinstance(key, str) and key:
                    names.add(key.lower())
    return names


def _check_names(statement: dict) -> None:
    known_tables = ALLOWED_TABLES | _cte_names(statement)
    aliases: set[str] = set()
    table_refs: list[str] = []
    column_refs: list[tuple[str, ...]] = []
    star_relations: list[str] = []
    for node in _walk(statement):
        alias = node.get("alias")
        if isinstance(alias, str) and alias:
            aliases.add(alias.lower())
        relation_name = node.get("relation_name")
        if isinstance(relation_name, str) and relation_name and node.get("type") == "STAR":
            star_relations.append(relation_name.lower())
        table_name = node.get("table_name")
        if isinstance(table_name, str) and table_name:
            table_refs.append(table_name.lower())
        column_names = node.get("column_names")
        if isinstance(column_names, list) and column_names:
            parts = tuple(str(part).lower() for part in column_names)
            column_refs.append(parts)
    allowed_qualifiers = known_tables | aliases
    # A query names its own columns too. `SELECT sum(value) AS val_q2 ...` then
    # reads `val_q2` further out. An alias is defined inside this statement, so
    # it is not an unknown identifier and it reaches no new data.
    allowed_columns = ALLOWED_COLUMNS | aliases
    for table in table_refs:
        if table not in known_tables:
            raise SQLRejected(f"unknown table {table!r}")
    for relation in star_relations:
        if relation not in known_tables:
            raise SQLRejected(f"unknown table {relation!r}")
    for parts in column_refs:
        qualifier, column = parts[:-1], parts[-1]
        if qualifier and qualifier[-1] not in allowed_qualifiers:
            raise SQLRejected(f"unknown table or alias {qualifier[-1]!r}")
        if column not in allowed_columns:
            raise SQLRejected(f"unknown column {column!r}")


def _node_projects_accession(node: dict) -> bool:
    """True when this query node projects accession_number.

    A set operation (UNION, EXCEPT, INTERSECT) holds no select list of its own.
    Both sides must project the column, because either side can supply a row.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "SET_OPERATION_NODE":
        return _node_projects_accession(
            node.get("left") or {}
        ) and _node_projects_accession(node.get("right") or {})
    select_list = node.get("select_list") or []
    for item in _walk(select_list):
        if item.get("type") == "STAR":
            return True
        column_names = item.get("column_names")
        if isinstance(column_names, list) and column_names:
            if str(column_names[-1]).lower() == "accession_number":
                return True
    return False


def _projects_accession_number(statement: dict) -> bool:
    return _node_projects_accession(statement.get("node") or {})


def _has_top_level_limit(statement: dict) -> bool:
    node = statement.get("node") or {}
    for modifier in node.get("modifiers") or []:
        if modifier.get("type") == "LIMIT_MODIFIER":
            return True
    return False


def check(sql: str) -> str:
    """Return a safe SQL string. Raise SQLRejected with a reason."""
    if not sql or not sql.strip():
        raise SQLRejected("empty statement")

    # Check 1: one statement only.
    statement_text = _strip_one_trailing_semicolon(sql)
    if ";" in statement_text:
        position = statement_text.index(";")
        raise SQLRejected(
            f"expected one statement, found a semicolon at position {position}"
        )

    # Check 2: must start with SELECT or WITH.
    body = _skip_leading_comments(statement_text)
    head = re.match(r"(select|with)\b", body, re.IGNORECASE)
    if not head:
        raise SQLRejected("statement must start with SELECT or WITH")

    # Checks 3 and 4: keywords and function names outside literals.
    code_only = _blank_literals_and_comments(body)
    for keyword in sorted(_FORBIDDEN_KEYWORDS):
        _reject_keyword(keyword, code_only, "keyword")
    for function in sorted(_FORBIDDEN_FUNCTIONS):
        _reject_keyword(function, code_only, "function")

    # Check 5: DuckDB parses it; one statement; known names only.
    parsed = _parse(body)
    _check_names(parsed)

    # Check 6: accession_number must appear in the projection.
    if not _projects_accession_number(parsed):
        raise SQLRejected("the projection must include accession_number")

    # Check 7: cap the row count.
    if not _has_top_level_limit(parsed):
        statement_text = f"{statement_text.rstrip()} LIMIT {_ROW_LIMIT}"
    return statement_text
