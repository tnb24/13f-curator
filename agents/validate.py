"""Static validation for model-written SQL.

The read-only connection in `agents.db` is the guarantee. This module reduces
noise: it rejects statements that are not a single select over the two known
tables before the database ever sees them.

Checks run in order:
1. One statement only (strip one trailing semicolon, reject any other).
2. Must start with SELECT or WITH, after whitespace and leading comments.
3. No forbidden keyword outside string literals.
4. No file- or environment-reading function name outside string literals.
5. DuckDB must parse it, it must be one statement, and every relation in the
   parse tree must be filings, holdings, a CTE, or a subquery. A table
   function (read_text, query, range, ...) is never a relation here, and a
   table name may carry no catalog and no schema other than main.
6. Every column name in the parse tree must be known.
7. The projection must include accession_number.
8. A limit of 1000 rows is appended on its own line when the statement has
   none, so a trailing line comment cannot swallow it.

String literals are blanked before checks 3 and 4, so
`WHERE upper(put_call) = 'CALL'` passes even though "call" is forbidden as a
keyword.
"""

from __future__ import annotations

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
    "read_csv", "read_parquet", "read_json", "read_text", "read_blob", "glob",
    "getenv", "shell", "system", "query", "query_table",
}


def _word_pattern(words: set[str]) -> re.Pattern[str]:
    """One pattern that finds any of the words, so each scan reads the text once."""
    alternatives = "|".join(sorted(re.escape(word) for word in words))
    return re.compile(rf"\b({alternatives})\b", re.IGNORECASE)


_KEYWORD_RE = _word_pattern(_FORBIDDEN_KEYWORDS)
_FUNCTION_RE = _word_pattern(_FORBIDDEN_FUNCTIONS)
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


def _reject_words(pattern: re.Pattern[str], text: str, kind: str) -> None:
    match = pattern.search(text)
    if match:
        word = match.group(1).lower()
        raise SQLRejected(f"forbidden {kind} {word!r} at position {match.start()}")


def _parse(sql: str, conn: duckdb.DuckDBPyConnection) -> dict:
    """Return DuckDB's parse tree for one statement. Nothing runs.

    The statement travels as a quoted literal, not a bound parameter. A bound
    parameter makes the DuckDB client import pandas when pandas is installed,
    and that import costs about 0.9 s. In a plain literal only a quote needs
    escaping, and doubling it is DuckDB's escape.
    """
    literal = "'" + sql.replace("'", "''") + "'"
    raw = conn.execute(f"SELECT json_serialize_sql({literal})").fetchone()[0]
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
        if (
            isinstance(column_names, list)
            and column_names
            and str(column_names[-1]).lower() == "accession_number"
        ):
            return True
    return False


def _projects_accession_number(statement: dict) -> bool:
    return _node_projects_accession(statement.get("node") or {})


def _has_top_level_limit(statement: dict) -> bool:
    node = statement.get("node") or {}
    return any(
        modifier.get("type") == "LIMIT_MODIFIER" for modifier in node.get("modifiers") or []
    )


def _check_relations(statement: dict) -> None:
    """Every FROM entry is a known base table, a CTE, a subquery, or a join of those."""
    for node in _walk(statement):
        kind = node.get("type")
        if kind == "TABLE_FUNCTION":
            function = (node.get("function") or {}).get("function_name") or "?"
            raise SQLRejected(f"table function {function!r} is not allowed")
        if kind == "BASE_TABLE":
            catalog = str(node.get("catalog_name") or "")
            schema = str(node.get("schema_name") or "").lower()
            if catalog or schema not in ("", "main"):
                qualified = ".".join(part for part in (catalog, schema) if part)
                raise SQLRejected(f"qualified table name {qualified!r} is not allowed")


def check(sql: str, conn: duckdb.DuckDBPyConnection) -> str:
    """Return a safe SQL string. Raise SQLRejected with a reason.

    `conn` parses the statement through json_serialize_sql and runs nothing.
    Any DuckDB connection works; the agent passes a cursor of its engine.
    """
    if not sql or not sql.strip():
        raise SQLRejected("empty statement")

    # Check 1: one statement only. A semicolon inside a literal or a comment
    # ends no statement, so the scan reads the blanked text.
    statement_text = _strip_one_trailing_semicolon(sql)
    position = _blank_literals_and_comments(statement_text).find(";")
    if position != -1:
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
    _reject_words(_KEYWORD_RE, code_only, "keyword")
    _reject_words(_FUNCTION_RE, code_only, "function")

    # Checks 5 and 6: DuckDB parses it; one statement; known relations and names.
    parsed = _parse(body, conn)
    _check_relations(parsed)
    _check_names(parsed)

    # Check 7: accession_number must appear in the projection.
    if not _projects_accession_number(parsed):
        raise SQLRejected("the projection must include accession_number")

    # An occurrence inside COUNT, a scalar subquery, or * EXCLUDE is not a
    # citation column. Bind (without executing) and check the actual output
    # names/types the answer collector will see. This cannot prove that a
    # model's citations are relevant to its question.
    columns = conn.execute(f"DESCRIBE {body}").fetchall()
    if not any("accession" in name.lower() and dtype in ("VARCHAR", "VARCHAR[]")
               for name, dtype, *_rest in columns):
        raise SQLRejected("the projection must expose a string accession_number column")

    # Check 8: cap the row count.
    if not _has_top_level_limit(parsed):
        statement_text = f"{statement_text.rstrip()}\nLIMIT {_ROW_LIMIT}"
    return statement_text
