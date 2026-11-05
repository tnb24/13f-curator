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

`check` says that the statement is safe. `check_facts` says that the statement
fits the dataset and the question. `check_facts` reads the parse tree that
`inspect` built, so it adds no second parse. One walk over the tree requires
that:

- every cusip, fund_name, and report_quarter literal exists in the dataset;
- every cusip literal is one that the user message listed;
- no filter tests name_of_issuer against a text literal;
- no GROUP BY splits a fund by accession_number, or groups by cik alone;
- the statement filters on each quarter that the question spells out.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Checked:
    """A statement that passed `inspect`, and the parse tree that proved it."""

    sql: str
    tree: dict


def check(sql: str, conn: duckdb.DuckDBPyConnection) -> str:
    """Return a safe SQL string. Raise SQLRejected with a reason."""
    return inspect(sql, conn).sql


def inspect(sql: str, conn: duckdb.DuckDBPyConnection) -> Checked:
    """Return the safe statement and its parse tree. Raise SQLRejected with a reason.

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
    return Checked(statement_text, parsed)


_UNGROUPED = re.compile(r'column "([^"]+)" must appear in the GROUP BY', re.IGNORECASE)


def describe_failure(failure: Exception, shape: str | None = None) -> str:
    """Return the repair message for a failed statement.

    The message holds the first line of the error. DuckDB puts its advice on
    the next lines, and the model never sees them, so a known error gets its
    fix here. A column beside an aggregate must join the GROUP BY or sit in an
    aggregate. The right fix depends on `shape`, the result shape that the
    model chose. A list has one row per group, so any_value would collapse it
    to one row. Without this advice a model repeats the same statement.
    """
    text = str(failure)
    message = f"{type(failure).__name__}: {(text.splitlines() or [''])[0]}"
    ungrouped = _UNGROUPED.search(text)
    if ungrouped:
        column = ungrouped.group(1)
        if shape == "list":
            message += (
                f" If {column} is the answer, add {column} to GROUP BY: the list has one row "
                f"per {column}, and any_value({column}) would collapse it to one row. "
                "Wrap any other ungrouped column in an aggregate such as max()."
            )
        else:
            message += (
                f" Fix it in one of two ways: wrap {column} in an aggregate such as "
                f"any_value({column}) or max({column}), or add {column} to GROUP BY. "
                "A scalar answer is one row, so the aggregate is usually the right fix."
            )
        message += (
            " A query with list(DISTINCT accession_number) or another aggregate "
            "may select only grouped columns and aggregates."
        )
    return message


# --- facts: does the statement fit the dataset and the question? ---------------


@dataclass(frozen=True)
class Facts:
    """What the dataset holds. Each field needs only `in`; the agent copies nothing.

    `offered` lists the cusips that the user message showed to the model. A
    cusip outside it is a guess, even when the dataset holds it. A value of
    None skips that check.
    """

    cusips: Collection[str]
    funds: Collection[str]
    quarters: Collection[str]
    offered: Collection[str] | None = None


_FACT_COLUMNS = frozenset({"cusip", "fund_name", "report_quarter"})
_EQUALITY = frozenset({
    "COMPARE_EQUAL", "COMPARE_NOTEQUAL", "COMPARE_DISTINCT_FROM", "COMPARE_NOT_DISTINCT_FROM",
})
_MEMBERSHIP = frozenset({"COMPARE_IN", "COMPARE_NOT_IN"})
# A filter is an expression in one of these slots of the parse tree.
_FILTER_SLOTS = ("where_clause", "having", "qualify", "condition", "when_expr", "filter")


def _column_name(node: object) -> str | None:
    """The lower-case column of a column reference, or None for any other node."""
    if isinstance(node, dict) and node.get("class") == "COLUMN_REF":
        names = node.get("column_names") or []
        if names:
            return str(names[-1]).lower()
    return None


def _text_literal(node: object) -> str | None:
    """The text of a string constant, or None for any other node."""
    if isinstance(node, dict) and node.get("class") == "CONSTANT":
        value = node.get("value") or {}
        if not value.get("is_null") and (value.get("type") or {}).get("id") == "VARCHAR":
            return str(value.get("value"))
    return None


def _literals_of(node: dict):
    """Yield (column, text) for each text literal that this node compares to a fact column.

    The node is `column = 'x'`, `'x' = column`, or `column IN ('x', 'y')`. A
    column inside a function, such as left(cusip, 6), is not a direct compare.
    """
    kind = node.get("type")
    if not isinstance(kind, str):  # a type node, such as {"id": "VARCHAR"}
        return
    if node.get("class") == "COMPARISON" and kind in _EQUALITY:
        left, right = node.get("left"), node.get("right")
        for column, other in ((left, right), (right, left)):
            name = _column_name(column)
            text = _text_literal(other)
            if name in _FACT_COLUMNS and text is not None:
                yield name, text
    elif kind in _MEMBERSHIP:
        children = node.get("children") or []
        name = _column_name(children[0]) if children else None
        if name in _FACT_COLUMNS:
            for child in children[1:]:
                text = _text_literal(child)
                if text is not None:
                    yield name, text


def _grouped_columns(node: dict) -> set[str]:
    """The lower-case columns that a SELECT node groups by.

    `GROUP BY 1` points at the first select item, and `GROUP BY alias` at the
    item with that alias. Both resolve to the column that the item reads. A
    table column wins over an alias of the same name, as in DuckDB.
    """
    select_list = node.get("select_list") or []
    by_alias = {
        str(item.get("alias")).lower(): item
        for item in select_list
        if isinstance(item, dict) and item.get("alias")
    }
    columns: set[str] = set()
    for key in node.get("group_expressions") or []:
        if isinstance(key, dict) and key.get("class") == "CONSTANT":
            position = (key.get("value") or {}).get("value")
            if isinstance(position, int) and 1 <= position <= len(select_list):
                key = select_list[position - 1]
        name = _column_name(key)
        if name in by_alias and name not in ALLOWED_COLUMNS:
            name = _column_name(by_alias[name])
        if name:
            columns.add(name)
    return columns


def _check_grouping(node: dict) -> None:
    """Reject a GROUP BY that splits a fund's rows by filing.

    One accession belongs to one quarter, so a sum per accession covers one
    filing, not the fund. A cik without fund_name drops the answer column.
    """
    columns = _grouped_columns(node)
    if "accession_number" in columns:
        raise SQLRejected(
            "do not group by accession_number: one accession belongs to one quarter, so each "
            "sum covers one filing. For one value across all rows, use no GROUP BY. "
            "For one value per fund, group by f.fund_name. "
            "list(DISTINCT accession_number) cites the filings and needs no GROUP BY"
        )
    if "cik" in columns and "fund_name" not in columns:
        raise SQLRejected(
            "do not group by cik alone: the answer names a fund. "
            "Group by f.fund_name, so that each sum covers every filing of the fund"
        )


def _atoms(expression: object):
    """Split a filter at AND, OR, and NOT. Yield each part that remains."""
    if not isinstance(expression, dict):
        return
    if expression.get("class") == "CONJUNCTION" or expression.get("type") == "OPERATOR_NOT":
        for child in expression.get("children") or []:
            yield from _atoms(child)
    else:
        yield expression


def _filters_issuer_name(part: dict) -> bool:
    """True when one part of a filter tests name_of_issuer against a text literal.

    The part is a comparison, IN, BETWEEN, or a function such as LIKE. A text
    literal must be one of its operands, and name_of_issuer must sit in
    another operand, as in `upper(name_of_issuer) LIKE '%APPLE%'`.
    """
    operands = [part.get(key) for key in ("left", "right", "input", "lower", "upper")]
    operands += part.get("children") or []
    if not any(_text_literal(operand) is not None for operand in operands):
        return False
    return any(
        _column_name(node) == "name_of_issuer" for operand in operands for node in _walk(operand)
    )


def check_facts(tree: dict, facts: Facts, named_quarters: Collection[str]) -> None:
    """Raise SQLRejected unless the statement fits the dataset and the question.

    `tree` is the parse tree that `inspect` returned. `named_quarters` are the
    quarters that the question spells out and that the dataset holds.
    """
    seen_quarters: set[str] = set()
    for node in _walk(tree):
        if node.get("group_expressions"):
            _check_grouping(node)
        for column, text in _literals_of(node):
            if column == "cusip" and text not in facts.cusips:
                raise SQLRejected(f"cusip {text!r} is not in the dataset; do not invent a cusip")
            if column == "cusip" and facts.offered is not None and text not in facts.offered:
                raise SQLRejected(
                    f"cusip {text!r} is not in the user message; use a cusip that it lists"
                )
            if column == "fund_name" and text not in facts.funds:
                raise SQLRejected(
                    f"fund_name {text!r} is not in the dataset; "
                    "write a fund exactly as the user message spells it"
                )
            if column == "report_quarter":
                if text not in facts.quarters:
                    raise SQLRejected(
                        f"report_quarter {text!r} is not in the dataset; "
                        f"it holds {', '.join(sorted(facts.quarters))}"
                    )
                seen_quarters.add(text)
        for slot in _FILTER_SLOTS:
            for part in _atoms(node.get(slot)):
                if _filters_issuer_name(part):
                    raise SQLRejected(
                        "do not filter on name_of_issuer; "
                        "filter an issuer by cusip, using a cusip from the user message"
                    )
    missing = sorted(set(named_quarters) - seen_quarters)
    if missing:
        raise SQLRejected(
            f"the question names {', '.join(missing)}, but the SQL has no "
            f"report_quarter filter on it; add report_quarter = '{missing[0]}'"
        )
