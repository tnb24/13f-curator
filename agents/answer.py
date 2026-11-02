"""The agent: a question in, a structured answer out.

    python -m agents.answer "which manager held the largest Apple position in 2026 Q2?"

stdout carries one JSON object and nothing else. Logs go to stderr.

    {
      "answer":  <number | string | list | null>,
      "unit":    "USD" | "SHARES" | "COUNT" | "PERCENT" | "NAME" | "DATE" | "NONE",
      "sources": ["0001423053-26-000012", ...]
    }

`sources` lists the accession numbers of the filings behind the answer, so a
researcher can check every answer against EDGAR.

The pipeline runs in five steps:

    1. reject a question about a quarter outside the dataset (no model call)
    2. find the fund and issuer spellings the dataset holds    agents/resolve.py
    3. ask the model for one SELECT statement                  agents/sql.py
    4. validate the statement against an allow list            agents/validate.py
    5. run it on a read-only DuckDB connection                 agents/db.py
"""

from __future__ import annotations

import datetime as _datetime
import decimal
import json
import sys
import traceback
from typing import Any

from agents import db, resolve, sql as sqlgen, validate

VALID_UNITS = {"USD", "SHARES", "COUNT", "PERCENT", "NAME", "DATE", "NONE"}


NULL_ANSWER: dict[str, Any] = {"answer": None, "unit": "NONE", "sources": []}


def _log(message: str) -> None:
    """Every diagnostic goes to stderr. stdout carries the JSON only."""
    print(message, file=sys.stderr)


def _plain(value: Any) -> Any:
    """Return a value that survives json.dumps.

    DuckDB hands back Decimal for an exact aggregate and date for a date32
    column. numpy scalars carry an .item() method.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (AttributeError, ValueError):
            pass
    return str(value)


def _accession_columns(names: list[str]) -> list[int]:
    """Return the index of every projected column that holds an accession."""
    return [i for i, name in enumerate(names) if "accession" in name.lower()]


def _collect_sources(rows: list[tuple], names: list[str]) -> list[str]:
    """Every accession in every accession column, sorted and deduplicated.

    DuckDB returns `list(DISTINCT accession_number)` as a Python list inside one
    cell, so a cell is flattened before its values are read.
    """
    found: set[str] = set()
    for index in _accession_columns(names):
        for row in rows:
            if index >= len(row):
                continue
            cell = row[index]
            values = cell if isinstance(cell, (list, tuple)) else [cell]
            for value in values:
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    found.add(text)
    return sorted(found)


def _answer_columns(names: list[str], width: int) -> list[int]:
    """Return the column indexes that carry the answer.

    The prompt puts `accession_number` straight after the answer column, so the
    second column of a pair is the accession unless it is excluded here. An
    accession is evidence, not an answer.
    """
    accessions = set(_accession_columns(names))
    columns = [i for i in range(width) if i not in accessions]
    return columns or list(range(width))


def _shape_answer(rows: list[tuple], shape: str, names: list[str]) -> Any:
    """Read the answer out of the rows the way `result_shape` names."""
    columns = _answer_columns(names, max(len(row) for row in rows))
    if shape == "list":
        return [_plain(row[columns[0]]) for row in rows if row]
    first = rows[0]
    if shape == "pair":
        # A pair needs two answer columns. When the query projected only one,
        # reporting it alone beats padding the pair with a null.
        if len(columns) >= 2:
            return [_plain(first[columns[0]]), _plain(first[columns[1]])]
        _log("agent: result_shape is pair but the query projected one answer column")
    return _plain(first[columns[0]])


def _generate_validated(
    question: str, funds: list[str], issuers: list[str]
) -> tuple[str, dict[str, Any]]:
    """Return validated SQL and the generation. One repair call is allowed."""
    generation = sqlgen.generate(question, funds, issuers)
    try:
        return validate.check(generation["sql"]), generation
    except validate.SQLRejected as rejected:
        _log(f"agent: the validator rejected the first statement: {rejected}")
        generation = sqlgen.generate(question, funds, issuers, error=str(rejected))
        return validate.check(generation["sql"]), generation


def main(question: str) -> dict[str, Any]:
    """Answer `question` against the dataset. Never raises."""
    try:
        # Step 1. An out-of-scope quarter returns before the agent spends a
        # token and before it opens the database.
        reason = resolve.out_of_scope(question)
        if reason:
            _log(f"agent: out of scope, so no model call happens: {reason}")
            return dict(NULL_ANSWER)

        connection = db.connect()
        funds = resolve.fund_candidates(question, connection)
        issuers = resolve.issuer_candidates(question, connection)
        _log(f"agent: {len(funds)} fund candidate(s), {len(issuers)} issuer candidate(s)")

        statement, generation = _generate_validated(question, funds, issuers)
        _log(f"agent: running {statement}")

        rows = db.query(connection, statement)
        names = [column[0] for column in (connection.description or [])]

        if not rows:
            _log(
                "agent: the query returned no row. The prompt asks for an aggregate, "
                "so an empty result means the query shape was wrong, not that the "
                "answer is No."
            )
            return dict(NULL_ANSWER)

        answer = _shape_answer(rows, generation["result_shape"], names)
        sources = _collect_sources(rows, names)
        if not _accession_columns(names):
            _log("agent: the projection carried no accession column, so sources is empty")

        unit = generation["unit"]
        if unit not in VALID_UNITS:
            _log(f"agent: the model returned unit {unit!r}, which is not allowed")
            unit = "NONE"
        if answer is None:
            unit = "NONE"

        return {"answer": answer, "unit": unit, "sources": sources}
    except Exception as exc:  # noqa: BLE001 - the agent must never raise.
        _log(f"agent: returning a null answer after {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        return dict(NULL_ANSWER)


def _cli() -> int:
    if len(sys.argv) < 2:
        print('usage: python -m agents.answer "your question"', file=sys.stderr)
        return 2

    result = main(sys.argv[1])

    # Check the shape here, so a mistake surfaces before a caller parses stdout.
    if not isinstance(result, dict):
        print(f"main() must return a dict, got {type(result).__name__}", file=sys.stderr)
        return 1
    missing = {"answer", "unit", "sources"} - set(result)
    if missing:
        print(f"result missing key(s): {sorted(missing)}", file=sys.stderr)
        return 1
    if result["unit"] not in VALID_UNITS:
        print(f"unit must be one of {sorted(VALID_UNITS)}, got {result['unit']!r}",
              file=sys.stderr)
        return 1
    if not isinstance(result["sources"], list):
        print("sources must be a list of accession numbers", file=sys.stderr)
        return 1

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
