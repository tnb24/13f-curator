"""The agent: a question in, a structured answer out.

    python -m agents.answer "Which manager held the largest Apple position in 2026 Q2?"
    python -m agents.answer --trace "..."     # also print the trace to stderr

stdout carries one JSON object and nothing else. Logs go to stderr.

    {
      "answer":  <number | string | list | null>,
      "unit":    "USD" | "SHARES" | "COUNT" | "PERCENT" | "NAME" | "DATE" | "NONE",
      "sources": ["0001423053-26-000012", ...]
    }

`sources` lists the accession numbers of the filings behind the answer, so a
researcher can check every answer against EDGAR.

One question runs in six steps:

    1. reject a question about a quarter outside the dataset (no model call)
    2. find the funds and issuers (by CUSIP) that it names     agents/resolve.py
    3. ask the model for one SELECT statement                  agents/sql.py
    4. validate the statement against an allow list            agents/validate.py
    5. run it on a private, read-only DuckDB                   agents/db.py
    6. check the result: a row, an answer, and a filing cited

The model may answer step 3 with names to look up instead of SQL, such as
"Alphabet Inc" for "Google". The agent resolves them with the same index as
step 2 and asks again. A failure in step 4, 5, or 6 goes back to the model as
the repair error. The call budget is at most one lookup and two statements, so
a question costs one call, or at most three.

`Agent` owns everything that outlives one question: the database, the report
quarters of the dataset (`Agent.quarters`), and the name indexes. A process
that asks many questions, such as the eval runner, builds one Agent and reuses
it. `main()` builds one for a single question.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import decimal
import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from agents import db, resolve, validate
from agents import sql as sqlgen
from agents.resolve import Issuer, IssuerIndex, Resolution, Resolver

VALID_UNITS = frozenset({"USD", "SHARES", "COUNT", "PERCENT", "NAME", "DATE", "NONE"})
NULL_ANSWER: dict[str, Any] = {"answer": None, "unit": "NONE", "sources": []}
_PREVIEW_ROWS = 20

# The call budget. A lookup is not a statement, so the worst case is one lookup,
# one statement, and one repaired statement: three calls. A question that needs
# no lookup and no repair costs one call.
MAX_LOOKUPS = 1
MAX_STATEMENTS = 2

# A DuckDB error that a second statement would most likely repeat. A timeout
# (db.QueryTimeout) is not a duckdb.Error, so it already passes through.
_NO_REPAIR = (duckdb.InterruptException, duckdb.OutOfMemoryException)


def _log(message: str) -> None:
    """Every diagnostic goes to stderr. stdout carries the JSON only."""
    print(message, file=sys.stderr)


def _plain(value: Any) -> Any:
    """Return a value that survives json.dumps.

    DuckDB returns Decimal for an exact aggregate and date for a date32 column.
    A numpy scalar has an .item() method.
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
            for value in cell if isinstance(cell, (list, tuple)) else [cell]:
                text = "" if value is None else str(value).strip()
                if text:
                    found.add(text)
    return sorted(found)


def _answer_columns(names: list[str], width: int) -> list[int]:
    """Return the column indexes that carry the answer.

    The prompt puts accession_number after the answer column. An accession is
    evidence, not an answer, so it is never read as one.
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


def _read_result(result: db.Result, shape: str) -> tuple[Any, list[str]]:
    """Return the answer and its sources. Raise SQLRejected when the result is no answer.

    Three results are failures, and each earns a repair call: no row, a NULL
    answer, and an answer that cites no accession. The agent returns a null
    answer only after the repair also fails.
    """
    if not result.rows:
        raise validate.SQLRejected(
            "the query returned no row; a filter probably matches nothing, so recheck "
            "each cusip, fund_name, quarter, and put_call filter"
        )
    answer = _shape_answer(result.rows, shape, result.columns)
    if answer is None or (isinstance(answer, list) and all(item is None for item in answer)):
        raise validate.SQLRejected(
            "the answer column is NULL; an aggregate over no row gives NULL, "
            "so recheck each filter"
        )
    sources = _collect_sources(result.rows, result.columns)
    if not sources:
        raise validate.SQLRejected(
            "the result cites no accession_number; the answer must name the filings it reads"
        )
    return answer, sources


@dataclass
class _Attempt:
    """One statement that passed every check, with its result."""

    statement: str
    generation: dict[str, Any]
    result: db.Result
    answer: Any
    sources: list[str]


@dataclass
class Trace:
    """How one answer came about. The CLI prints it on --trace; the evals store it."""

    question: str
    out_of_scope: str | None = None
    terms: list[str] = field(default_factory=list)
    funds: list[str] = field(default_factory=list)
    issuers: list[dict[str, Any]] = field(default_factory=list)
    lookups: list[dict[str, Any]] = field(default_factory=list)
    sql: str | None = None
    rejected: list[str] = field(default_factory=list)
    model_calls: int = 0
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class Answer:
    answer: Any
    unit: str
    sources: list[str]
    trace: Trace

    def contract(self) -> dict[str, Any]:
        """The three keys that the CLI prints and the grader reads."""
        return {"answer": self.answer, "unit": self.unit, "sources": list(self.sources)}

    def to_dict(self) -> dict[str, Any]:
        return {**self.contract(), "trace": asdict(self.trace)}


class _Clock:
    """Milliseconds per step, into the trace."""

    def __init__(self, trace: Trace) -> None:
        self._trace = trace
        self._started = time.perf_counter()
        self._mark = self._started

    def lap(self, step: str) -> None:
        now = time.perf_counter()
        self._trace.timings_ms[step] = round((now - self._mark) * 1000, 2)
        self._mark = now

    def total(self) -> None:
        self._trace.timings_ms["total"] = round((time.perf_counter() - self._started) * 1000, 2)


_SPELLINGS_SQL = """
SELECT cusip, name_of_issuer, title_of_class, count(*) AS n
FROM holdings GROUP BY ALL ORDER BY cusip, n DESC, name_of_issuer, title_of_class
"""


def build_resolver(engine: db.Engine) -> Resolver:
    """Build the fund and issuer indexes from one scan of each table.

    One grouped scan of holdings returns every (cusip, spelling, class) with
    its row count, most common first within each CUSIP. That ordering gives
    each CUSIP its usual name and class, and the rows give every spelling.
    """
    funds = [row[0] for row in engine.query(
        "SELECT DISTINCT fund_name FROM filings ORDER BY 1", limit=10_000).rows]

    groups = engine.query(_SPELLINGS_SQL, limit=10_000_000).rows
    names: dict[str, dict[str, int]] = {}
    titles: dict[str, dict[str, int]] = {}
    for cusip, name, title, count in groups:
        spelled = names.setdefault(cusip, {})
        spelled[name] = spelled.get(name, 0) + count
        classed = titles.setdefault(cusip, {})
        classed[title] = classed.get(title, 0) + count

    def usual(counts: dict[str, int]) -> str:
        return min(counts, key=lambda text: (-counts[text], text))

    issuers = {
        cusip: Issuer(
            cusip=cusip,
            name=usual(spelled),
            title_of_class=usual(titles[cusip]),
            rows=sum(spelled.values()),
        )
        for cusip, spelled in names.items()
    }
    spellings = [(cusip, name) for cusip, spelled in names.items() for name in spelled]
    return Resolver(funds, IssuerIndex(spellings, issuers))


class Agent:
    """The dataset, its indexes, and a model, ready to answer questions."""

    def __init__(
        self,
        sources: dict[str, Any] | None = None,
        complete: sqlgen.Complete | None = None,
    ) -> None:
        started = time.perf_counter()
        if complete is None:
            from agents.llm import complete as live_complete

            complete = live_complete
        self._complete = complete
        self.engine = db.Engine(sources)
        try:
            # The report quarters of the dataset, read once. The scope check
            # and the system prompt both use them.
            self.quarters = db.dataset_quarters(sources)
            self.resolver = build_resolver(self.engine)
        except BaseException:
            self.engine.close()
            raise
        self.startup_ms = round((time.perf_counter() - started) * 1000, 1)

    def close(self) -> None:
        self.engine.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def ask(self, question: str) -> Answer:
        """Answer one question. Never raises: a failure is a null answer and a trace."""
        trace = Trace(question=question)
        clock = _Clock(trace)
        try:
            return self._ask(question, trace, clock)
        except Exception as exc:  # The contract: ask() never raises; the trace keeps why.
            trace.error = f"{type(exc).__name__}: {exc}"
            _log(f"agent: returning a null answer after {trace.error}")
            traceback.print_exc(file=sys.stderr)
            return Answer(None, "NONE", [], trace)
        finally:
            clock.total()

    def _ask(self, question: str, trace: Trace, clock: _Clock) -> Answer:
        reason = resolve.out_of_scope(question, self.quarters)
        if reason:
            trace.out_of_scope = reason
            _log(f"agent: out of scope, so no model call happens: {reason}")
            return Answer(None, "NONE", [], trace)

        resolution = self.resolver.resolve(question)
        trace.terms = resolution.terms
        trace.funds = resolution.funds
        trace.issuers = [asdict(issuer) for issuer in resolution.issuers]
        clock.lap("resolve")

        attempt = self._generate_and_run(question, resolution, trace, clock)
        trace.sql = attempt.statement
        trace.columns = attempt.result.columns
        trace.row_count = len(attempt.result.rows)
        trace.rows = [[_plain(cell) for cell in row] for row in attempt.result.rows[:_PREVIEW_ROWS]]
        unit = attempt.generation["unit"]
        return Answer(attempt.answer, unit, attempt.sources, trace)

    def _generate_and_run(
        self, question: str, resolution: Resolution, trace: Trace, clock: _Clock
    ) -> _Attempt:
        """Return the first statement that passes every check. Raise SQLRejected if none does.

        The budget is MAX_STATEMENTS statements and MAX_LOOKUPS lookup. A
        statement fails at one of four points:

        - the validator rejects the statement;
        - a literal in the statement is not in the dataset, a cusip is not one
          that the user message listed, or the statement ignores a quarter
          that the question names;
        - DuckDB fails to bind or run the statement (a column beside an
          aggregate gets its fix in the message);
        - the result holds no row, a NULL answer, or no accession.

        The failure message goes back to the model as the repair error. A
        timeout and an out-of-memory error get no repair, because a second
        statement most likely fails the same way.
        """
        offered = {issuer.cusip for issuer in resolution.issuers}
        facts = validate.Facts(
            cusips=self.resolver.cusips,
            funds=self.resolver.fund_set,
            quarters=frozenset(self.quarters),
            offered=offered,
        )
        named = resolve.named_quarters(question) & facts.quarters
        cursor = self.engine.cursor()
        try:
            error: str | None = None
            previous: str | None = None
            looked_up: list[resolve.Lookup] = []
            statements = 0
            while statements < MAX_STATEMENTS:
                trace.model_calls += 1
                generation = sqlgen.generate(
                    question, resolution.funds, resolution.issuers, self._complete,
                    error=error, lookups=looked_up, allow_lookup=len(trace.lookups) < MAX_LOOKUPS,
                    previous=previous, quarters=self.quarters,
                )
                clock.lap(f"model_{trace.model_calls}")
                if "lookup" in generation and len(trace.lookups) < MAX_LOOKUPS:
                    looked_up = self.resolver.lookup(generation["lookup"])
                    offered.update(issuer.cusip for found in looked_up for issuer in found.issuers)
                    trace.lookups = [
                        {"name": found.name, "funds": found.funds,
                         "issuers": [issuer.cusip for issuer in found.issuers]}
                        for found in looked_up
                    ]
                    clock.lap("lookup")
                    continue
                statements += 1
                try:
                    if "lookup" in generation:
                        raise validate.SQLRejected("no second lookup is available; reply with sql")
                    trace.sql = generation["sql"]
                    checked = validate.inspect(generation["sql"], cursor)
                    validate.check_facts(checked.tree, facts, named)
                    clock.lap("validate")
                    result = self.engine.query(checked.sql)
                    clock.lap("query")
                    answer, sources = _read_result(result, generation["result_shape"])
                    clock.lap("check_result")
                    return _Attempt(checked.sql, generation, result, answer, sources)
                except (validate.SQLRejected, duckdb.Error) as failure:
                    if isinstance(failure, _NO_REPAIR):
                        raise
                    error = validate.describe_failure(failure, generation.get("result_shape"))
                    previous = generation.get("sql") or previous
                    trace.rejected.append(error)
                    clock.lap(f"failed_{trace.model_calls}")
                    _log(f"agent: statement {trace.model_calls} failed: {error}")
            raise validate.SQLRejected(f"two statements failed; last: {error}")
        finally:
            cursor.close()


def _scope_reason(question: str, sources: dict[str, Path] | None = None) -> str | None:
    """Check a question against the dataset before any Agent exists.

    The check reads the quarters from the filings file. It opens no database.
    """
    return resolve.out_of_scope(question, db.dataset_quarters(sources))


def main(question: str) -> dict[str, Any]:
    """Answer one question in a fresh Agent. Never raises.

    The scope check runs first, so an out-of-scope question opens no database
    and makes no model call.
    """
    try:
        reason = _scope_reason(question)
        if reason:
            _log(f"agent: out of scope, so no model call happens: {reason}")
            return dict(NULL_ANSWER)
        with Agent() as agent:
            return agent.ask(question).contract()
    except Exception as exc:  # The contract: main() never raises, even if startup fails.
        _log(f"agent: returning a null answer after {type(exc).__name__}: {exc}")
        return dict(NULL_ANSWER)


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="python -m agents.answer")
    parser.add_argument("question")
    parser.add_argument("--trace", action="store_true", help="print the trace to stderr")
    parser.add_argument("--data-dir", type=Path, help="directory containing the two Parquet tables")
    args = parser.parse_args()

    sources = (
        {name: args.data_dir / f"{name}.parquet" for name in ("filings", "holdings")}
        if args.data_dir is not None else None
    )
    failed = False
    try:
        reason = _scope_reason(args.question, sources)
        if reason:
            result = dict(NULL_ANSWER)
            if args.trace:
                _log(json.dumps({"out_of_scope": reason}, indent=2))
            else:
                _log(f"agent: out of scope, so no model call happens: {reason}")
        else:
            with Agent(sources=sources) as agent:
                answer = agent.ask(args.question)
            result = answer.contract()
            failed = answer.trace.error is not None
            if args.trace:
                _log(json.dumps(asdict(answer.trace), indent=2, default=str))
    except Exception as exc:
        _log(f"agent: returning a null answer after {type(exc).__name__}: {exc}")
        result = dict(NULL_ANSWER)
        failed = True

    # Check the shape here, so a mistake surfaces before a caller parses stdout.
    missing = {"answer", "unit", "sources"} - set(result)
    if missing or result["unit"] not in VALID_UNITS or not isinstance(result["sources"], list):
        _log(f"agent: malformed result {result!r}")
        return 1
    print(json.dumps(result))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_cli())
