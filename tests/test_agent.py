"""The agent, end to end, with a fake model and no network.

`Agent(complete=...)` takes the model call as a function, so each test scripts
the replies. The unit tests run over two tiny Parquet files in tmp_path. The
integration test replays the eval set's reference SQL over the real tables.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agents import answer as agent_module
from agents import db, llm, sql
from agents.answer import NULL_ANSWER, VALID_UNITS, Agent, _collect_sources, _shape_answer, main
from agents.prompts import system_prompt
from curator.schema import FILINGS_SCHEMA, HOLDINGS_SCHEMA
from evals.grade import citation, grade

ROOT = Path(__file__).resolve().parents[1]
HAVE_TABLES = (ROOT / "output" / "holdings.parquet").exists()

OUT_OF_SCOPE = "What was the average portfolio value across all managers in 2026 Q3?"
APPLE_QUESTION = "Which manager held the largest Apple position in 2026 Q2?"
APPLE_SQL = (
    "SELECT f.fund_name AS answer, list(DISTINCT h.accession_number) AS accession_number "
    "FROM holdings h JOIN filings f USING (accession_number) "
    "WHERE h.cusip IN ('037833100') AND h.report_quarter = '2026Q2' AND h.put_call IS NULL "
    "GROUP BY f.fund_name ORDER BY sum(h.value) DESC LIMIT 1"
)


def _filing(accession: str, fund: str, cik: str) -> dict:
    return {
        "accession_number": accession, "cik": cik, "fund_name": fund, "filing_manager": fund,
        "form_type": "13F-HR", "report_period": date(2026, 6, 30), "report_quarter": "2026Q2",
        "filing_date": date(2026, 8, 10), "is_amendment": False, "amendment_no": None,
        "amendment_type": None, "report_type": "13F HOLDINGS REPORT",
        "form_13f_file_number": None, "crd_number": None, "sec_file_number": None,
        "other_included_managers_count": 0, "table_entry_total": 2, "table_value_total": 0,
    }


def _holding(accession: str, cik: str, name: str, cusip: str, value: int,
             put_call: str | None = None) -> dict:
    return {
        "accession_number": accession, "cik": cik, "report_quarter": "2026Q2",
        "name_of_issuer": name, "title_of_class": "COM", "cusip": cusip, "figi": None,
        "value": value, "ssh_prnamt": value // 10, "ssh_prnamt_type": "SH",
        "put_call": put_call, "investment_discretion": "SOLE", "other_manager": None,
        "voting_sole": 0, "voting_shared": 0, "voting_none": 0,
    }


@pytest.fixture(scope="module")
def sources(tmp_path_factory) -> dict[str, Path]:
    """Alpha holds more Apple shares; Beta holds a bigger Apple call option.

    Beta files Apple under a second spelling, so only a CUSIP filter sees both.
    """
    folder = tmp_path_factory.mktemp("tables")
    filings = [_filing("A-1", "Alpha Capital LLC", "0000000001"),
               _filing("B-1", "Beta Partners LP", "0000000002")]
    holdings = [
        _holding("A-1", "0000000001", "APPLE INC", "037833100", 5000),
        _holding("A-1", "0000000001", "NVIDIA CORPORATION", "67066G104", 100),
        _holding("A-1", "0000000001", "ALPHABET INC", "02079K305", 700),
        _holding("B-1", "0000000002", "Apple Inc - US", "037833100", 3000),
        _holding("B-1", "0000000002", "APPLE INC", "037833100", 9000, put_call="Call"),
    ]
    pq.write_table(pa.Table.from_pylist(filings, schema=FILINGS_SCHEMA), folder / "f.parquet")
    pq.write_table(pa.Table.from_pylist(holdings, schema=HOLDINGS_SCHEMA), folder / "h.parquet")
    return {"filings": folder / "f.parquet", "holdings": folder / "h.parquet"}


class Script:
    """A fake model: returns the scripted replies in order, records every call."""

    def __init__(self, *replies: dict | str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []
        self.schemas: list[dict | None] = []

    def __call__(self, messages, schema=None, max_tokens=0) -> str:
        self.calls.append(list(messages))
        self.schemas.append(schema)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def _reply(sql: str, unit: str = "NAME", shape: str = "scalar") -> dict:
    return {"sql": sql, "unit": unit, "result_shape": shape}


def test_the_happy_path(sources) -> None:
    model = Script(_reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.contract() == {"answer": "Alpha Capital LLC", "unit": "NAME", "sources": ["A-1"]}
    trace = result.trace
    assert trace.model_calls == 1 and trace.rejected == [] and trace.error is None
    assert trace.issuers[0]["cusip"] == "037833100"
    assert {"resolve", "model_1", "validate", "query", "total"} <= set(trace.timings_ms)


def test_the_prompt_carries_the_cusip_and_a_stable_system_prompt(sources) -> None:
    model = Script(_reply(APPLE_SQL), _reply(NVIDIA_OK))
    with Agent(sources, complete=model) as agent:
        agent.ask(APPLE_QUESTION)
        agent.ask("Which manager held Nvidia in 2026 Q2?")
    first, second = model.calls
    assert first[0] == second[0]  # one cacheable system prompt
    assert "037833100  APPLE INC (COM)" in first[1]["content"]
    assert "67066G104" in second[1]["content"]
    assert APPLE_QUESTION in first[1]["content"]


def test_a_rejected_statement_gets_one_repair(sources) -> None:
    model = Script(_reply("SELECT f.fund_name FROM filings f"), _reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.answer == "Alpha Capital LLC"
    assert result.trace.model_calls == 2
    assert "accession_number" in result.trace.rejected[0]
    assert "accession_number" in model.calls[1][1]["content"]  # the reason goes back


def test_a_statement_duckdb_cannot_bind_gets_one_repair(sources) -> None:
    unbound = ("SELECT f.fund_name AS answer, list(DISTINCT f.accession_number) "
               "AS accession_number FROM filings f")
    model = Script(_reply(unbound), _reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.answer == "Alpha Capital LLC"
    assert result.trace.rejected[0].startswith("BinderException")
    repair = model.calls[1][1]["content"]
    assert "any_value(fund_name)" in repair  # the fix, which DuckDB puts on a later line
    assert f"Your previous statement:\n{unbound}" in repair  # the model sees what it wrote


def test_two_failures_give_a_null_answer_and_a_reason(sources) -> None:
    model = Script(_reply("DROP TABLE holdings"), _reply("DROP TABLE holdings"))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.contract() == NULL_ANSWER
    assert result.trace.model_calls == 2
    assert "two statements failed" in result.trace.error


def test_a_model_failure_gives_a_null_answer(sources) -> None:
    model = Script(llm.LLMError("endpoint down"))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.contract() == NULL_ANSWER
    assert "endpoint down" in result.trace.error


def test_a_reply_inside_a_markdown_fence_still_parses(sources) -> None:
    model = Script("```json\n" + json.dumps(_reply(APPLE_SQL)) + "\n```")
    with Agent(sources, complete=model) as agent:
        assert agent.ask(APPLE_QUESTION).answer == "Alpha Capital LLC"


def test_an_empty_result_is_null_not_no(sources) -> None:
    empty = ("SELECT fund_name AS answer, accession_number FROM filings "
             "WHERE report_quarter = '2026Q2' AND false")
    model = Script(_reply(empty), _reply(empty))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.contract() == NULL_ANSWER
    assert result.trace.model_calls == 2  # the empty result earned one repair call
    assert "no row" in result.trace.rejected[0]
    assert "two statements failed" in result.trace.error


def test_out_of_scope_makes_no_model_call_and_opens_no_database(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the engine must not open for an out-of-scope question")

    monkeypatch.setattr(agent_module.db, "Engine", refuse)
    before = llm.usage()["calls"]
    assert main(OUT_OF_SCOPE) == NULL_ANSWER
    assert llm.usage()["calls"] == before


def _dataset(folder: Path, quarters: tuple[str, ...]) -> dict[str, Path]:
    """One fund, one filing and one Apple row for each quarter."""
    filings, holdings = [], []
    for position, quarter in enumerate(quarters):
        accession = f"A-{position}"
        year, number = int(quarter[:4]), int(quarter[5])
        filing = _filing(accession, "Alpha Capital LLC", "0000000001")
        filing.update(report_quarter=quarter, report_period=date(year, 3 * number, 28))
        holding = _holding(accession, "0000000001", "APPLE INC", "037833100", 5000)
        holding["report_quarter"] = quarter
        filings.append(filing)
        holdings.append(holding)
    pq.write_table(pa.Table.from_pylist(filings, schema=FILINGS_SCHEMA), folder / "f.parquet")
    pq.write_table(pa.Table.from_pylist(holdings, schema=HOLDINGS_SCHEMA), folder / "h.parquet")
    return {"filings": folder / "f.parquet", "holdings": folder / "h.parquet"}


def test_the_system_prompt_is_byte_identical_for_the_shipped_quarters() -> None:
    """The prompt text came from a fixed string before the quarters became data.

    The fixture holds that string. Git may change its line ends on a checkout.
    """
    fixture = ROOT / "tests" / "fixtures" / "system_prompt_2026q1_q2.txt"
    expected = fixture.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert system_prompt(("2026Q1", "2026Q2")) == expected


def test_the_system_prompt_names_the_quarters_of_the_dataset() -> None:
    three = system_prompt(("2026Q1", "2026Q2", "2026Q3"))
    assert "report_quarter VARCHAR ('2026Q1', '2026Q2' or '2026Q3')" in three
    # The examples use the last quarter, and the one before it for a change.
    assert "h.report_quarter = '2026Q3' AND h.put_call IS NULL" in three
    assert "WHEN h.report_quarter = '2026Q2' THEN" in three
    assert "WHEN h.report_quarter = '2026Q1' THEN" not in three

    one = system_prompt(("2025Q4",))
    assert "report_quarter VARCHAR ('2025Q4')" in one
    assert "f.report_quarter = '2025Q4'" in one
    assert "change between quarters" not in one
    assert "2026" not in one


def test_a_dataset_with_no_quarter_has_no_system_prompt() -> None:
    with pytest.raises(ValueError, match="no quarter"):
        system_prompt(())


def test_the_agent_reads_its_quarters_once_from_the_dataset(tmp_path) -> None:
    sources = _dataset(tmp_path, ("2025Q4", "2026Q1"))
    assert db.dataset_quarters(sources) == ("2025Q4", "2026Q1")
    with Agent(sources, complete=Script()) as agent:
        assert agent.quarters == ("2025Q4", "2026Q1")


def test_the_scope_check_and_the_prompt_follow_the_dataset(tmp_path) -> None:
    sources = _dataset(tmp_path, ("2025Q4", "2026Q1"))
    model = Script(_reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        # 2026 Q2 is outside this dataset, though the shipped dataset holds it.
        outside = agent.ask(APPLE_QUESTION)
        assert outside.contract() == NULL_ANSWER
        assert "2025Q4 and 2026Q1" in outside.trace.out_of_scope
        assert model.calls == []

        agent.ask("Which manager held the largest Apple position in 2025 Q4?")
    system = model.calls[0][0]["content"]
    assert "('2025Q4' or '2026Q1')" in system
    assert "2026Q2" not in system


def test_a_one_quarter_dataset_gets_a_prompt_without_a_change_example(tmp_path) -> None:
    sources = _dataset(tmp_path, ("2026Q2",))
    model = Script(_reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        assert agent.quarters == ("2026Q2",)
        assert agent.ask(APPLE_QUESTION).answer == "Alpha Capital LLC"
    assert model.calls[0][0]["content"] == system_prompt(("2026Q2",))


def test_the_check_before_any_agent_reads_the_quarters_of_the_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "TABLE_SOURCES", _dataset(tmp_path, ("2026Q3",)))
    assert agent_module._scope_reason("What did Citadel hold in 2026 Q3?") is None
    assert "2026Q3" in agent_module._scope_reason("What did Citadel hold in 2026 Q2?")


def test_cli_queries_and_scopes_the_selected_data_directory(tmp_path, monkeypatch, capsys):
    paths = _dataset(tmp_path, ("2025Q4",))
    for name, path in paths.items():
        path.rename(tmp_path / f"{name}.parquet")
    model = Script(_reply(APPLE_SQL.replace("2026Q2", "2025Q4")))
    monkeypatch.setattr(agent_module, "Agent", lambda sources: Agent(sources, complete=model))
    question = "Which manager held the largest Apple position in 2025 Q4?"
    monkeypatch.setattr(sys, "argv", ["agents.answer", "--data-dir", str(tmp_path), question])
    assert agent_module._cli() == 0
    assert json.loads(capsys.readouterr().out)["answer"] == "Alpha Capital LLC"
    assert "2025Q4" in model.calls[0][0]["content"]

    monkeypatch.setattr(sys, "argv", ["agents.answer", "--data-dir", str(tmp_path), APPLE_QUESTION])
    assert agent_module._cli() == 0
    assert json.loads(capsys.readouterr().out) == NULL_ANSWER
    assert len(model.calls) == 1


def test_cli_reports_a_missing_data_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["agents.answer", "--data-dir", str(tmp_path), APPLE_QUESTION])
    assert agent_module._cli() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == NULL_ANSWER
    assert "filings.parquet" in captured.err and "missing" in captured.err


def test_a_missing_table_names_the_build_step(tmp_path) -> None:
    sources = {"filings": tmp_path / "none.parquet", "holdings": tmp_path / "none2.parquet"}
    with pytest.raises(FileNotFoundError, match=r"run main\.py"):
        db.dataset_quarters(sources)


def test_main_never_raises_and_keeps_stdout_empty(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")  # the mock reply is not a generation
    for question in ["", "???", "DROP TABLE holdings", APPLE_QUESTION, OUT_OF_SCOPE]:
        result = main(question)
        assert set(result) == {"answer", "unit", "sources"}
        assert result["unit"] in VALID_UNITS
        json.dumps(result)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "rows, shape, names, expected",
    [
        ([("Citadel", "0001-26-1")], "scalar", ["answer", "accession_number"], "Citadel"),
        ([("A", "x"), ("B", "y")], "list", ["answer", "accession_number"], ["A", "B"]),
        ([("A", 5, "x")], "pair", ["answer", "n", "accession_number"], ["A", 5]),
        # The accession is never read as the second half of a pair.
        ([("A", "x")], "pair", ["answer", "accession_number"], "A"),
    ],
    ids=["scalar", "list", "pair", "pair-skips-the-accession"],
)
def test_the_shape_reader(rows, shape, names, expected) -> None:
    assert _shape_answer(rows, shape, names) == expected


def test_sources_flatten_a_list_column_and_sort() -> None:
    rows = [(["0002-26-2", "0001-26-1"],), (["0001-26-1", None],)]
    assert _collect_sources(rows, ["accession_number"]) == ["0001-26-1", "0002-26-2"]
    assert _collect_sources([(5,)], ["answer"]) == []


_SHAPES = {"name": "scalar", "number": "scalar", "yesno": "scalar", "date": "scalar",
           "names": "list", "labels": "list"}


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_TABLES, reason="output/*.parquet is absent")
def test_the_reference_sql_passes_through_the_whole_agent() -> None:
    """Play each eval question's reference SQL as the model's reply.

    The validator must accept it, the engine must run it, the shape reader
    must read the expected answer out of it, and the sources must be exact.
    A label question joins its label in the SQL, as the prompt asks.
    """
    items = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
    with Agent(complete=lambda *a, **k: "") as agent:
        for item in items:
            if "reference_sql" not in item:
                continue
            sql = item["reference_sql"]
            if item["expect"]["kind"] == "labels":
                sql = sql.replace("SELECT f.fund_name, CASE", "SELECT f.fund_name || ' (' || CASE")
                sql = sql.replace("'shrank' END,", "'shrank' END || ')',")
            reply = _reply(sql, shape=_SHAPES[item["expect"]["kind"]])
            # A question with a "lookup" field names a company that no filer spells.
            lookup = [{"lookup": item["lookup"]}] if "lookup" in item else []
            agent._complete = Script(*lookup, reply)
            result = agent.ask(item["question"])
            assert result.trace.error is None, (item["id"], result.trace.error)
            assert bool(result.trace.lookups) == ("lookup" in item), item["id"]
            assert grade(item, result.answer), (item["id"], result.answer)
            assert citation(item["sources"], result.sources) == (1.0, 1.0), item["id"]


# --- run-time checks and the lookup round: a failed check is the repair error -----

NVIDIA_QUESTION = "Which manager held the largest Nvidia position in 2026 Q2?"


def _nvidia(where: str) -> str:
    return (
        "SELECT f.fund_name AS answer, list(DISTINCT h.accession_number) AS accession_number "
        f"FROM holdings h JOIN filings f USING (accession_number) WHERE {where} "
        "GROUP BY f.fund_name ORDER BY sum(h.value) DESC LIMIT 1"
    )


NVIDIA_OK = _nvidia("h.cusip IN ('67066G104') AND h.report_quarter = '2026Q2'")


def _repaired(sources, first_sql: str, fragment: str) -> None:
    """Play a bad first reply and a good second one. Check the repair round trip."""
    model = Script(_reply(first_sql), _reply(NVIDIA_OK))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(NVIDIA_QUESTION)
    assert result.answer == "Alpha Capital LLC", result.trace.error
    assert result.trace.model_calls == 2
    assert fragment in result.trace.rejected[0]
    assert fragment in model.calls[1][1]["content"]  # the reason goes back to the model


@pytest.mark.parametrize(
    "bad_where, fragment",
    [
        ("h.cusip IN ('999999999') AND h.report_quarter = '2026Q2'", "do not invent a cusip"),
        # A real cusip that the user message never listed is a guess from memory.
        ("h.cusip IN ('02079K305') AND h.report_quarter = '2026Q2'", "not in the user message"),
        ("h.cusip IN ('67066G104') AND h.report_quarter = '2026Q9'", "report_quarter '2026Q9'"),
        ("h.cusip IN ('67066G104') AND h.report_quarter = '2026Q2' AND f.fund_name = 'Nobody'",
         "fund_name 'Nobody'"),
        ("upper(h.name_of_issuer) LIKE '%NVIDIA%' AND h.report_quarter = '2026Q2'",
         "do not filter on name_of_issuer"),
        ("h.cusip IN ('67066G104')", "no report_quarter filter on it"),
    ],
    ids=["invented-cusip", "unoffered-cusip", "invented-quarter", "invented-fund", "issuer-name",
         "missing-quarter"],
)
def test_a_failed_fact_check_is_the_repair_error(sources, bad_where, fragment) -> None:
    _repaired(sources, _nvidia(bad_where), fragment)


def test_a_list_with_an_ungrouped_column_is_told_to_group(sources) -> None:
    unbound = ("SELECT f.fund_name AS answer, list(DISTINCT f.accession_number) "
               "AS accession_number FROM filings f WHERE f.report_quarter = '2026Q2'")
    model = Script(_reply(unbound, shape="list"), _reply(unbound + " GROUP BY f.fund_name",
                                                         shape="list"))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(NVIDIA_QUESTION)
    assert sorted(result.answer) == ["Alpha Capital LLC", "Beta Partners LP"]
    repair = model.calls[1][1]["content"]
    assert "collapse it to one row" in repair and "add fund_name to GROUP BY" in repair


def test_a_group_by_accession_is_the_repair_error(sources) -> None:
    split = NVIDIA_OK.replace("GROUP BY f.fund_name", "GROUP BY f.fund_name, h.accession_number")
    assert split != NVIDIA_OK
    _repaired(sources, split, "do not group by accession_number")


def test_a_question_with_a_bare_year_needs_no_quarter_filter(tmp_path) -> None:
    sources = _dataset(tmp_path, ("2026Q1", "2026Q2", "2026Q3", "2026Q4"))
    model = Script(_reply(APPLE_SQL.replace("AND h.report_quarter = '2026Q2' ", "")))
    with Agent(sources, complete=model) as agent:
        result = agent.ask("Which manager held the largest Apple position in 2026?")
    assert result.answer == "Alpha Capital LLC"
    assert result.trace.model_calls == 1


@pytest.mark.parametrize("question", [
    "How did Apple change between 2026 Q2 and 2026 Q3?",
    "How did Apple change between 2026 Q2 and Q3?",
    "What was the total value of Apple in 2026?",
])
def test_partial_scope_never_calls_the_model(sources, question) -> None:
    model = Script()
    with Agent(sources, complete=model) as agent:
        result = agent.ask(question)
    assert result.contract() == NULL_ANSWER
    assert "unavailable" in result.trace.out_of_scope
    assert model.calls == []


@pytest.mark.parametrize(
    "bad_sql, fragment",
    [
        ("SELECT fund_name AS answer, accession_number FROM filings "
         "WHERE report_quarter = '2026Q2' AND false", "no row"),
        ("SELECT sum(value) AS answer, list(DISTINCT accession_number) AS accession_number "
         "FROM holdings WHERE report_quarter = '2026Q2' AND value < 0", "NULL"),
        ("SELECT fund_name AS answer, nullif(accession_number, accession_number) "
         "AS accession_number FROM filings WHERE report_quarter = '2026Q2'",
         "cites no accession_number"),
    ],
    ids=["no-row", "null-answer", "no-accession"],
)
def test_a_failed_result_check_is_the_repair_error(sources, bad_sql, fragment) -> None:
    _repaired(sources, bad_sql, fragment)


def test_a_zero_count_is_an_answer(sources) -> None:
    """A count of zero is a number, not a NULL. The LEFT JOIN cites the filings it searched."""
    zero = ("SELECT count(h.cusip) AS answer, "
            "list(DISTINCT f.accession_number) AS accession_number "
            "FROM filings f LEFT JOIN holdings h ON h.accession_number = f.accession_number "
            "AND h.value < 0 WHERE f.report_quarter = '2026Q2'")
    model = Script(_reply(zero, "COUNT"))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(NVIDIA_QUESTION)
    assert result.answer == 0 and result.trace.model_calls == 1


@pytest.mark.parametrize(
    "failure",
    [duckdb.OutOfMemoryException("Out of Memory Error"),
     agent_module.db.QueryTimeout("query exceeded 10s")],
    ids=["out-of-memory", "timeout"],
)
def test_a_timeout_and_an_out_of_memory_error_get_no_repair(
    sources, monkeypatch, failure
) -> None:
    model = Script(_reply(NVIDIA_OK), _reply(NVIDIA_OK))
    with Agent(sources, complete=model) as agent:
        def fail(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(agent.engine, "query", fail)
        result = agent.ask(NVIDIA_QUESTION)
    assert result.contract() == NULL_ANSWER
    assert result.trace.model_calls == 1
    assert type(failure).__name__ in result.trace.error


# --- the lookup round -----------------------------------------------------------

GOOGLE_QUESTION = "Which manager held the largest Google position in 2026 Q2?"
GOOGLE_SQL = (
    "SELECT f.fund_name AS answer, list(DISTINCT h.accession_number) AS accession_number "
    "FROM holdings h JOIN filings f USING (accession_number) "
    "WHERE h.cusip IN ('02079K305') AND h.report_quarter = '2026Q2' "
    "GROUP BY f.fund_name ORDER BY sum(h.value) DESC LIMIT 1"
)
LOOKUP = {"lookup": ["Alphabet Inc"]}


def test_a_lookup_resolves_the_names_and_asks_again(sources) -> None:
    model = Script(LOOKUP, _reply(GOOGLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(GOOGLE_QUESTION)
    assert result.contract() == {"answer": "Alpha Capital LLC", "unit": "NAME", "sources": ["A-1"]}
    trace = result.trace
    assert trace.model_calls == 2 and trace.rejected == []
    assert trace.lookups == [{"name": "Alphabet Inc", "funds": [], "issuers": ["02079K305"]}]
    first, second = (call[1]["content"] for call in model.calls)
    assert "<looked_up_names>" not in first
    assert '"Alphabet Inc"' in second and "02079K305  ALPHABET INC (COM)" in second
    assert "<resolved_names>" in second  # the first block stays
    assert model.schemas[0] == sql.REPLY_SCHEMA  # the first call may ask for a lookup
    assert model.schemas[1] == sql.GENERATION_SCHEMA  # after the lookup: a statement only
    assert model.calls[0][0] == model.calls[1][0]  # one cacheable system prompt
    assert "ALPHABET" not in model.calls[0][0]["content"]  # no filing text in it


def test_a_question_that_needs_no_lookup_costs_one_call(sources) -> None:
    model = Script(_reply(APPLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(APPLE_QUESTION)
    assert result.trace.model_calls == 1 and result.trace.lookups == []


def test_a_second_lookup_counts_as_a_failed_statement(sources) -> None:
    model = Script(LOOKUP, LOOKUP, _reply(GOOGLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(GOOGLE_QUESTION)
    assert result.answer == "Alpha Capital LLC"
    assert result.trace.model_calls == 3
    assert "no second lookup" in result.trace.rejected[0]


def test_the_worst_case_is_three_calls(sources) -> None:
    bad = _nvidia("h.cusip IN ('999999999')")
    model = Script(LOOKUP, _reply(bad), _reply(bad), _reply(GOOGLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(GOOGLE_QUESTION)
    assert result.contract() == NULL_ANSWER
    assert result.trace.model_calls == 3
    assert len(model.replies) == 1  # the fourth call never happens
    assert "two statements failed" in result.trace.error


def test_a_rejected_statement_can_lead_to_a_lookup(sources) -> None:
    invented = _nvidia("h.cusip IN ('02079K999') AND h.report_quarter = '2026Q2'")
    model = Script(_reply(invented), LOOKUP, _reply(GOOGLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(GOOGLE_QUESTION)
    assert result.answer == "Alpha Capital LLC"
    assert result.trace.model_calls == 3
    assert "ask for a lookup" in model.calls[1][1]["content"]
    assert "do not invent a cusip" in model.calls[2][1]["content"]  # the error stays in view


def test_a_lookup_that_finds_nothing_still_reaches_the_model(sources) -> None:
    model = Script({"lookup": ["Zzyzx Holdings"]}, _reply(GOOGLE_SQL))
    with Agent(sources, complete=model) as agent:
        result = agent.ask(GOOGLE_QUESTION)
    assert result.trace.lookups[0]["issuers"] == []
    assert "(none matched)" in model.calls[1][1]["content"].split("<looked_up_names>")[1]
