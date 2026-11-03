"""The agent, end to end, with a fake model and no network.

`Agent(complete=...)` takes the model call as a function, so each test scripts
the replies. The unit tests run over two tiny Parquet files in tmp_path. The
integration test replays the eval set's reference SQL over the real tables.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agents import answer as agent_module
from agents import llm
from agents.answer import NULL_ANSWER, VALID_UNITS, Agent, _collect_sources, _shape_answer, main
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

    def __call__(self, messages, schema=None, max_tokens=0) -> str:
        self.calls.append(list(messages))
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
    model = Script(_reply(APPLE_SQL), _reply(APPLE_SQL))
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
    empty = "SELECT fund_name AS answer, accession_number FROM filings WHERE false"
    with Agent(sources, complete=Script(_reply(empty))) as agent:
        assert agent.ask(APPLE_QUESTION).contract() == NULL_ANSWER


def test_out_of_scope_makes_no_model_call_and_opens_no_database(monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("the engine must not open for an out-of-scope question")

    monkeypatch.setattr(agent_module.db, "Engine", refuse)
    before = llm.usage()["calls"]
    assert main(OUT_OF_SCOPE) == NULL_ANSWER
    assert llm.usage()["calls"] == before


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
            agent._complete = Script(reply)
            result = agent.ask(item["question"])
            assert result.trace.error is None, (item["id"], result.trace.error)
            assert grade(item, result.answer), (item["id"], result.answer)
            assert citation(item["sources"], result.sources) == (1.0, 1.0), item["id"]
