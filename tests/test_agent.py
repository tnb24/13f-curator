"""The agent contract: the result shape, and zero model calls when out of scope.

No test here contacts the network. Every test sets `LLM_MODE` itself with
`monkeypatch.setenv`, so the module passes whatever the shell holds.
"""

from __future__ import annotations

import json

import pytest

from agents import llm
from agents.answer import VALID_UNITS, main

OUT_OF_SCOPE = "What was the average portfolio value across all managers in 2026 Q3?"
IN_SCOPE = "Which manager held the largest Apple position in 2026 Q2?"


def test_out_of_scope_makes_zero_model_calls(monkeypatch) -> None:
    """The strongest evidence that the zero-token path really is zero-cost.

    This test needs no database: the out-of-scope check returns before the
    agent opens DuckDB or builds a prompt.
    """
    monkeypatch.setenv("LLM_MODE", "mock")
    before = llm.usage()["calls"]
    result = main(OUT_OF_SCOPE)
    assert llm.usage()["calls"] == before
    assert result == {"answer": None, "unit": "NONE", "sources": []}


def test_out_of_scope_writes_its_reason_to_stderr(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    main(OUT_OF_SCOPE)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "2026Q3" in captured.err


def test_mock_mode_returns_a_null_answer(monkeypatch) -> None:
    """The mock returns an object with no `sql` key, so generate raises."""
    monkeypatch.setenv("LLM_MODE", "mock")
    assert main(IN_SCOPE) == {"answer": None, "unit": "NONE", "sources": []}


def test_the_agent_never_raises(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    for question in ["", "???", "DROP TABLE holdings", IN_SCOPE, OUT_OF_SCOPE]:
        result = main(question)
        assert isinstance(result, dict)


def test_the_result_always_has_the_contract_shape(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    result = main(IN_SCOPE)
    assert set(result) == {"answer", "unit", "sources"}
    assert result["unit"] in VALID_UNITS
    assert isinstance(result["sources"], list)


def test_the_result_survives_json_dumps(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    assert json.loads(json.dumps(main(IN_SCOPE)))["unit"] == "NONE"


def test_main_writes_nothing_to_stdout(monkeypatch, capsys) -> None:
    """`_cli` prints the JSON. `main` logs to stderr only."""
    monkeypatch.setenv("LLM_MODE", "mock")
    main(IN_SCOPE)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "rows, shape, names, expected",
    [
        ([("Citadel", "0001-26-1")], "scalar", ["answer", "accession_number"], "Citadel"),
        ([("A", "x"), ("B", "y")], "list", ["answer", "accession_number"], ["A", "B"]),
        ([("A", 5, "x")], "pair", ["answer", "n", "accession_number"], ["A", 5]),
        # Rule 3 reserves column two for the accession, so a pair must skip it.
        ([("A", "x")], "pair", ["answer", "accession_number"], "A"),
    ],
    ids=["scalar", "list", "pair", "pair-skips-the-accession"],
)
def test_the_shape_reader_skips_the_accession_column(rows, shape, names, expected) -> None:
    from agents.answer import _shape_answer

    assert _shape_answer(rows, shape, names) == expected


def test_sources_flatten_a_list_column_and_sort() -> None:
    """DuckDB returns list(DISTINCT accession_number) as a list in one cell."""
    from agents.answer import _collect_sources

    rows = [(["0002-26-2", "0001-26-1"],), (["0001-26-1"],)]
    assert _collect_sources(rows, ["accession_number"]) == ["0001-26-1", "0002-26-2"]


def test_sources_are_empty_when_no_accession_is_projected() -> None:
    from agents.answer import _collect_sources

    assert _collect_sources([(5,)], ["answer"]) == []
