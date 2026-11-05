"""The scope value: how it parses, what it rejects, and what it derives."""

from __future__ import annotations

import pytest

from curator.scope import (
    DEFAULT_CUTOFF,
    DEFAULT_QUARTERS,
    Scope,
    ScopeError,
    parse_quarter,
    previous_quarter,
    quarter_end,
)


def test_the_default_scope_is_the_shipped_dataset() -> None:
    scope = Scope()
    assert scope.quarters == ("2026Q1", "2026Q2") == DEFAULT_QUARTERS
    assert scope.cutoff == "2026-08-18" == DEFAULT_CUTOFF
    assert scope.report_periods == ("2026-03-31", "2026-06-30")
    assert scope.expected_per_filer == 2
    assert scope.last_quarter == "2026Q2"
    assert Scope.parse("2026Q1,2026Q2", "2026-08-18") == scope


@pytest.mark.parametrize("quarter, end", [
    ("2026Q1", "2026-03-31"),
    ("2026Q2", "2026-06-30"),
    ("2026Q3", "2026-09-30"),
    ("2025Q4", "2025-12-31"),
])
def test_a_quarter_ends_on_the_report_date(quarter: str, end: str) -> None:
    assert quarter_end(quarter) == end


@pytest.mark.parametrize("quarter, before", [
    ("2026Q2", "2026Q1"),
    ("2026Q1", "2025Q4"),
    ("2026Q4", "2026Q3"),
])
def test_the_quarter_before(quarter: str, before: str) -> None:
    assert previous_quarter(quarter) == before


@pytest.mark.parametrize("text", ["2026", "2026Q5", "2026Q0", "26Q1", "Q1 2026", "2026-Q1", ""])
def test_a_text_that_is_not_a_quarter_is_refused(text: str) -> None:
    with pytest.raises(ScopeError, match="YYYYQN"):
        parse_quarter(text)


def test_the_command_line_text_sorts_and_ignores_case_and_spaces() -> None:
    scope = Scope.parse(" 2026q2 , 2026Q1,", " 2026-08-18 ")
    assert scope.quarters == ("2026Q1", "2026Q2")
    assert scope.cutoff == "2026-08-18"


def test_the_quarter_count_gives_the_expected_filings_per_filer() -> None:
    assert Scope.parse("2025Q4", "2026-02-17").expected_per_filer == 1
    assert Scope.parse("2026Q1,2026Q2,2026Q3", "2026-11-16").expected_per_filer == 3


def test_the_scope_records_itself_for_the_manifest() -> None:
    assert Scope.parse("2025Q4,2026Q1", "2026-05-15").as_dict() == {
        "quarters": ["2025Q4", "2026Q1"],
        "report_periods": ["2025-12-31", "2026-03-31"],
        "filing_date_cutoff": "2026-05-15",
    }


def test_a_quarter_that_ends_after_the_cutoff_is_refused_with_the_fix() -> None:
    with pytest.raises(ScopeError) as caught:
        Scope.parse("2026Q1,2026Q2,2026Q3", "2026-08-18")
    message = str(caught.value)
    assert "quarter 2026Q3 ends after the filing cutoff 2026-08-18" in message
    assert "--cutoff to 2026-09-30 or later" in message
    assert "remove the quarter from --quarters" in message


def test_every_late_quarter_is_named() -> None:
    with pytest.raises(ScopeError, match=r"quarters 2026Q2, 2026Q3 end after"):
        Scope.parse("2026Q1,2026Q2,2026Q3", "2026-05-01")


def test_a_cutoff_on_the_last_day_of_the_quarter_stands() -> None:
    assert Scope.parse("2026Q2", "2026-06-30").report_periods == ("2026-06-30",)
    with pytest.raises(ScopeError):
        Scope.parse("2026Q2", "2026-06-29")


@pytest.mark.parametrize("cutoff", ["2026-8-18", "18/08/2026", "20260818", "2026-02-30", ""])
def test_a_cutoff_that_is_not_a_date_is_refused(cutoff: str) -> None:
    with pytest.raises(ScopeError, match="cutoff"):
        Scope.parse("2026Q1", cutoff)


def test_a_repeated_or_missing_quarter_is_refused() -> None:
    with pytest.raises(ScopeError, match="repeat"):
        Scope.parse("2026Q1,2026Q1", "2026-08-18")
    with pytest.raises(ScopeError, match="at least one quarter"):
        Scope.parse(" , ", "2026-08-18")
    with pytest.raises(ScopeError, match="ascending"):
        Scope(("2026Q2", "2026Q1"), "2026-08-18")
