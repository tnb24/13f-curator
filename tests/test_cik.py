"""Tests for curator.cik: name normalisation and the 13F confirmation gate.

A fake cache serves every body from a dictionary. No test touches the
network, and the reduced lookup file lands in ``tmp_path`` through the
cache's ``root`` property, exactly as the real cache behaves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curator import cik
from curator.cache import FetchError
from curator.scope import Scope

SCOPE = Scope()  # the default scope: 2026Q1 and 2026Q2, filed by 2026-08-18

_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:0>10}.json"

_FILER_CIK = "0001002958"  # BAUPOST GROUP LLC/MA: filed a 13F in scope.
_NONFILER_CIK = "0001999999"  # BAUPOST GROUP LLC /ADV: filed nothing in scope.


class FakeCache:
    """Return canned bodies by URL. Raise FetchError when no body exists."""

    def __init__(self, root: Path, bodies: dict[str, bytes]) -> None:
        self.root = Path(root)
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        if url not in self._bodies:
            raise FetchError(f"GET {url} failed (no canned body)")
        return self._bodies[url]


def _submissions_body(
    name: str, forms: list[str], report_dates: list[str], filing_dates: list[str]
) -> bytes:
    """Build a minimal submissions API payload with parallel arrays."""
    return json.dumps(
        {
            "name": name,
            "filings": {
                "recent": {
                    "form": forms,
                    "reportDate": report_dates,
                    "filingDate": filing_dates,
                }
            },
        }
    ).encode("utf-8")


def _fake_cache(root: Path) -> FakeCache:
    """Serve a lookup file whose one normalised name maps to two CIKs."""
    lookup = (
        f"BAUPOST GROUP LLC/MA:{_FILER_CIK}:\n"
        f"BAUPOST GROUP LLC /ADV:{_NONFILER_CIK}:\n"
    ).encode("latin-1")
    bodies = {
        _LOOKUP_URL: lookup,
        _SUBMISSIONS_URL.format(
            int(_FILER_CIK)
        ): _submissions_body(
            "BAUPOST GROUP LLC/MA", ["13F-HR"], ["2026-03-31"], ["2026-05-15"]
        ),
        _SUBMISSIONS_URL.format(
            int(_NONFILER_CIK)
        ): _submissions_body(
            "BAUPOST GROUP LLC /ADV", ["NT 10-K"], ["2025-12-31"], ["2026-01-02"]
        ),
    }
    return FakeCache(root, bodies)


def test_normalisation_removes_a_parenthetical():
    assert (
        cik.normalise("DME Capital Management LP (Greenlight)")
        == "DME CAPITAL MANAGEMENT LP"
    )


def test_normalisation_removes_a_slash_suffix():
    assert cik.normalise("BAUPOST GROUP LLC/MA") == "BAUPOST GROUP LLC"
    assert cik.normalise("BAUPOST GROUP LLC /ADV") == "BAUPOST GROUP LLC"


def test_normalisation_removes_a_leading_article():
    assert cik.normalise("The Baupost Group LLC") == "BAUPOST GROUP LLC"


def test_normalisation_maps_an_ampersand():
    assert cik.normalise("D. E. Shaw & Co. Inc.") == "D E SHAW AND CO INC"


def test_a_collision_needs_the_13f_test(tmp_path):
    """Two candidates share one normalised name. Only the filer survives."""
    cache = _fake_cache(tmp_path)
    roster = [{"fund_name": "Baupost Group LLC/MA", "cik": _FILER_CIK}]

    records = cik.resolve_filers(roster, cache, SCOPE)

    assert len(records) == 1
    record = records[0]
    assert record.cik == _FILER_CIK
    assert record.cik != _NONFILER_CIK
    assert record.edgar_name == "BAUPOST GROUP LLC/MA"
    assert record.cik_source == "given"
    # Both colliding candidates went through the submissions check.
    assert _SUBMISSIONS_URL.format(int(_NONFILER_CIK)) in cache.requested
    assert _SUBMISSIONS_URL.format(int(_FILER_CIK)) in cache.requested


def test_two_names_that_resolve_to_one_filer_are_rejected(tmp_path):
    roster = [{"fund_name": "Baupost Group LLC/MA"}, {"fund_name": "Baupost Group LLC"}]
    with pytest.raises(ValueError, match="resolve to the same SEC filer"):
        cik.resolve_filers(roster, _fake_cache(tmp_path), SCOPE)


def _cache_with(root: Path, lookup: str, submissions: dict[str, bytes]) -> FakeCache:
    bodies = {_LOOKUP_URL: lookup.encode("latin-1")}
    bodies.update({_SUBMISSIONS_URL.format(int(c)): body for c, body in submissions.items()})
    return FakeCache(root, bodies)


def _files_13f(name: str) -> bytes:
    return _submissions_body(name, ["13F-HR"], ["2026-06-30"], ["2026-08-10"])


def test_an_input_cik_that_names_another_filer_is_corrected_by_name(tmp_path):
    cache = _cache_with(tmp_path, "ALPHA CAPITAL LLC:0000000111:\n",
                        {"0000000111": _files_13f("ALPHA CAPITAL LLC")})
    record = cik.resolve_filers([{"fund_name": "Alpha Capital LLC", "cik": "999"}], cache, SCOPE)[0]
    assert (record.cik, record.cik_source) == ("0000000111", "corrected")


def test_a_name_with_no_input_cik_resolves(tmp_path):
    cache = _cache_with(tmp_path, "ALPHA CAPITAL LLC:0000000111:\n",
                        {"0000000111": _files_13f("ALPHA CAPITAL LLC")})
    for row in ({"fund_name": "Alpha Capital LLC"}, {"fund_name": "Alpha Capital LLC", "cik": ""}):
        record = cik.resolve_filers([row], cache, SCOPE)[0]
        assert (record.cik, record.cik_source) == ("0000000111", "resolved")


def test_the_fuzzy_stage_runs_only_when_no_exact_candidate_files(tmp_path):
    """TUDOR INVESTMENT CORP ET AL is the shape of the real case: the exact
    entity files nothing, and a longer variant of the name does."""
    lookup = "TUDOR INVESTMENT CORP:0000000001:\nTUDOR INVESTMENT CORP ET AL:0000000002:\n"
    cache = _cache_with(tmp_path, lookup, {
        "0000000001": _submissions_body("TUDOR INVESTMENT CORP", ["D"], [""], ["2020-01-01"]),
        "0000000002": _files_13f("TUDOR INVESTMENT CORP ET AL"),
    })
    row = {"fund_name": "Tudor Investment Corp", "cik": "1"}
    record = cik.resolve_filers([row], cache, SCOPE)[0]
    assert record.cik == "0000000002"
    assert record.edgar_name == "TUDOR INVESTMENT CORP ET AL"


def test_no_filing_candidate_raises_and_names_what_it_tried(tmp_path):
    cache = _cache_with(tmp_path, "OMEGA FUND LP:0000000005:\n", {
        "0000000005": _submissions_body("OMEGA FUND LP", ["D"], [""], ["2020-01-01"]),
    })
    with pytest.raises(LookupError, match=r"0000000005\(100\.0\)"):
        cik.resolve_filers([{"fund_name": "Omega Fund LP", "cik": "5"}], cache, SCOPE)


def test_a_late_filing_does_not_confirm_a_candidate(tmp_path):
    """The filing-date cutoff (2026-08-18, inclusive) applies to confirmation too."""
    late = _submissions_body("LATE LLC", ["13F-HR"], ["2026-06-30"], ["2026-08-19"])
    cache = _cache_with(tmp_path, "LATE LLC:0000000007:\n", {"0000000007": late})
    with pytest.raises(LookupError):
        cik.resolve_filers([{"fund_name": "Late LLC", "cik": "7"}], cache, SCOPE)


def test_a_filing_on_an_older_page_confirms_a_candidate(tmp_path):
    """`recent` holds no 13F, and the in-scope one sits on an older submissions page."""
    page_name = "CIK0000000009-submissions-001.json"
    recent = json.loads(_submissions_body("OLDPAGE LLC", ["ADV"], [""], ["2026-07-01"]))
    recent["filings"]["files"] = [
        {"name": page_name, "filingFrom": "2026-01-01", "filingTo": "2026-05-30"},
        {"name": "CIK0000000009-submissions-002.json",
         "filingFrom": "2010-01-01", "filingTo": "2025-12-31"},
    ]
    page = {"accessionNumber": ["0000000009-26-000001"], "form": ["13F-HR"],
            "reportDate": ["2026-03-31"], "filingDate": ["2026-05-12"]}
    cache = _cache_with(tmp_path, "OLDPAGE LLC:0000000009:\n", {"0000000009": b"{}"})
    cache._bodies[_SUBMISSIONS_URL.format(9)] = json.dumps(recent).encode()
    cache._bodies["https://data.sec.gov/submissions/" + page_name] = json.dumps(page).encode()

    record = cik.resolve_filers([{"fund_name": "Oldpage LLC", "cik": "9"}], cache, SCOPE)[0]

    assert record.cik == "0000000009"
    # The page that ends before 2026-03-31 holds no 13F in scope, so it is never read.
    assert not any("submissions-002" in url for url in cache.requested)


def test_the_scope_decides_which_quarter_confirms_a_candidate(tmp_path):
    body = _submissions_body("ALPHA CAPITAL LLC", ["13F-HR"], ["2025-12-31"], ["2026-02-10"])
    cache = _cache_with(tmp_path, "ALPHA CAPITAL LLC:0000000111:\n", {"0000000111": body})
    row = {"fund_name": "Alpha Capital LLC", "cik": "111"}
    with pytest.raises(LookupError, match=r"periods \('2026-03-31', '2026-06-30'\)"):
        cik.resolve_filers([row], cache, SCOPE)
    other = Scope.parse("2025Q4", "2026-02-17")
    assert cik.resolve_filers([row], cache, other)[0].cik == "0000000111"
    with pytest.raises(LookupError, match=r"filed by 2026-02-09"):
        cik.resolve_filers([row], cache, Scope.parse("2025Q4", "2026-02-09"))


def test_filers_csv_sorts_by_integer_cik(tmp_path):
    records = [
        cik.FilerRecord("B", "0000000020", "given", "B"),
        cik.FilerRecord("A", "0000000003", "corrected", "A"),
    ]
    cik.write_filers_csv(records, tmp_path / "filers.csv")
    assert (tmp_path / "filers.csv").read_text(encoding="utf-8").splitlines() == [
        "fund_name,cik,cik_source", "A,3,corrected", "B,20,given",
    ]
