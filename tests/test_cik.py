"""Tests for curator.cik: name normalisation and the 13F confirmation gate.

A fake cache serves every body from a dictionary. No test touches the
network, and the reduced lookup file lands in ``tmp_path`` through the
cache's ``root`` property, exactly as the real cache behaves.
"""

from __future__ import annotations

import json
from pathlib import Path

from curator import cik
from curator.http import FetchError

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

    records = cik.verify(roster, cache)

    assert len(records) == 1
    record = records[0]
    assert record.cik == _FILER_CIK
    assert record.cik != _NONFILER_CIK
    assert record.edgar_name == "BAUPOST GROUP LLC/MA"
    assert record.cik_source == "given"
    # Both colliding candidates went through the submissions check.
    assert _SUBMISSIONS_URL.format(int(_NONFILER_CIK)) in cache.requested
    assert _SUBMISSIONS_URL.format(int(_FILER_CIK)) in cache.requested
