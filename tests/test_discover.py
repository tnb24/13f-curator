"""Discovery over the submissions API: older pages, and the filings expected per filer.

A fake cache serves canned bodies and records each request. No test touches the network.
"""

from __future__ import annotations

import json

import pytest

from curator.cache import FetchError
from curator.discover import discover, filing_frames, payload_frames
from curator.records import FilerRecord
from curator.scope import Scope

_CIK = "0001234567"
_URL = f"https://data.sec.gov/submissions/CIK{_CIK}.json"
_PAGE_URL = "https://data.sec.gov/submissions/{name}"
_FILER = FilerRecord("TEST FUND LP", _CIK, "given", "TEST FUND LP")


class FakeCache:
    """Serve canned bodies by URL, and record every request."""

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self._bodies = bodies
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        if url not in self._bodies:
            raise FetchError(f"GET {url} failed (no canned body)", status=404)
        return self._bodies[url]


def _frame(*filings: tuple[str, str, str, str]) -> dict[str, list[str]]:
    """A recent-style frame of (accession, form, report date, filing date) rows."""
    accession, form, report, filed = (list(column) for column in zip(*filings, strict=True))
    return {"accessionNumber": accession, "form": form, "reportDate": report,
            "filingDate": filed, "primaryDocument": ["doc.xml"] * len(accession)}


def _payload(recent: dict, files: list[dict] | None = None) -> bytes:
    filings: dict = {"recent": recent}
    if files is not None:
        filings["files"] = files
    return json.dumps({"name": "TEST FUND LP", "filings": filings}).encode()


def _page(name: str, first: str | None, last: str | None) -> dict:
    entry: dict = {"name": name, "filingCount": 3}
    if first is not None:
        entry["filingFrom"] = first
    if last is not None:
        entry["filingTo"] = last
    return entry


Q1 = ("0001234567-26-000001", "13F-HR", "2026-03-31", "2026-05-12")
Q2 = ("0001234567-26-000002", "13F-HR", "2026-06-30", "2026-08-11")


def _cache_with_page(first: str | None, last: str | None) -> FakeCache:
    """The Q2 filing is in `recent`. The Q1 filing sits on one older page."""
    name = "CIK0001234567-submissions-001.json"
    return FakeCache({
        _URL: _payload(_frame(Q2), [_page(name, first, last)]),
        _PAGE_URL.format(name=name): json.dumps(_frame(Q1)).encode(),
    })


def test_an_older_page_is_read_though_recent_is_not_full() -> None:
    cache = _cache_with_page("2026-01-01", "2026-05-12")
    frames, name = filing_frames(cache, _CIK, Scope())
    assert name == "TEST FUND LP"
    assert len(frames) == 2
    assert _PAGE_URL.format(name="CIK0001234567-submissions-001.json") in cache.requested

    refs = discover([_FILER], cache, Scope())
    assert [ref.accession for ref in refs] == [Q1[0], Q2[0]]


@pytest.mark.parametrize("first, last, read", [
    ("2026-01-01", "2026-05-12", True),   # the range meets the scope
    ("2026-03-31", "2026-03-31", True),   # a filing on the first report date
    ("2025-01-01", "2026-03-30", False),  # the last filing is before the first quarter ends
    ("2026-08-19", "2026-09-30", False),  # the first filing is after the cutoff
    ("2026-08-18", "2026-09-30", True),   # the cutoff day itself is in
    (None, None, True),                   # no range: read it, to be safe
    ("2026-01-01", None, True),
    (None, "2025-12-31", False),
])
def test_a_page_is_read_when_its_dates_meet_the_scope(first, last, read) -> None:
    cache = _cache_with_page(first, last)
    list(payload_frames(json.loads(cache.get(_URL)), cache, Scope()))
    page = _PAGE_URL.format(name="CIK0001234567-submissions-001.json")
    assert (page in cache.requested) is read


def test_an_older_quarter_scope_reads_the_page_that_holds_that_quarter() -> None:
    """2019Q4 filings sit on a page from the 2010s. The default quarters skip that page."""
    old = ("0001234567-20-000001", "13F-HR", "2019-12-31", "2020-02-12")
    new = ("0001234567-26-000002", "13F-HR", "2026-06-30", "2026-08-11")
    scope = Scope.parse("2019Q4", "2020-02-14")
    first, second = "CIK0001234567-submissions-001.json", "CIK0001234567-submissions-002.json"
    cache = FakeCache({
        _URL: _payload(_frame(new), [
            _page(first, "2020-01-02", "2021-06-30"),
            _page(second, "2013-01-01", "2019-06-30"),
        ]),
        _PAGE_URL.format(name=first): json.dumps(_frame(old)).encode(),
    })
    refs = discover([_FILER], cache, scope)
    assert [ref.accession for ref in refs] == [old[0]]
    assert _PAGE_URL.format(name=second) not in cache.requested  # ends before the quarter does

    other = _cache_with_page("2013-01-01", "2019-06-30")
    discover([_FILER], other, Scope())
    assert _PAGE_URL.format(name="CIK0001234567-submissions-001.json") not in other.requested


def test_a_page_loads_only_when_the_caller_asks_for_it() -> None:
    cache = _cache_with_page("2026-01-01", "2026-05-12")
    frames = payload_frames(json.loads(cache.get(_URL)), cache, Scope())
    next(frames)
    assert cache.requested == [_URL]


def test_a_page_wrapped_in_filings_recent_still_reads() -> None:
    """An extra page holds the columns at the top, as above, or inside filings.recent."""
    name = "CIK0001234567-submissions-001.json"
    wrapped = {"filings": {"recent": _frame(Q1)}}
    cache = FakeCache({
        _URL: _payload(_frame(Q2), [_page(name, "2026-01-01", "2026-05-12")]),
        _PAGE_URL.format(name=name): json.dumps(wrapped).encode(),
    })
    refs = discover([_FILER], cache, Scope())
    assert [ref.accession for ref in refs] == [Q1[0], Q2[0]]


def test_the_filters_follow_the_quarters_and_the_cutoff() -> None:
    from curator.discover import _in_scope

    q3 = ("0001234567-26-000003", "13F-HR", "2026-09-30", "2026-11-10")
    late = ("0001234567-26-000004", "13F-HR", "2026-09-30", "2026-11-17")
    frame = _frame(Q1, Q2, q3, late)

    def accessions(scope: Scope) -> list[str]:
        return [ref.accession for ref in _in_scope(frame, _FILER, scope)]

    assert accessions(Scope()) == [Q1[0], Q2[0]]
    assert accessions(Scope.parse("2026Q3", "2026-11-16")) == [q3[0]]
    assert accessions(Scope.parse("2026Q1,2026Q2,2026Q3", "2026-11-17")) == [
        Q1[0], Q2[0], q3[0], late[0],
    ]
    assert accessions(Scope.parse("2026Q1,2026Q2,2026Q3", "2026-11-10")) == [
        Q1[0], Q2[0], q3[0],
    ]  # the cutoff day itself stays in


def _warnings(capsys) -> str:
    return capsys.readouterr().err


def test_discovery_expects_one_filing_for_each_quarter(capsys) -> None:
    cache = FakeCache({_URL: _payload(_frame(Q1, Q2))})
    discover([_FILER], cache, Scope())
    assert "warning" not in _warnings(capsys)

    discover([_FILER], cache, Scope.parse("2026Q1,2026Q2,2026Q3", "2026-11-16"))
    assert "yielded 2 filings in scope, expected 3" in _warnings(capsys)

    discover([_FILER], cache, Scope.parse("2026Q2", "2026-08-18"))  # Q1 is out of scope
    assert "warning" not in _warnings(capsys)


def test_a_one_quarter_scope_warns_when_a_filer_has_two(capsys) -> None:
    cache = FakeCache({_URL: _payload(_frame(Q2, ("0001234567-26-000003", "13F-HR/A",
                                                  "2026-06-30", "2026-08-15")))})
    discover([_FILER], cache, Scope.parse("2026Q2", "2026-08-18"))
    assert "yielded 2 filings in scope, expected 1" in _warnings(capsys)
