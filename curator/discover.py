"""Filing discovery: select the 13F filings in scope for each verified filer.

The submissions API lists every filing an entity ever sent. This module keeps
only forms 13F-HR, 13F-HR/A, 13F-NT, and 13F-NT/A whose report date is the
last day of a quarter in the scope and whose filing date is on or before the
cutoff. The period filter reads ``reportDate``, never ``filingDate``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator

from curator.cache import Cache
from curator.records import FilerRecord, FilingRef
from curator.scope import IN_SCOPE_FORMS, SUBMISSIONS_URL, Scope

_PAGES_URL = "https://data.sec.gov/submissions/{name}"


def discover(filers: list[FilerRecord], cache: Cache, scope: Scope) -> list[FilingRef]:
    """Return every in-scope filing, sorted by cik, report_date, accession."""
    refs: list[FilingRef] = []
    expected = scope.expected_per_filer
    for filer in filers:
        found: list[FilingRef] = []
        frames, _ = filing_frames(cache, filer.cik, scope)
        for frame in frames:
            found.extend(_in_scope(frame, filer, scope))
        if len(found) != expected:
            print(
                f"curator.discover: warning: {filer.fund_name} ({filer.cik}) "
                f"yielded {len(found)} filings in scope, "
                f"expected {expected}",
                file=sys.stderr,
            )
        refs.extend(found)
    refs.sort(key=lambda ref: (ref.cik, ref.report_date, ref.accession))
    return refs


def filing_frames(cache: Cache, cik: str, scope: Scope) -> tuple[list[dict], str]:
    """Return every filing frame for one CIK that can hold a filing in scope.

    Also return the entity name. See `payload_frames` for which pages load.
    """
    payload = json.loads(cache.get(SUBMISSIONS_URL.format(int(cik))))
    return list(payload_frames(payload, cache, scope)), str(payload.get("name") or "")


def payload_frames(payload: dict, cache: Cache, scope: Scope) -> Iterator[dict]:
    """Yield the recent frame, then each older page that can hold a filing in scope.

    ``filings.recent`` holds the newest filings only. The API moves older
    filings into the files that ``filings.files`` names. Each entry states the
    first and the last filing date of its page. The function loads a page when
    that date range meets the scope, and when the range is missing. So an old
    quarter is found whatever the size of ``recent``. A page loads on demand:
    a caller that stops early fetches no more pages.
    """
    filings = payload.get("filings") or {}
    yield filings.get("recent") or {}
    for entry in filings.get("files") or []:
        page = entry.get("name") if isinstance(entry, dict) else None
        if not page or not _page_in_range(entry, scope):
            continue
        yield _page_frame(json.loads(cache.get(_PAGES_URL.format(name=page))))


def _page_in_range(entry: dict, scope: Scope) -> bool:
    """False only when the page's filing dates all fall outside the scope.

    A report for a quarter is filed after the quarter ends. So a page whose last
    filing is older than the first report date holds nothing in scope. A page
    whose first filing is newer than the cutoff holds nothing in scope either.
    """
    first, last = entry.get("filingFrom"), entry.get("filingTo")
    if isinstance(last, str) and last and last < scope.report_periods[0]:
        return False
    return not (isinstance(first, str) and first and first > scope.cutoff)


def _page_frame(page: dict) -> dict:
    """Return the recent-style frame inside one extra submissions page."""
    inner = page.get("filings")
    source = inner if isinstance(inner, dict) else page
    frame = source.get("recent")
    if isinstance(frame, dict):
        return frame
    if isinstance(source.get("accessionNumber"), list):
        return source
    return {}


def _in_scope(frame: dict, filer: FilerRecord, scope: Scope) -> list[FilingRef]:
    """Apply the three scope filters to one parallel-array frame."""
    periods = scope.report_periods
    columns = zip(
        frame.get("accessionNumber") or [],
        frame.get("form") or [],
        frame.get("reportDate") or [],
        frame.get("filingDate") or [],
        frame.get("primaryDocument") or [], strict=False,
    )
    refs: list[FilingRef] = []
    for accession, form, report_date, filing_date, primary_document in columns:
        if form not in IN_SCOPE_FORMS:
            continue
        if report_date not in periods:
            continue
        filed = str(filing_date or "")
        if not filed or filed > scope.cutoff:
            continue
        refs.append(
            FilingRef(
                cik=filer.cik,
                fund_name=filer.fund_name,
                accession=str(accession),
                form=str(form),
                report_date=str(report_date),
                filing_date=filed,
                primary_document=str(primary_document or ""),
            )
        )
    return refs
