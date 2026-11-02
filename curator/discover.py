"""Filing discovery: select the 13F filings in scope for each verified filer.

The submissions API lists every filing an entity ever sent. This module keeps
only forms 13F-HR, 13F-HR/A, 13F-NT, and 13F-NT/A whose report date is
2026-03-31 or 2026-06-30 and whose filing date is on or before the cutoff.
The period filter reads ``reportDate``, never ``filingDate``.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from curator.cache import Cache
from curator.cik import FilerRecord
from curator.scope import (
    FILING_DATE_CUTOFF,
    IN_SCOPE_FORMS,
    REPORT_PERIODS,
    SUBMISSIONS_URL,
)

_PAGES_URL = "https://data.sec.gov/submissions/{name}"
_RECENT_MAX = 1000
_EXPECTED_PER_FILER = 2


@dataclass(frozen=True)
class FilingRef:
    """One in-scope filing, exactly as the submissions API describes it."""

    cik: str                # zero-padded to 10
    fund_name: str
    accession: str          # dashed form, from the API
    form: str
    report_date: str        # YYYY-MM-DD
    filing_date: str        # YYYY-MM-DD
    primary_document: str


def discover(filers: list[FilerRecord], cache: Cache) -> list[FilingRef]:
    """Return every in-scope filing, sorted by cik, report_date, accession."""
    refs: list[FilingRef] = []
    for filer in filers:
        found: list[FilingRef] = []
        for frame in _frames(cache, filer.cik):
            found.extend(_in_scope(frame, filer))
        if len(found) != _EXPECTED_PER_FILER:
            print(
                f"curator.discover: warning: {filer.fund_name} ({filer.cik}) "
                f"yielded {len(found)} filings in scope, "
                f"expected {_EXPECTED_PER_FILER}",
                file=sys.stderr,
            )
        refs.extend(found)
    refs.sort(key=lambda ref: (ref.cik, ref.report_date, ref.accession))
    return refs


def _frames(cache: Cache, cik: str) -> list[dict]:
    """Return the recent frame plus every extra page frame for one CIK.

    ``filings.recent`` holds at most 1000 entries. A full array means the
    API moved older filings into the files named in ``filings.files``.
    """
    payload = json.loads(cache.get(SUBMISSIONS_URL.format(cik)))
    filings = payload.get("filings") or {}
    recent = filings.get("recent") or {}
    frames = [recent]
    count = len(recent.get("accessionNumber") or [])
    if count < _RECENT_MAX:
        return frames
    for entry in filings.get("files") or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not name:
            continue
        frames.append(_page_frame(json.loads(cache.get(_PAGES_URL.format(name=name)))))
    return frames


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


def _in_scope(frame: dict, filer: FilerRecord) -> list[FilingRef]:
    """Apply the three scope filters to one parallel-array frame."""
    columns = zip(
        frame.get("accessionNumber") or [],
        frame.get("form") or [],
        frame.get("reportDate") or [],
        frame.get("filingDate") or [],
        frame.get("primaryDocument") or [],
    )
    refs: list[FilingRef] = []
    for accession, form, report_date, filing_date, primary_document in columns:
        if form not in IN_SCOPE_FORMS:
            continue
        if report_date not in REPORT_PERIODS:
            continue
        filed = str(filing_date or "")
        if not filed or filed > FILING_DATE_CUTOFF:
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
