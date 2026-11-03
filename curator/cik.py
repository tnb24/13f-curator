"""Filer resolution: map every manager name in the input to its SEC filer.

A hand-typed CIK can be stale, or it can belong to another filer. So this
module finds each filer by name only. It reads EDGAR's name-to-CIK lookup file
(see `curator/lookup.py`), narrows each name to candidate entities, and
confirms every candidate against the submissions API. A candidate passes only
when it filed a 13F in scope.

A CIK in the input is never used to find the filer. `cik_source` records how
it compares with the result: "given" when it agrees, "corrected" when it names
another filer, and "resolved" when the input has no CIK.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from curator.cache import Cache, FetchError
from curator.lookup import LookupIndex, normalise
from curator.records import FilerRecord
from curator.scope import (
    FILING_DATE_CUTOFF,
    IN_SCOPE_FORMS,
    REPORT_PERIODS,
    SUBMISSIONS_URL,
)

__all__ = ["FilerRecord", "load_lookup", "normalise", "resolve_filers", "write_filers_csv"]

_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_EXACT_SCORE = 100.0


def load_lookup(cache: Cache) -> LookupIndex:
    """Return SEC's name-to-CIK lookup, normalised, from the raw file."""
    return LookupIndex.from_body(cache.get(_LOOKUP_URL))


def resolve_filers(roster: list[dict[str, str]], cache: Cache) -> list[FilerRecord]:
    """Return one FilerRecord per input row, in input order."""
    lookup = load_lookup(cache)
    records: list[FilerRecord] = []
    for row in roster:
        record = _resolve_row(row, lookup, cache)
        records.append(record)
        print(
            f"curator.cik: {record.fund_name} -> {record.cik} "
            f"({record.cik_source}) as {record.edgar_name}",
            file=sys.stderr,
        )
    return records


def _resolve_row(row: dict[str, str], lookup: LookupIndex, cache: Cache) -> FilerRecord:
    """Resolve one input row through the exact, confirm, and fuzzy stages."""
    fund_name = row["fund_name"]
    given_cik = str(row.get("cik") or "").strip()
    target = normalise(fund_name)

    tried: dict[str, float] = dict.fromkeys(lookup.exact(target), _EXACT_SCORE)
    passing = _confirm_all(tried, cache)

    if not passing:
        fuzzy = lookup.fuzzy(target)
        passing = _confirm_all(fuzzy, cache)
        for cik, score in fuzzy.items():
            tried[cik] = max(tried.get(cik, 0.0), score)

    if not passing:
        detail = ", ".join(f"{cik}({score:.1f})" for cik, score in sorted(tried.items()))
        raise LookupError(
            f"{fund_name}: no candidate filed a 13F in scope "
            f"(periods {REPORT_PERIODS}, filed by {FILING_DATE_CUTOFF}); "
            f"candidates tried: {detail or 'none'}"
        )

    selected_cik, edgar_name = _select(passing)
    if not given_cik:
        source = "resolved"
    elif given_cik.isdigit() and int(given_cik) == int(selected_cik):
        source = "given"
    else:
        source = "corrected"
    return FilerRecord(
        fund_name=fund_name,
        cik=selected_cik.zfill(10),
        cik_source=source,
        edgar_name=edgar_name,
    )


def _confirm_all(
    candidates: dict[str, float],
    cache: Cache,
) -> dict[str, tuple[float, str]]:
    """Confirm every candidate. Return the passers with their scores."""
    passing: dict[str, tuple[float, str]] = {}
    for cik, score in candidates.items():
        in_scope, edgar_name = _submissions(cik, cache)
        if in_scope:
            passing[cik] = (score, edgar_name)
    return passing


def _submissions(cik: str, cache: Cache) -> tuple[bool, str]:
    """Return (files a 13F in scope, entity name) for one zero-padded CIK.

    Every response goes through the cache, so discovery reads the same bodies.
    """
    try:
        payload = json.loads(cache.get(SUBMISSIONS_URL.format(int(cik))))
    except FetchError:
        return False, ""
    edgar_name = str(payload.get("name") or "")
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    report_dates = recent.get("reportDate") or []
    filing_dates = recent.get("filingDate") or []
    for form, report_date, filing_date in zip(forms, report_dates, filing_dates, strict=False):
        if form not in IN_SCOPE_FORMS:
            continue
        if report_date not in REPORT_PERIODS:
            continue
        if not filing_date or str(filing_date) > FILING_DATE_CUTOFF:
            continue
        return True, edgar_name
    return False, edgar_name


def _select(passing: dict[str, tuple[float, str]]) -> tuple[str, str]:
    """Pick one winner: highest score, then lowest integer CIK."""
    ranked = min(passing.items(), key=lambda item: (-item[1][0], int(item[0])))
    return ranked[0], ranked[1][1]


def write_filers_csv(records: list[FilerRecord], path: Path) -> None:
    """Write output/filers.csv: fund_name, cik, cik_source, sorted by int CIK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["fund_name", "cik", "cik_source"])
        for record in sorted(records, key=lambda item: int(item.cik)):
            writer.writerow([record.fund_name, str(int(record.cik)), record.cik_source])
