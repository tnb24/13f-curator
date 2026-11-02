"""CIK verification: resolve every roster fund name to a 13F-filing entity.

The roster names are right and the roster CIKs are not. This module reads
EDGAR's name-to-CIK lookup file through the shared cache, narrows each roster
row to candidate entities, and confirms every candidate against the
submissions API. A candidate passes only when it filed a 13F in scope.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

from curator.cache import Cache
from curator.http import FetchError
from curator.scope import FILING_DATE_CUTOFF, REPORT_PERIODS, SUBMISSIONS_URL

_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
_FUZZY_THRESHOLD = 85.0  # TUDOR INVESTMENT CORP ET AL scores 87.5
_FUZZY_LIMIT = 10
_EXACT_SCORE = 100.0

_PARENS_RE = re.compile(r"\([^()]*\)")
_SLASH_SUFFIX_RE = re.compile(r"\s*/[A-Z]{1,4}/?\s*$")
_PUNCT_RE = re.compile(r"[.,'\"]")
_WHITESPACE_RE = re.compile(r"\s+")


# Manual exceptions, fund name to CIK. Empty: the automatic match resolves every row.
OVERRIDES: dict[str, str] = {}


@dataclass(frozen=True)
class FilerRecord:
    fund_name: str      # exactly as supplied
    cik: str            # zero-padded to 10
    cik_source: str     # "given" or "corrected"
    edgar_name: str     # entity name from the submissions API


def normalise(name: str) -> str:
    """Apply the eight normalisation steps."""
    text = html.unescape(name)
    text = text.upper()
    text = _PARENS_RE.sub("", text)
    text = _SLASH_SUFFIX_RE.sub("", text)
    text = text.replace("&", " AND ")
    text = _PUNCT_RE.sub("", text)
    if text.startswith("THE "):
        text = text[len("THE "):]
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_lookup(cache: Cache) -> dict[str, list[str]]:
    """Return normalised name to list of zero-padded CIKs."""
    reduced_path = _reduced_path(cache)
    if reduced_path.is_file():
        return _read_reduced(reduced_path)
    table = _parse_lookup(cache.get(_LOOKUP_URL))
    _write_reduced(reduced_path, table)
    return table


def verify(roster: list[dict[str, str]], cache: Cache) -> list[FilerRecord]:
    """Return one FilerRecord per roster row, in roster order."""
    lookup = load_lookup(cache)
    names = list(lookup)
    records: list[FilerRecord] = []
    for row in roster:
        record = _verify_row(row, lookup, names, cache)
        records.append(record)
        print(
            f"curator.cik: {record.fund_name} -> {record.cik} "
            f"({record.cik_source}) as {record.edgar_name}",
            file=sys.stderr,
        )
    return records


def _verify_row(
    row: dict[str, str],
    lookup: dict[str, list[str]],
    names: list[str],
    cache: Cache,
) -> FilerRecord:
    """Resolve one roster row through the exact, confirm, and fuzzy stages."""
    fund_name = row["fund_name"]
    given_cik = str(row.get("cik") or "").strip()
    target = normalise(fund_name)

    tried: dict[str, float] = {}
    for cik in lookup.get(target, []):
        tried[cik] = _EXACT_SCORE
    passing = _confirm_all(tried, cache)

    if not passing:
        fuzzy = _fuzzy_candidates(target, names, lookup)
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

    override_cik = OVERRIDES.get(fund_name)
    if override_cik is not None:
        selected_cik = str(override_cik).zfill(10)
        _, edgar_name = _submissions(selected_cik, cache)
        print(
            f"curator.cik: override applied for {fund_name}: -> {selected_cik}",
            file=sys.stderr,
        )

    source = (
        "given"
        if given_cik.isdigit() and int(given_cik) == int(selected_cik)
        else "corrected"
    )
    return FilerRecord(
        fund_name=fund_name,
        cik=selected_cik.zfill(10),
        cik_source=source,
        edgar_name=edgar_name,
    )


def _fuzzy_candidates(
    target: str,
    names: list[str],
    lookup: dict[str, list[str]],
) -> dict[str, float]:
    """Return up to ten fuzzy name matches at or above the threshold."""
    hits = process.extract(
        target,
        names,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=_FUZZY_THRESHOLD,
        limit=_FUZZY_LIMIT,
    )
    candidates: dict[str, float] = {}
    for _name, score, index in hits:
        for cik in lookup[names[index]]:
            candidates[cik] = max(candidates.get(cik, 0.0), float(score))
    return candidates


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
    for form, report_date, filing_date in zip(forms, report_dates, filing_dates):
        if not isinstance(form, str) or not form.startswith("13F"):
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


def _reduced_path(cache: Cache) -> Path:
    return Path(cache.root) / "cik_lookup.csv"


def _parse_lookup(body: bytes) -> dict[str, list[str]]:
    """Parse the NAME:CIK: lookup file into a normalised index."""
    table: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for line in body.decode("latin-1").splitlines():
        line = line.strip()
        if not line:
            continue
        last = line.rfind(":")
        if last == -1:
            continue
        previous = line.rfind(":", 0, last)
        if previous == -1:
            continue
        raw_name = line[:previous]
        raw_cik = line[previous + 1:last]
        if not raw_cik.isdigit() or not raw_name:
            continue
        pair = (normalise(raw_name), raw_cik.zfill(10))
        if pair in seen or not pair[0]:
            continue
        seen.add(pair)
        table.setdefault(pair[0], []).append(pair[1])
    return table


def _write_reduced(path: Path, table: dict[str, list[str]]) -> None:
    """Store the two-column reduced copy with normalised names.

    The write is atomic, so a later run never reads a half-written copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["name", "cik"])
            for name in sorted(table):
                for cik in sorted(table[name]):
                    writer.writerow([name, cik])
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_reduced(path: Path) -> dict[str, list[str]]:
    """Read back the reduced copy written by _write_reduced."""
    table: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table.setdefault(row["name"], []).append(row["cik"])
    return table


def write_filers_csv(records: list[FilerRecord], path: Path) -> None:
    """Write output/filers.csv: fund_name, cik, cik_source, sorted by int CIK."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["fund_name", "cik", "cik_source"])
        for record in sorted(records, key=lambda item: int(item.cik)):
            writer.writerow([record.fund_name, str(int(record.cik)), record.cik_source])
