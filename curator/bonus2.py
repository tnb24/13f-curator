"""Bonus 2: check every 2026 Q2 CUSIP against the official 13F list.

Every unmatched value carries a label and the evidence behind it.

The official list is fixed width. It carries no header and no delimiter, and an
issuer name holds spaces, so every field comes from a character slice.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from curator.cache import Cache

_Q2_URL = "https://www.sec.gov/files/investment/13flist2026q2-txt.txt"
# The Q1 file does not follow the Q2 naming pattern. `13flist2026q1-txt.txt`
# returns 404. Verified live.
_Q1_URL = "https://www.sec.gov/files/investment/13flist2026q1.txt"

_QUARTER = "2026Q2"
_MIN_LINE = 10

# Character slices, zero-based and half-open.
_CUSIP = slice(0, 9)
_OPTION = slice(9, 10)
_ISSUER = slice(10, 40)
_CLASS = slice(40, 70)

_ADDED_FLAG = "*A*"

_FIELDS = [
    "accession_number",
    "cik",
    "fund_name",
    "cusip",
    "on_official_list",
    "issuer_from_filing",
    "issuer_from_list",
    "rows",
    "assessment",
]


def normalise(value: str | None) -> str:
    """Uppercase, strip whitespace, and drop any hyphen.

    Rule out a formatting difference before calling anything a mismatch.
    """
    if value is None:
        return ""
    return "".join(value.split()).upper().replace("-", "")


def _character_value(char: str) -> int | None:
    """Map one CUSIP character to its numeric value."""
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    return {"*": 36, "@": 37, "#": 38}.get(char)


def check_digit(cusip: str) -> int | None:
    """Return the modulus 10 double-add-double check digit for the first 8."""
    if len(cusip) != 9:
        return None
    total = 0
    for index, char in enumerate(cusip[:8], start=1):
        value = _character_value(char)
        if value is None:
            return None
        if index % 2 == 0:
            value *= 2
        total += value // 10 + value % 10
    return (10 - (total % 10)) % 10


def check_digit_ok(cusip: str) -> bool | None:
    """True when position 9 agrees with the computed check digit."""
    expected = check_digit(cusip)
    if expected is None or not cusip[8].isdigit():
        return None
    return int(cusip[8]) == expected


def parse_official(body: str) -> dict[str, dict[str, Any]]:
    """Parse the fixed-width official list into a map keyed by CUSIP."""
    entries: dict[str, dict[str, Any]] = {}
    for line in body.splitlines():
        if len(line) < _MIN_LINE or not line.strip():
            continue
        cusip = normalise(line[_CUSIP])
        if not cusip:
            continue
        class_text = line[_CLASS].strip()
        added = class_text.endswith(_ADDED_FLAG)
        if added:
            class_text = class_text[: -len(_ADDED_FLAG)].strip()
        entries.setdefault(
            cusip,
            {
                "cusip": cusip,
                "option_marker": line[_OPTION].strip(),
                "issuer": line[_ISSUER].strip(),
                "class": class_text,
                "added_this_quarter": added,
            },
        )
    return entries


def _assess(cusip: str, q1: dict[str, dict[str, Any]]) -> str:
    """Apply the four assessment rules in order."""
    if check_digit_ok(cusip) is False:
        return "LIKELY_FILER_ERROR"
    if cusip in q1:
        return "TIMING"
    if cusip[:1].isalpha() and check_digit_ok(cusip) is True:
        return "CINS_FOREIGN"
    return "UNRESOLVED"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the report with \\n line endings, through a temporary file."""
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(handle)
    temp = Path(temp_name)
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_bonus2(dataset, cache: Cache, output: Path) -> None:
    """Check every 2026 Q2 CUSIP against the official 13F list."""
    output = Path(output)

    q2 = parse_official(cache.get(_Q2_URL).decode("latin-1"))
    q1 = parse_official(cache.get(_Q1_URL).decode("latin-1"))
    print(
        f"curator.bonus2: official list holds {len(q2)} Q2 CUSIPs "
        f"and {len(q1)} Q1 CUSIPs",
        file=sys.stderr,
    )
    for sample in list(q2.values())[:5]:
        print(f"curator.bonus2:   sample {sample}", file=sys.stderr)

    names = {
        str(f["accession_number"]): str(f["fund_name"]) for f in dataset.filings
    }

    # Rule out our own error before any comparison starts.
    quarter_rows = [h for h in dataset.holdings if str(h["report_quarter"]) == _QUARTER]
    bad_length = sum(1 for h in quarter_rows if len(normalise(h["cusip"])) != 9)
    print(
        f"curator.bonus2: {len(quarter_rows)} holdings in {_QUARTER}; "
        f"{bad_length} fail the 9-character check",
        file=sys.stderr,
    )

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for holding in quarter_rows:
        accession = str(holding["accession_number"])
        stored = str(holding["cusip"])
        key = (accession, stored)
        entry = pairs.get(key)
        if entry is None:
            pairs[key] = {
                "accession_number": accession,
                "cik": str(holding["cik"]),
                "fund_name": names.get(accession, ""),
                "cusip": stored,
                "issuer_from_filing": str(holding["name_of_issuer"]),
                "rows": 1,
            }
        else:
            entry["rows"] += 1

    report: list[dict[str, Any]] = []
    labels: dict[str, int] = {}
    matched = 0
    for entry in pairs.values():
        key = normalise(entry["cusip"])
        official = q2.get(key)
        if official is not None:
            matched += 1
            entry["on_official_list"] = True
            entry["issuer_from_list"] = official["issuer"]
            entry["assessment"] = ""
        else:
            label = _assess(key, q1)
            labels[label] = labels.get(label, 0) + 1
            entry["on_official_list"] = False
            entry["issuer_from_list"] = ""
            entry["assessment"] = label
        report.append(entry)

    report.sort(key=lambda row: (row["accession_number"], row["cusip"]))
    _write_csv(report, output / "bonus_cusip_validation.csv")

    print(
        f"curator.bonus2: {len(report)} pairs; {matched} matched, "
        f"{len(report) - matched} unmatched",
        file=sys.stderr,
    )
    for label in sorted(labels):
        print(f"curator.bonus2:   {label}: {labels[label]}", file=sys.stderr)
