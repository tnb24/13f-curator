"""CUSIP check: match every CUSIP of the last quarter in scope against the official 13F list.

Every unmatched value carries a label and the evidence behind it. The list for
the quarter before the last one gives the TIMING label.

The official list is fixed width. It carries no header and no delimiter, and an
issuer name holds spaces, so every field comes from a character slice.

The stage works on columns. Arrow kernels slice the official list, match each
(filing, CUSIP) pair against it, and render the report as CSV. Python code runs
once per unmatched pair only. `parse_official` and `normalise` are the reference
rules: a body or a column outside printable ASCII goes through them, and the
tests compare both paths.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Container
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from curator.cache import Cache, FetchError
from curator.parse import ParsedDataset
from curator.scope import Scope, parse_quarter, previous_quarter

_LIST_BASE = "https://www.sec.gov/files/investment/"
# Two quarters have a verified URL. The Q1 file does not follow the Q2 naming
# pattern: `13flist2026q1-txt.txt` returns 404. Verified live.
_KNOWN_LISTS = {
    "2026Q2": _LIST_BASE + "13flist2026q2-txt.txt",
    "2026Q1": _LIST_BASE + "13flist2026q1.txt",
}

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

OFFICIAL_SCHEMA = pa.schema([
    ("cusip", pa.string()),
    ("option_marker", pa.string()),
    ("issuer", pa.string()),
    ("class", pa.string()),
    ("added_this_quarter", pa.bool_()),
])

# On printable ASCII text, the only whitespace is the space, Python's `upper`
# changes a-z only, and one byte is one character. Each Arrow kernel below then
# gives exactly the result of the Python rule that it replaces.
_PRINTABLE = r"^[\x20-\x7e]*$"
# Python's csv module quotes a field that holds one of these characters.
_CSV_SPECIAL = '[",\r\n]'


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
    """Parse the fixed-width official list into a map keyed by CUSIP.

    This is the reference rule. `official_table` gives the same rows.
    """
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


def _plain_lines(body: bytes) -> pa.StringArray | None:
    """Split a body of printable ASCII lines into an Arrow array, or return None.

    On such a body, a split on newline bytes gives what `str.splitlines` gives.
    """
    raw = np.frombuffer(body, np.uint8)
    newline = raw == 0x0A
    if not np.all(newline | ((raw >= 0x20) & (raw <= 0x7E))):
        return None
    ends = np.flatnonzero(newline)
    starts = np.concatenate(([0], ends + 1))
    if starts[-1] < len(raw):
        ends = np.append(ends, len(raw))  # the last line has no newline
    else:
        starts = starts[:-1]  # splitlines gives no empty line after the last newline
    offsets = np.zeros(len(starts) + 1, np.int32)
    np.cumsum(ends - starts, out=offsets[1:])
    data = raw[~newline]
    return pa.StringArray.from_buffers(len(starts), pa.py_buffer(offsets), pa.py_buffer(data))


def _slice(lines: pa.Array, field: slice) -> pa.Array:
    return pc.utf8_slice_codeunits(lines, field.start, field.stop)


def _strip(values: pa.Array) -> pa.Array:
    return pc.utf8_trim(values, " ")


def _ascii_normalise(values: pa.Array) -> pa.Array:
    no_space = pc.replace_substring(values, " ", "")
    return pc.ascii_upper(pc.replace_substring(no_space, "-", ""))


def official_table(body: bytes) -> pa.Table:
    """Parse the fixed-width official list into a table, one row per CUSIP.

    The rows are the values of `parse_official`, in the same order: file order,
    and the first line for a CUSIP wins.
    """
    lines = _plain_lines(body)
    if lines is None:
        entries = parse_official(body.decode("latin-1"))
        return pa.Table.from_pylist(list(entries.values()), schema=OFFICIAL_SCHEMA)

    lines = lines.filter(pc.greater_equal(pc.utf8_length(lines), _MIN_LINE))
    # A blank line has an empty CUSIP, so this filter also drops blank lines.
    cusip = _ascii_normalise(_slice(lines, _CUSIP))
    keep = pc.not_equal(cusip, "")
    lines, cusip = lines.filter(keep), cusip.filter(keep)

    codes = pc.dictionary_encode(cusip).indices.to_numpy()
    _, first = np.unique(codes, return_index=True)
    first = pa.array(np.sort(first))
    lines, cusip = lines.take(first), cusip.take(first)

    class_text = _strip(_slice(lines, _CLASS))
    added = pc.ends_with(class_text, _ADDED_FLAG)
    flag_removed = _strip(pc.utf8_slice_codeunits(class_text, 0, -len(_ADDED_FLAG)))
    return pa.table(
        [
            cusip,
            _strip(_slice(lines, _OPTION)),
            _strip(_slice(lines, _ISSUER)),
            pc.if_else(added, flag_removed, class_text),
            added,
        ],
        schema=OFFICIAL_SCHEMA,
    )


def normalise_column(values: pa.Array) -> pa.Array:
    """Apply `normalise` to each value of a string column. A null becomes ""."""
    values = pc.fill_null(values, "")
    if pc.all(pc.match_substring_regex(values, _PRINTABLE)).as_py():
        return _ascii_normalise(values)
    return pa.array([normalise(value) for value in values.to_pylist()], pa.string())


def official_list_urls(quarter: str) -> tuple[str, ...]:
    """Return the URLs that can hold the official list for one quarter, in the order to try.

    The two verified quarters have one exact URL each. The SEC names any other
    file in one of two ways, so the function returns both names.
    """
    known = _KNOWN_LISTS.get(quarter)
    if known is not None:
        return (known,)
    year, number = parse_quarter(quarter)
    stem = f"{_LIST_BASE}13flist{year:04d}q{number}"
    return (f"{stem}.txt", f"{stem}-txt.txt")


def fetch_official(cache: Cache, quarter: str) -> bytes:
    """Return the official list for one quarter. Try each URL until one answers.

    A 404 moves on to the next URL. Every other failure stops at once.
    When no URL answers, the error names the quarter and every URL tried.
    """
    missing: list[str] = []
    for url in official_list_urls(quarter):
        try:
            return cache.get(url)
        except FetchError as exc:
            if exc.status != 404:
                raise
            missing.append(url)
    raise FetchError(
        f"no official 13F list for {quarter}: the SEC returned 404 for {', '.join(missing)}. "
        "The SEC has not published that list, or it names the file in a new way. "
        "Use an earlier last quarter in --quarters, or add the file name to "
        "curator/cusip_check.py.",
        status=404,
    )


def _assess(cusip: str, prior: Container[str]) -> str:
    """Apply the four assessment rules in order."""
    if check_digit_ok(cusip) is False:
        return "LIKELY_FILER_ERROR"
    if cusip in prior:
        return "TIMING"
    if cusip[:1].isalpha() and check_digit_ok(cusip) is True:
        return "CINS_FOREIGN"
    return "UNRESOLVED"


def _csv_field(values: pa.Array) -> pa.Array:
    """Render one column as `csv.writer` renders each value of it."""
    if pa.types.is_boolean(values.type):
        return pc.fill_null(pc.if_else(values, "True", "False"), "")
    if not pa.types.is_string(values.type):
        return pc.fill_null(pc.cast(values, pa.string()), "")
    values = pc.fill_null(values, "")
    special = pc.match_substring_regex(values, _CSV_SPECIAL)
    if not pc.any(special).as_py():
        return values
    escaped = pc.replace_substring(values, '"', '""')
    return pc.if_else(special, pc.binary_join_element_wise('"', escaped, '"', ""), values)


def csv_bytes(table: pa.Table) -> bytes:
    """Render a table as `csv.writer(lineterminator="\\n")` renders its rows.

    Each line is a join of the rendered fields. One string array then holds
    every line, so its data buffer is the CSV body.
    """
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(table.column_names)
    # Arrow and Python differ when formatting floats, timestamps and nested
    # values. The report uses only strings, integers and booleans; preserve
    # csv.writer semantics for other callers without weakening that hot path.
    if any(not (pa.types.is_string(c.type) or pa.types.is_integer(c.type)
                or pa.types.is_boolean(c.type) or pa.types.is_null(c.type))
           for c in table.columns):
        writer.writerows(zip(*(c.to_pylist() for c in table.columns), strict=True))
        return stream.getvalue().encode("utf-8")
    header = stream.getvalue()
    if table.num_rows == 0:
        return header.encode("utf-8")
    fields = [_csv_field(column.combine_chunks()) for column in table.columns]
    if len(fields) == 1:
        # csv.writer distinguishes a single empty field from an empty record.
        fields[0] = pc.if_else(pc.equal(fields[0], ""), '""', fields[0])
    lines = pc.binary_join_element_wise(*fields, ",")
    lines = pc.binary_join_element_wise(lines, "", "\n")
    offsets = np.frombuffer(lines.buffers()[1], np.int32, len(lines) + 1, lines.offset * 4)
    body = memoryview(lines.buffers()[2])[offsets[0]:offsets[-1]]
    return header.encode("utf-8") + bytes(body)


def _write_bytes(data: bytes, path: Path) -> None:
    """Write through a temporary file beside the target, then rename into place."""
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(handle)
    temp = Path(temp_name)
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_cusip_check(dataset: ParsedDataset, cache: Cache, output: Path, scope: Scope) -> None:
    """Check every CUSIP of the last quarter in scope against the official 13F list.

    The comparison list is the one for the quarter before the last quarter. That
    quarter need not be in the scope: a one-quarter scope still needs it.
    """
    output = Path(output)

    last = scope.last_quarter
    before = previous_quarter(last)
    current = official_table(fetch_official(cache, last))
    prior = official_table(fetch_official(cache, before))
    print(
        f"curator.cusip_check: official list holds {current.num_rows} CUSIPs for {last} "
        f"and {prior.num_rows} CUSIPs for {before}",
        file=sys.stderr,
    )
    for sample in current.slice(0, 5).to_pylist():
        print(f"curator.cusip_check:   sample {sample}", file=sys.stderr)

    # One row per (filing, stored CUSIP) pair, sorted. use_threads=False keeps
    # the aggregation ordered, so "first" is the first row in source order.
    holdings = dataset.holdings
    in_last = holdings.filter(pc.equal(holdings["report_quarter"], last))
    pairs = (
        in_last.group_by(["accession_number", "cusip"], use_threads=False)
        .aggregate([("cik", "first"), ("name_of_issuer", "first"), ([], "count_all")])
        .sort_by([("accession_number", "ascending"), ("cusip", "ascending")])
    )
    rows = pairs["count_all"].combine_chunks()
    key = normalise_column(pairs["cusip"].combine_chunks())

    # Rule out our own error before any comparison starts.
    bad_length = pc.sum(pc.if_else(pc.not_equal(pc.utf8_length(key), 9), rows, 0)).as_py()
    print(
        f"curator.cusip_check: {in_last.num_rows} holdings in {last}; "
        f"{bad_length or 0} fail the 9-character check",
        file=sys.stderr,
    )

    found = pc.index_in(key, value_set=current["cusip"])
    on_list = pc.is_valid(found)
    unmatched = np.flatnonzero(~on_list.to_numpy(zero_copy_only=False))
    prior_cusips = frozenset(prior["cusip"].to_pylist()) if len(unmatched) else frozenset()
    labels = [_assess(cusip, prior_cusips) for cusip in key.take(unmatched).to_pylist()]
    assessment = pc.replace_with_mask(
        pa.nulls(len(key), pa.string()), pc.invert(on_list), pa.array(labels, pa.string()))

    # Accession numbers are unique in filings: verify.py checks the key.
    filings = dataset.filings
    fund = filings["fund_name"].take(
        pc.index_in(pairs["accession_number"], value_set=filings["accession_number"]))
    report = pa.table(
        [
            pairs["accession_number"],
            pairs["cik_first"],
            fund,
            pairs["cusip"],
            on_list,
            pairs["name_of_issuer_first"],
            current["issuer"].take(found),
            rows,
            pc.fill_null(assessment, ""),
        ],
        names=_FIELDS,
    )
    _write_bytes(csv_bytes(report), output / "cusip_check.csv")

    matched = report.num_rows - len(unmatched)
    print(
        f"curator.cusip_check: {report.num_rows} pairs; {matched} matched, "
        f"{len(unmatched)} unmatched",
        file=sys.stderr,
    )
    for label, count in sorted(Counter(labels).items()):
        print(f"curator.cusip_check:   {label}: {count}", file=sys.stderr)
