#!/usr/bin/env python3
"""Check the Parquet output against the schema in docs/SCHEMA.md.

    python verify.py

Checks structure only: that both files exist, that every required column is present
with the specified type, and that columns marked non-null contain no nulls.

It does not check whether the values are correct.

Exit codes: 0 clean, 1 problems found, 2 files unreadable.

The challenge organisers supplied this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_FILINGS = 40  # 20 managers x 2 quarters

# (column, arrow type, nullable) — from docs/SCHEMA.md. Extra columns are permitted;
# these are the ones we require.
FILINGS: list[tuple[str, pa.DataType, bool]] = [
    ("accession_number", pa.string(), False),
    ("cik", pa.string(), False),
    ("fund_name", pa.string(), False),
    ("filing_manager", pa.string(), False),
    ("form_type", pa.string(), False),
    ("report_period", pa.date32(), False),
    ("report_quarter", pa.string(), False),
    ("filing_date", pa.date32(), False),
    ("is_amendment", pa.bool_(), False),
    ("amendment_no", pa.int32(), True),
    ("amendment_type", pa.string(), True),
    ("report_type", pa.string(), False),
    ("form_13f_file_number", pa.string(), True),
    ("crd_number", pa.string(), True),
    ("sec_file_number", pa.string(), True),
    ("other_included_managers_count", pa.int32(), True),
    ("table_entry_total", pa.int64(), True),
    ("table_value_total", pa.int64(), True),
]

HOLDINGS: list[tuple[str, pa.DataType, bool]] = [
    ("accession_number", pa.string(), False),
    ("cik", pa.string(), False),
    ("report_quarter", pa.string(), False),
    ("name_of_issuer", pa.string(), False),
    ("title_of_class", pa.string(), False),
    ("cusip", pa.string(), False),
    ("figi", pa.string(), True),
    ("value", pa.int64(), False),
    ("ssh_prnamt", pa.int64(), False),
    ("ssh_prnamt_type", pa.string(), False),
    ("put_call", pa.string(), True),
    ("investment_discretion", pa.string(), False),
    ("other_manager", pa.string(), True),
    ("voting_sole", pa.int64(), False),
    ("voting_shared", pa.int64(), False),
    ("voting_none", pa.int64(), False),
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

problems: list[str] = []
warnings: list[str] = []
passed = 0


def ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"  {GREEN}PASS{RESET} {msg}")


def fail(msg: str, hint: str = "") -> None:
    problems.append(msg)
    print(f"  {RED}FAIL{RESET} {msg}")
    if hint:
        print(f"       {DIM}{hint}{RESET}")


def warn(msg: str, hint: str = "") -> None:
    warnings.append(msg)
    print(f"  {YELLOW}WARN{RESET} {msg}")
    if hint:
        print(f"       {DIM}{hint}{RESET}")


def check_table(table: pa.Table, spec: list[tuple[str, pa.DataType, bool]], label: str) -> None:
    """Required columns present, correctly typed, and non-null where specified."""
    names = set(table.column_names)

    # A leaked pandas index is common enough to name explicitly — the generic
    # "unexpected column" message sends people looking in the wrong place.
    leaked = [c for c in table.column_names if c.startswith("__index_level_")]
    if leaked:
        fail(f"{label}: no pandas index column",
             f"found {leaked} — write with index=False")

    missing = [c for c, _, _ in spec if c not in names]
    if missing:
        fail(f"{label}: all required columns present", f"missing: {missing}")
        return
    ok(f"{label}: all {len(spec)} required columns present")

    bad_types: list[str] = []
    for col, want, _ in spec:
        got = table.schema.field(col).type
        if not got.equals(want):
            bad_types.append(f"{col}: expected {want}, got {got}")
    if bad_types:
        hint = "; ".join(bad_types)
        if any("double" in b or "float" in b for b in bad_types):
            # float64 cannot represent every integer above 2^53 and grading compares
            # exact integers, so this is a correctness bug rather than a style issue.
            hint += " — cast numeric columns to int64 before writing, not float"
        fail(f"{label}: column types match schema", hint)
    else:
        ok(f"{label}: column types")

    nulls = [
        f"{col} ({table.column(col).null_count})"
        for col, _, nullable in spec
        if not nullable and table.column(col).null_count
    ]
    if nulls:
        fail(f"{label}: non-nullable columns contain no nulls", "; ".join(nulls))
    else:
        ok(f"{label}: nullability")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("output"))
    args = ap.parse_args()

    fpath = args.output / "filings.parquet"
    hpath = args.output / "holdings.parquet"

    tables: dict[str, pa.Table] = {}
    for path in (fpath, hpath):
        if not path.exists():
            print(f"{RED}cannot find {path}{RESET}")
            print("Your pipeline must write both Parquet files there. See docs/SCHEMA.md.")
            return 2
        try:
            tables[path.name] = pq.read_table(path)
        except Exception as exc:
            print(f"{RED}cannot read {path}: {exc}{RESET}")
            return 2

    filings, holdings = tables["filings.parquet"], tables["holdings.parquet"]
    print(f"\n  filings.parquet   {filings.num_rows:>8,} rows x {filings.num_columns} cols")
    print(f"  holdings.parquet  {holdings.num_rows:>8,} rows x {holdings.num_columns} cols\n")

    print("filings.parquet")
    check_table(filings, FILINGS, "filings")

    print("\nholdings.parquet")
    check_table(holdings, HOLDINGS, "holdings")

    print("\nintegrity")
    if filings.num_rows == EXPECTED_FILINGS:
        ok(f"filings row count is {EXPECTED_FILINGS}")
    else:
        # Warning, not failure: the count is a strong signal something upstream is
        # wrong, but this script checks structure and we do not want to reject a
        # submission on a number that a deliberate, documented choice could change.
        warn(f"filings row count is {EXPECTED_FILINGS}",
             f"found {filings.num_rows}; 20 managers x 2 quarters should be "
             f"{EXPECTED_FILINGS}. Check your filing discovery.")

    if "accession_number" in filings.column_names and "accession_number" in holdings.column_names:
        known = set(filings.column("accession_number").to_pylist())
        referenced = set(holdings.column("accession_number").to_pylist())
        orphans = referenced - known
        if orphans:
            fail("every holding references a filing",
                 f"{len(orphans)} accession(s) in holdings are absent from filings, "
                 f"e.g. {sorted(orphans)[:3]}")
        else:
            ok("holdings join to filings")

    print(f"\n{'-' * 60}")
    if problems:
        print(f"{RED}{len(problems)} problem(s){RESET}")
        for p in problems:
            print(f"  - {p}")
        return 1

    note = f", {len(warnings)} warning(s)" if warnings else ""
    print(f"{GREEN}{passed} checks passed{note}{RESET}")
    print(f"{DIM}Structure is valid. This says nothing about whether your values "
          f"are correct.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
