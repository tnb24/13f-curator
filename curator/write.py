"""Parquet writer: turn the parsed corpus into the two output tables.

Both tables come from an explicit Arrow schema. No type is inferred, no pandas
frame is involved, and every nullable column keeps a real null when a value is
absent. Each file lands through a temporary path in its target directory, so a
crash never leaves a half-written Parquet that looks valid.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from curator.parse import ParsedDataset

# docs/SCHEMA.md, in that file's column order. The nullable flag is part of the
# contract: verify.py reads it back field by field.
FILINGS_SCHEMA = pa.schema(
    [
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("fund_name", pa.string(), nullable=False),
        pa.field("filing_manager", pa.string(), nullable=False),
        pa.field("form_type", pa.string(), nullable=False),
        pa.field("report_period", pa.date32(), nullable=False),
        pa.field("report_quarter", pa.string(), nullable=False),
        pa.field("filing_date", pa.date32(), nullable=False),
        pa.field("is_amendment", pa.bool_(), nullable=False),
        pa.field("amendment_no", pa.int32(), nullable=True),
        pa.field("amendment_type", pa.string(), nullable=True),
        pa.field("report_type", pa.string(), nullable=False),
        pa.field("form_13f_file_number", pa.string(), nullable=True),
        pa.field("crd_number", pa.string(), nullable=True),
        pa.field("sec_file_number", pa.string(), nullable=True),
        pa.field("other_included_managers_count", pa.int32(), nullable=True),
        pa.field("table_entry_total", pa.int64(), nullable=True),
        pa.field("table_value_total", pa.int64(), nullable=True),
    ]
)

HOLDINGS_SCHEMA = pa.schema(
    [
        pa.field("accession_number", pa.string(), nullable=False),
        pa.field("cik", pa.string(), nullable=False),
        pa.field("report_quarter", pa.string(), nullable=False),
        pa.field("name_of_issuer", pa.string(), nullable=False),
        pa.field("title_of_class", pa.string(), nullable=False),
        pa.field("cusip", pa.string(), nullable=False),
        pa.field("figi", pa.string(), nullable=True),
        pa.field("value", pa.int64(), nullable=False),
        pa.field("ssh_prnamt", pa.int64(), nullable=False),
        pa.field("ssh_prnamt_type", pa.string(), nullable=False),
        pa.field("put_call", pa.string(), nullable=True),
        pa.field("investment_discretion", pa.string(), nullable=False),
        pa.field("other_manager", pa.string(), nullable=True),
        pa.field("voting_sole", pa.int64(), nullable=False),
        pa.field("voting_shared", pa.int64(), nullable=False),
        pa.field("voting_none", pa.int64(), nullable=False),
    ]
)


def write_parquet(dataset: ParsedDataset, output: Path) -> None:
    """Write output/filings.parquet and output/holdings.parquet.

    Filings sort on cik, report_period, accession_number. Holdings keep the
    dataset order: filing order, then source order inside each filing.
    """
    filings = sorted(
        dataset.filings,
        key=lambda row: (row["cik"], row["report_period"], row["accession_number"]),
    )
    filings_table = pa.Table.from_pylist(
        [_filing_row(row) for row in filings], schema=FILINGS_SCHEMA
    )
    holdings_table = pa.Table.from_pylist(
        [_holding_row(row) for row in dataset.holdings], schema=HOLDINGS_SCHEMA
    )
    _write_atomic(filings_table, Path(output) / "filings.parquet")
    _write_atomic(holdings_table, Path(output) / "holdings.parquet")


def _filing_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one filing record onto FILINGS_SCHEMA with exact types."""
    return {
        "accession_number": str(row["accession_number"]),
        "cik": str(row["cik"]),
        "fund_name": str(row["fund_name"]),
        "filing_manager": str(row["filing_manager"]),
        "form_type": str(row["form_type"]),
        "report_period": _day(row["report_period"]),
        "report_quarter": str(row["report_quarter"]),
        "filing_date": _day(row["filing_date"]),
        "is_amendment": bool(row["is_amendment"]),
        "amendment_no": _int(row.get("amendment_no")),
        "amendment_type": _text(row.get("amendment_type")),
        "report_type": str(row["report_type"]),
        "form_13f_file_number": _text(row.get("form_13f_file_number")),
        "crd_number": _text(row.get("crd_number")),
        "sec_file_number": _text(row.get("sec_file_number")),
        "other_included_managers_count": _int(row.get("other_included_managers_count")),
        "table_entry_total": _int(row.get("table_entry_total")),
        "table_value_total": _int(row.get("table_value_total")),
    }


def _holding_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one holding record onto HOLDINGS_SCHEMA with exact types."""
    return {
        "accession_number": str(row["accession_number"]),
        "cik": str(row["cik"]),
        "report_quarter": str(row["report_quarter"]),
        "name_of_issuer": str(row["name_of_issuer"]),
        "title_of_class": str(row["title_of_class"]),
        "cusip": str(row["cusip"]),
        "figi": _text(row.get("figi")),
        "value": int(row["value"]),
        "ssh_prnamt": int(row["ssh_prnamt"]),
        "ssh_prnamt_type": str(row["ssh_prnamt_type"]),
        "put_call": _text(row.get("put_call")),
        "investment_discretion": str(row["investment_discretion"]),
        "other_manager": _text(row.get("other_manager")),
        "voting_sole": int(row["voting_sole"]),
        "voting_shared": int(row["voting_shared"]),
        "voting_none": int(row["voting_none"]),
    }


def _text(value: Any) -> str | None:
    """Return the string, or a real null when no value exists."""
    return None if value is None else str(value)


def _int(value: Any) -> int | None:
    """Return the exact integer, or a real null when no value exists."""
    return None if value is None else int(value)


def _day(value: Any) -> date:
    """Return the calendar date. Arrow maps a datetime.date to date32."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"expected a date, got {type(value).__name__}")


def _write_atomic(table: pa.Table, path: Path) -> None:
    """Write through a temporary file beside the target, then rename into place."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table, tmp, compression="snappy", version="2.6")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
