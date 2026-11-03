"""Byte-stable output: the same rows always write the same bytes.

tests/test_pipeline.py checks whole runs. This module checks the writer alone.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pyarrow as pa
import pytest

from curator.parse import ParsedDataset, QualityReport, parse_filing
from curator.records import FilingDocs, FilingRef
from curator.schema import ATTRIBUTED_SCHEMA, FILINGS_SCHEMA, HOLDINGS_SCHEMA
from curator.write import write_parquet, write_table_atomic

FIXTURES = Path(__file__).parent / "fixtures"

_FILING = {
    "accession_number": "0001350693-26-000001",
    "cik": "0001350693",
    "fund_name": "TEST FUND LP",
    "filing_manager": "TEST MANAGER LLC",
    "form_type": "13F-HR",
    "report_period": date(2026, 3, 31),
    "report_quarter": "2026Q1",
    "filing_date": date(2026, 5, 15),
    "is_amendment": False,
    "amendment_no": None,
    "amendment_type": None,
    "report_type": "13F HOLDINGS REPORT",
    "form_13f_file_number": "028-12345",
    "crd_number": None,
    "sec_file_number": None,
    "other_included_managers_count": 0,
    "table_entry_total": 2,
    "table_value_total": 3000,
}


def _holding(cusip: str, value: int, put_call: str | None = None) -> dict:
    return {
        "accession_number": "0001350693-26-000001", "cik": "0001350693",
        "report_quarter": "2026Q1", "name_of_issuer": "ACME CORP", "title_of_class": "COM",
        "cusip": cusip, "figi": None, "value": value, "ssh_prnamt": 250,
        "ssh_prnamt_type": "SH", "put_call": put_call, "investment_discretion": "SOLE",
        "other_manager": None, "voting_sole": 100, "voting_shared": 0, "voting_none": 0,
    }


def _dataset(holdings: pa.Table | None = None) -> ParsedDataset:
    rows = [_holding("023135106", 1000), _holding("G11448100", 2000, "Call")]
    return ParsedDataset(
        filings=pa.Table.from_pylist([_FILING], schema=FILINGS_SCHEMA),
        holdings=holdings or pa.Table.from_pylist(rows, schema=HOLDINGS_SCHEMA),
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_two_writes_agree(tmp_path) -> None:
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        write_parquet(_dataset(), tmp_path / name)
    for table in ("filings.parquet", "holdings.parquet"):
        assert _digest(tmp_path / "first" / table) == _digest(tmp_path / "second" / table)


def test_chunk_layout_does_not_change_the_bytes(tmp_path) -> None:
    """parse_all concatenates one table per filing. The bytes must not depend on that."""
    whole = _dataset().holdings
    chunked = pa.concat_tables([whole.slice(0, 1), whole.slice(1)])
    assert chunked.column(0).num_chunks == 2
    write_table_atomic(whole, tmp_path / "whole.parquet")
    write_table_atomic(chunked, tmp_path / "chunked.parquet")
    assert _digest(tmp_path / "whole.parquet") == _digest(tmp_path / "chunked.parquet")


def test_a_table_with_the_wrong_schema_never_reaches_disk(tmp_path) -> None:
    loose = pa.Table.from_pylist([_holding("023135106", 1000)])  # inferred types, nullable
    with pytest.raises(TypeError):
        write_parquet(_dataset(holdings=loose), tmp_path)
    assert not (tmp_path / "holdings.parquet").exists()


def test_the_bonus_writer_agrees(tmp_path) -> None:
    row = {**_holding("023135106", 1000), "attributed_to_cik": "0001336528"}
    table = pa.Table.from_pylist([row], schema=ATTRIBUTED_SCHEMA)
    write_table_atomic(table, tmp_path / "a.parquet")
    write_table_atomic(table, tmp_path / "b.parquet")
    assert _digest(tmp_path / "a.parquet") == _digest(tmp_path / "b.parquet")


def test_no_temporary_file_is_left_behind(tmp_path) -> None:
    write_table_atomic(_dataset().holdings, tmp_path / "h.parquet")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["h.parquet"]


def test_parse_is_stable() -> None:
    ref = FilingRef(
        cik="0001350693", fund_name="TEST FUND LP", accession="0001350693-26-000001",
        form="13F-HR", report_date="2026-03-31", filing_date="2026-05-15",
        primary_document="cover_regular.xml",
    )
    docs = FilingDocs(
        ref=ref, info_table_path=FIXTURES / "info_hazards.xml",
        primary_doc_path=FIXTURES / "cover_regular.xml",
    )
    first = parse_filing(docs, QualityReport())
    second = parse_filing(docs, QualityReport())
    assert first.filing == second.filing
    assert first.holdings.equals(second.holdings)
