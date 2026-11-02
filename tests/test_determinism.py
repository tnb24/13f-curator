"""Tests for byte-stable output: the same rows always write the same bytes."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

from curator.discover import FilingRef
from curator.download import FilingDocs
from curator.parse import ParsedDataset, QualityReport, parse_filing
from curator.write import write_parquet

FIXTURES = Path(__file__).parent / "fixtures"


def _dataset() -> ParsedDataset:
    """Return one small hand-built corpus with real nulls where allowed."""
    filing = {
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
        "table_value_total": 4000,
    }
    base_holding = {
        "accession_number": "0001350693-26-000001",
        "cik": "0001350693",
        "report_quarter": "2026Q1",
        "name_of_issuer": "ACME CORP",
        "title_of_class": "COM",
        "cusip": "023135106",
        "figi": None,
        "value": 1000,
        "ssh_prnamt": 250,
        "ssh_prnamt_type": "SH",
        "investment_discretion": "SOLE",
        "other_manager": None,
        "voting_sole": 100,
        "voting_shared": 0,
        "voting_none": 0,
    }
    call_holding = {
        **base_holding,
        "cusip": "G11448100",
        "value": 2000,
        "ssh_prnamt": 300,
        "put_call": "Call",
        "voting_sole": 0,
        "voting_shared": 40,
        "voting_none": 10,
    }
    return ParsedDataset(filings=[filing], holdings=[base_holding, call_holding])


def _hazard_docs() -> FilingDocs:
    ref = FilingRef(
        cik="0001350693",
        fund_name="TEST FUND LP",
        accession="0001350693-26-000001",
        form="13F-HR",
        report_date="2026-03-31",
        filing_date="2026-05-15",
        primary_document="cover_regular.xml",
    )
    return FilingDocs(
        ref=ref,
        info_table_path=FIXTURES / "info_hazards.xml",
        primary_doc_path=FIXTURES / "cover_regular.xml",
    )


def _digests(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("filings.parquet", "holdings.parquet")
    }


def test_two_writes_agree(tmp_path):
    """Writing the same rows twice gives equal SHA-256 digests."""
    dataset = _dataset()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    write_parquet(dataset, first_dir)
    write_parquet(dataset, second_dir)
    assert _digests(first_dir) == _digests(second_dir)


def test_parse_is_stable():
    """Parsing the same fixture twice gives equal records."""
    first_quality = QualityReport()
    second_quality = QualityReport()
    first = parse_filing(_hazard_docs(), first_quality)
    second = parse_filing(_hazard_docs(), second_quality)
    assert first.filing == second.filing
    assert first.holdings == second.holdings


def test_the_bonus_writer_agrees(tmp_path):
    """Writing the same bonus rows twice gives equal SHA-256 digests."""
    import pyarrow as pa

    from curator.bonus1 import BONUS1_SCHEMA
    from curator.write import _write_atomic

    dataset = _dataset()
    row = {field.name: dataset.holdings[0].get(field.name) for field in BONUS1_SCHEMA}
    row["attributed_to_cik"] = "0001336528"
    table = pa.Table.from_pylist([row], schema=BONUS1_SCHEMA)

    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    _write_atomic(table, first)
    _write_atomic(table, second)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(first) == digest(second)
