"""End-to-end test over canned bodies: discover, download, parse, write.

The fake cache serves a submissions payload, two archive indexes, and four
small XML documents. The run touches no network and writes only into
``tmp_path``. It exercises the real discover, download, parse, and write
code paths and checks the written Parquet against the schema contract.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq

from curator.cache import Cache  # noqa: F401  (interface reference)
from curator.cik import FilerRecord
from curator.discover import discover
from curator.download import download
from curator.parse import ParsedDataset, QualityReport, parse_all
from curator.write import FILINGS_SCHEMA, HOLDINGS_SCHEMA, write_parquet

FIXTURES = Path(__file__).parent / "fixtures"

_CIK = "0001350693"
_HR_ACC = "0001350693-26-000001"
_NT_ACC = "0001350693-26-000002"
_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/1350693/"
_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{int(_CIK):010}.json"


class FakeCache:
    """Return canned bodies by URL. Raise KeyError when no body exists."""

    def __init__(self, root: Path, bodies: dict[str, bytes]) -> None:
        self.root = Path(root)
        self._bodies = bodies

    def get(self, url: str) -> bytes:
        return self._bodies[url]


def _index(names: list[str]) -> bytes:
    return json.dumps({"directory": {"item": [{"name": name} for name in names]}}).encode(
        "utf-8"
    )


def _bodies() -> dict[str, bytes]:
    submissions = {
        "name": "TEST MANAGER LLC",
        "filings": {
            "recent": {
                "accessionNumber": [_HR_ACC, _NT_ACC],
                "form": ["13F-HR", "13F-NT"],
                "reportDate": ["2026-03-31", "2026-06-30"],
                "filingDate": ["2026-05-15", "2026-08-03"],
                "primaryDocument": ["cover.xml", "notice_cover.xml"],
            }
        },
    }
    cover = (FIXTURES / "cover_regular.xml").read_bytes()
    notice = (FIXTURES / "cover_notice.xml").read_bytes()
    info_table = (FIXTURES / "info_prefixed.xml").read_bytes()
    return {
        _SUBMISSIONS_URL: json.dumps(submissions).encode("utf-8"),
        _ARCHIVE + f"{_HR_ACC.replace('-', '')}/index.json": _index(
            ["cover.xml", "infotable.xml"]
        ),
        _ARCHIVE + f"{_HR_ACC.replace('-', '')}/cover.xml": cover,
        _ARCHIVE + f"{_HR_ACC.replace('-', '')}/infotable.xml": info_table,
        _ARCHIVE + f"{_NT_ACC.replace('-', '')}/index.json": _index(["notice_cover.xml"]),
        _ARCHIVE + f"{_NT_ACC.replace('-', '')}/notice_cover.xml": notice,
    }


def _filer() -> FilerRecord:
    return FilerRecord(
        fund_name="TEST FUND LP",
        cik=_CIK,
        cik_source="given",
        edgar_name="TEST MANAGER LLC",
    )


def test_pipeline_runs_from_submissions_to_parquet(tmp_path):
    cache = FakeCache(tmp_path, _bodies())

    refs = discover([_filer()], cache)
    assert [(ref.accession, ref.form) for ref in refs] == [
        (_HR_ACC, "13F-HR"),
        (_NT_ACC, "13F-NT"),
    ]

    docs = download(refs, cache, tmp_path / "filings")
    assert len(docs) == 2

    quality = QualityReport()
    dataset: ParsedDataset = parse_all(docs, quality)
    # Both fixtures are clean: no anomaly, no substitution.
    assert quality.entries == []
    assert len(dataset.filings) == 2
    assert len(dataset.holdings) == 1

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    write_parquet(dataset, out_dir)

    filings = pq.read_table(out_dir / "filings.parquet")
    holdings = pq.read_table(out_dir / "holdings.parquet")
    assert filings.schema.equals(FILINGS_SCHEMA, check_metadata=False)
    assert holdings.schema.equals(HOLDINGS_SCHEMA, check_metadata=False)
    assert filings.num_rows == 2
    assert holdings.num_rows == 1

    filing_rows = filings.to_pylist()
    holding_rows = holdings.to_pylist()
    notice_row = next(row for row in filing_rows if row["form_type"] == "13F-NT")
    assert notice_row["table_entry_total"] is None
    report = next(row for row in filing_rows if row["form_type"] == "13F-HR")
    assert report["report_period"] == date(2026, 3, 31)
    assert report["report_quarter"] == "2026Q1"
    assert holding_rows[0]["accession_number"] == report["accession_number"]
