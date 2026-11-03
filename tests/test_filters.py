"""Tests for curator.discover: the three scope filters.

Each test feeds one parallel-array frame straight into the module's own
filter, ``_in_scope``. No cache, no network, no re-implemented filter.
"""

from __future__ import annotations

from curator.discover import _in_scope
from curator.records import FilerRecord

_FILER = FilerRecord(
    fund_name="TEST FUND LP",
    cik="0001234567",
    cik_source="given",
    edgar_name="TEST FUND LP",
)


def _entry(
    accession: str = "0001234567-26-000001",
    form: str = "13F-HR",
    report_date: str = "2026-03-31",
    filing_date: str = "2026-05-15",
    primary_document: str = "doc.xml",
) -> dict[str, str]:
    return {
        "accession": accession,
        "form": form,
        "report_date": report_date,
        "filing_date": filing_date,
        "primary_document": primary_document,
    }


def _frame(entries: list[dict[str, str]]) -> dict[str, list[str]]:
    """Build one submissions-style parallel-array frame."""
    columns: dict[str, list[str]] = {
        "accessionNumber": [],
        "form": [],
        "reportDate": [],
        "filingDate": [],
        "primaryDocument": [],
    }
    for entry in entries:
        columns["accessionNumber"].append(entry["accession"])
        columns["form"].append(entry["form"])
        columns["reportDate"].append(entry["report_date"])
        columns["filingDate"].append(entry["filing_date"])
        columns["primaryDocument"].append(entry["primary_document"])
    return columns


def test_report_period_filter():
    """reportDate drives the period filter, never filingDate."""
    excluded = _frame([_entry(report_date="2025-12-31")])
    assert _in_scope(excluded, _FILER) == []
    included = _frame([_entry(report_date="2026-03-31")])
    refs = _in_scope(included, _FILER)
    assert [ref.accession for ref in refs] == ["0001234567-26-000001"]
    assert [ref.report_date for ref in refs] == ["2026-03-31"]


def test_filing_date_cutoff():
    """The cutoff day 2026-08-18 is inclusive; the next day is out."""
    excluded = _frame([_entry(filing_date="2026-08-19")])
    assert _in_scope(excluded, _FILER) == []
    included = _frame([_entry(filing_date="2026-08-18")])
    refs = _in_scope(included, _FILER)
    assert [ref.filing_date for ref in refs] == ["2026-08-18"]


def test_form_filter():
    """13F-NT stays in scope. NT 10-K shares a prefix and must not."""
    frame = _frame(
        [
            _entry(accession="0001234567-26-000009", form="NT 10-K"),
            _entry(accession="0001234567-26-000002", form="13F-NT"),
        ]
    )
    refs = _in_scope(frame, _FILER)
    assert len(refs) == 1
    assert refs[0].form == "13F-NT"
    assert refs[0].accession == "0001234567-26-000002"


def test_amendment_forms():
    """13F-HR/A survives, and its form keeps the /A suffix parse reads."""
    refs = _in_scope(_frame([_entry(form="13F-HR/A")]), _FILER)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.form == "13F-HR/A"
    # curator.parse derives is_amendment from exactly this suffix.
    assert ref.form.endswith("/A")
