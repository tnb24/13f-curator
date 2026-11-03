"""Tests for curator.parse: namespaces, nulls, entities, and filed values.

Every test reads real fixture files from tests/fixtures, so the file-reading
path runs each time. No test contacts the network and no test needs output/.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from curator.parse import QualityReport, parse_all, parse_filing
from curator.records import FilingDocs, FilingRef

FIXTURES = Path(__file__).parent / "fixtures"


def _ref(accession: str = "0001350693-26-000001", form: str = "13F-HR") -> FilingRef:
    return FilingRef(
        cik="0001350693",
        fund_name="TEST FUND LP",
        accession=accession,
        form=form,
        report_date="2026-03-31",
        filing_date="2026-05-15",
        primary_document="cover_regular.xml",
    )


def _docs(
    primary_doc: str = "cover_regular.xml", info_table: str | None = "info_hazards.xml",
    ref: FilingRef | None = None,
) -> FilingDocs:
    return FilingDocs(
        ref=ref or _ref(),
        info_table_path=None if info_table is None else FIXTURES / info_table,
        primary_doc_path=FIXTURES / primary_doc,
    )


def _parse(docs: FilingDocs | None = None):
    """Parse one docs pair. Return (filing record, holdings, quality report)."""
    quality = QualityReport()
    parsed = parse_filing(docs or _docs(), quality)
    return parsed.filing, parsed.holdings.to_pylist(), quality


def test_default_namespace():
    filing, holdings, _ = _parse(_docs(info_table=None))
    assert holdings == []
    assert filing["accession_number"] == "0001350693-26-000001"
    assert filing["cik"] == "0001350693"
    assert filing["filing_manager"] == "TEST MANAGER LLC"
    assert filing["form_type"] == "13F-HR"
    assert filing["report_period"] == date(2026, 3, 31)
    assert filing["report_quarter"] == "2026Q1"
    assert filing["filing_date"] == date(2026, 5, 15)
    assert filing["is_amendment"] is False
    assert filing["report_type"] == "13F HOLDINGS REPORT"
    assert filing["table_entry_total"] == 4
    assert filing["table_value_total"] == 123456


def test_prefixed_namespace():
    _, holdings, _ = _parse(_docs(info_table="info_prefixed.xml"))
    assert len(holdings) == 1
    assert holdings[0]["name_of_issuer"] == "PREFCO INC"
    assert holdings[0]["cusip"] == "037833100"


def test_mixed_namespaces_in_one_filing():
    docs = _docs(primary_doc="cover_regular.xml", info_table="info_prefixed.xml")
    filing, holdings, _ = _parse(docs)
    assert filing["filing_manager"] == "TEST MANAGER LLC"
    assert len(holdings) == 1
    assert holdings[0]["accession_number"] == filing["accession_number"]
    assert holdings[0]["name_of_issuer"] == "PREFCO INC"


def test_an_empty_element():
    filing, _, _ = _parse(
        _docs(
            primary_doc="cover_notice.xml", info_table=None,
            ref=_ref(form="13F-NT"),
        )
    )
    # Null: neither the integer 0 nor an empty string.
    assert filing["table_entry_total"] is None
    assert filing["table_value_total"] is None


def test_the_none_element():
    _, holdings, _ = _parse()
    row = holdings[1]
    assert row["voting_sole"] == 0
    assert row["voting_shared"] == 40
    assert row["voting_none"] == 10


def test_an_xml_entity():
    _, holdings, _ = _parse()
    assert holdings[0]["name_of_issuer"] == "S&P GLOBAL INC"


def test_a_cusip_with_a_leading_zero():
    _, holdings, _ = _parse()
    cusip = holdings[0]["cusip"]
    assert isinstance(cusip, str)
    assert cusip == "023135106"


def test_a_cins_code():
    _, holdings, _ = _parse()
    assert holdings[1]["cusip"] == "G11448100"


def test_a_duplicate_cusip():
    _, holdings, _ = _parse()
    assert len(holdings) == 4
    duplicates = [row for row in holdings if row["cusip"] == "023135106"]
    assert len(duplicates) == 2


def test_source_order():
    _, holdings, _ = _parse()
    assert [row["cusip"] for row in holdings] == [
        "023135106",
        "G11448100",
        "023135106",
        "12345678",
    ]
    assert [row["name_of_issuer"] for row in holdings] == [
        "S&P GLOBAL INC",
        "ACME CORP",
        "S&P GLOBAL INC",
        "SHORTCO INC",
    ]


def test_a_notice():
    filing, holdings, _ = _parse(
        _docs(primary_doc="cover_notice.xml", info_table=None, ref=_ref(form="13F-NT"))
    )
    assert holdings == []
    assert filing["form_type"] == "13F-NT"
    assert filing["report_type"] == "13F NOTICE"
    assert filing["report_period"] == date(2026, 6, 30)
    assert filing["report_quarter"] == "2026Q2"


def test_an_absent_put_call():
    _, holdings, _ = _parse()
    assert holdings[0]["put_call"] is None


def test_mixed_capitalisation():
    """put_call stays as filed. 'CALL' would be wrong here."""
    _, holdings, _ = _parse()
    assert holdings[1]["put_call"] == "Call"
    assert holdings[2]["put_call"] == "call"


def test_a_short_cusip():
    """An 8-character CUSIP survives as filed and names itself in quality."""
    _, holdings, quality = _parse()
    assert holdings[3]["cusip"] == "12345678"
    matches = [
        entry
        for entry in quality.entries
        if entry["field"] == "cusip"
        and entry["issue"] == "cusip_length_not_9"
        and entry.get("detail") == "12345678"
        and entry["row_index"] == 3
    ]
    assert len(matches) == 1


def test_an_absent_required_field_gets_a_default_and_a_quality_entry(tmp_path):
    """Absence is never silent: the row survives and the report names it."""
    table = (FIXTURES / "info_default.xml").read_text(encoding="utf-8")
    broken = table.replace("<value>", "<!-- no value --><unused>", 1).replace(
        "</value>", "</unused>", 1
    )
    path = tmp_path / "info_missing_value.xml"
    path.write_text(broken, encoding="utf-8")
    _, holdings, quality = _parse(FilingDocs(
        ref=_ref(), info_table_path=path, primary_doc_path=FIXTURES / "cover_regular.xml"))
    assert holdings[0]["value"] == 0
    assert any(
        e["field"] == "value" and e["issue"] == "absent_non_nullable" and e["row_index"] == 0
        for e in quality.entries
    )


def test_an_unparseable_integer_is_reported(tmp_path):
    table = (FIXTURES / "info_default.xml").read_text(encoding="utf-8")
    path = tmp_path / "info_bad_int.xml"
    path.write_text(table.replace("<sshPrnamt>", "<sshPrnamt>1,", 1), encoding="utf-8")
    _, holdings, quality = _parse(FilingDocs(
        ref=_ref(), info_table_path=path, primary_doc_path=FIXTURES / "cover_regular.xml"))
    assert holdings[0]["ssh_prnamt"] == 0
    assert any(e["issue"] == "unparseable_int" and e["field"] == "ssh_prnamt"
               for e in quality.entries)


def test_parse_all_sorts_filings_and_keeps_holdings_with_them():
    late = _docs(ref=_ref(accession="0001350693-26-000009"))
    early = _docs(ref=_ref(accession="0001350693-26-000001"))
    quality = QualityReport()
    dataset = parse_all([late, early], quality)
    assert dataset.filings.column("accession_number").to_pylist() == [
        "0001350693-26-000001", "0001350693-26-000009",
    ]
    assert dataset.holdings.column("accession_number").to_pylist() == (
        ["0001350693-26-000001"] * 4 + ["0001350693-26-000009"] * 4
    )
    # Both filings carry the hazards fixture, so each reports its short CUSIP.
    assert len([e for e in quality.entries if e["issue"] == "cusip_length_not_9"]) == 2


def test_a_declared_entity_is_not_expanded(tmp_path):
    """The parser resolves no declared entity, so a DTD cannot inject text."""
    table = (FIXTURES / "info_default.xml").read_text(encoding="utf-8")
    hostile = table.replace(
        '<?xml version="1.0"?>',
        '<?xml version="1.0"?>\n<!DOCTYPE x [<!ENTITY evil "INJECTED">]>',
    ).replace("<titleOfClass>COM</titleOfClass>", "<titleOfClass>&evil;</titleOfClass>", 1)
    path = tmp_path / "hostile.xml"
    path.write_text(hostile, encoding="utf-8")
    _, holdings, _ = _parse(FilingDocs(
        ref=_ref(), info_table_path=path, primary_doc_path=FIXTURES / "cover_regular.xml"))
    assert "INJECTED" not in holdings[0]["title_of_class"]
