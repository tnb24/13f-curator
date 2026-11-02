"""XML parser: turn each filing into one filing record and its holding rows.

Filers bind three different prefixes to one namespace URI. Every lookup in
this module matches the local name only, so the prefix choice never matters.
An empty element means null, never zero and never an empty string.

The parser never drops a row and never drops a filing. It records every
substitution and every anomaly in a QualityReport and continues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from lxml import etree

from curator.download import FilingDocs

_COVER_ROOT = "edgarSubmission"
_INFO_ROOT = "informationTable"
_ROW = "infoTable"


def _local_name(element) -> str | None:
    """Return the local name of one element, or None for a comment or PI."""
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def _child_map(parent) -> dict[str, Any]:
    """Map each child local name to its first element with that name.

    This is the only child lookup in the module. ``child_text`` and the row
    reader share it, so both read the same first match. A parent of None maps
    to nothing.
    """
    if parent is None:
        return {}
    found: dict[str, Any] = {}
    for candidate in parent:
        name = _local_name(candidate)
        if name is not None and name not in found:
            found[name] = candidate
    return found


def _text_of(element) -> str | None:
    """Return the stripped text of one element, or None when it has no text."""
    if element is None:
        return None
    text = element.text
    if text is None:
        return None
    stripped = text.strip()
    return stripped if stripped else None


def child_text(parent, local_name: str) -> str | None:
    """Return the stripped text of the first child with this local name.

    Return None when the element is absent.
    Return None when the element is present and its text is empty.
    """
    return _text_of(_child_map(parent).get(local_name))


class QualityReport:
    """Collect every substitution and every anomaly. Write output/quality_report.json."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(
        self,
        accession_number: str,
        row_index: int | None,
        field: str,
        issue: str,
        action: str,
        detail: str | None = None,
    ) -> None:
        """Append one entry. ``detail`` carries the offending value when any."""
        entry: dict[str, Any] = {
            "accession_number": accession_number,
            "row_index": row_index,
            "field": field,
            "issue": issue,
            "action": action,
        }
        if detail is not None:
            entry["detail"] = detail
        self.entries.append(entry)

    def write(self, path: Path) -> None:
        """Write every entry sorted on a full key, so bytes match each run."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = sorted(self.entries, key=_entry_sort_key)
        payload = json.dumps(entries, indent=2, ensure_ascii=False, sort_keys=True)
        path.write_bytes((payload + "\n").encode("utf-8"))


def _entry_sort_key(entry: dict[str, Any]) -> tuple:
    row_index = entry["row_index"]
    return (
        entry["accession_number"],
        row_index is None,
        row_index if row_index is not None else -1,
        entry["field"],
        entry["issue"],
        entry["action"],
        entry.get("detail", ""),
    )


@dataclass(frozen=True)
class ParsedFiling:
    """One parsed filing: its record and the holdings it carries."""

    filing: dict[str, Any]
    holdings: list[dict[str, Any]]


@dataclass(frozen=True)
class ParsedDataset:
    """The whole corpus: filings sorted, holdings in their filing's order."""

    filings: list[dict[str, Any]]
    holdings: list[dict[str, Any]]


def parse_filing(docs: FilingDocs, quality: QualityReport) -> ParsedFiling:
    """Parse one stored filing into its record and its holding rows."""
    filing = _parse_cover(docs, quality)
    holdings = _parse_rows(docs, filing, quality)
    return ParsedFiling(filing=filing, holdings=holdings)


def parse_all(docs: list[FilingDocs], quality: QualityReport) -> ParsedDataset:
    """Parse every filing. Sort filings by cik, report_period, accession.

    Holdings follow their filing and keep source order.
    """
    parsed = [parse_filing(doc, quality) for doc in docs]
    order = sorted(
        range(len(parsed)),
        key=lambda i: (
            parsed[i].filing["cik"],
            parsed[i].filing["report_period"],
            parsed[i].filing["accession_number"],
        ),
    )
    filings = [parsed[i].filing for i in order]
    holdings: list[dict[str, Any]] = []
    for i in order:
        holdings.extend(parsed[i].holdings)
    return ParsedDataset(filings=filings, holdings=holdings)


def _parse_cover(docs: FilingDocs, quality: QualityReport) -> dict[str, Any]:
    """Read the cover page into one filing record."""
    ref = docs.ref
    accession = ref.accession
    root = etree.parse(str(docs.primary_doc_path)).getroot()
    root_name = _local_name(root)
    if root_name != _COVER_ROOT:
        quality.record(
            accession, None, "cover_page", "unexpected_root", "continued",
            detail=root_name or "",
        )

    header_data = _child_map(root).get("headerData")
    filer_info = _child_map(header_data).get("filerInfo")
    filer = _child_map(filer_info).get("filer")
    credentials = _child_map(filer).get("credentials")

    cik_text = child_text(credentials, "cik")
    if cik_text is None:
        quality.record(
            accession, None, "cik", "absent_non_nullable",
            "fallback_to_submissions_api", detail=ref.cik,
        )
        cik_text = ref.cik
    cik = cik_text.strip().zfill(10)

    form_data = _child_map(root).get("formData")
    cover = _child_map(form_data).get("coverPage")
    summary = _child_map(form_data).get("summaryPage")
    if cover is None:
        quality.record(accession, None, "cover_page", "absent_non_nullable", "continued")

    manager = _child_map(cover).get("filingManager")
    filing_manager = child_text(manager, "name")
    if filing_manager is None:
        quality.record(
            accession, None, "filing_manager", "absent_non_nullable", "default_applied"
        )
        filing_manager = ""

    period_text = child_text(cover, "reportCalendarOrQuarter")
    report_period = _parse_mmddyyyy(period_text)
    if report_period is None:
        quality.record(
            accession, None, "report_period",
            "absent_non_nullable" if period_text is None else "unparseable_date",
            "fallback_to_submissions_api", detail=ref.report_date,
        )
        try:
            report_period = date.fromisoformat(ref.report_date)
        except ValueError as exc:
            raise RuntimeError(
                f"{accession}: unusable report period {ref.report_date!r}"
            ) from exc

    amendment_type = child_text(_child_map(cover).get("amendmentInfo"), "amendmentType")

    return {
        "accession_number": accession,
        "cik": cik,
        "fund_name": ref.fund_name,
        "filing_manager": filing_manager,
        "form_type": ref.form,
        "report_period": report_period,
        "report_quarter": (
            f"{report_period.year}Q{(report_period.month - 1) // 3 + 1}"
        ),
        "filing_date": _parse_isodate(ref.filing_date, accession),
        "is_amendment": ref.form.endswith("/A"),
        "amendment_no": _nullable_int(
            child_text(cover, "amendmentNo"),
            accession=accession, column="amendment_no", quality=quality,
        ),
        "amendment_type": amendment_type,
        "report_type": _required_text(
            child_text(cover, "reportType"),
            accession=accession, row_index=None, column="report_type", quality=quality,
        ),
        "form_13f_file_number": child_text(cover, "form13FFileNumber"),
        "crd_number": child_text(cover, "crdNumber"),
        "sec_file_number": child_text(cover, "secFileNumber"),
        "other_included_managers_count": _nullable_int(
            child_text(summary, "otherIncludedManagersCount"),
            accession=accession,
            column="other_included_managers_count",
            quality=quality,
        ),
        "table_entry_total": _nullable_int(
            child_text(summary, "tableEntryTotal"),
            accession=accession, column="table_entry_total", quality=quality,
        ),
        "table_value_total": _nullable_int(
            child_text(summary, "tableValueTotal"),
            accession=accession, column="table_value_total", quality=quality,
        ),
    }


def _parse_rows(
    docs: FilingDocs, filing: dict[str, Any], quality: QualityReport
) -> list[dict[str, Any]]:
    """Read the information table into holding rows in source order."""
    if docs.info_table_path is None:
        return []
    accession = filing["accession_number"]
    root = etree.parse(str(docs.info_table_path)).getroot()
    root_name = _local_name(root)
    if root_name != _INFO_ROOT:
        quality.record(
            accession, None, "info_table", "unexpected_root", "continued",
            detail=root_name or "",
        )
    rows: list[dict[str, Any]] = []
    index = 0
    for element in root:
        if _local_name(element) != _ROW:
            continue
        rows.append(_parse_row(element, index, filing, quality))
        index += 1
    return rows


def _parse_row(
    info_table, row_index: int, filing: dict[str, Any], quality: QualityReport
) -> dict[str, Any]:
    """Read one infoTable element into one holding row.

    The row keeps its source position through ``row_index``. Text fields stay
    as filed: lxml decodes entities once when this code reads .text, and no
    second decode happens. ``other_manager`` keeps its raw filed text, spaces
    included; an all-whitespace body still means null.
    """
    accession = filing["accession_number"]
    children = _child_map(info_table)

    shrs = children.get("shrsOrPrnAmt")
    voting = children.get("votingAuthority")

    def required_int(text: str | None, column: str) -> int:
        return _required_int(
            text,
            accession=accession, row_index=row_index, column=column, quality=quality,
        )

    def required_text(text: str | None, column: str) -> str:
        return _required_text(
            text,
            accession=accession, row_index=row_index, column=column, quality=quality,
        )

    cusip = required_text(_text_of(children.get("cusip")), "cusip")
    if cusip and len(cusip) != 9:
        quality.record(
            accession, row_index, "cusip", "cusip_length_not_9",
            "stored_as_filed", detail=cusip,
        )

    other_element = children.get("otherManager")
    other_manager = other_element.text if other_element is not None else None
    if other_manager is not None and not other_manager.strip():
        other_manager = None

    return {
        "accession_number": accession,
        "cik": filing["cik"],
        "report_quarter": filing["report_quarter"],
        "name_of_issuer": required_text(_text_of(children.get("nameOfIssuer")), "name_of_issuer"),
        "title_of_class": required_text(_text_of(children.get("titleOfClass")), "title_of_class"),
        "cusip": cusip,
        "figi": _text_of(children.get("figi")),
        "value": required_int(_text_of(children.get("value")), "value"),
        "ssh_prnamt": required_int(child_text(shrs, "sshPrnamt"), "ssh_prnamt"),
        "ssh_prnamt_type": required_text(child_text(shrs, "sshPrnamtType"), "ssh_prnamt_type"),
        "put_call": _text_of(children.get("putCall")),
        "investment_discretion": required_text(
            _text_of(children.get("investmentDiscretion")), "investment_discretion"
        ),
        "other_manager": other_manager,
        "voting_sole": required_int(child_text(voting, "Sole"), "voting_sole"),
        "voting_shared": required_int(child_text(voting, "Shared"), "voting_shared"),
        "voting_none": required_int(child_text(voting, "None"), "voting_none"),
    }


def _required_text(
    text: str | None, *, accession: str, row_index: int | None,
    column: str, quality: QualityReport,
) -> str:
    """Return the text, or "" when absent. Record every substitution."""
    if text is None:
        quality.record(accession, row_index, column, "absent_non_nullable", "default_applied")
        return ""
    return text


def _required_int(
    text: str | None, *, accession: str, row_index: int | None,
    column: str, quality: QualityReport,
) -> int:
    """Return the integer, or 0 when absent or unparseable. Record both cases."""
    if text is None:
        quality.record(accession, row_index, column, "absent_non_nullable", "default_applied")
        return 0
    try:
        return int(text)
    except ValueError:
        quality.record(
            accession, row_index, column, "unparseable_int",
            "default_applied", detail=text,
        )
        return 0


def _nullable_int(
    text: str | None, *, accession: str, column: str, quality: QualityReport
) -> int | None:
    """Return the integer, or null when absent. Absence is legal here."""
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        quality.record(
            accession, None, column, "unparseable_int", "null_stored", detail=text
        )
        return None


def _parse_mmddyyyy(text: str | None) -> date | None:
    """Parse MM-DD-YYYY cover page text into a date."""
    if text is None:
        return None
    try:
        return datetime.strptime(text, "%m-%d-%Y").date()
    except ValueError:
        return None


def _parse_isodate(text: str, accession: str) -> date:
    """Parse a YYYY-MM-DD submissions API date into a date."""
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(f"{accession}: unusable filing date {text!r}") from exc
