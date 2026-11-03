"""XML parser: turn each filing into one filing record and a holdings table.

Filers bind three different prefixes to one namespace URI. Every lookup in
this module matches the local name only, so the prefix choice never matters.
An empty element means null, never zero and never an empty string.

The parser never drops a row and never drops a filing. It records every
substitution and every anomaly in a QualityReport and continues.
"""

from __future__ import annotations

import json
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
from lxml import etree

from curator.records import FilingDocs
from curator.schema import FILINGS_SCHEMA, HOLDINGS_SCHEMA

_COVER_ROOT = "edgarSubmission"
_INFO_ROOT = "informationTable"
_ROW = "infoTable"

# The XML comes from third parties. The parser resolves no declared entity and
# opens no network connection. The five predefined entities still decode.
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)


def _parse_xml(path: Path):
    """Return the root element of one stored XML document."""
    return etree.parse(str(path), _PARSER).getroot()


def _local_name(element) -> str | None:
    """Return the local name of one element, or None for a comment or PI.

    lxml stores a namespaced tag as `{uri}name`. A string split is about ten
    times faster than `etree.QName`, and the parser calls this once per element.
    """
    tag = element.tag
    if not isinstance(tag, str):
        return None
    return tag.rpartition("}")[2]


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
    """One parsed filing: its record and its holdings, in source order."""

    filing: dict[str, Any]
    holdings: pa.Table  # HOLDINGS_SCHEMA


@dataclass(frozen=True)
class ParsedDataset:
    """The whole corpus as two tables.

    Filings sort on cik, report_period, accession_number. Holdings follow
    their filing and keep source order inside it.
    """

    filings: pa.Table  # FILINGS_SCHEMA
    holdings: pa.Table  # HOLDINGS_SCHEMA


def parse_filing(docs: FilingDocs, quality: QualityReport) -> ParsedFiling:
    """Parse one stored filing into its record and its holdings table."""
    filing = _parse_cover(docs, quality)
    holdings = _parse_rows(docs, filing, quality)
    return ParsedFiling(filing=filing, holdings=holdings)


def warm_up() -> None:
    """Prepare one worker process. The pipeline calls this once per worker.

    Unpickling this function imports the module, and so lxml and pyarrow. The
    first Arrow array in a process also loads pyarrow's pandas shim when pandas
    is installed. Both costs fall here, while the main process still verifies
    CIKs, and not inside the first parse task.
    """
    pa.array([], pa.string())


def parse_task(docs: FilingDocs) -> tuple[ParsedFiling, list[dict[str, Any]]]:
    """Parse one filing. The quality entries travel back with the result."""
    quality = QualityReport()
    return parse_filing(docs, quality), quality.entries


def parse_all(
    docs: list[FilingDocs], quality: QualityReport, pool: Executor | None = None
) -> ParsedDataset:
    """Parse every filing, on `pool` when one is given.

    Each filing is independent, so each one is a task. lxml creates one Python
    object for every element that Python touches, and that work holds the GIL.
    So the pipeline passes a process pool, not a thread pool. A worker returns
    an Arrow table, which crosses the process boundary as raw buffers.
    The largest files go first, so a long task never starts last. The results
    are put back in input order, so the output never depends on scheduling.
    """
    if pool is None:
        results = [parse_task(doc) for doc in docs]
    else:
        sizes = [(doc.info_table_path or doc.primary_doc_path).stat().st_size for doc in docs]
        order = sorted(range(len(docs)), key=lambda i: -sizes[i])
        futures = {i: pool.submit(parse_task, docs[i]) for i in order}
        results = [futures[i].result() for i in range(len(docs))]

    parsed: list[ParsedFiling] = []
    for filing, entries in results:
        parsed.append(filing)
        quality.entries.extend(entries)
    parsed.sort(
        key=lambda item: (
            item.filing["cik"], item.filing["report_period"], item.filing["accession_number"]
        )
    )
    filings = pa.Table.from_pylist([item.filing for item in parsed], schema=FILINGS_SCHEMA)
    holdings = pa.concat_tables([item.holdings for item in parsed]).combine_chunks()
    return ParsedDataset(filings=filings, holdings=holdings)


def _parse_cover(docs: FilingDocs, quality: QualityReport) -> dict[str, Any]:
    """Read the cover page into one filing record."""
    ref = docs.ref
    accession = ref.accession
    root = _parse_xml(docs.primary_doc_path)
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
) -> pa.Table:
    """Read the information table into one Arrow table, in source order.

    Each field goes straight into its own column list, so no row object is
    ever built. Each row costs three child maps: the row, its share block,
    and its voting block.
    """
    if docs.info_table_path is None:
        return HOLDINGS_SCHEMA.empty_table()
    accession = filing["accession_number"]
    root = _parse_xml(docs.info_table_path)
    root_name = _local_name(root)
    if root_name != _INFO_ROOT:
        quality.record(
            accession, None, "info_table", "unexpected_root", "continued",
            detail=root_name or "",
        )

    names: list[str] = []
    titles: list[str] = []
    cusips: list[str] = []
    figis: list[str | None] = []
    values: list[int] = []
    amounts: list[int] = []
    amount_types: list[str] = []
    put_calls: list[str | None] = []
    discretions: list[str] = []
    other_managers: list[str | None] = []
    sole: list[int] = []
    shared: list[int] = []
    none: list[int] = []

    def text(element, column: str, row: int) -> str:
        found = _text_of(element)
        if found is None:
            quality.record(accession, row, column, "absent_non_nullable", "default_applied")
            return ""
        return found

    def integer(element, column: str, row: int) -> int:
        found = _text_of(element)
        if found is not None and found.isdecimal():
            return int(found)
        return _required_int(
            found, accession=accession, row_index=row, column=column, quality=quality
        )

    row = 0
    for element in root:
        if _local_name(element) != _ROW:
            continue
        children = _child_map(element)
        shrs = _child_map(children.get("shrsOrPrnAmt"))
        voting = _child_map(children.get("votingAuthority"))

        names.append(text(children.get("nameOfIssuer"), "name_of_issuer", row))
        titles.append(text(children.get("titleOfClass"), "title_of_class", row))
        cusip = text(children.get("cusip"), "cusip", row)
        if cusip and len(cusip) != 9:
            quality.record(
                accession, row, "cusip", "cusip_length_not_9", "stored_as_filed", detail=cusip
            )
        cusips.append(cusip)
        figis.append(_text_of(children.get("figi")))
        values.append(integer(children.get("value"), "value", row))
        amounts.append(integer(shrs.get("sshPrnamt"), "ssh_prnamt", row))
        amount_types.append(text(shrs.get("sshPrnamtType"), "ssh_prnamt_type", row))
        put_calls.append(_text_of(children.get("putCall")))
        discretions.append(
            text(children.get("investmentDiscretion"), "investment_discretion", row)
        )
        # otherManager keeps its raw filed text, spaces included. An
        # all-whitespace body still means null.
        other = children.get("otherManager")
        other_text = other.text if other is not None else None
        other_managers.append(other_text if other_text and other_text.strip() else None)
        sole.append(integer(voting.get("Sole"), "voting_sole", row))
        shared.append(integer(voting.get("Shared"), "voting_shared", row))
        none.append(integer(voting.get("None"), "voting_none", row))
        row += 1

    def repeat(value: str) -> pa.Array:
        return pa.array([value] * row, pa.string())

    return pa.table(
        {
            "accession_number": repeat(accession),
            "cik": repeat(filing["cik"]),
            "report_quarter": repeat(filing["report_quarter"]),
            "name_of_issuer": names,
            "title_of_class": titles,
            "cusip": cusips,
            "figi": figis,
            "value": values,
            "ssh_prnamt": amounts,
            "ssh_prnamt_type": amount_types,
            "put_call": put_calls,
            "investment_discretion": discretions,
            "other_manager": other_managers,
            "voting_sole": sole,
            "voting_shared": shared,
            "voting_none": none,
        },
        schema=HOLDINGS_SCHEMA,
    )


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
