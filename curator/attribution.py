"""Attribution: recover the notice filer's rows from its parent's filing.

A 13F notice carries no holdings. It says the holdings sit in another manager's
report. This module finds that report, finds the sequence number the parent
uses for the notice filer, and selects the information table rows that carry it.

Nothing here is hard-coded. The parent accession, the parent CIK, and the
sequence number all come out of the filings themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
from lxml import etree

from curator.cache import Cache
from curator.discover import filing_frames
from curator.download import download
from curator.parse import ParsedDataset, QualityReport, child_text, parse_filing
from curator.records import FilingRef
from curator.schema import ATTRIBUTED_SCHEMA
from curator.scope import Scope
from curator.write import write_table_atomic

_PARENT_FORMS = ("13F-HR", "13F-HR/A")


def _local(element) -> str | None:
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def _find_all(root, local_name: str) -> list:
    """Every element with this local name, whatever namespace declares it."""
    return [node for node in root.iter() if _local(node) == local_name]


def _tokens(value: str | None) -> list[str]:
    """Split an otherManager list into whole tokens.

    The value reads like `1, 2, 3, 4`. Splitting on a comma and stripping each
    piece keeps `1` from matching `11`.
    """
    if not value:
        return []
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def _digits(value: str | None) -> str:
    """Return the CIK without leading zeros, for comparison."""
    return (value or "").strip().lstrip("0")


def _notice_parents(cover_path: Path) -> list[dict[str, str | None]]:
    """Every otherManager the notice cover page names."""
    root = etree.parse(str(cover_path)).getroot()
    managers: list[dict[str, str | None]] = []
    for info in _find_all(root, "otherManagersInfo"):
        for manager in _find_all(info, "otherManager"):
            managers.append(
                {
                    "cik": child_text(manager, "cik"),
                    "file_number": child_text(manager, "form13FFileNumber"),
                    "name": child_text(manager, "name"),
                }
            )
    return managers


def _sequence_map(cover_path: Path) -> dict[str, dict[str, str | None]]:
    """Map each sequence number in the parent cover page to its manager."""
    root = etree.parse(str(cover_path)).getroot()
    mapping: dict[str, dict[str, str | None]] = {}
    for block in _find_all(root, "otherManagers2Info"):
        for manager in _find_all(block, "otherManager2"):
            sequence = child_text(manager, "sequenceNumber")
            if not sequence:
                continue
            # The identity sits one level deeper: otherManager2 holds the
            # sequence number, and a nested otherManager holds the CIK.
            inner = _find_all(manager, "otherManager")
            source = inner[0] if inner else manager
            mapping[sequence.strip()] = {
                "cik": child_text(source, "cik"),
                "file_number": child_text(source, "form13FFileNumber"),
                "name": child_text(source, "name"),
            }
    return mapping


def _find_parent_ref(
    cache: Cache, cik: str, report_date: str, scope: Scope
) -> FilingRef | None:
    """Find the parent's 13F-HR for the same period through the submissions API.

    The search reads every page of the parent's filing history that can hold the
    period, as discovery does, so an old parent with more than 1000 filings is
    still found.
    """
    frames, name = filing_frames(cache, cik, scope)
    for frame in frames:
        columns = zip(
            frame.get("accessionNumber") or [],
            frame.get("form") or [],
            frame.get("reportDate") or [],
            frame.get("filingDate") or [],
            frame.get("primaryDocument") or [],
            strict=False,
        )
        for accession, form, reported, filed, primary in columns:
            if form not in _PARENT_FORMS or reported != report_date:
                continue
            if not filed or filed > scope.cutoff:
                continue
            return FilingRef(
                cik=f"{int(cik):010d}",
                fund_name=name,
                accession=accession,
                form=form,
                report_date=reported,
                filing_date=filed,
                primary_document=primary or "",
            )
    return None


def run_attribution(dataset: ParsedDataset, cache: Cache, output: Path, scope: Scope) -> None:
    """Attribute the notice filer's rows out of the parent filing."""
    output = Path(output)
    # The parent is not in the input list, so its documents must not land under
    # output/filings/, which holds the listed filers only.
    work = Path(cache.root) / "attribution"
    work.mkdir(parents=True, exist_ok=True)

    attributed: list[pa.Table] = []
    quality = QualityReport()

    notices = [
        f for f in dataset.filings.to_pylist() if str(f["form_type"]).startswith("13F-NT")
    ]
    print(f"curator.attribution: {len(notices)} notice filing(s)", file=sys.stderr)

    for notice in notices:
        notice_cik = str(notice["cik"])
        accession = str(notice["accession_number"])
        stem = accession.replace("-", "")
        cover = output / "filings" / str(int(notice_cik)) / f"{stem}.primary_doc.xml"
        if not cover.exists():
            cover = output / "filings" / str(int(notice_cik)) / f"{stem}.xml"
        if not cover.exists():
            print(f"curator.attribution: no cover page for {accession}", file=sys.stderr)
            continue

        for manager in _notice_parents(cover):
            parent_cik = (manager.get("cik") or "").strip()
            if not parent_cik:
                continue
            ref = _find_parent_ref(cache, parent_cik, str(notice["report_period"]), scope)
            if ref is None:
                print(
                    f"curator.attribution: no parent 13F-HR for CIK {parent_cik}",
                    file=sys.stderr,
                )
                continue

            docs = download([ref], cache, work)[0]
            parsed = parse_filing(docs, quality)
            sequences = _sequence_map(docs.primary_doc_path)

            wanted = [
                number
                for number, entry in sequences.items()
                if _digits(entry.get("cik")) == _digits(notice_cik)
                or (
                    entry.get("file_number")
                    and entry.get("file_number") == manager.get("file_number")
                )
            ]
            print(
                f"curator.attribution: parent {ref.accession} declares {len(sequences)} "
                f"manager(s); the notice maps to sequence {wanted}",
                file=sys.stderr,
            )
            if not wanted:
                continue

            sequence = wanted[0]
            carries = [
                sequence in _tokens(value)
                for value in parsed.holdings.column("other_manager").to_pylist()
            ]
            matched = parsed.holdings.filter(pa.array(carries, pa.bool_()))
            print(
                f"curator.attribution: the parent holds {parsed.holdings.num_rows} row(s); "
                f"{matched.num_rows} carry sequence {sequence}",
                file=sys.stderr,
            )
            if not matched.num_rows:
                # A justified zero beats an unjustified set of rows.
                print(
                    f"curator.attribution: sequence {sequence} appears in no otherManager "
                    "value, so this module attributes no row",
                    file=sys.stderr,
                )
                continue

            owner = pa.array([f"{int(notice_cik):010d}"] * matched.num_rows, pa.string())
            attributed.append(
                matched.append_column(ATTRIBUTED_SCHEMA.field("attributed_to_cik"), owner)
            )

    table = (
        pa.concat_tables(attributed) if attributed else ATTRIBUTED_SCHEMA.empty_table()
    )
    write_table_atomic(table, output / "attributed_holdings.parquet")
    print(f"curator.attribution: wrote {table.num_rows} attributed row(s)", file=sys.stderr)
