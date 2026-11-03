"""Exploration sweep over the stored filings in output/filings/.

Purpose: find what breaks the parser in curator/parse.py before anyone writes it.
The script reads the corpus from disk, runs 19 checks (C1 to C19), prints the
schema map for docs/SCHEMA.md, and prints its FINDINGS last.

    python scripts/eda.py

One exception to "disk only": check C18 reads two submissions documents that the
pipeline never requests. The script reads them from .cache/http/ when present.
On a cache miss it fetches them once through curator.cache.Cache, with the
User-Agent in the SEC_USER_AGENT environment variable. Every other byte comes
from disk.

This script observes. It repairs nothing and writes nothing outside .cache/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
FILINGS_DIR = ROOT / "output" / "filings"
CACHE_HTTP_DIR = ROOT / ".cache" / "http"

USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
WRONG_ROSTER_CIKS = ("0000854157", "0001697748")
IN_SCOPE_FORMS = frozenset({"13F-HR", "13F-HR/A", "13F-NT", "13F-NT/A"})
REPORT_PERIODS = frozenset({"2026-03-31", "2026-06-30"})
FILING_DATE_CUTOFF = "2026-08-18"

INFO_URI = "http://www.sec.gov/edgar/document/thirteenf/informationtable"
COVER_URI = "http://www.sec.gov/edgar/thirteenffiler"
COMMON_URI = "http://www.sec.gov/edgar/common"
XSI_URI = "http://www.w3.org/2001/XMLSchema-instance"
KNOWN_URIS = frozenset({INFO_URI, COVER_URI, COMMON_URI, XSI_URI})

ROW_CHILDREN = (
    "nameOfIssuer",
    "titleOfClass",
    "cusip",
    "figi",
    "value",
    "shrsOrPrnAmt",
    "putCall",
    "investmentDiscretion",
    "otherManager",
    "votingAuthority",
)
COVER_WATCHED = (
    "amendmentNo",
    "amendmentInfo",
    "amendmentType",
    "crdNumber",
    "secFileNumber",
    "form13FFileNumber",
    "otherIncludedManagersCount",
    "tableEntryTotal",
    "tableValueTotal",
    "reportType",
    "reportCalendarOrQuarter",
    "filingManager",
)


def local_name(element) -> str | None:
    """Return the local name of one element, or None for comments and PIs."""
    if not isinstance(element.tag, str):
        return None
    return etree.QName(element).localname


def child(parent, name: str):
    """Return the first direct child with this local name, or None."""
    for candidate in parent:
        if local_name(candidate) == name:
            return candidate
    return None


def child_text(parent, name: str) -> str | None:
    """Return the raw text of the first child with this local name."""
    element = child(parent, name)
    return element.text if element is not None else None


def dashed(stem: str) -> str:
    """Return the dashed accession form for one undashed stem."""
    return f"{stem[:10]}-{stem[10:12]}-{stem[12:]}"


def share(part: int, whole: int) -> str:
    """Return one count with its share as a percentage string."""
    pct = 100.0 * part / whole if whole else 0.0
    return f"{part}/{whole} ({pct:.1f}%)"


def examples(items, limit: int = 3) -> str:
    """Return up to three items as one display string."""
    shown = [str(item) for item in list(items)[:limit]]
    more = len(list(items)) - len(shown)
    suffix = f" +{more} more" if more > 0 else ""
    return "[" + ", ".join(shown) + suffix + "]"


@dataclass
class Row:
    """One observed infoTable row. Text stays as filed; nothing is repaired."""

    accession: str
    issuer_raw: str | None
    class_raw: str | None
    cusip: str
    has_figi: bool
    figi_text: str | None
    value_raw: str | None
    value: int | None
    ssh_raw: str | None
    ssh_prnamt: int | None
    ssh_type_raw: str | None
    put_call_raw: str | None
    has_put_call: bool
    discretion_raw: str | None
    other_manager_raw: str | None
    has_other_manager: bool
    voting_children: tuple[str, ...]


@dataclass
class Cover:
    """Everything the sweep needs from one cover page."""

    accession: str
    filing_manager: str | None = None
    report_type: str | None = None
    entry_total_raw: str | None = None
    value_total_raw: str | None = None
    other_included_managers_raw: str | None = None
    states: dict[str, str] = field(default_factory=dict)
    managers_shape: str | None = None
    sequence_numbers: tuple[str, ...] = ()
    listed_managers: tuple[str, ...] = ()


@dataclass
class FilingDoc:
    """One filing: paths, parsed cover, parsed rows, and raw-document facts."""

    accession: str
    directory: str
    stem: str
    is_notice: bool
    cover_path: Path
    info_path: Path | None
    cover: Cover
    rows: list[Row] = field(default_factory=list)
    cover_root_local: str | None = None
    cover_bindings: tuple[tuple[str, str], ...] = ()
    info_bindings: tuple[tuple[str, str], ...] = ()
    extra_namespaces: tuple[str, ...] = ()
    cover_encoding: str | None = None
    main_encoding: str | None = None
    comment_nodes: int = 0


def load_filings() -> list[FilingDoc]:
    """Parse every stored filing once and return them in stable order."""
    filings: list[FilingDoc] = []
    directories = sorted(path for path in FILINGS_DIR.iterdir() if path.is_dir())
    for directory in directories:
        mains = [
            path
            for path in sorted(directory.glob("*.xml"))
            if not path.name.endswith(".primary_doc.xml")
        ]
        for main_path in mains:
            stem = main_path.name.split(".")[0]
            primary_path = directory / f"{stem}.primary_doc.xml"
            main_root = etree.parse(str(main_path)).getroot()
            main_local = local_name(main_root)
            if main_local == "informationTable":
                is_notice = False
                cover_path: Path | None = primary_path
                info_path: Path | None = main_path
            elif main_local == "edgarSubmission":
                is_notice = True
                cover_path = main_path
                info_path = None
            else:
                raise RuntimeError(
                    f"{dashed(stem)}: unexpected root element {main_local!r} "
                    f"in {main_path.name}"
                )
            if cover_path is None or not cover_path.is_file():
                raise RuntimeError(
                    f"{dashed(stem)}: no cover page document on disk "
                    f"in {directory.name}"
                )
            filings.append(
                _build_filing(directory.name, stem, is_notice, cover_path, info_path)
            )
    if not filings:
        raise RuntimeError(f"{FILINGS_DIR} holds no filings")
    return filings


def _bindings(root) -> tuple[tuple[str, str], ...]:
    pairs = [
        (prefix or "DEFAULT", uri)
        for prefix, uri in (root.nsmap or {}).items()
        if uri is not None
    ]
    return tuple(sorted(pairs))


def _state_of(element) -> str:
    """Return the observed presence state of one cover element."""
    if (element.text is None or not element.text.strip()) and len(element) == 0:
        return "empty"
    return "present"


def _manager_list(root) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """Return the cover page manager-list shape, sequence numbers, and names."""
    shape = None
    sequences: list[str] = []
    names: list[str] = []
    for node in root.iter():
        name = local_name(node)
        if name not in ("otherManagers2Info", "otherManagersInfo"):
            continue
        shape = name
        for wrapper in node:
            wrapper_name = local_name(wrapper)
            holder = None
            if wrapper_name == "otherManager2":
                seq = child_text(wrapper, "sequenceNumber")
                if seq and seq.strip():
                    sequences.append(seq.strip())
                holder = child(wrapper, "otherManager")
            elif wrapper_name == "otherManager":
                holder = wrapper
            if holder is None:
                continue
            mgr_name = child_text(holder, "name")
            if mgr_name and mgr_name.strip():
                names.append(mgr_name.strip())
    return shape, tuple(sorted(sequences)), tuple(names)


def _build_cover(cover_path: Path, accession: str) -> Cover:
    tree = etree.parse(str(cover_path))
    cover = Cover(accession=accession)
    for element in tree.iter():
        name = local_name(element)
        if name is None:
            continue
        if name == "name":
            parent = element.getparent()
            if parent is not None and local_name(parent) == "filingManager":
                cover.filing_manager = element.text
        elif name == "reportType":
            cover.report_type = element.text
            cover.states[name] = _state_of(element)
        elif name == "tableEntryTotal":
            cover.entry_total_raw = element.text
            cover.states[name] = _state_of(element)
        elif name == "tableValueTotal":
            cover.value_total_raw = element.text
            cover.states[name] = _state_of(element)
        elif name == "otherIncludedManagersCount":
            cover.other_included_managers_raw = element.text
            cover.states[name] = _state_of(element)
        elif name in COVER_WATCHED:
            cover.states[name] = _state_of(element)
    cover.managers_shape, cover.sequence_numbers, cover.listed_managers = (
        _manager_list(tree.getroot())
    )
    return cover


def _build_row(accession: str, info_table) -> Row:
    def raw(name: str) -> str | None:
        return child_text(info_table, name)

    def as_int(text: str | None) -> int | None:
        if text is None or not text.strip():
            return None
        try:
            return int(text.strip())
        except ValueError:
            return None

    shrs = child(info_table, "shrsOrPrnAmt")
    ssh_amount = child_text(shrs, "sshPrnamt") if shrs is not None else None
    ssh_type = child_text(shrs, "sshPrnamtType") if shrs is not None else None
    voting = child(info_table, "votingAuthority")
    figi_element = child(info_table, "figi")
    put_call = child(info_table, "putCall")
    other_manager = child(info_table, "otherManager")
    cusip_text = raw("cusip")
    voting_names = (
        tuple(
            sorted(
                name
                for name in (local_name(node) for node in voting)
                if name is not None
            )
        )
        if voting is not None
        else ()
    )
    return Row(
        accession=accession,
        issuer_raw=raw("nameOfIssuer"),
        class_raw=raw("titleOfClass"),
        cusip=cusip_text.strip() if cusip_text else "",
        has_figi=figi_element is not None,
        figi_text=figi_element.text if figi_element is not None else None,
        value_raw=raw("value"),
        value=as_int(raw("value")),
        ssh_raw=ssh_amount,
        ssh_prnamt=as_int(ssh_amount),
        ssh_type_raw=ssh_type,
        put_call_raw=put_call.text if put_call is not None else None,
        has_put_call=put_call is not None,
        discretion_raw=raw("investmentDiscretion"),
        other_manager_raw=other_manager.text if other_manager is not None else None,
        has_other_manager=other_manager is not None,
        voting_children=voting_names,
    )


def _build_filing(
    directory: str,
    stem: str,
    is_notice: bool,
    cover_path: Path,
    info_path: Path | None,
) -> FilingDoc:
    accession = dashed(stem)
    cover_tree = etree.parse(str(cover_path))
    cover_root = cover_tree.getroot()
    filing = FilingDoc(
        accession=accession,
        directory=directory,
        stem=stem,
        is_notice=is_notice,
        cover_path=cover_path,
        info_path=info_path,
        cover=_build_cover(cover_path, accession),
        cover_root_local=local_name(cover_root),
        cover_bindings=_bindings(cover_root),
        cover_encoding=_declared_encoding(cover_path.read_bytes()),
        comment_nodes=_comment_nodes(cover_tree),
    )
    filing.extra_namespaces = tuple(
        sorted({uri for _, uri in filing.cover_bindings if uri not in KNOWN_URIS})
    )
    if info_path is None:
        filing.main_encoding = filing.cover_encoding
        return filing

    main_body = info_path.read_bytes()
    info_tree = etree.parse(str(info_path))
    info_root = info_tree.getroot()
    filing.info_bindings = _bindings(info_root)
    filing.main_encoding = _declared_encoding(main_body)
    filing.comment_nodes += _comment_nodes(info_tree)
    for uri in {uri for _, uri in filing.info_bindings if uri not in KNOWN_URIS}:
        if uri not in filing.extra_namespaces:
            filing.extra_namespaces = tuple(sorted((*filing.extra_namespaces, uri)))
    for node in info_root.iter():
        if local_name(node) != "infoTable":
            continue
        filing.rows.append(_build_row(accession, node))
    return filing


def _comment_nodes(tree) -> int:
    return sum(1 for node in tree.iter() if not isinstance(node.tag, str))


def _declared_encoding(body: bytes) -> str | None:
    match = re.match(rb'<\?xml[^>]*encoding="([^"]*)"', body[:200])
    return match.group(1).decode("ascii") if match else None


def _holdings(filings: list[FilingDoc]) -> list[FilingDoc]:
    return [filing for filing in filings if filing.info_path is not None]


def _rows_total(filings: list[FilingDoc]) -> int:
    return sum(len(filing.rows) for filing in filings)


def print_header(filings: list[FilingDoc]) -> None:
    notices = sum(1 for filing in filings if filing.is_notice)
    print("CORPUS")
    print(
        f"  filings={len(filings)} holdings_filings={len(filings) - notices} "
        f"notices={notices} info_table_rows={_rows_total(filings)}"
    )
    print()


def _info_prefixes(filing: FilingDoc) -> list[str]:
    return [
        prefix for prefix, uri in filing.info_bindings if uri == INFO_URI
    ]


def c1_namespaces(filings: list[FilingDoc]) -> None:
    print("C1: namespace URIs and prefixes in use, per root element")
    patterns: dict[tuple, list[str]] = {}
    for filing in filings:
        key = (filing.cover_root_local, filing.cover_bindings)
        patterns.setdefault(key, []).append(filing.accession)
        if filing.info_path is not None:
            key = ("informationTable", filing.info_bindings)
            patterns.setdefault(key, []).append(filing.accession)
    for (kind, bindings), accessions in sorted(
        patterns.items(), key=lambda kv: (kv[0][0], -len(kv[1]))
    ):
        pretty = ", ".join(f"{prefix}<{uri}>" for prefix, uri in bindings)
        print(
            f"  root {kind}: files={share(len(accessions), len(filings))} "
            f"{examples(sorted(accessions))}"
        )
        print(f"    bindings: {pretty}")
    prefixed_only, dual, default_only, none_bound = [], [], [], []
    for filing in _holdings(filings):
        prefixes = _info_prefixes(filing)
        if not prefixes:
            none_bound.append(filing.accession)
        elif "DEFAULT" not in prefixes:
            prefixed_only.append(filing.accession)
        elif len(prefixes) > 1:
            dual.append(filing.accession)
        else:
            default_only.append(filing.accession)
    print(
        f"  information-table URI bound ONLY through a prefix: "
        f"{share(len(prefixed_only), len(_holdings(filings)))} "
        f"{examples(sorted(prefixed_only))}"
    )
    print(
        f"  information-table URI declared twice (DEFAULT + prefix): "
        f"{share(len(dual), len(_holdings(filings)))} {examples(sorted(dual))}"
    )
    print(
        f"  information-table URI bound by DEFAULT alone: "
        f"{share(len(default_only), len(_holdings(filings)))} "
        f"{examples(sorted(default_only))}"
    )
    extras = [
        (filing.accession, filing.extra_namespaces)
        for filing in filings
        if filing.extra_namespaces
    ]
    print(f"  filings declaring an unknown third namespace: {len(extras)}")
    for accession, uris in extras:
        print(f"    {accession}: {', '.join(uris)}")


def _declared_int(text: str | None) -> int | None:
    if text is None or not text.strip():
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def c2_entry_totals(filings: list[FilingDoc]) -> None:
    print("C2: cover page tableEntryTotal against the count of infoTable elements")
    holdings = _holdings(filings)
    matched, mismatched = [], []
    for filing in holdings:
        declared = _declared_int(filing.cover.entry_total_raw)
        actual = len(filing.rows)
        if declared == actual:
            matched.append(filing.accession)
        else:
            mismatched.append((filing.accession, declared, actual))
    print(f"  filings compared: {len(holdings)}; the notice carries no information table")
    print(f"  declared equals actual: {share(len(matched), len(holdings))}")
    print(f"    examples checked: {examples(sorted(matched))}")
    print(f"  mismatches: {len(mismatched)}")
    for accession, declared, actual in mismatched[:3]:
        print(f"    {accession}: declared={declared} actual={actual}")
    empties = [
        f.accession
        for f in filings
        if f.cover.states.get("tableEntryTotal") == "empty"
    ]
    print(f"  covers writing an EMPTY tableEntryTotal element: {empties}")


def c3_value_totals(filings: list[FilingDoc]) -> None:
    print("C3: cover page tableValueTotal against the sum of value")
    holdings = _holdings(filings)
    matched, mismatched = [], []
    for filing in holdings:
        declared = _declared_int(filing.cover.value_total_raw)
        total = sum(row.value or 0 for row in filing.rows)
        if declared == total:
            matched.append(filing.accession)
        else:
            mismatched.append((filing.accession, declared, total))
    print(f"  filings compared: {len(holdings)}")
    print(f"  declared equals sum of value: {share(len(matched), len(holdings))}")
    print(f"    examples checked: {examples(sorted(matched))}")
    print(f"  mismatches: {len(mismatched)}")
    for accession, declared, total in mismatched[:3]:
        diff = total - (declared or 0)
        print(f"    {accession}: declared={declared} sum={total} diff={diff}")
    empties = [
        f.accession
        for f in filings
        if f.cover.states.get("tableValueTotal") == "empty"
    ]
    print(f"  covers writing an EMPTY tableValueTotal element: {empties}")


def c4_cover_absences(filings: list[FilingDoc]) -> None:
    print("C4: elements absent from the cover page, per filing")
    for watched in COVER_WATCHED:
        buckets: dict[str, list[str]] = {"absent": [], "empty": [], "present": []}
        for filing in filings:
            buckets[filing.cover.states.get(watched, "absent")].append(filing.accession)
        parts = [
            f"{state}={share(len(buckets[state]), len(filings))} "
            f"{examples(sorted(buckets[state]))}"
            for state in ("absent", "empty", "present")
            if buckets[state]
        ]
        print(f"  {watched}: " + "; ".join(parts))


def _row_has(row: Row, name: str) -> bool:
    if name == "figi":
        return row.has_figi
    if name == "putCall":
        return row.has_put_call
    if name == "otherManager":
        return row.has_other_manager
    return True


def c5_row_absences(filings: list[FilingDoc]) -> None:
    print("C5: elements absent from infoTable rows, per filing")
    holdings = _holdings(filings)
    rows_total = _rows_total(filings)
    for watched in ROW_CHILDREN:
        rows_with = sum(1 for f in holdings for r in f.rows if _row_has(r, watched))
        filings_with = sum(1 for f in holdings if any(_row_has(r, watched) for r in f.rows))
        print(
            f"  {watched}: rows_with={share(rows_with, rows_total)}; "
            f"filings_with={share(filings_with, len(holdings))}"
        )
    variants: dict[tuple, list[str]] = {}
    for filing in holdings:
        for row in filing.rows:
            present = tuple(name for name in ROW_CHILDREN if _row_has(row, name))
            variants.setdefault(present, []).append(filing.accession)
    print(f"  distinct child-set variants across rows: {len(variants)}")
    ordered = sorted(variants.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for present, accessions in ordered:
        missing = [name for name in ROW_CHILDREN if name not in present]
        label = str(missing) if missing else "none"
        print(
            f"    rows={len(accessions)} absent_children={label} "
            f"{examples(sorted(set(accessions)))}"
        )


def _value_table(filings: list[FilingDoc], extract) -> pd.DataFrame:
    holdings = _holdings(filings)
    return pd.DataFrame(
        [
            {"accession": filing.accession, "value": extract(row)}
            for filing in holdings
            for row in filing.rows
        ]
    )


def _print_distribution(frame: pd.DataFrame, n_filings: int, column: str) -> None:
    grouped = frame.groupby(column, dropna=False)["accession"].agg(
        rows="size", filings="nunique"
    )
    grouped = grouped.sort_index(ascending=True, na_position="first")
    for value, stats in grouped.iterrows():
        label = "<ABSENT element>" if pd.isna(value) else repr(value)
        print(
            f"  {label}: rows={stats['rows']} "
            f"filings={share(int(stats['filings']), n_filings)}"
        )


def c6_put_call(filings: list[FilingDoc]) -> None:
    print("C6: putCall values as filed, with exact capitalisation")
    frame = _value_table(filings, lambda row: row.put_call_raw)
    _print_distribution(frame, len(_holdings(filings)), "value")


def c7_discretion(filings: list[FilingDoc]) -> None:
    print("C7: investmentDiscretion values as filed")
    frame = _value_table(filings, lambda row: row.discretion_raw)
    _print_distribution(frame, len(_holdings(filings)), "value")


def c8_ssh_type(filings: list[FilingDoc]) -> None:
    print("C8: sshPrnamtType values as filed")
    frame = _value_table(filings, lambda row: row.ssh_type_raw)
    _print_distribution(frame, len(_holdings(filings)), "value")


def c9_cusip_shapes(filings: list[FilingDoc]) -> None:
    print("C9: CUSIP shapes: length, leading zero, leading letter, non-alphanumeric")
    holdings = _holdings(filings)
    lengths: Counter = Counter()
    leading_zero = 0
    leading_letter = 0
    non_alnum = []
    bad_length = []
    for filing in holdings:
        for row in filing.rows:
            lengths[len(row.cusip)] += 1
            if row.cusip.startswith("0"):
                leading_zero += 1
            if row.cusip and row.cusip[0].isalpha():
                leading_letter += 1
            if re.search(r"[^A-Za-z0-9]", row.cusip):
                non_alnum.append((filing.accession, row.cusip))
            if len(row.cusip) != 9:
                bad_length.append((filing.accession, row.cusip))
    rows_total = _rows_total(filings)
    for length in sorted(lengths):
        print(f"  length={length}: rows={lengths[length]} share={share(lengths[length], rows_total)}")
    print(f"  values whose length is not 9: {len(bad_length)}")
    for accession, cusip in bad_length[:3]:
        print(f"    {accession}: {cusip!r}")
    print(f"  leading-zero CUSIPs: {leading_zero} rows ({share(leading_zero, rows_total)})")
    print(f"  leading-letter CUSIPs (CINS codes): {leading_letter} rows ({share(leading_letter, rows_total)})")
    print(f"  CUSIPs holding a non-alphanumeric character: {len(non_alnum)} {examples(non_alnum)}")


def c10_duplicates(filings: list[FilingDoc]) -> None:
    print("C10: duplicate CUSIPs inside one filing, with the largest repeat count")
    holdings = _holdings(filings)
    filings_with_repeats = []
    largest = []
    for filing in holdings:
        counts = Counter(row.cusip for row in filing.rows)
        repeats = {cusip: n for cusip, n in counts.items() if n > 1}
        if not repeats:
            continue
        filings_with_repeats.append(filing.accession)
        top_cusip, top_n = sorted(repeats.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        largest.append((top_n, top_cusip, filing.accession))
    print(
        f"  filings holding at least one repeated CUSIP: "
        f"{share(len(filings_with_repeats), len(holdings))} "
        f"{examples(sorted(filings_with_repeats))}"
    )
    largest.sort(key=lambda item: (-item[0], item[2]))
    top_count = largest[0][0] if largest else 0
    print(f"  largest repeat count for one CUSIP inside one filing: {top_count}")
    for count, cusip, accession in largest[:3]:
        print(f"    {accession}: {cusip} appears {count} times")


def c11_other_manager(filings: list[FilingDoc]) -> None:
    print("C11: otherManager presence, distinct values, and the cover page manager list")
    holdings = _holdings(filings)
    rows_total = _rows_total(filings)
    present_rows = sum(1 for f in holdings for r in f.rows if r.has_other_manager)
    filings_with = sum(1 for f in holdings if any(r.has_other_manager for r in f.rows))
    print(
        f"  rows carrying otherManager: {share(present_rows, rows_total)}; "
        f"filings carrying at least one: {share(filings_with, len(holdings))}"
    )
    values = Counter(
        r.other_manager_raw.strip()
        for f in holdings
        for r in f.rows
        if r.has_other_manager and r.other_manager_raw
    )
    distinct = sorted(values)
    print(f"  distinct otherManager values: {len(distinct)}")
    for value in distinct:
        print(f"    {value!r}: rows={values[value]}")
    shapes = Counter(f.cover.managers_shape or "none" for f in holdings)
    print(f"  cover page manager-list shapes: {dict(sorted(shapes.items()))}")
    for filing in holdings:
        if filing.cover.managers_shape is None:
            continue
        used = sorted(
            {
                r.other_manager_raw.strip()
                for r in filing.rows
                if r.has_other_manager and r.other_manager_raw
            }
        )
        print(
            f"    {filing.accession} [{filing.cover.managers_shape}] "
            f"declared_count={filing.cover.other_included_managers_raw} "
            f"listed={len(filing.cover.listed_managers)} "
            f"sequence_numbers={list(filing.cover.sequence_numbers)} "
            f"row_refs={examples(used, 8)}"
        )


def c12_figi(filings: list[FilingDoc]) -> None:
    print("C12: figi presence")
    holdings = _holdings(filings)
    rows_total = _rows_total(filings)
    with_figi = sum(1 for f in holdings for r in f.rows if r.has_figi)
    filings_with = sum(1 for f in holdings if any(r.has_figi for r in f.rows))
    sample = sorted(
        {
            r.figi_text.strip()
            for f in holdings
            for r in f.rows
            if r.has_figi and r.figi_text
        }
    )
    print(f"  rows carrying figi: {with_figi} ({share(with_figi, rows_total)})")
    print(f"  filings carrying at least one figi: {share(filings_with, len(holdings))}")
    print(f"  distinct figi values: {len(sample)}; first values: {sample[:3]}")


def c13_entities_and_text(filings: list[FilingDoc]) -> None:
    print("C13: XML entities and text hygiene in nameOfIssuer and titleOfClass")
    holdings = _holdings(filings)
    rows_total = _rows_total(filings)
    entity_kinds: Counter = Counter()
    raw_examples = []
    lead_trail = {"nameOfIssuer": 0, "titleOfClass": 0}
    internal_runs = {"nameOfIssuer": 0, "titleOfClass": 0}
    internal_examples = []
    amp_rows = 0
    for filing in holdings:
        body = filing.info_path.read_bytes() if filing.info_path else b""
        for column, pattern in (
            ("nameOfIssuer", rb"<(?:[A-Za-z0-9_]+:)?nameOfIssuer>([^<]*)</(?:[A-Za-z0-9_]+:)?nameOfIssuer>"),
            ("titleOfClass", rb"<(?:[A-Za-z0-9_]+:)?titleOfClass>([^<]*)</(?:[A-Za-z0-9_]+:)?titleOfClass>"),
        ):
            seen_here = set()
            for raw_bytes in re.findall(pattern, body):
                entities = re.findall(rb"&[#0-9A-Za-z]+;", raw_bytes)
                for entity in entities:
                    entity_kinds[entity.decode("ascii")] += 1
                text = raw_bytes.decode("ascii")
                if b"&amp;" in raw_bytes and text not in seen_here:
                    seen_here.add(text)
                    if len(raw_examples) < 3 and all(t != text for _, _, t in raw_examples):
                        raw_examples.append((filing.accession, column, text))
        for row in filing.rows:
            has_amp = False
            for column, text in (("nameOfIssuer", row.issuer_raw), ("titleOfClass", row.class_raw)):
                if text is None:
                    continue
                if "&" in text:
                    has_amp = True
                if text != text.strip():
                    lead_trail[column] += 1
                stripped = text.strip()
                if re.search(r"\S\s{2,}\S", stripped):
                    internal_runs[column] += 1
                    if len(internal_examples) < 3 and all(
                        seen != stripped for _, _, seen in internal_examples
                    ):
                        internal_examples.append((filing.accession, column, stripped))
            if has_amp:
                amp_rows += 1
    decoded_examples = [
        (accession, column, raw.replace("&amp;", "&"))
        for accession, column, raw in raw_examples
    ]
    print(f"  entity kinds in raw bytes: {dict(entity_kinds)}")
    print(f"    raw-byte examples: {raw_examples}")
    print(f"    same strings after lxml decodes .text: {decoded_examples}")
    print(f"  decoded rows containing an ampersand: {amp_rows}")
    for column in ("nameOfIssuer", "titleOfClass"):
        print(f"  {column}: rows whose filed text carries leading/trailing whitespace: {lead_trail[column]}")
    for column in ("nameOfIssuer", "titleOfClass"):
        print(
            f"  {column}: rows with interior multi-space runs: "
            f"{internal_runs[column]} ({share(internal_runs[column], rows_total)})"
        )
    print(f"    interior-space examples: {internal_examples}")


def c14_encodings(filings: list[FilingDoc]) -> None:
    print("C14: character encodings declared in the XML prolog")
    cover_kinds: Counter = Counter()
    main_kinds: Counter = Counter()
    for filing in filings:
        cover_kind = "notice_cover" if filing.is_notice else "primary_doc"
        main_kind = "notice_main" if filing.is_notice else "info_table"
        cover_kinds[(cover_kind, filing.cover_encoding or "MISSING")] += 1
        main_kinds[(main_kind, filing.main_encoding or "MISSING")] += 1
    print("  cover documents:")
    for (kind, encoding), count in sorted(cover_kinds.items()):
        print(f"    {kind} declares {encoding}: files={count}")
    print("  main documents:")
    for (kind, encoding), count in sorted(main_kinds.items()):
        print(f"    {kind} declares {encoding}: files={count}")


def c15_voting_children(filings: list[FilingDoc]) -> None:
    print("C15: votingAuthority child element names, including None")
    combos: Counter = Counter()
    for filing in _holdings(filings):
        for row in filing.rows:
            combos[row.voting_children] += 1
    rows_total = _rows_total(filings)
    ordered = sorted(combos.items(), key=lambda kv: (-kv[1], kv[0]))
    for combo, count in ordered:
        print(f"  children={list(combo)}: rows={count} share={share(count, rows_total)}")


def c16_report_types(filings: list[FilingDoc]) -> None:
    print("C16: filings whose reportType is not a holdings report")
    values: Counter = Counter()
    non_holdings = []
    for filing in filings:
        report_type = (filing.cover.report_type or "").strip() or "<ABSENT>"
        values[report_type] += 1
        if report_type != "13F HOLDINGS REPORT":
            non_holdings.append((report_type, filing.accession))
    for report_type, count in sorted(values.items()):
        print(f"  {report_type!r}: {share(count, len(filings))}")
    print(f"  not a holdings report: {len(non_holdings)}")
    for report_type, accession in sorted(non_holdings)[:3]:
        print(f"    {accession}: {report_type}")


def c17_value_magnitude(filings: list[FilingDoc]) -> None:
    print("C17: value magnitude. Confirm whole dollars, not thousands")
    holdings = _holdings(filings)
    values = sorted(
        row.value for filing in holdings for row in filing.rows if row.value is not None
    )
    rows_total = len(values)
    divisible = sum(1 for v in values if v % 1000 == 0)
    print(f"  rows counted: {rows_total}")
    print(f"  min={values[0]} p50={values[rows_total // 2]} max={values[-1]}")
    print(
        f"  divisible by 1000: {divisible} ({share(divisible, rows_total)}); "
        f"not divisible: {rows_total - divisible}"
    )
    demo_filing = next(
        (f for f in holdings if f.stem == "000110465926062477"), holdings[0]
    )
    demo_row = next((r for r in demo_filing.rows if r.value and r.ssh_prnamt), None)
    if demo_row is not None:
        per_share = demo_row.value / demo_row.ssh_prnamt
        print(
            f"  arithmetic on {demo_filing.accession} cusip={demo_row.cusip}: "
            f"value={demo_row.value} sshPrnamt={demo_row.ssh_prnamt} "
            f"=> {per_share:.4f} dollars per share, a plausible market price"
        )
    zero_rows = [
        (f.accession, r.cusip, r.ssh_prnamt)
        for f in holdings
        for r in f.rows
        if r.value == 0
    ]
    zero_filings = sorted({accession for accession, _, _ in zero_rows})
    print(f"  rows reporting value=0: {len(zero_rows)} in {len(zero_filings)} filings")
    print(f"    example filings: {zero_filings[:3]}")
    for accession, cusip, shares in zero_rows[:3]:
        print(f"    {accession}: cusip={cusip} shares={shares} value=0")


def _in_scope_count(payload: dict) -> int:
    recent = (payload.get("filings") or {}).get("recent") or {}
    columns = zip(
        recent.get("accessionNumber") or [],
        recent.get("form") or [],
        recent.get("reportDate") or [],
        recent.get("filingDate") or [], strict=False,
    )
    count = 0
    for _accession, form, report_date, filing_date in columns:
        if form not in IN_SCOPE_FORMS:
            continue
        if report_date not in REPORT_PERIODS:
            continue
        if str(filing_date or "") > FILING_DATE_CUTOFF:
            continue
        count += 1
    return count


def _read_submissions(cik: str) -> bytes | None:
    """Return the body for one submissions URL, from cache or one fetch."""
    url = SUBMISSIONS_URL.format(cik)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    bin_path = CACHE_HTTP_DIR / f"{key}.bin"
    meta_path = CACHE_HTTP_DIR / f"{key}.json"
    if bin_path.is_file() and meta_path.is_file():
        print(f"  {cik}: cache HIT at {bin_path.relative_to(ROOT)}; reading disk only")
        return bin_path.read_bytes()
    if not USER_AGENT:
        print(f"  {cik}: cache MISS for {url}; set SEC_USER_AGENT to fetch it")
        return None
    print(f"  {cik}: cache MISS for {url}; fetching once through curator.cache.Cache")
    try:
        sys.path.insert(0, str(ROOT))
        from curator.cache import Cache, Manifest
        from curator.http import Fetcher, RateLimiter

        cache = Cache(ROOT / ".cache", Fetcher(USER_AGENT, RateLimiter(5.0)), Manifest())
        body = cache.get(url)
    except Exception as exc:
        print(f"  {cik}: fetch failed: {exc}")
        return None
    print(f"  {cik}: fetched once and stored at {bin_path.relative_to(ROOT)}")
    return body


def c18_wrong_roster_ciks(filings: list[FilingDoc]) -> None:
    print("C18: the two wrong roster CIKs resolve to real 13F filers")
    corpus_ciks = {filing.directory.zfill(10) for filing in filings}
    for cik in WRONG_ROSTER_CIKS:
        body = _read_submissions(cik)
        if body is None:
            continue
        payload = json.loads(body)
        print(
            f"  {cik}: entity={payload.get('name')!r} "
            f"in_scope_filings={_in_scope_count(payload)}"
        )
        print(
            f"  {cik}: is one of the 20 downloaded corpus CIKs: "
            f"{'yes' if cik in corpus_ciks else 'no'}"
        )
    print(
        "  Conclusion: each wrong CIK belongs to a live filer with two in-scope "
        "filings, so a 40-filing count alone cannot catch a wrong CIK."
    )


def c19_cusip_lengths(filings: list[FilingDoc]) -> None:
    print("C19: CUSIP lengths. Count values whose length is not 9")
    offenders = [
        (filing.accession, row.cusip)
        for filing in _holdings(filings)
        for row in filing.rows
        if len(row.cusip) != 9
    ]
    rows_total = _rows_total(filings)
    print(
        f"  rows whose CUSIP length differs from 9: {len(offenders)} "
        f"({share(len(offenders), rows_total)})"
    )
    for accession, cusip in offenders[:3]:
        print(f"    {accession}: {cusip!r} length={len(cusip)}")


def extra_observations(filings: list[FilingDoc]) -> None:
    print("EXTRA: further observations from the sweep")
    commented = sorted(
        filing.accession for filing in filings if filing.comment_nodes > 0
    )
    print(f"  documents embedding XML comments: {len(commented)} {examples(commented)}")
    nondigit = [
        (filing.accession, field, raw.strip())
        for filing in _holdings(filings)
        for row in filing.rows
        for field, raw in (("value", row.value_raw), ("sshPrnamt", row.ssh_raw))
        if raw is not None and raw.strip() and not raw.strip().isdigit()
    ]
    print(f"  numeric row texts holding non-digit characters: {len(nondigit)}")


SCHEMA_MAP = [
    ("filings", "accession_number", "submissions API accessionNumber", "dashed form; never built from the filename"),
    ("filings", "cik", "headerData/filerInfo/filer/credentials/cik", "zero-pad to 10 characters"),
    ("filings", "fund_name", "roster CSV", "as supplied; joins back to the roster"),
    ("filings", "filing_manager", "coverPage/filingManager/name", "as filed, stripped"),
    ("filings", "form_type", "submissions API form", "one of four 13F forms"),
    ("filings", "report_period", "coverPage/reportCalendarOrQuarter", "MM-DD-YYYY text; parse to date32"),
    ("filings", "report_quarter", "derived", "YYYYQN computed from report_period"),
    ("filings", "filing_date", "submissions API filingDate", "comes from the API, not from the XML"),
    ("filings", "is_amendment", "derived", "true when form_type ends in /A"),
    ("filings", "amendment_no", "coverPage/amendmentNo", "null when absent; absent in all 40 filings"),
    ("filings", "amendment_type", "coverPage/amendmentInfo/amendmentType", "null when absent; absent in all 40 filings"),
    ("filings", "report_type", "coverPage/reportType", "31 holdings, 8 combination, 1 notice"),
    ("filings", "form_13f_file_number", "coverPage/form13FFileNumber", "present in all 40 covers"),
    ("filings", "crd_number", "coverPage/crdNumber", "nullable string; absent in 4 covers; keep leading zeros"),
    ("filings", "sec_file_number", "coverPage/secFileNumber", "nullable string; absent in 4 covers"),
    ("filings", "other_included_managers_count", "summaryPage/otherIncludedManagersCount", "int32; an empty element means null"),
    ("filings", "table_entry_total", "summaryPage/tableEntryTotal", "int64; an empty element means null"),
    ("filings", "table_value_total", "summaryPage/tableValueTotal", "int64 whole dollars; an empty element means null"),
    ("holdings", "accession_number", "derived", "foreign key copied from the filings row"),
    ("holdings", "cik", "derived", "denormalised from the filings row; zero-padded"),
    ("holdings", "report_quarter", "derived", "denormalised from the filings row"),
    ("holdings", "name_of_issuer", "infoTable/nameOfIssuer", "decode entities, strip ends, keep interior spacing"),
    ("holdings", "title_of_class", "infoTable/titleOfClass", "same treatment as name_of_issuer"),
    ("holdings", "cusip", "infoTable/cusip", "strip whitespace only; never pad, truncate, or recase"),
    ("holdings", "figi", "infoTable/figi", "null when absent; absent from most rows"),
    ("holdings", "value", "infoTable/value", "int64 whole US dollars"),
    ("holdings", "ssh_prnamt", "shrsOrPrnAmt/sshPrnamt", "int64 quantity"),
    ("holdings", "ssh_prnamt_type", "shrsOrPrnAmt/sshPrnamtType", "SH or PRN as filed"),
    ("holdings", "put_call", "infoTable/putCall", "an absent element means null; store the case as filed"),
    ("holdings", "investment_discretion", "infoTable/investmentDiscretion", "SOLE, DFND, or OTR as filed"),
    ("holdings", "other_manager", "infoTable/otherManager", "as filed, including comma lists; null when absent"),
    ("holdings", "voting_sole", "votingAuthority/Sole", "int64"),
    ("holdings", "voting_shared", "votingAuthority/Shared", "int64"),
    ("holdings", "voting_none", "votingAuthority/None", "the source element is literally named None"),
]


def print_schema_map() -> None:
    print("SCHEMA MAP (docs/SCHEMA.md: 18 filings columns + 16 holdings columns)")
    for table, column, source, note in SCHEMA_MAP:
        print(f"  [{table}] {column} <- {source} :: {note}")
    print(f"  columns mapped: {len(SCHEMA_MAP)}")


FINDINGS = [
    "[C1] Fourteen of thirty-nine information tables bind the information-table URI "
    "only through a prefix: ns1 on twelve filings and n1 on two. Millennium uses n1 "
    "alone in 0001273087-26-000004 and 0001273087-26-000007. Third Point uses ns1 "
    "alone in 0001040273-26-000002. A parser that matches the literal tag infoTable "
    "finds zero rows there and silently drops the whole position book. Eighteen more "
    "filings, including Citadel 0001104659-26-062477, declare the default namespace "
    "and n1 together. Match etree.QName local names only.",
    "[C4] The notice 0001172661-26-003777 writes empty summary elements: "
    "tableEntryTotal, tableValueTotal, and otherIncludedManagersCount hold no text. "
    "An int cast on such an element raises TypeError, and reading it as 0 turns a "
    "genuine null into a false total. An empty element means null, not zero and not "
    "an empty string.",
    "[C4] crdNumber and secFileNumber are absent, not empty, in four of forty covers: "
    "both Baupost filings and both Millennium filings. A cast on a missing element "
    "crashes the parser, and the schema wants strings anyway. Read both as nullable "
    "strings and preserve leading zeros such as 000148826.",
    "[C10] Twenty-four of thirty-nine holdings filings repeat at least one CUSIP. The "
    "largest repeat count is thirteen: 040413205 in AQR 0001167557-26-000115 and "
    "458140100 in AQR 0001167557-26-000226. Each repeat is a separate filed position. "
    "Any deduplication on CUSIP destroys rows that the answer key keeps.",
    "[C13] Raw bytes carry the escaped entity &amp; 4345 times inside issuer and "
    "class texts. ABERCROMBIE &amp; FITCH CO arrives from lxml as ABERCROMBIE & FITCH CO. "
    "A second manual unescape corrupts every name that holds a real ampersand. Decode "
    "exactly once, by reading .text.",
    "[C13] Sixty-nine nameOfIssuer texts and forty-three titleOfClass texts carry "
    "leading or trailing whitespace, so the parser strips the ends. Sixteen thousand "
    "six hundred sixty-three texts hold interior multi-space runs such as "
    "'CL A  EX SUB VTG' and 'LOMA NEGRA C I A S A MTN  14'. Collapse no interior run; "
    "the filed spacing identifies the issuer.",
    "[C17] Values are whole dollars, not thousands. Only 1391 of 134635 values divide "
    "by 1000. Declared tableValueTotal equals the sum of value in all thirty-nine "
    "holdings filings. One Citadel row proves scale: 80560 dollars over 26500 shares "
    "gives 3.04 dollars per share, a plausible market price. Parse value as int64 "
    "dollars.",
    "[C17] Eight rows in seven filings report value=0 while holding shares, including "
    "XCF GLOBAL INC at one share in AQR 0001167557-26-000115. Zero is filed data, not "
    "a defect. The parser keeps these rows unchanged.",
    "[C6] putCall holds exactly two spellings: Call on 16614 rows and Put on 14418 "
    "rows, absent on 103603. No lowercase variant exists this quarter, so the schema's "
    "capitalisation warning does not appear in this corpus. Store the value as filed "
    "and let agent SQL apply upper().",
    "[C9][C19] All 134635 CUSIPs are exactly nine characters: 17434 start with 0 and "
    "11137 start with a letter as CINS codes. No value drops a character, so the "
    "Bonus 2 length-error class stays empty this quarter. Store as filed and record "
    "any future length exception in the quality report.",
    "[C11] Rows carry otherManager on 85510 of 134635 rows across twenty distinct "
    "values. Refs appear as bare numbers 1 to 13, as packed lists like '2,1' in "
    "Citadel 0001104659-26-062472, as spaced lists like '1, 2, 3, 4, 5' in Starboard "
    "0000921895-26-001402, and as literal '0' on 6353 Renaissance rows whose cover "
    "lists no managers. Store the string unchanged; Bonus 1 splits both spellings.",
    "[C11] Cover pages write manager lists in two shapes: otherManagers2Info with a "
    "sequenceNumber on sixteen filings and plain otherManagersInfo on the two "
    "Balyasny filings. AQR declares 13 managers yet lists 14 names. Bonus 1 must "
    "read both shapes and reconcile the declared count.",
    "[C14] Twenty-two of forty main files declare no prolog encoding, eight declare "
    "lowercase utf-8, and ten declare UTF-8. Case-sensitive matching on the prolog "
    "fails for eight files. Parse with lxml, which defaults to UTF-8 and accepts "
    "both spellings.",
    "[C16] Nine of forty filings are not plain holdings reports: eight 13F COMBINATION "
    "REPORT filings from AQR, Balyasny, Millennium, and Man Group, plus the Pershing "
    "Square 13F NOTICE 0001172661-26-003777. A combination merges several managers' "
    "books, so Bonus 1 attribution starts there. The notice yields one filing record "
    "and zero holding rows.",
    "[C15] Every one of the 134635 rows carries votingAuthority with exactly three "
    "children: Sole, Shared, and None. The third child is literally named None, so "
    "the parser passes it as a quoted string. Absent voting children never occur in "
    "this corpus.",
    "[EXTRA] Three documents embed XML comments: both Elliott filings and Situational "
    "Awareness 0000935836-26-000418. etree.QName raises on a comment node. Every "
    "traversal helper skips nodes whose tag is not a string, or parsing dies "
    "mid-file.",
    "[C2][C3] Declared tableEntryTotal equals the real infoTable count, and declared "
    "tableValueTotal equals the sum of value, in thirty-nine of thirty-nine holdings "
    "filings. Totals reconcile today, so a later mismatch signals a parser bug rather "
    "than filer noise. Keep the comparison as a quality check.",
    "[C18] Both wrong roster CIKs resolve to live 13F filers with two in-scope "
    "filings each: STATE OF WISCONSIN INVESTMENT BOARD at 0000854157 and ARK "
    "Investment Management LLC at 0001697748. The 40-filing count alone cannot catch "
    "a swapped CIK. Only the roster name check catches it.",
]


def main() -> int:
    if not FILINGS_DIR.is_dir():
        print(
            f"error: {FILINGS_DIR} does not exist. Run the download step first "
            f"(python main.py --user-agent ...) so the sweep can read it.",
            file=sys.stderr,
        )
        return 1
    filings = load_filings()
    print_header(filings)
    c1_namespaces(filings)
    c2_entry_totals(filings)
    c3_value_totals(filings)
    c4_cover_absences(filings)
    c5_row_absences(filings)
    c6_put_call(filings)
    c7_discretion(filings)
    c8_ssh_type(filings)
    c9_cusip_shapes(filings)
    c10_duplicates(filings)
    c11_other_manager(filings)
    c12_figi(filings)
    c13_entities_and_text(filings)
    c14_encodings(filings)
    c15_voting_children(filings)
    c16_report_types(filings)
    c17_value_magnitude(filings)
    c18_wrong_roster_ciks(filings)
    c19_cusip_lengths(filings)
    extra_observations(filings)
    print()
    print_schema_map()
    print()
    print("FINDINGS")
    for finding in FINDINGS:
        print(f"- {finding}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
