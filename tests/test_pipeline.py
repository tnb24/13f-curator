"""The whole pipeline, end to end, over the synthetic EDGAR in tests/world.py.

No test here touches the network. Each run writes into tmp_path only.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from curator.pipeline import run
from curator.schema import ATTRIBUTED_SCHEMA, FILINGS_SCHEMA, HOLDINGS_SCHEMA
from tests import world

OUTPUTS = (
    "filings.parquet",
    "holdings.parquet",
    "attributed_holdings.parquet",
    "cusip_check.csv",
    "filers.csv",
    "quality_report.json",
)


def _run(root: Path, name: str, cache: Path | None = None, workers: int = 1):
    source = world.FakeSource(world.bodies())
    output = root / name
    run(
        "Test User test@example.com",
        output,
        cache_dir=cache or root / f"{name}-cache",
        fetcher=source,
        roster=world.write_roster(root),
        workers=workers,
    )
    return output, source


def _digests(directory: Path) -> dict[str, str]:
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in OUTPUTS}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    output, source = _run(root, "first")
    return root, output, source


def test_every_table_matches_its_schema(built) -> None:
    _, output, _ = built
    assert pq.read_schema(output / "filings.parquet").equals(FILINGS_SCHEMA)
    assert pq.read_schema(output / "holdings.parquet").equals(HOLDINGS_SCHEMA)
    assert pq.read_schema(output / "attributed_holdings.parquet").equals(ATTRIBUTED_SCHEMA)


def test_an_input_cik_that_names_another_filer_is_corrected(built) -> None:
    _, output, _ = built
    with (output / "filers.csv").open(newline="") as handle:
        rows = {row["fund_name"]: row for row in csv.DictReader(handle)}
    assert rows["Alpha Capital LLC"]["cik_source"] == "given"
    # The roster says 1000099. Two lookup entries share the name, and only
    # 1000002 filed a 13F, so the /ADV entry 1000098 must lose.
    assert rows["Beta Partners LP"] == {
        "fund_name": "Beta Partners LP", "cik": "1000002", "cik_source": "corrected",
    }


def test_filings_hold_every_quarter_and_the_notice(built) -> None:
    _, output, _ = built
    filings = pq.read_table(output / "filings.parquet").to_pylist()
    assert [(f["fund_name"], f["report_quarter"], f["form_type"]) for f in filings] == [
        ("Alpha Capital LLC", "2026Q1", "13F-HR"),
        ("Alpha Capital LLC", "2026Q2", "13F-HR"),
        ("Beta Partners LP", "2026Q1", "13F-HR"),
        ("Beta Partners LP", "2026Q2", "13F-HR"),
        ("Gamma Advisors Inc", "2026Q1", "13F-HR"),
        ("Gamma Advisors Inc", "2026Q2", "13F-NT"),
    ]
    notice = filings[-1]
    assert notice["table_entry_total"] is None
    assert notice["report_type"] == "13F NOTICE"


def test_holdings_keep_filing_order_then_source_order(built) -> None:
    _, output, _ = built
    holdings = pq.read_table(output / "holdings.parquet").to_pylist()
    assert len(holdings) == 11
    alpha_q2 = [h for h in holdings if h["accession_number"] == "0001000001-26-000002"]
    assert [(h["cusip"], h["put_call"]) for h in alpha_q2] == [
        ("037833100", None), ("037833100", "Call"), ("000360206", None),
    ]
    # The prefixed namespace (ns1:) parses exactly like the default one.
    assert alpha_q2[0]["name_of_issuer"] == "APPLE INC"
    assert alpha_q2[0]["voting_sole"] == 12


def test_attribution_takes_whole_sequence_tokens_only(built) -> None:
    _, output, _ = built
    rows = pq.read_table(output / "attributed_holdings.parquet").to_pylist()
    # Sequence 1 matches "1" and "2, 1". It must not match "11".
    assert [(r["name_of_issuer"], r["other_manager"]) for r in rows] == [
        ("APPLE INC", "1"), ("ACME WIDGET CO", "2, 1"),
    ]
    assert {r["attributed_to_cik"] for r in rows} == {"0001000003"}
    # The parent is not in the input list, so its documents stay out of output/filings.
    assert not (output / "filings" / "1000004").exists()


def test_the_cusip_check_labels_every_unmatched_cusip(built) -> None:
    _, output, _ = built
    with (output / "cusip_check.csv").open(newline="") as handle:
        rows = {row["cusip"]: row for row in csv.DictReader(handle)}
    assert rows["037833100"]["on_official_list"] == "True"
    assert rows["037833100"]["rows"] == "2"  # the share row and the call row
    assert rows["037833101"]["assessment"] == "LIKELY_FILER_ERROR"
    assert rows["123456782"]["assessment"] == "TIMING"
    assert rows["G3922B107"]["assessment"] == "CINS_FOREIGN"


def test_a_second_run_reads_every_body_from_the_cache(tmp_path) -> None:
    cache = tmp_path / "cache"
    first, source = _run(tmp_path, "first", cache)
    second, again = _run(tmp_path, "second", cache)
    assert source.requested
    assert again.requested == []
    assert _digests(first) == _digests(second)
    manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))["run"]
    assert manifest["cache_misses"] == 0
    assert manifest["requests"] == manifest["cache_hits"] > 0


def test_parallel_parse_writes_the_same_bytes(tmp_path) -> None:
    """The process pool must not change a byte: results return in input order."""
    serial, _ = _run(tmp_path, "serial", workers=1)
    parallel, _ = _run(tmp_path, "parallel", workers=2)
    assert _digests(serial) == _digests(parallel)


def test_stdout_stays_empty(tmp_path, capsys) -> None:
    _run(tmp_path, "quiet")
    assert capsys.readouterr().out == ""
