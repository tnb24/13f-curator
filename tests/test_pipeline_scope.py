"""The pipeline with other scopes, over the synthetic EDGAR in tests/world.py.

The default scope is covered in tests/test_pipeline.py. Here the quarters and
the cutoff change: one quarter, three quarters, an older page, a missing list.
No test touches the network. Each run writes into tmp_path only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from curator.pipeline import run
from curator.scope import Scope
from tests import world

ONE = ("2025Q4",)
ONE_CUTOFF = "2026-02-17"
THREE = ("2026Q1", "2026Q2", "2026Q3")
THREE_CUTOFF = "2026-11-16"
ROOT = Path(__file__).resolve().parents[1]


def _build(root: Path, name: str, served: dict[str, bytes], scope: Scope):
    source = world.FakeSource(served)
    output = root / name
    run(
        "Test User test@example.com",
        output,
        cache_dir=root / f"{name}-cache",
        fetcher=source,
        roster=world.write_roster(root),
        workers=1,
        scope=scope,
    )
    return output, source


def _filings(output: Path) -> list[tuple[str, str, str]]:
    rows = pq.read_table(output / "filings.parquet").to_pylist()
    return [(r["fund_name"], r["report_quarter"], r["form_type"]) for r in rows]


def _manifest(output: Path) -> dict:
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_the_default_scope_goes_into_the_manifest(tmp_path) -> None:
    output, _ = _build(tmp_path, "default", world.bodies(), Scope())
    assert _manifest(output)["scope"] == {
        "quarters": ["2026Q1", "2026Q2"],
        "report_periods": ["2026-03-31", "2026-06-30"],
        "filing_date_cutoff": "2026-08-18",
    }
    assert list(_manifest(output)) == ["run", "scope", "stages", "entries"]
    assert _manifest(output)["stages"] == {"attribution": True, "cusip_check": True}


def test_a_run_without_a_scope_uses_the_default(tmp_path) -> None:
    source = world.FakeSource(world.bodies())
    run("Test User test@example.com", tmp_path / "out", cache_dir=tmp_path / "c",
        fetcher=source, roster=world.write_roster(tmp_path), workers=1)
    assert _manifest(tmp_path / "out")["scope"]["quarters"] == ["2026Q1", "2026Q2"]


def test_one_quarter_builds_one_filing_for_each_filer(tmp_path, capsys) -> None:
    scope = Scope.parse(",".join(ONE), ONE_CUTOFF)
    output, source = _build(tmp_path, "one", world.bodies(ONE), scope)

    assert _filings(output) == [
        ("Alpha Capital LLC", "2025Q4", "13F-HR"),
        ("Beta Partners LP", "2025Q4", "13F-HR"),
        ("Gamma Advisors Inc", "2025Q4", "13F-NT"),
    ]
    assert len(pq.read_table(output / "holdings.parquet")) == 7
    assert "warning" not in capsys.readouterr().err  # one filing each, as one quarter expects
    manifest = _manifest(output)
    assert manifest["scope"]["quarters"] == ["2025Q4"]
    assert manifest["scope"]["report_periods"] == ["2025-12-31"]
    assert manifest["scope"]["filing_date_cutoff"] == ONE_CUTOFF

    # The check reads the list of the last quarter and the list of the quarter before it.
    lists = [url for url in source.requested if "/files/investment/" in url]
    assert lists == [world.list_url("2025Q4"), world.list_url("2025Q3")]
    with (output / "cusip_check.csv").open(newline="") as handle:
        rows = {row["cusip"]: row for row in csv.DictReader(handle)}
    assert rows["123456782"]["assessment"] == "TIMING"  # on the 2025Q3 list only
    # The notice attributes the parent's rows from the same quarter.
    attributed = pq.read_table(output / "attributed_holdings.parquet").to_pylist()
    assert [r["name_of_issuer"] for r in attributed] == ["APPLE INC", "ACME WIDGET CO"]
    assert {r["report_quarter"] for r in attributed} == {"2025Q4"}


def test_three_quarters_build_three_filings_for_each_filer(tmp_path, capsys) -> None:
    scope = Scope.parse(",".join(THREE), THREE_CUTOFF)
    output, source = _build(tmp_path, "three", world.bodies(THREE), scope)

    assert _filings(output) == [
        (fund, quarter, "13F-NT" if (fund, quarter) == ("Gamma Advisors Inc", "2026Q3")
         else "13F-HR")
        for fund in ("Alpha Capital LLC", "Beta Partners LP", "Gamma Advisors Inc")
        for quarter in THREE
    ]
    assert "warning" not in capsys.readouterr().err
    assert _manifest(output)["scope"]["quarters"] == list(THREE)

    # 2026Q3 has no verified URL, so it uses the pattern. 2026Q2 keeps its exact URL.
    lists = [url for url in source.requested if "/files/investment/" in url]
    assert lists == [world.list_url("2026Q3"), world.Q2_LIST_URL]
    assert (output / "cusip_check.csv").is_file()
    attributed = pq.read_table(output / "attributed_holdings.parquet").to_pylist()
    assert len(attributed) == 2


def test_a_narrower_scope_leaves_the_other_quarter_out(tmp_path) -> None:
    scope = Scope.parse("2026Q2", "2026-08-18")
    output, source = _build(tmp_path, "narrow", world.bodies(), scope)
    assert {quarter for _, quarter, _ in _filings(output)} == {"2026Q2"}
    assert len(_filings(output)) == 3
    # Q2 is the last quarter, so the check still compares with Q1.
    assert world.Q1_LIST_URL in source.requested


def test_the_cutoff_day_is_in_and_the_next_day_is_out(tmp_path) -> None:
    """Gamma files on 2026-02-11 and its parent Delta on 2026-02-12."""
    scope = Scope.parse(",".join(ONE), "2026-02-11")
    output, _ = _build(tmp_path, "edge", world.bodies(ONE), scope)
    assert ("Gamma Advisors Inc", "2025Q4", "13F-NT") in _filings(output)
    # The parent's report is one day late, so the notice finds no rows to attribute.
    assert pq.read_table(output / "attributed_holdings.parquet").num_rows == 0

    with pytest.raises(LookupError, match="filed by 2026-02-10"):
        _build(tmp_path, "early", world.bodies(ONE), Scope.parse(",".join(ONE), "2026-02-10"))


def test_an_older_page_does_not_lose_a_quarter(tmp_path, capsys) -> None:
    """Alpha's first two quarters move to an older page. `recent` holds one filing."""
    served = world.bodies(THREE)
    pages = world.move_to_page(served, 1000001, keep=1, decoy=True)
    scope = Scope.parse(",".join(THREE), THREE_CUTOFF)

    output, source = _build(tmp_path, "paged", served, scope)

    assert [q for fund, q, _ in _filings(output) if fund == "Alpha Capital LLC"] == list(THREE)
    assert "warning" not in capsys.readouterr().err
    assert pages[0] in source.requested
    assert pages[1] not in source.requested  # it ends in 2012, before the first quarter


def test_a_list_name_the_sec_changed_is_found_by_the_second_name(tmp_path) -> None:
    scope = Scope.parse(",".join(THREE), THREE_CUTOFF)
    output, source = _build(tmp_path, "renamed", world.bodies(THREE, list_suffix="-txt"), scope)
    assert (output / "cusip_check.csv").is_file()
    tried = [url for url in source.requested if "13flist2026q3" in url]
    assert tried == [world.list_url("2026Q3"), world.list_url("2026Q3", "-txt")]


def test_a_missing_list_costs_the_check_and_not_the_core_tables(tmp_path, capsys) -> None:
    scope = Scope.parse(",".join(ONE), ONE_CUTOFF)
    output, _ = _build(tmp_path, "nolist", world.bodies(ONE, list_suffix=None), scope)

    err = capsys.readouterr().err
    assert "no official 13F list for 2025Q4: the SEC returned 404 for" in err
    assert "13flist2025q4.txt" in err and "13flist2025q4-txt.txt" in err
    assert "main: cusip_check failed; the core output stands" in err
    assert not (output / "cusip_check.csv").exists()
    assert len(_filings(output)) == 3
    assert (output / "manifest.json").is_file()  # the run still ends cleanly


def test_a_quarter_after_the_cutoff_stops_the_command_line_at_once(tmp_path) -> None:
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "main.py", "--user-agent", "Test User test@example.com",
         "--quarters", "2026Q1,2026Q3", "--cutoff", "2026-08-18",
         "--output", str(tmp_path / "out"), "--cache-dir", str(tmp_path / "cache")],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 2
    assert "quarter 2026Q3 ends after the filing cutoff 2026-08-18" in done.stderr
    assert "--cutoff to 2026-09-30 or later" in done.stderr
    assert not (tmp_path / "out").exists()  # it stopped before any file or request
    assert not (tmp_path / "cache").exists()
