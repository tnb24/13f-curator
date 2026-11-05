"""The pipeline: fetch from EDGAR, parse, write the dataset.

`main.py` at the repository root is the command line entry point:

    python main.py --user-agent "Jane Doe jane@example.com"
    python main.py --user-agent "..." --quarters 2026Q1,2026Q2,2026Q3 --cutoff 2026-11-16

`--quarters` and `--cutoff` set the scope. The defaults build the shipped
dataset. A quarter that ends after the cutoff is an error.

The run works from a clean checkout with nothing in output/. A second run can
reuse cached responses; the manifest includes run-specific timings and cache stats.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path

from curator.attribution import run_attribution
from curator.cache import Cache, Manifest, Source
from curator.cik import RosterError, resolve_filers, write_filers_csv
from curator.cusip_check import run_cusip_check
from curator.discover import discover
from curator.download import download
from curator.http import Fetcher, RateLimiter
from curator.parse import QualityReport, parse_all, warm_up
from curator.scope import DEFAULT_CUTOFF, DEFAULT_QUARTERS, Scope, ScopeError
from curator.write import write_parquet

ROOT = Path(__file__).resolve().parents[1]
FILERS = ROOT / "filers.csv"
OUTPUT = ROOT / "output"
CACHE_DIR = ROOT / ".cache"
_MAX_PARSE_WORKERS = 8

# SEC rejects requests without a contact address, and a run that gets the department
# blocked is worth failing fast on. Loose on purpose: we check an address is present,
# not that it is well-formed.
UA_PATTERN = re.compile(r"^\S.*\s+[^@\s]+@[^@\s]+\.[a-z]{2,}\s*$", re.I)


def load_filers(path: Path = FILERS) -> list[dict[str, str]]:
    """Read and validate a roster before fetching data or changing outputs."""
    if not path.is_file():
        raise RosterError(f"{path}: roster file does not exist")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "fund_name" not in fields or len(fields) != len(set(fields)):
            raise RosterError(f"{path}: CSV needs a unique fund_name column")
        rows = []
        seen: set[str] = set()
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise RosterError(f"{path}: row {line} has more fields than the header")
            name = (row.get("fund_name") or "").strip()
            if not name:
                raise RosterError(f"{path}: row {line} has an empty fund_name")
            key = name.casefold()
            if key in seen:
                raise RosterError(f"{path}: row {line} repeats fund_name {name!r}")
            seen.add(key)
            row["fund_name"] = name
            rows.append(row)
    if not rows:
        raise RosterError(f"{path}: CSV contains no managers")
    return rows


def _run_extra(function, name: str, filename: str, records, cache, output, scope) -> bool:
    """Run one extra stage. A failure there never stops the core pipeline.

    The two Parquet tables are already on disk by this point, so a traceback
    here costs the extra output only.
    """
    target = output / filename
    # A failed rerun must not leave a report from an older dataset in place.
    target.unlink(missing_ok=True)
    try:
        function(records, cache, output, scope)
        if not target.is_file():
            raise RuntimeError(f"{name} did not produce {filename}")
        return True
    except Exception:  # An extra stage must never cost the core tables.
        target.unlink(missing_ok=True)
        print(f"main: {name} failed; the core output stands", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False


@contextlib.contextmanager
def _stage(name: str) -> Iterator[None]:
    """Log how long one pipeline stage takes, to stderr."""
    started = time.perf_counter()
    yield
    print(f"main: {name} took {time.perf_counter() - started:.2f}s", file=sys.stderr)


def run(
    user_agent: str,
    output: Path,
    cache_dir: Path = CACHE_DIR,
    fetcher: Source | None = None,
    roster: Path = FILERS,
    workers: int | None = None,
    scope: Scope | None = None,
) -> dict[str, bool]:
    """Build the dataset, one step at a time, over the shared cache.

        1. resolve each manager name to its SEC filer     curator/cik.py
        2. find in-scope filings via the submissions API  curator/discover.py
        3. download the filings through the cache         curator/download.py
        4. parse every filing into records and holdings   curator/parse.py
        5. write output/filings.parquet and holdings      curator/write.py
        6. run the two extra stages     curator/attribution.py, curator/cusip_check.py
        7. record every request in output/manifest.json   curator/cache.py

    `fetcher` replaces the EDGAR client and `roster` replaces filers.csv. The
    tests pass a fetcher that serves fixture bodies, so the whole pipeline
    runs with no network. `workers=1` parses in this process. stdout stays empty.
    Progress goes to stderr.

    When calling this from a script with multiple workers, use the standard
    ``if __name__ == "__main__"`` guard: workers start with multiprocessing spawn.

    Return the success status of each optional report. A failed report is absent,
    while the core tables and the manifest remain available.

    `scope` names the quarters and the filing cutoff. None means the default
    scope. The run passes the scope to every stage and records it in the manifest.
    """
    scope = Scope() if scope is None else scope
    filers = load_filers(roster)
    owned = fetcher is None
    source: Source = Fetcher(user_agent, RateLimiter(5.0)) if fetcher is None else fetcher
    manifest = Manifest()
    cache = Cache(cache_dir, source, manifest)
    try:
        return _build(
            cache, manifest, output, filers,
            parse_workers() if workers is None else workers, scope,
        )
    finally:
        if owned and isinstance(source, Fetcher):
            source.close()


def parse_workers() -> int:
    """How many parse processes to start. One means parse in this process."""
    return min(os.cpu_count() or 1, _MAX_PARSE_WORKERS)


def _build(
    cache: Cache, manifest: Manifest, output: Path, roster: list[dict[str, str]],
    workers: int, scope: Scope,
) -> dict[str, bool]:
    filings_dir = output / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted or failed rebuild must not retain an old success manifest.
    (output / "manifest.json").unlink(missing_ok=True)

    # The parse workers start now. Each one imports lxml and pyarrow while this
    # process resolves filers, so no parse task waits for an interpreter start.
    # Arrow and other libraries may have started threads before a rebuild. Forking
    # that process can deadlock; spawn gives every worker a clean interpreter.
    pool = (
        ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1 else None
    )
    try:
        if pool is not None:
            for _ in range(workers):
                pool.submit(warm_up)
        return _stages(cache, manifest, output, roster, filings_dir, pool, scope)
    finally:
        if pool is not None:
            pool.shutdown()


def _stages(
    cache: Cache,
    manifest: Manifest,
    output: Path,
    roster: list[dict[str, str]],
    filings_dir: Path,
    pool: Executor | None,
    scope: Scope,
) -> dict[str, bool]:
    with _stage("resolve"):
        filers = resolve_filers(roster, cache, scope)
    corrected = sum(1 for filer in filers if filer.cik_source == "corrected")
    print(
        f"main: resolved {len(filers)} filers; {corrected} input CIKs named another filer",
        file=sys.stderr,
    )
    write_filers_csv(filers, output / "filers.csv")

    with _stage("discover"):
        refs = discover(filers, cache, scope)
    print(f"main: discovered {len(refs)} filings in scope", file=sys.stderr)
    with _stage("download"):
        docs = download(refs, cache, filings_dir)
    notices = sum(1 for doc in docs if doc.info_table_path is None)
    print(
        f"main: stored {len(docs)} filings under {filings_dir} "
        f"({notices} notices)",
        file=sys.stderr,
    )

    quality = QualityReport()
    with _stage("parse"):
        records = parse_all(docs, quality, pool)
    with _stage("write"):
        write_parquet(records, output)
        quality.write(output / "quality_report.json")
    print(
        f"main: wrote {len(records.filings)} filings and "
        f"{len(records.holdings)} holdings; "
        f"quality report holds {len(quality.entries)} entries",
        file=sys.stderr,
    )

    with _stage("attribution"):
        attribution = _run_extra(
            run_attribution, "attribution", "attributed_holdings.parquet",
            records, cache, output, scope,
        )
    with _stage("cusip_check"):
        cusip_check = _run_extra(
            run_cusip_check, "cusip_check", "cusip_check.csv", records, cache, output, scope
        )

    manifest_path = output / "manifest.json"
    stages = {"attribution": attribution, "cusip_check": cusip_check}
    manifest.write(manifest_path, scope.as_dict(), stages)
    run_block = json.loads(manifest_path.read_text(encoding="utf-8"))["run"]
    print(f"main: manifest run: {json.dumps(run_block)}", file=sys.stderr)
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True,
                    help='required by SEC: "FirstName LastName email@example.com"')
    ap.add_argument("--filers", type=Path, default=FILERS,
                    help="CSV of managers: fund_name, and optionally cik (default: filers.csv)")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                    help="where raw EDGAR responses are kept (default: .cache/)")
    ap.add_argument("--quarters", default=",".join(DEFAULT_QUARTERS),
                    help="report quarters to collect, comma separated, as YYYYQN "
                         "(default: %(default)s)")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                    help="last filing date to accept, YYYY-MM-DD, inclusive "
                         "(default: %(default)s)")
    args = ap.parse_args()

    if not UA_PATTERN.match(args.user_agent):
        sys.exit(
            "invalid --user-agent.\n"
            "SEC requires a contact address and rejects requests without one.\n"
            '  python main.py --user-agent "Jane Doe jane@example.com"'
        )
    try:
        scope = Scope.parse(args.quarters, args.cutoff)
    except ScopeError as exc:
        ap.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        stages = run(args.user_agent, args.output, args.cache_dir, roster=args.filers, scope=scope)
    except RosterError as exc:
        ap.error(str(exc))
    if not all(stages.values()):
        print("main: one or more reports failed; see manifest.json and stderr", file=sys.stderr)
        return 1
    return 0

