"""The pipeline: fetch from EDGAR, parse, write the dataset.

`main.py` at the repository root is the command line entry point:

    python main.py --user-agent "Jane Doe jane@example.com"

The run works from a clean checkout with nothing in output/. A second run reads
every response from .cache/ and writes byte-identical output.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
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
from curator.cik import resolve_filers, write_filers_csv
from curator.cusip_check import run_cusip_check
from curator.discover import discover
from curator.download import download
from curator.http import Fetcher, RateLimiter
from curator.parse import QualityReport, parse_all, warm_up
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
    """The managers to collect: a `fund_name` column and an optional `cik` column."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _run_extra(function, name: str, records, cache, output) -> None:
    """Run one extra stage. A failure there never stops the core pipeline.

    The two Parquet tables are already on disk by this point, so a traceback
    here costs the extra output only.
    """
    try:
        function(records, cache, output)
    except Exception:  # An extra stage must never cost the core tables.
        print(f"main: {name} failed; the core output stands", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


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
) -> None:
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
    """
    owned = fetcher is None
    source: Source = Fetcher(user_agent, RateLimiter(5.0)) if fetcher is None else fetcher
    manifest = Manifest()
    cache = Cache(cache_dir, source, manifest)
    try:
        _build(cache, manifest, output, roster, parse_workers() if workers is None else workers)
    finally:
        if owned and isinstance(source, Fetcher):
            source.close()


def parse_workers() -> int:
    """How many parse processes to start. One means parse in this process."""
    return min(os.cpu_count() or 1, _MAX_PARSE_WORKERS)


def _build(cache: Cache, manifest: Manifest, output: Path, roster: Path, workers: int) -> None:
    filings_dir = output / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    # The parse workers start now. Each one imports lxml and pyarrow while this
    # process resolves filers, so no parse task waits for an interpreter start.
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        if pool is not None:
            for _ in range(workers):
                pool.submit(warm_up)
        _stages(cache, manifest, output, roster, filings_dir, pool)
    finally:
        if pool is not None:
            pool.shutdown()


def _stages(
    cache: Cache,
    manifest: Manifest,
    output: Path,
    roster: Path,
    filings_dir: Path,
    pool: Executor | None,
) -> None:
    with _stage("resolve"):
        filers = resolve_filers(load_filers(roster), cache)
    corrected = sum(1 for filer in filers if filer.cik_source == "corrected")
    print(
        f"main: resolved {len(filers)} filers; {corrected} input CIKs named another filer",
        file=sys.stderr,
    )
    write_filers_csv(filers, output / "filers.csv")

    with _stage("discover"):
        refs = discover(filers, cache)
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
        _run_extra(run_attribution, "attribution", records, cache, output)
    with _stage("cusip_check"):
        _run_extra(run_cusip_check, "cusip_check", records, cache, output)

    manifest_path = output / "manifest.json"
    manifest.write(manifest_path)
    run_block = json.loads(manifest_path.read_text(encoding="utf-8"))["run"]
    print(f"main: manifest run: {json.dumps(run_block)}", file=sys.stderr)


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
    args = ap.parse_args()

    if not UA_PATTERN.match(args.user_agent):
        sys.exit(
            "invalid --user-agent.\n"
            "SEC requires a contact address and rejects requests without one.\n"
            '  python main.py --user-agent "Jane Doe jane@example.com"'
        )

    args.output.mkdir(parents=True, exist_ok=True)
    run(args.user_agent, args.output, args.cache_dir, roster=args.filers)
    return 0

