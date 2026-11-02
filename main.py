#!/usr/bin/env python3
"""Run the whole pipeline: fetch from EDGAR, parse, write the dataset.

    python main.py --user-agent "Jane Doe jane@example.com"

The run works from a clean checkout with nothing in output/. A second run reads
every response from .cache/ and writes byte-identical output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path

from curator.bonus1 import run_bonus1
from curator.bonus2 import run_bonus2
from curator.cache import Cache, Manifest
from curator.cik import verify, write_filers_csv
from curator.discover import discover
from curator.download import download
from curator.http import Fetcher, RateLimiter
from curator.parse import QualityReport, parse_all
from curator.write import write_parquet

ROOT = Path(__file__).resolve().parent
FILERS = ROOT / "filers.csv"
OUTPUT = ROOT / "output"

# SEC rejects requests without a contact address, and a run that gets the department
# blocked is worth failing fast on. Loose on purpose: we check an address is present,
# not that it is well-formed.
UA_PATTERN = re.compile(r"^\S.*\s+[^@\s]+@[^@\s]+\.[a-z]{2,}\s*$", re.I)


def load_filers() -> list[dict[str, str]]:
    """The roster the researcher supplied. At least one CIK in here is wrong."""
    with FILERS.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _run_bonus(function, name: str, records, cache, output) -> None:
    """Run one bonus module. A bonus failure never stops the core pipeline.

    The two Parquet tables are already on disk by this point, so a traceback
    here costs the bonus only.
    """
    try:
        function(records, cache, output)
    except Exception:  # noqa: BLE001 - the core result must survive.
        print(f"main: {name} failed; the core output stands", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def run(user_agent: str, output: Path) -> None:
    """Build the dataset, one step at a time, over the shared cache.

        1. verify the CIKs against SEC's lookup file      curator/cik.py
        2. find in-scope filings via the submissions API  curator/discover.py
        3. download the filings through the cache         curator/download.py
        4. parse every filing into records and holdings   curator/parse.py
        5. write output/filings.parquet and holdings      curator/write.py
        6. run the two bonus stages                       curator/bonus1.py, bonus2.py
        7. record every request in output/manifest.json   curator/cache.py

    stdout stays empty. Progress goes to stderr.
    """
    limiter = RateLimiter(5.0)
    fetcher = Fetcher(user_agent, limiter)
    manifest = Manifest()
    cache = Cache(ROOT / ".cache", fetcher, manifest)

    filings_dir = output / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    filers = verify(load_filers(), cache)
    corrected = sum(1 for filer in filers if filer.cik_source == "corrected")
    print(
        f"main: verified {len(filers)} filers, corrected {corrected} CIKs",
        file=sys.stderr,
    )
    write_filers_csv(filers, output / "filers.csv")

    refs = discover(filers, cache)
    print(f"main: discovered {len(refs)} filings in scope", file=sys.stderr)
    docs = download(refs, cache, filings_dir)
    notices = sum(1 for doc in docs if doc.info_table_path is None)
    print(
        f"main: stored {len(docs)} filings under {filings_dir} "
        f"({notices} notices)",
        file=sys.stderr,
    )

    quality = QualityReport()
    records = parse_all(docs, quality)
    write_parquet(records, output)
    quality.write(output / "quality_report.json")
    print(
        f"main: wrote {len(records.filings)} filings and "
        f"{len(records.holdings)} holdings; "
        f"quality report holds {len(quality.entries)} entries",
        file=sys.stderr,
    )

    _run_bonus(run_bonus1, "bonus1", records, cache, output)
    _run_bonus(run_bonus2, "bonus2", records, cache, output)

    manifest_path = output / "manifest.json"
    manifest.write(manifest_path)
    run_block = json.loads(manifest_path.read_text(encoding="utf-8"))["run"]
    print(f"main: manifest run: {json.dumps(run_block)}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-agent", required=True,
                    help='required by SEC: "FirstName LastName email@example.com"')
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()

    if not UA_PATTERN.match(args.user_agent):
        sys.exit(
            "invalid --user-agent.\n"
            "SEC requires a contact address and rejects requests without one.\n"
            '  python main.py --user-agent "Jane Doe jane@example.com"'
        )

    args.output.mkdir(parents=True, exist_ok=True)
    run(args.user_agent, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
