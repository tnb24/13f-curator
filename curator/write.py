"""Parquet writer: store the two parsed tables as output/*.parquet.

Parse builds both tables against the schemas in `curator/schema.py`. No type
is inferred and no pandas frame is involved. Each file lands through a
temporary path in its target directory, so a crash never leaves a half-written
Parquet file that looks valid.
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from curator.parse import ParsedDataset
from curator.schema import FILINGS_SCHEMA, HOLDINGS_SCHEMA

__all__ = ["FILINGS_SCHEMA", "HOLDINGS_SCHEMA", "write_parquet", "write_table_atomic"]


def write_parquet(dataset: ParsedDataset, output: Path) -> None:
    """Write output/filings.parquet and output/holdings.parquet.

    The tables arrive in their final order. A table must match its schema
    exactly, nullable flags included, or the write stops.
    """
    for table, schema, name in (
        (dataset.filings, FILINGS_SCHEMA, "filings.parquet"),
        (dataset.holdings, HOLDINGS_SCHEMA, "holdings.parquet"),
    ):
        if not table.schema.equals(schema):
            raise TypeError(f"{name}: table schema differs from curator/schema.py")
        write_table_atomic(table, Path(output) / name)


def write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write through a temporary file beside the target, then rename into place.

    One contiguous chunk per column gives one fixed page layout, so the same
    rows always write the same bytes.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        pq.write_table(table.combine_chunks(), tmp, compression="snappy", version="2.6")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
