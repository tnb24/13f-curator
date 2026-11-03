"""The records that pass between pipeline stages.

This module imports nothing outside the standard library. The parse workers
import it, and a worker that starts faster finishes sooner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilerRecord:
    """One input row after filer resolution."""

    fund_name: str      # exactly as supplied
    cik: str            # zero-padded to 10
    cik_source: str     # "given", "corrected", or "resolved" (no input CIK)
    edgar_name: str     # entity name from the submissions API


@dataclass(frozen=True)
class FilingRef:
    """One in-scope filing, exactly as the submissions API describes it."""

    cik: str                # zero-padded to 10
    fund_name: str
    accession: str          # dashed form, from the API
    form: str
    report_date: str        # YYYY-MM-DD
    filing_date: str        # YYYY-MM-DD
    primary_document: str


@dataclass(frozen=True)
class FilingDocs:
    """The stored documents for one filing.

    ``primary_doc_path`` always points at a real file. For a notice it
    points at the ``{accession}.xml`` file, which holds the cover page.
    """

    ref: FilingRef
    info_table_path: Path | None    # None for a notice
    primary_doc_path: Path
