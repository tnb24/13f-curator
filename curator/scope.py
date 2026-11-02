"""The dataset scope and the EDGAR endpoints that every stage shares."""

from __future__ import annotations

# The period filter reads reportDate, never filingDate.
REPORT_PERIODS = ("2026-03-31", "2026-06-30")
FILING_DATE_CUTOFF = "2026-08-18"  # inclusive
IN_SCOPE_FORMS = frozenset({"13F-HR", "13F-HR/A", "13F-NT", "13F-NT/A"})

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:0>10}.json"
