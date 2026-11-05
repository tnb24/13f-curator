"""The dataset scope and the EDGAR endpoints that every stage shares.

A `Scope` names the report quarters to collect and the last filing date to
accept. `main.py` builds one `Scope` and the pipeline passes it to every stage.
No stage keeps a scope of its own.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date

# The scope that `main.py` uses when it gets no --quarters and no --cutoff.
DEFAULT_QUARTERS = ("2026Q1", "2026Q2")
DEFAULT_CUTOFF = "2026-08-18"  # inclusive

IN_SCOPE_FORMS = frozenset({"13F-HR", "13F-HR/A", "13F-NT", "13F-NT/A"})

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{:0>10}.json"

_QUARTER = re.compile(r"^(\d{4})Q([1-4])$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ScopeError(ValueError):
    """The quarters or the cutoff do not form a usable scope."""


def parse_quarter(quarter: str) -> tuple[int, int]:
    """Return (year, quarter number) for a text such as 2026Q2."""
    match = _QUARTER.match(quarter)
    if match is None:
        raise ScopeError(f"{quarter!r} is not a quarter; write it as YYYYQN, for example 2026Q2")
    return int(match.group(1)), int(match.group(2))


def quarter_end(quarter: str) -> str:
    """Return the last day of the quarter, as YYYY-MM-DD. This is the 13F report date."""
    year, number = parse_quarter(quarter)
    month = 3 * number
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def previous_quarter(quarter: str) -> str:
    """Return the quarter before this one, so 2026Q1 gives 2025Q4."""
    year, number = parse_quarter(quarter)
    return f"{year - 1:04d}Q4" if number == 1 else f"{year:04d}Q{number - 1}"


@dataclass(frozen=True)
class Scope:
    """The quarters to collect and the last filing date to accept.

    `quarters` holds YYYYQN texts in ascending order, without repeats. A filing
    matches when its report date is the last day of one of these quarters and its
    filing date is on or before `cutoff`. The check reads the report date, never
    the filing date, to choose the quarter.
    """

    quarters: tuple[str, ...] = DEFAULT_QUARTERS
    cutoff: str = DEFAULT_CUTOFF

    def __post_init__(self) -> None:
        if not self.quarters:
            raise ScopeError("the scope needs at least one quarter")
        for quarter in self.quarters:
            parse_quarter(quarter)
        if len(set(self.quarters)) != len(self.quarters):
            raise ScopeError(f"the quarters {', '.join(self.quarters)} repeat a quarter")
        if list(self.quarters) != sorted(self.quarters):
            raise ScopeError(f"the quarters {', '.join(self.quarters)} are not in ascending order")
        if not _ISO_DATE.match(self.cutoff):
            raise ScopeError(f"the cutoff {self.cutoff!r} is not a date; write it as YYYY-MM-DD")
        try:
            date.fromisoformat(self.cutoff)
        except ValueError as exc:
            raise ScopeError(f"the cutoff {self.cutoff!r} is not a calendar date") from exc
        late = [q for q in self.quarters if quarter_end(q) > self.cutoff]
        if late:
            latest = quarter_end(self.quarters[-1])
            noun = "quarter" if len(late) == 1 else "quarters"
            verb = "ends" if len(late) == 1 else "end"
            raise ScopeError(
                f"{noun} {', '.join(late)} {verb} after the filing cutoff {self.cutoff}. "
                "A 13F report cannot exist before its quarter ends. "
                f"Set --cutoff to {latest} or later, or remove the quarter from --quarters."
            )

    @classmethod
    def parse(cls, quarters: str, cutoff: str) -> Scope:
        """Build a scope from the two command line texts.

        `quarters` is a comma separated list, for example 2026Q1,2026Q2. The
        order and the letter case do not matter. A repeated quarter is an error.
        """
        names = sorted(part.strip().upper() for part in quarters.split(",") if part.strip())
        return cls(tuple(names), cutoff.strip())

    @property
    def report_periods(self) -> tuple[str, ...]:
        """The report date of each quarter, in the same order."""
        return tuple(quarter_end(quarter) for quarter in self.quarters)

    @property
    def expected_per_filer(self) -> int:
        """How many filings one filer files in scope: one for each quarter."""
        return len(self.quarters)

    @property
    def last_quarter(self) -> str:
        return self.quarters[-1]

    def as_dict(self) -> dict[str, object]:
        """The scope as `manifest.json` records it."""
        return {
            "quarters": list(self.quarters),
            "report_periods": list(self.report_periods),
            "filing_date_cutoff": self.cutoff,
        }
