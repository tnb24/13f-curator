"""Deterministic question scoping and name resolution.

No model call happens in this module. Two jobs live here:

1. `parse_scope` and `out_of_scope` decide whether the dataset can answer a
   question at all. An out-of-scope question costs zero model calls.
2. `fund_candidates` and `issuer_candidates` find the spellings the dataset
   actually holds. A 31B model cannot know whether a filing spells a company
   `NVIDIA CORP` or `NVIDIA CORPORATION`; the distinct-value scan does.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

IN_SCOPE = {"2026Q1", "2026Q2"}

_QUARTER_INDEX = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
}

_YEAR = r"(20\d{2})"

# "2026 Q1", "2026Q1", "2026-Q1"
_YEAR_THEN_QUARTER = re.compile(
    rf"(?<!\d){_YEAR}\s*-?\s*q([1-4])(?!\d)", re.IGNORECASE
)
# "Q1 2026"
_QUARTER_THEN_YEAR = re.compile(
    rf"\bq([1-4])\s*{_YEAR}(?!\d)", re.IGNORECASE
)
# "first quarter of 2026", "2nd quarter 2026"
_NAMED_QUARTER = re.compile(
    rf"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?{_YEAR}(?!\d)",
    re.IGNORECASE,
)
# A bare year stands for every quarter of that year.
_BARE_YEAR = re.compile(rf"(?<!\d){_YEAR}(?!\d)")

_WORD = re.compile(r"[A-Za-z0-9&.'\-]+")

_MAX_GROUP_WORDS = 5
_MIN_SCORE = 60

# A question word that carries no company identity. `fuzz.WRatio` scores a short
# query against a long name by partial match, so the bare word "in" scores 90
# against "Grindr Inc - US". Every real match scores 90 too, and the alphabetical
# tiebreak then decides. Dropping these words is what makes the scan select on
# identity instead of on filler.
_STOPWORDS = frozenset("""
a an the of in on at by for to from with and or not is are was were be been
do does did has have had can could will would should may might must
what which who whom whose when where why how
many much more most least large largest small smallest big biggest top bottom
high highest low lowest total sum amount number count distinct average mean
value values worth size
hold holds held holding holdings position positions stake stakes
share shares shared sole none security securities stock stocks
report reported reports filing filings filed file
manager managers fund funds adviser advisers firm firms company companies
issuer issuers portfolio quarter quarters year years period periods
between across all any each every per both same different
change changed grew grow shrank shrink increase increased decrease decreased
compare compared versus vs percent percentage ratio
call calls put puts option options price prices
its their his her there that this these those it they them
""".split())

# Process-wide cache. `scripts/run_examples.py` asks ten questions in one
# process, and the distinct scans walk ~134635 holding rows each time they run.
_CACHE: dict[str, list[tuple[str, str, frozenset[str]]]] = {}


def parse_scope(question: str) -> set[str]:
    """Return every quarter named, as YYYYQN."""
    quarters: set[str] = set()
    consumed: list[tuple[int, int]] = []

    def _collect(pattern: re.Pattern[str], build) -> None:
        for match in pattern.finditer(question):
            quarters.update(build(match))
            consumed.append(match.span())

    _collect(_YEAR_THEN_QUARTER, lambda m: {f"{m.group(1)}Q{m.group(2)}"})
    _collect(_QUARTER_THEN_YEAR, lambda m: {f"{m.group(2)}Q{m.group(1)}"})
    _collect(
        _NAMED_QUARTER,
        lambda m: {f"{m.group(2)}Q{_QUARTER_INDEX[m.group(1).lower()]}"},
    )

    for match in _BARE_YEAR.finditer(question):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        year = match.group(1)
        quarters.update(f"{year}Q{number}" for number in range(1, 5))
    return quarters


def out_of_scope(question: str) -> str | None:
    """Return a reason when the question asks for data the dataset lacks."""
    named = parse_scope(question)
    if not named:
        return None
    if named & IN_SCOPE:
        return None
    outside = ", ".join(sorted(named))
    return (
        f"the dataset holds only {' and '.join(sorted(IN_SCOPE))}; "
        f"the question asks only about {outside}"
    )


def fund_candidates(question: str, conn, limit: int = 5) -> list[str]:
    """Return the `fund_name` values that best match the question."""
    return _candidates(question, conn, "fund", "filings", "fund_name", limit)


def issuer_candidates(question: str, conn, limit: int = 8) -> list[str]:
    """Return the `name_of_issuer` values that best match the question."""
    return _candidates(question, conn, "issuer", "holdings", "name_of_issuer", limit)


def _candidates(
    question: str, conn, key: str, table: str, column: str, limit: int
) -> list[str]:
    # rapidfuzz 3.10.1 leaves WRatio's processor off by default, so an
    # all-caps filed name scores 0 against a lower-case question word.
    # Both sides are normalised once here instead of inside every call.
    entries = _CACHE.get(key)
    if entries is None:
        rows = conn.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall()
        entries = sorted(
            (_entry(str(row[0])) for row in rows if row[0] is not None),
            key=lambda entry: entry[0],
        )
        _CACHE[key] = entries

    seen: dict[str, frozenset[str]] = {}
    for group in set(_word_groups(question)):
        processed = default_process(group)
        seen[processed] = _content(processed)
    groups = sorted(seen.items())
    if not groups:
        return []

    scored: list[tuple[int, str]] = []
    for value, processed, value_words in entries:
        best = 0.0
        shares_a_word = False
        for group, group_words in groups:
            score = fuzz.WRatio(group, processed)
            if score > best:
                best = score
            if score >= _MIN_SCORE and group_words & value_words:
                shares_a_word = True
        # WRatio's partial-match and long-string branches hand almost any
        # short query an 85 or 90 on some long filed name, so no raw-score
        # cutoff separates a real name from noise. A survivor must also
        # share one content word with the name.
        if best >= _MIN_SCORE and shares_a_word:
            scored.append((-int(best), value))
    scored.sort()
    return [value for _, value in scored[:limit]]


def _entry(value: str) -> tuple[str, str, frozenset[str]]:
    processed = default_process(value)
    return value, processed, _content(processed)


def _content(processed: str) -> frozenset[str]:
    """The words of a processed name that can carry identity."""
    return frozenset(
        word for word in processed.split() if word not in _STOPWORDS
    )


def _carries_identity(word: str) -> bool:
    """True when the word can name a company rather than describe a question."""
    lowered = word.lower()
    if lowered in _STOPWORDS:
        return False
    if len(word) < 3:
        return False
    return not lowered.isdigit()


def _word_groups(question: str) -> list[str]:
    """Every contiguous word sequence that holds at least one naming word.

    A group of pure filler words matches a long company name by partial ratio
    and floods the result. Such a group never names a company, so it is dropped.
    """
    words = _WORD.findall(question)
    groups: list[str] = []
    for size in range(1, min(_MAX_GROUP_WORDS, len(words)) + 1):
        for start in range(len(words) - size + 1):
            chunk = words[start:start + size]
            if not any(_carries_identity(word) for word in chunk):
                continue
            groups.append(" ".join(chunk))
    return groups
