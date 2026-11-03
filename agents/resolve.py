"""Deterministic question scoping and name resolution. No model call happens here.

Two jobs live in this module:

1. `parse_scope` and `out_of_scope` decide whether the dataset can answer a
   question at all. An out-of-scope question costs zero model calls.
2. `NameIndex` finds the funds and issuers that a question names. A 31B model
   cannot know whether a filer spelled a company `NVIDIA CORPORATION` or
   `Nvidia Corp - US`. The index can, and for an issuer it returns the CUSIP,
   which every spelling shares.

The index is data-oriented. It stores every name as a row of a sparse
name-by-word matrix, weighted by inverse document frequency (IDF), so a rare
word such as "pershing" counts for more than "capital". A question becomes a
sparse vector over the same words. One matrix-vector product then scores
every name at once, in a few milliseconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process

IN_SCOPE = {"2026Q1", "2026Q2"}

# --- scope ---------------------------------------------------------------------

_QUARTER_INDEX = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
}
_YEAR = r"(20\d{2})"
# "2026 Q1", "2026Q1", "2026-Q1"
_YEAR_THEN_QUARTER = re.compile(rf"(?<!\d){_YEAR}\s*-?\s*q([1-4])(?!\d)", re.IGNORECASE)
# "Q1 2026"
_QUARTER_THEN_YEAR = re.compile(rf"\bq([1-4])\s*{_YEAR}(?!\d)", re.IGNORECASE)
# "first quarter of 2026", "2nd quarter 2026"
_NAMED_QUARTER = re.compile(
    rf"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+(?:of\s+)?{_YEAR}(?!\d)",
    re.IGNORECASE,
)
# A bare year stands for every quarter of that year.
_BARE_YEAR = re.compile(rf"(?<!\d){_YEAR}(?!\d)")


def parse_scope(question: str) -> set[str]:
    """Return every quarter that the question names, as YYYYQN."""
    quarters: set[str] = set()
    consumed: list[tuple[int, int]] = []

    def collect(pattern: re.Pattern[str], build) -> None:
        for match in pattern.finditer(question):
            quarters.update(build(match))
            consumed.append(match.span())

    collect(_YEAR_THEN_QUARTER, lambda m: {f"{m.group(1)}Q{m.group(2)}"})
    collect(_QUARTER_THEN_YEAR, lambda m: {f"{m.group(2)}Q{m.group(1)}"})
    collect(_NAMED_QUARTER, lambda m: {f"{m.group(2)}Q{_QUARTER_INDEX[m.group(1).lower()]}"})

    for match in _BARE_YEAR.finditer(question):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        quarters.update(f"{match.group(1)}Q{number}" for number in range(1, 5))
    return quarters


def out_of_scope(question: str) -> str | None:
    """Return a reason when the question asks only for data the dataset lacks."""
    named = parse_scope(question)
    if not named or named & IN_SCOPE:
        return None
    return (
        f"the dataset holds only {' and '.join(sorted(IN_SCOPE))}; "
        f"the question asks only about {', '.join(sorted(named))}"
    )


# --- words ---------------------------------------------------------------------

# Words that carry no identity: question filler, and the legal and share-class
# words that most company names end with. Neither side of a match uses them.
STOPWORDS = frozenset("""
a an the of in on at by for to from with and or not no is are was were be been
do does did done has have had can could will would should may might must
what which who whom whose when where why how than then that this these those
it its they them their there here his her he she we our you your
many much more most least large larger largest small smaller smallest big bigger biggest
top bottom high higher highest low lower lowest total sum amount number count
distinct average mean median value values worth size dollar dollars usd
hold holds held holding holdings position positions stake stakes exposure
share shares shared sole none security securities stock stocks equity equities
report reported reports reporting filing filings filed file files
manager managers fund funds adviser advisers advisor advisors firm firms
company companies issuer issuers portfolio portfolios roster
quarter quarters year years period periods q1 q2 q3 q4 13f nt hr form forms
between across all any each every per both same different other only also
change changed changes grew grow grown growth shrank shrink shrunk
increase increased decrease decreased added add adding cut sold bought buy
compare compared versus vs percent percentage ratio rank ranked
call calls put puts option options price prices directly instead
did whether list name names give show tell find
inc corp corporation co company llc lp ltd limited plc sa ag nv se
holdings holding hldgs group grp trust tr class cl com new del us
""".split())

_STOPWORD_SET = pa.array(sorted(STOPWORDS), pa.string())
_FUZZY_TERM_MIN_LENGTH = 5
_FUZZY_TERM_CUTOFF = 85.0
_KEEP_FRACTION = 0.6  # a candidate needs this share of the best score


def tokenize(texts: pa.Array) -> pa.ListArray:
    """Split each text into its identity words, as Arrow string kernels.

    Lower case, every run of characters that are neither letters nor digits
    becomes one space, then a split on whitespace. A word survives when it has
    two or more characters, is not all digits, and is not a stop word. Names
    and questions both pass through this one function, so the two sides of a
    match always split words the same way.
    """
    words = pc.utf8_split_whitespace(
        pc.replace_substring_regex(pc.utf8_lower(texts), r"[^\p{L}\p{N}]+", " ")
    )
    flat = pc.list_flatten(words)
    owner = pc.list_parent_indices(words)
    keep = pc.and_(
        pc.and_(pc.greater_equal(pc.utf8_length(flat), 2), pc.invert(pc.utf8_is_digit(flat))),
        pc.invert(pc.is_in(flat, value_set=_STOPWORD_SET)),
    )
    flat = flat.filter(keep)
    owner = np.asarray(owner.filter(keep), np.int64)
    offsets = np.searchsorted(owner, np.arange(len(texts) + 1), side="left")
    return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)), flat)


_ASCII_NON_WORD = re.compile(r"[^a-z0-9]+")


def question_terms(question: str) -> list[str]:
    """The distinct identity words of a question, in order of first use.

    Eight Arrow kernel calls cost about 3 ms on a one-string array, so an ASCII
    question takes a scalar path. On ASCII text the two paths agree: after
    lower case, the letters and digits are exactly a-z and 0-9, and the only
    whitespace left is a space. A test checks the two paths on random text.
    """
    if not question.isascii():
        return list(dict.fromkeys(tokenize(pa.array([question], pa.string()))[0].as_py()))
    words = _ASCII_NON_WORD.sub(" ", question.lower()).split()
    return list(dict.fromkeys(
        word for word in words
        if len(word) >= 2 and not word.isdigit() and word not in STOPWORDS
    ))


# --- the index -----------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    """One scored name. `terms` are the question words that it used."""

    row: int
    score: float
    coverage: float
    terms: frozenset[str]


class NameIndex:
    """Names as a CSR matrix of IDF-weighted content words.

    Row i of the matrix is name i. `offsets[i]:offsets[i + 1]` slices its word
    ids out of `words`. A name with no content word gets no row, because no
    question can name it.
    """

    def __init__(self, values: list[str]) -> None:
        tokens = tokenize(pa.array(values, pa.string()))
        flat = pc.list_flatten(tokens)
        owner = np.asarray(pc.list_parent_indices(tokens), np.int64)
        encoded = pc.dictionary_encode(flat) if len(flat) else None
        vocabulary = encoded.dictionary.to_pylist() if encoded is not None else []
        word = np.asarray(encoded.indices, np.int64) if encoded is not None else owner

        # One entry per (name, word) pair, sorted by name: the CSR layout.
        pairs = np.unique(owner * max(len(vocabulary), 1) + word)
        rows_of_pairs = pairs // max(len(vocabulary), 1)
        rows, starts = np.unique(rows_of_pairs, return_index=True)

        self.values = values
        self.vocabulary = vocabulary
        self._ids = {text: index for index, text in enumerate(vocabulary)}
        self._rows = rows
        self._words = pairs % max(len(vocabulary), 1)
        self._offsets = np.append(starts, len(pairs)).astype(np.int64)
        frequency = np.bincount(self._words, minlength=len(vocabulary))
        count = max(len(rows), 1)
        self._idf = np.log((count + 1) / (frequency + 1)) + 1.0
        self._totals = (
            np.add.reduceat(self._idf[self._words], self._offsets[:-1])
            if len(rows) else np.zeros(0)
        )

    def similarities(self, terms: list[str]) -> tuple[np.ndarray, dict[int, set[str]]]:
        """Return each vocabulary word's best similarity to any term, and which terms.

        A term matches its own word at 1.0. A term of five or more letters also
        matches a near spelling ("Nvdia", "Berkshre") at its fuzz.ratio / 100.
        Few words match, so the terms behind each match sit in a sparse map.
        """
        similarity = np.zeros(len(self.vocabulary), np.float64)
        sources: dict[int, set[str]] = {}
        for term in terms:
            word = self._ids.get(term)
            if word is not None:
                similarity[word] = 1.0
                sources.setdefault(word, set()).add(term)
        long_terms = [term for term in terms if len(term) >= _FUZZY_TERM_MIN_LENGTH]
        if long_terms and self.vocabulary:
            scores = process.cdist(
                long_terms, self.vocabulary, scorer=fuzz.ratio,
                score_cutoff=_FUZZY_TERM_CUTOFF, dtype=np.float32,
            )
            for t, w in zip(*np.nonzero(scores), strict=True):
                word = int(w)
                similarity[word] = max(similarity[word], float(scores[t, w]) / 100.0)
                sources.setdefault(word, set()).add(long_terms[t])
        return similarity, sources

    def match(self, terms: list[str], limit: int) -> list[Match]:
        """Score every name against the terms in one pass. Return the best few.

        score    = sum of IDF x similarity over the name's matched words
        coverage = score / sum of IDF over all of the name's words

        Rank: higher score, then higher coverage, so "Apple" puts APPLE INC
        ahead of APPLE HOSPITALITY REIT INC.
        """
        if not terms or not len(self._rows):
            return []
        similarity, sources = self.similarities(terms)
        if not similarity.any():
            return []
        contribution = self._idf[self._words] * similarity[self._words]
        scores = np.add.reduceat(contribution, self._offsets[:-1])
        hit = np.flatnonzero(scores > 0)
        best = scores[hit].max()
        hit = hit[scores[hit] >= _KEEP_FRACTION * best]
        coverage = scores[hit] / self._totals[hit]
        order = np.lexsort((hit, -coverage, -scores[hit]))[:limit]

        matches = []
        for k in order:
            index = hit[k]
            span = self._words[self._offsets[index]:self._offsets[index + 1]]
            used = frozenset().union(*(sources[w] for w in span.tolist() if w in sources))
            matches.append(
                Match(int(self._rows[index]), float(scores[index]), float(coverage[k]), used)
            )
        return matches


@dataclass(frozen=True)
class Issuer:
    """One security: the CUSIP that every filed spelling shares."""

    cusip: str
    name: str            # the spelling most rows use
    title_of_class: str  # the class most rows use
    rows: int            # holdings rows that carry this CUSIP


class IssuerIndex:
    """Issuer spellings scored by NameIndex, then grouped by CUSIP.

    The index holds each distinct spelling once and maps it to its CUSIPs. One
    spelling can cover many securities: ISHARES TR covers 312. A CUSIP takes
    the best score of any of its spellings, so "Apple" finds 037833100 through
    APPLE INC, APPLE   INC, and Apple Inc - US at once.
    """

    def __init__(self, spellings: list[tuple[str, str]], issuers: dict[str, Issuer]) -> None:
        names = sorted({name for _, name in spellings})
        position = {name: index for index, name in enumerate(names)}
        self._cusips_of: list[list[str]] = [[] for _ in names]
        for cusip, name in sorted(spellings):
            self._cusips_of[position[name]].append(cusip)
        self._names = NameIndex(names)
        self._issuers = issuers

    def match(self, terms: list[str], limit: int) -> list[Issuer]:
        best: dict[str, tuple[float, float]] = {}
        # A CUSIP may have arbitrarily many spellings. Truncating names before
        # grouping lets one security crowd out a better-ranked security (the
        # final tie-break is holding-row count). Apply the limit to CUSIPs only.
        for found in self._names.match(terms, limit=len(self._names.values)):
            for cusip in self._cusips_of[found.row]:
                current = best.get(cusip)
                if current is None or (found.score, found.coverage) > current:
                    best[cusip] = (found.score, found.coverage)
        ranked = sorted(
            best.items(),
            key=lambda item: (-item[1][0], -item[1][1], -self._issuers[item[0]].rows, item[0]),
        )
        return [self._issuers[cusip] for cusip, _ in ranked[:limit]]


@dataclass(frozen=True)
class Resolution:
    """What the resolver found for one question."""

    terms: list[str]
    funds: list[str]
    issuers: list[Issuer]


class Resolver:
    """Funds and issuers for one dataset. Built once, then read per question."""

    def __init__(self, fund_names: list[str], issuer_index: IssuerIndex) -> None:
        self._fund_names = fund_names
        self._funds = NameIndex(fund_names)
        self._issuers = issuer_index

    def resolve(self, question: str, fund_limit: int = 3, issuer_limit: int = 6) -> Resolution:
        """Match the funds first. Words a matched fund used do not also name an issuer.

        "Did Pershing Square report Microsoft?" then offers Microsoft, not
        Pershing Square Tontine, and "Third Point's largest position" offers no
        issuer at all, not Fifth Third Bancorp.
        """
        terms = question_terms(question)
        funds = self._funds.match(terms, limit=fund_limit)
        used = frozenset().union(*(found.terms for found in funds)) if funds else frozenset()
        rest = [term for term in terms if term not in used]
        issuers = self._issuers.match(rest, limit=issuer_limit) if rest else []
        return Resolution(
            terms=terms,
            funds=[self._fund_names[found.row] for found in funds],
            issuers=issuers,
        )

