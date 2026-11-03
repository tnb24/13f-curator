"""SEC's name-to-CIK lookup file as a columnar index.

The file holds about one million `NAME:CIK:` lines, Latin-1 encoded. A Python
loop that splits and normalises each line takes about 7 seconds. This module
works on the raw bytes instead:

    bytes --numpy--> line and colon offsets --numpy--> name bytes, uppercased
          --Arrow, only lines with ( / or &--> parentheses, suffix, & to AND
          --numpy--> punctuation, leading THE, and whitespace rules
          --> one Arrow string column of names, one of CIKs

About 99.7% of lines are "simple": printable ASCII, no leading space, and a
space after every `&`. On a simple line every byte rule above gives exactly the result of
`normalise`. Each other line goes through `normalise` itself. A test compares
the two paths on every line of the real file and on random strings.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process

_PARENS_RE = re.compile(r"\([^()]*\)")
_SLASH_SUFFIX_RE = re.compile(r"\s*/[A-Z]{1,4}/?\s*$")
_PUNCT_RE = re.compile(r"[.,'\"]")
_WHITESPACE_RE = re.compile(r"\s+")

FUZZY_THRESHOLD = 85.0  # TUDOR INVESTMENT CORP ET AL scores 87.5
FUZZY_LIMIT = 10

_NL, _SPACE, _QUOTE, _AMP, _APOS = 10, 32, 34, 38, 39
_LPAREN, _COMMA, _DOT, _SLASH, _COLON = 40, 44, 46, 47, 58
_THE = np.frombuffer(b"THE ", np.uint8)


def normalise(name: str) -> str:
    """Return the comparison key for one entity name. The reference definition."""
    text = html.unescape(name)
    text = text.upper()
    text = _PARENS_RE.sub("", text)
    text = _SLASH_SUFFIX_RE.sub("", text)
    text = text.replace("&", " AND ")
    text = _PUNCT_RE.sub("", text)
    if text.startswith("THE "):
        text = text[len("THE "):]
    return _WHITESPACE_RE.sub(" ", text).strip()


def split_line(line: str) -> tuple[str, str] | None:
    """Split one `NAME:CIK:` line. A name can hold a colon, so split from the right."""
    line = line.strip()
    last = line.rfind(":")
    if last == -1:
        return None
    previous = line.rfind(":", 0, last)
    if previous == -1:
        return None
    raw_name, raw_cik = line[:previous], line[previous + 1:last]
    if not raw_cik.isdigit() or not raw_name:
        return None
    return raw_name, raw_cik


# --- ragged-array helpers ---------------------------------------------------
# A ragged array is one flat uint8 buffer plus int64 offsets: segment i spans
# data[offsets[i]:offsets[i + 1]]. It is the layout of an Arrow string column.
#
# The flags these helpers count are rare: a non-ASCII byte, a parenthesis, a
# full stop. So each count starts from the flagged positions alone. A running
# sum over all 40 million bytes would move 320 MB through memory per count.


def _owners(positions: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Return the index of the segment that holds each position."""
    return np.searchsorted(starts, positions, side="right") - 1


def _flag_counts(flags: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Count the set flags inside each [start, end) span. Spans can have gaps."""
    positions = np.flatnonzero(flags)
    owner = _owners(positions, starts)
    inside = positions < ends[owner]
    return np.bincount(owner[inside], minlength=len(starts))


def _select(buf: np.ndarray, starts: np.ndarray, ends: np.ndarray):
    """Copy the sorted, disjoint spans [start, end) into one compact ragged array.

    A +1 at each start and a -1 at each end, summed in int8, mark the bytes
    to keep. That costs one byte of scratch memory per input byte.
    """
    marks = np.zeros(len(buf) + 1, np.int8)
    marks[starts] = 1
    marks[ends] -= 1  # an empty span puts its -1 on top of its own +1
    inside = np.cumsum(marks[:-1], dtype=np.int8).view(bool)
    offsets = np.zeros(len(starts) + 1, np.int64)
    np.cumsum(ends - starts, out=offsets[1:])
    return buf[inside], offsets


def _compact(data: np.ndarray, offsets: np.ndarray, drop: np.ndarray):
    """Remove the bytes where `drop` is set and recompute the offsets."""
    removed = np.bincount(_owners(np.flatnonzero(drop), offsets[:-1]),
                          minlength=len(offsets) - 1)
    new_offsets = np.zeros_like(offsets)
    np.cumsum(np.diff(offsets) - removed, out=new_offsets[1:])
    return data[~drop], new_offsets


def _to_arrow(data: np.ndarray, offsets: np.ndarray) -> pa.Array:
    """Wrap a ragged array of UTF-8 bytes as an Arrow string column, no copy."""
    return pa.Array.from_buffers(
        pa.string(), len(offsets) - 1,
        [None, pa.py_buffer(offsets.astype(np.int32)), pa.py_buffer(data)],
    )


def _from_arrow(column: pa.Array):
    """Return a writable ragged array with the bytes of an Arrow string column."""
    column = pa.concat_arrays([column])  # a fresh array starts at offset zero
    _, offsets, data = column.buffers()
    offsets = np.frombuffer(offsets, np.int32, len(column) + 1).astype(np.int64)
    return np.frombuffer(data, np.uint8, offsets[-1]).copy(), offsets


# --- the byte-level normaliser ----------------------------------------------


def _normalise_simple(data: np.ndarray, offsets: np.ndarray) -> pa.Array:
    """`normalise` for simple names, whole column at once.

    The steps run in the order `normalise` runs them: upper, parentheses,
    slash suffix, `&` to ` AND `, punctuation, leading THE, whitespace.
    """
    data[(data >= 97) & (data <= 122)] -= 32

    # Parentheses, the slash suffix, and `&` need string surgery, and few names
    # hold one. Arrow runs on those rows only; replace_with_mask writes them back.
    starts, ends = offsets[:-1], offsets[1:]
    marked = (data == _LPAREN) | (data == _SLASH) | (data == _AMP)
    special = _flag_counts(marked, starts, ends) > 0
    names = _to_arrow(data, offsets)
    subset = names.filter(special)
    subset = pc.replace_substring_regex(subset, r"\([^()]*\)", "")
    subset = pc.replace_substring_regex(subset, r" */[A-Z]{1,4}/? *$", "")
    subset = pc.replace_substring(subset, "&", " AND ")
    data, offsets = _from_arrow(pc.replace_with_mask(names, pa.array(special), subset))

    # Punctuation goes before the prefix test, so "THE. X" loses its THE.
    punct = (data == _DOT) | (data == _COMMA) | (data == _APOS) | (data == _QUOTE)
    data, offsets = _compact(data, offsets, punct)

    # One pass drops a leading "THE " and every space that follows a space.
    # A space after a dropped "THE " follows a space too, so it goes as well.
    starts, lengths = offsets[:-1], np.diff(offsets)
    long_enough = np.flatnonzero(lengths >= 4)
    heads = data[starts[long_enough, None] + np.arange(4)]
    the = long_enough[(heads == _THE).all(axis=1)]
    space = data == _SPACE
    drop = np.zeros(len(data), bool)
    drop[1:] = space[1:] & space[:-1]
    drop[(starts[the, None] + np.arange(4)).ravel()] = True
    data, offsets = _compact(data, offsets, drop)

    # What remains is at most one space at each end. Arrow trims it in C++.
    return pc.utf8_trim_whitespace(_to_arrow(data, offsets))


class LookupIndex:
    """Normalised names and their CIKs, as two parallel Arrow columns."""

    def __init__(self, names: pa.Array, ciks: pa.Array) -> None:
        self.names = names
        self.ciks = ciks

    @classmethod
    def from_body(cls, body: bytes) -> LookupIndex:
        if not body:
            return cls(pa.array([], pa.string()), pa.array([], pa.string()))
        buf = np.frombuffer(body, np.uint8)
        # Latin-1 str.splitlines also recognises CR, VT, FF, FS/GS/RS and NEL.
        # Treat CRLF as two adjacent separators: the empty span is discarded.
        # The reference splits BEFORE stripping/normalising each name.
        breaks = (buf >= 10) & (buf <= 13)
        breaks |= ((buf >= 28) & (buf <= 30)) | (buf == 133)
        newlines = np.flatnonzero(breaks)
        starts = np.concatenate(([0], newlines + 1))
        ends = np.concatenate((newlines, [len(buf)]))

        # A simple line holds printable ASCII and no leading space, and each `&`
        # in it has a space after it. An entity needs a letter or `#` after its
        # `&`, so html.unescape has nothing to decode on such a line.
        bad = ((buf < _SPACE) & (buf != _NL)) | (buf > 126)
        bad[:-1] |= (buf[:-1] == _AMP) & (buf[1:] != _SPACE)
        bad[-1:] |= buf[-1:] == _AMP
        first = buf[np.minimum(starts, max(len(buf) - 1, 0))]
        simple = (_flag_counts(bad, starts, ends) == 0) & ~(
            (ends > starts) & (first == _SPACE)
        )

        # The last two colons of each line bound the CIK.
        colons = np.flatnonzero(buf == _COLON)
        after = np.searchsorted(colons, ends, side="left")
        last = colons[np.maximum(after - 1, 0)] if len(colons) else np.zeros_like(ends)
        second = colons[np.maximum(after - 2, 0)] if len(colons) else np.zeros_like(ends)
        shaped = np.flatnonzero(
            simple & (after >= 2) & (second > starts) & (last - second > 1)
        )
        cik_data, cik_offsets = _select(buf, second[shaped] + 1, last[shaped])
        nondigit = (cik_data < 48) | (cik_data > 57)
        digits = _flag_counts(nondigit, cik_offsets[:-1], cik_offsets[1:]) == 0
        valid = shaped[digits]
        name_data, name_offsets = _select(buf, starts[valid], second[valid])
        names = [_normalise_simple(name_data, name_offsets)]
        cik_column = _to_arrow(cik_data, cik_offsets).filter(pa.array(digits))
        ciks = [pc.utf8_lpad(cik_column, 10, "0")]

        rest_names: list[str] = []
        rest_ciks: list[str] = []
        others = np.flatnonzero(~simple)
        for start, end in zip(starts[others].tolist(), ends[others].tolist(), strict=True):
            split = split_line(body[start:end].decode("latin-1"))
            if split is not None:
                rest_names.append(normalise(split[0]))
                rest_ciks.append(split[1].zfill(10))
        names.append(pa.array(rest_names, pa.string()))
        ciks.append(pa.array(rest_ciks, pa.string()))

        names_column = pa.concat_arrays(names)
        ciks_column = pa.concat_arrays(ciks)
        nonempty = pc.greater(pc.binary_length(names_column), 0)
        return cls(names_column.filter(nonempty), ciks_column.filter(nonempty))

    def __len__(self) -> int:
        return len(self.names)

    def exact(self, target: str) -> list[str]:
        """Every CIK whose normalised name equals `target`, sorted."""
        return sorted(set(self.ciks.filter(pc.equal(self.names, target)).to_pylist()))

    def fuzzy(self, target: str) -> dict[str, float]:
        """CIKs of the best fuzzy name matches at or above the threshold.

        token_sort_ratio is 100 * (1 - d / (la + lb)), and the edit distance d
        is at least |la - lb|. So a name far from the target in length cannot
        reach the threshold, and a vectorized length test drops it before any
        string reaches Python. The scan of the rest runs on every core.
        Ties sort on the name, so the result never depends on file order.
        """
        slack = (100.0 - FUZZY_THRESHOLD) / 100.0
        lengths = pc.utf8_length(self.names)  # code points, as rapidfuzz counts
        size = len(" ".join(target.split()))  # token_sort_ratio collapses whitespace
        # |l - n| <= slack * (l + n)  <=>  n(1-s)/(1+s) <= l <= n(1+s)/(1-s).
        # One extra character each side absorbs float rounding at the bound.
        low = size * (1 - slack) / (1 + slack) - 1
        high = size * (1 + slack) / (1 - slack) + 1
        near = pc.and_(pc.greater_equal(lengths, low), pc.less_equal(lengths, high))
        choices = pc.unique(self.names.filter(near)).to_pylist()

        scores = process.cdist(
            [target], choices, scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_THRESHOLD - 1.0, workers=-1,
        )[0]
        hits = []
        for index in np.flatnonzero(scores):
            # cdist returns float32. The exact float64 score decides the cut.
            score = fuzz.token_sort_ratio(target, choices[index])
            if score >= FUZZY_THRESHOLD:
                hits.append((-score, choices[index]))
        hits.sort()
        wanted = {name: -negative for negative, name in hits[:FUZZY_LIMIT]}

        mask = pc.is_in(self.names, value_set=pa.array(list(wanted), pa.string()))
        candidates: dict[str, float] = {}
        for name, cik in zip(
            self.names.filter(mask).to_pylist(), self.ciks.filter(mask).to_pylist(),
            strict=True,
        ):
            candidates[cik] = max(candidates.get(cik, 0.0), wanted[name])
        return dict(sorted(candidates.items()))
