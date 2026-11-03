"""The byte-level lookup index must agree with `normalise`, line for line.

`LookupIndex.from_body` normalises about 99.7% of lines with numpy and Arrow
kernels and the rest with `normalise` itself. These tests feed it lines whose
reference result is known, and random lines built from the characters that
each rule acts on, and compare every pair with the Python reference.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import pytest

from curator.lookup import LookupIndex, normalise, split_line

ROOT = Path(__file__).resolve().parents[1]
_REAL_LOOKUP = ROOT / ".cache" / "http" / (
    hashlib.sha256(b"https://www.sec.gov/Archives/edgar/cik-lookup-data.txt").hexdigest() + ".bin"
)


def _reference(lines: list[str]) -> set[tuple[str, str]]:
    """What the original one-line-at-a-time parser produces."""
    pairs = set()
    for line in lines:
        split = split_line(line)
        if split is None:
            continue
        name = normalise(split[0])
        if name:
            pairs.add((name, split[1].zfill(10)))
    return pairs


def _body_pairs(body: bytes) -> set[tuple[str, str]]:
    index = LookupIndex.from_body(body)
    return set(zip(index.names.to_pylist(), index.ciks.to_pylist(), strict=True))


def _index_pairs(lines: list[str]) -> set[tuple[str, str]]:
    return _body_pairs("\n".join(lines).encode("latin-1"))


def _body_reference(body: bytes) -> set[tuple[str, str]]:
    return _reference(body.decode("latin-1").splitlines())


HAZARDS = [
    "BAUPOST GROUP LLC/MA:1061768:",
    "BAUPOST GROUP LLC /ADV:1054420:",
    "THE BAUPOST GROUP, L.L.C.:1:",
    "THE. GROUP:2:",                    # punctuation goes before the THE test
    "THEATER CO:3:",                    # THE without a space is a word, not an article
    "  LEADING SPACE INC:4:",           # leading space: the reference path
    "TRAILING SPACE INC   :5:",
    "A (B (C) D) E:6:",                 # one pass removes the inner pair only
    "X (OLD) /MA:7:",                   # parentheses first, then the suffix
    "X/MA (OLD):8:",
    "NAME/ABCDE:9:",                    # five letters is no suffix
    "NAME/AB/:10:",
    "NAME /A1:11:",                     # a digit is no suffix
    "SMITH & JONES:12:",                # & with a space after: the byte path
    "AT&T INC:13:",                     # & before a letter: the reference path
    "S&amp;P GLOBAL:14:",               # an entity: the reference path
    "CAFÉ HOLDINGS:15:",           # Latin-1 letter: the reference path
    "D. E. SHAW & CO., L.P.:16:",
    "NAME: WITH: COLONS:17:",           # a name can hold a colon
    "NO CIK HERE:",
    "BAD CIK:12A:",
    ":18:",                             # an empty name is dropped
    "(ONLY PARENS):19:",                # normalises to empty and is dropped
    "QUOTE \"Q\" 'S' CO:20:",
    "lower case llc:21:",
    "TABS\tINSIDE:22:",                 # a control character: the reference path
    "",
]


def test_known_hazards_match_the_reference() -> None:
    assert _index_pairs(HAZARDS) == _reference(HAZARDS)


@pytest.mark.parametrize("seed", range(4))
def test_random_lines_match_the_reference(seed: int) -> None:
    """Random names over the alphabet of every rule, 5000 per seed."""
    alphabet = "ABCTHEthe ()/.,'\"&:;-1234567890é"
    rng = random.Random(seed)
    lines = []
    for _ in range(5000):
        name = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        if rng.random() < 0.3:
            name = "THE " + name
        if rng.random() < 0.3:
            name += rng.choice([" /MA", "/ADV", " (OLD)", "/A/", " "])
        lines.append(f"{name}:{rng.randint(0, 10**7)}:")
    assert _index_pairs(lines) == _reference(lines)


def test_exact_returns_every_cik_for_a_name_sorted() -> None:
    body = b"BAUPOST GROUP LLC/MA:1061768:\nBAUPOST GROUP LLC /ADV:1054420:\n"
    index = LookupIndex.from_body(body)
    assert index.exact("BAUPOST GROUP LLC") == ["0001054420", "0001061768"]
    assert index.exact("NOBODY") == []


def test_fuzzy_ties_break_on_name_not_on_file_order() -> None:
    """Fourteen names tie at one score, and the limit keeps ten of them.

    Each name differs from the target in one character of the token that
    sorts first, so every name scores 100 * (1 - 2/66).
    """
    letters = "NMLKJIHGFEDCBA"
    lines = [f"ACME GLOBAL OPPORTUNITIES FUND 1{c}:{100 + i}:" for i, c in enumerate(letters)]
    target = "ACME GLOBAL OPPORTUNITIES FUND 1Z"
    forward = LookupIndex.from_body("\n".join(lines).encode()).fuzzy(target)
    backward = LookupIndex.from_body("\n".join(reversed(lines)).encode()).fuzzy(target)
    assert forward == backward
    # The ten names that sort first are 1A to 1J, so the four last (1K to 1N) lose.
    kept = {f"{100 + letters.index(c):010d}" for c in "ABCDEFGHIJ"}
    assert set(forward) == kept
    assert all(score == pytest.approx(100 * (1 - 2 / 66)) for score in forward.values())


def test_fuzzy_keeps_only_names_at_or_above_the_threshold() -> None:
    index = LookupIndex.from_body(
        b"TUDOR INVESTMENT CORP ET AL:923093:\nTOTALLY DIFFERENT NAME:1:\n"
    )
    assert index.fuzzy("TUDOR INVESTMENT CORP") == {"0000923093": 87.5}


def test_the_fuzzy_length_bound_collapses_spaces_as_the_scorer_does() -> None:
    index = LookupIndex.from_body(b"TUDOR INVESTMENT CORP:1:")
    assert index.fuzzy("TUDOR" + " " * 100 + "INVESTMENT CORP") == {"0000000001": 100.0}


def test_an_empty_body_gives_an_empty_index() -> None:
    assert len(LookupIndex.from_body(b"")) == 0


# The tests above join lines with "\n". The tests below feed raw bytes, so they
# also check how the fast path finds the end of each line.

@pytest.mark.parametrize("separator", [b"\n", b"\r", b"\r\n", b"\v", b"\f",
                                       b"\x1c", b"\x1d", b"\x1e", b"\x85"])
def test_each_line_break_that_splitlines_knows_ends_a_record(separator: bytes) -> None:
    body = separator.join([b"A:1:", b"THE B:2:", b"", b"C:3:"])
    assert _body_pairs(body) == _body_reference(body)


def test_random_bodies_over_every_latin1_byte_match_the_reference() -> None:
    rng = random.Random(191)
    for _ in range(1_000):
        lines = [bytes(rng.randrange(256) for _ in range(rng.randrange(30))) + f":{i}:".encode()
                 for i in range(rng.randrange(1, 10))]
        body = rng.choice([b"\n", b"\r", b"\r\n", b"\x85"]).join(lines)
        assert _body_pairs(body) == _body_reference(body), body


@pytest.mark.integration
@pytest.mark.skipif(not _REAL_LOOKUP.exists(), reason="SEC lookup file not in the local cache")
def test_the_real_lookup_file_matches_the_reference() -> None:
    """All 1,044,809 (name, cik) pairs of the real file, both ways."""
    body = _REAL_LOOKUP.read_bytes()
    assert _body_pairs(body) == _body_reference(body)  # raw bytes, as the pipeline reads them
