"""Scope parsing and name resolution, on synthetic names and on the real tables."""

from __future__ import annotations

import random
import string
from pathlib import Path

import pyarrow as pa
import pytest

from agents.resolve import (
    Issuer,
    IssuerIndex,
    NameIndex,
    Resolver,
    out_of_scope,
    parse_scope,
    question_terms,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[1]
HAVE_TABLES = (ROOT / "output" / "holdings.parquet").exists()


@pytest.mark.parametrize(
    "question, quarters",
    [
        ("in 2026 Q1", {"2026Q1"}),
        ("in 2026Q2", {"2026Q2"}),
        ("in 2026-Q2", {"2026Q2"}),
        ("in Q1 2026", {"2026Q1"}),
        ("in the first quarter of 2026", {"2026Q1"}),
        ("in the 2nd quarter 2026", {"2026Q2"}),
        ("between 2026 Q1 and 2026 Q2", {"2026Q1", "2026Q2"}),
        ("at the end of 2025", {"2025Q1", "2025Q2", "2025Q3", "2025Q4"}),
        ("how many filings", set()),
        ("what about 13F-HR forms", set()),  # 13 is no year
    ],
)
def test_parse_scope(question: str, quarters: set[str]) -> None:
    assert parse_scope(question) == quarters


def test_out_of_scope_needs_every_named_quarter_outside() -> None:
    assert out_of_scope("in 2026 Q3") is not None
    assert out_of_scope("between 2026 Q2 and 2026 Q3") is None  # Q2 is answerable
    assert out_of_scope("no quarter at all") is None


def test_question_terms_drop_filler_and_keep_names() -> None:
    assert question_terms("Which manager held the largest Apple position in 2026 Q2?") == ["apple"]
    assert question_terms("Did D. E. Shaw & Co. report it?") == ["shaw"]
    assert question_terms("13F-NT filers in Q2 2026") == ["filers"]


def test_the_ascii_path_matches_the_arrow_tokenizer() -> None:
    """question_terms takes a scalar path for ASCII; tokenize is the definition."""
    rng = random.Random(7)
    alphabet = string.ascii_letters + string.digits + " .,&'-/()?!\t"
    texts = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))) for _ in range(5000)]
    texts += ["Which manager held APPLE in 2026 Q2?", "AT&T, 3M and S&P 500", "13F-HR/A"]
    for text, words in zip(texts, tokenize(pa.array(texts)).to_pylist(), strict=True):
        assert question_terms(text) == list(dict.fromkeys(words)), text


def test_the_ascii_path_matches_on_every_pair_of_ascii_characters() -> None:
    texts = [f"Alpha{chr(c)}BETA{chr(d)}42" for c in range(128) for d in range(128)]
    for text, words in zip(texts, tokenize(pa.array(texts)).to_pylist(), strict=True):
        assert question_terms(text) == list(dict.fromkeys(words)), repr(text)


def test_a_non_ascii_question_uses_the_arrow_tokenizer() -> None:
    assert question_terms("Did Nestlé report Café holdings?") == ["nestlé", "café"]


def _issuers(rows: list[tuple[str, str, str, int]]) -> IssuerIndex:
    spellings = [(cusip, name) for cusip, name, _, _ in rows]
    issuers: dict[str, Issuer] = {}
    for cusip, name, title, count in rows:
        if cusip not in issuers or count > issuers[cusip].rows:
            issuers[cusip] = Issuer(cusip, name, title, count)
    return IssuerIndex(spellings, issuers)


ISSUERS = _issuers([
    ("037833100", "APPLE INC", "COM", 73),
    ("037833100", "Apple Inc - US", "Equity", 6),
    ("03784Y200", "APPLE HOSPITALITY REIT INC", "COM NEW", 26),
    ("594918104", "MICROSOFT CORP", "COM", 82),
    ("67066G104", "NVIDIA CORPORATION", "COM", 83),
    ("71713U102", "PERSHING SQUARE TONTINE HLDGS", "CL A", 3),
])


def test_a_spelling_variant_resolves_to_its_cusip() -> None:
    found = ISSUERS.match(["apple"], limit=5)
    assert [issuer.cusip for issuer in found] == ["037833100", "03784Y200"]


def test_coverage_breaks_a_score_tie() -> None:
    """Both Apple names hold "apple"; APPLE INC has nothing else, so it ranks first."""
    assert ISSUERS.match(["apple"], limit=1)[0].name == "APPLE INC"


def test_a_typo_still_resolves() -> None:
    assert ISSUERS.match(["nvdia"], limit=1)[0].cusip == "67066G104"


def test_a_name_with_no_shared_word_never_matches() -> None:
    assert ISSUERS.match(["tesla"], limit=5) == []


def test_many_spellings_of_one_cusip_cannot_hide_a_better_cusip() -> None:
    """The limit applies to CUSIPs, after the spellings of each CUSIP merge."""
    index = _issuers([("low", "ACME" + " " * i + "INC", "COM", 1) for i in range(1, 30)]
                     + [("high", "ACME LTD", "COM", 1000)])
    assert [issuer.cusip for issuer in index.match(["acme"], limit=1)] == ["high"]


def test_rare_words_outweigh_common_ones() -> None:
    funds = NameIndex([
        "AQR Capital Management LLC",
        "Pershing Square Capital Management L.P.",
        "Lone Pine Capital LLC",
    ])
    found = funds.match(question_terms("Pershing Square Capital Management"), limit=3)
    assert found[0].row == 1


def test_a_fund_match_consumes_its_words() -> None:
    """Pershing Square names the fund, so it must not also name the Tontine issuer."""
    funds = ["Pershing Square Capital Management L.P.", "Citadel Advisors LLC"]
    resolver = Resolver(funds, ISSUERS)
    found = resolver.resolve("Did Pershing Square report any Microsoft holdings in 2026 Q2?")
    assert found.funds == ["Pershing Square Capital Management L.P."]
    assert [issuer.cusip for issuer in found.issuers] == ["594918104"]


def test_a_question_with_no_names_resolves_nothing() -> None:
    resolver = Resolver(["Citadel Advisors LLC"], ISSUERS)
    found = resolver.resolve("Which manager reported the most call options in 2026 Q2?")
    assert (found.funds, found.issuers) == ([], [])


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_TABLES, reason="output/*.parquet is absent")
@pytest.mark.parametrize(
    "question, fund, cusip",
    [
        # The three cases that once ranked alphabetical noise first.
        ("Which manager added the most Nvidia shares?", None, "67066G104"),
        ("Did Pershing Square Capital Management report any Microsoft holdings?",
         "Pershing Square Capital Management L.P.", "594918104"),
        ("Which manager held Tesla in both quarters?", None, "88160R101"),
        ("Which manager held the largest Apple position in 2026 Q2?", None, "037833100"),
        ("How much Berkshre Hathaway class B did Bridgewater hold?",
         "Bridgewater Associates LP", "084670702"),
        ("Does D. E. Shaw hold Alphabet class C?", "D. E. Shaw & Co. Inc.", None),
        ("How many distinct issuers did Renaissance Technologies LLC report?",
         "Renaissance Technologies LLC", None),
    ],
)
def test_real_names_resolve(question: str, fund: str | None, cusip: str | None) -> None:
    from agents.answer import build_resolver
    from agents.db import Engine

    with Engine() as engine:
        found = build_resolver(engine).resolve(question)
    if fund is not None:
        assert found.funds[0] == fund
    if cusip is not None:
        assert cusip in [issuer.cusip for issuer in found.issuers[:2]]
    if "Renaissance" in question:
        assert found.issuers == []  # the fund used every word
