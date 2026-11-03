"""The two bonus stages, rule by rule. tests/test_pipeline.py runs them whole."""

from __future__ import annotations

import csv
import io
import random

import pyarrow as pa
import pytest

from curator.bonus1 import _tokens
from curator.bonus2 import (
    _assess,
    check_digit,
    check_digit_ok,
    csv_bytes,
    normalise,
    normalise_column,
    official_table,
    parse_official,
)


@pytest.mark.parametrize("cusip, digit", [
    ("037833100", 0),  # Apple
    ("67066G104", 4),  # Nvidia: a letter inside
    ("G3922B107", 7),  # a CINS code starts with a letter
    ("02079K107", 7),  # Alphabet class C
])
def test_the_check_digit(cusip: str, digit: int) -> None:
    assert check_digit(cusip) == digit
    assert check_digit_ok(cusip) is True


def test_a_wrong_check_digit_is_caught() -> None:
    assert check_digit_ok("037833101") is False


def test_a_cusip_of_the_wrong_length_has_no_check_digit() -> None:
    assert check_digit("12345678") is None
    assert check_digit_ok("12345678") is None


def test_normalise_rules_out_formatting_before_a_mismatch() -> None:
    assert normalise(" 0378-3310 0 ") == "037833100"
    assert normalise(None) == ""


def test_the_official_list_parses_by_column() -> None:
    body = (
        "037833100 APPLE INC                     COM                           \n"
        "88160R101*TESLA INC                     COM *A*                       \n"
        "short\n"
    )
    entries = parse_official(body)
    assert set(entries) == {"037833100", "88160R101"}
    assert entries["88160R101"]["option_marker"] == "*"
    assert entries["88160R101"]["added_this_quarter"] is True
    assert entries["88160R101"]["class"] == "COM"
    assert entries["037833100"]["issuer"] == "APPLE INC"


def test_the_assessment_rules_apply_in_order() -> None:
    q1 = {"123456782": {}}
    assert _assess("037833101", q1) == "LIKELY_FILER_ERROR"  # bad digit wins first
    assert _assess("123456782", q1) == "TIMING"
    assert _assess("G3922B107", {}) == "CINS_FOREIGN"
    assert _assess("037833100", {}) == "UNRESOLVED"


@pytest.mark.parametrize("value, tokens", [
    ("1", ["1"]),
    ("1, 11", ["1", "11"]),
    ("2,5", ["2", "5"]),
    (" 3 ,", ["3"]),
    (None, []),
    ("", []),
])
def test_other_manager_tokens_are_whole(value, tokens) -> None:
    assert _tokens(value) == tokens
    if value and "11" in value:
        assert "1" in _tokens(value) and _tokens(value).count("1") == 1


# The columnar paths of bonus 2 must give exactly what the reference rules give.

_LIST_HAZARDS = [
    b"",
    b"\n",
    b"\n\n",
    b"037833100 APPLE INC                     COM                           \n",
    b"037833100 APPLE INC                     COM",  # no final newline
    b"short\n0123456789\n012345678\n",  # 5, 10, and 9 characters
    b"          \n" + b" " * 80 + b"\n",  # blank lines of 10 and 80 characters
    b"0378-3310 0X\n03783 3100Y\n",  # a hyphen and a space inside the CUSIP field
    b"037833100 FIRST\n037833100 SECOND\n",  # the first line for a CUSIP wins
    b"abcdefghi lower case                  com *A*  \n",
    b"88160R101*TESLA INC                     COM *A*                       \n",
    b"88160R101 TESLA INC                     *A*\n",  # the flag and no class
    b"88160R101 TESLA INC                     COM*A*X\n",  # the flag is not at the end
    b"- - - - - X\n",  # the CUSIP field normalises to nothing
    b"037833100 APPLE INC\r\n88160R101 TESLA\r\n",  # CRLF: the reference path
    b"037833100 CAF\xc9 SOCI\xc9T\xc9        COM\n",  # Latin-1: the reference path
    b"037833100\tTAB INC\n",  # a tab: the reference path
    b"037833100 A\x85B\x0bC\x0cD\x1cE\n",  # line breaks that splitlines knows
]


def _random_list_body(rng: random.Random) -> bytes:
    alphabet = "0123456789ABCabc -*A/&.,'"
    lines: list[str] = []
    for _ in range(rng.randrange(0, 12)):
        width = rng.choice([0, 3, 9, 10, 11, 40, 69, 70, 71, 80, 90])
        line = "".join(rng.choice(alphabet) for _ in range(width))
        if rng.random() < 0.3 and len(line) >= 47:
            line = line[:40] + rng.choice(["COM *A*", "*A*    ", " *A*   "]) + line[47:]
        if lines and rng.random() < 0.2:
            line = lines[-1][:9] + line[9:]  # a repeated CUSIP field
        lines.append(line)
    body = "\n".join(lines) + rng.choice(["", "\n"])
    return body.encode("ascii")


def _reference_rows(body: bytes) -> list[dict]:
    return list(parse_official(body.decode("latin-1")).values())


@pytest.mark.parametrize("body", _LIST_HAZARDS)
def test_the_official_table_matches_the_reference_on_hazards(body: bytes) -> None:
    assert official_table(body).to_pylist() == _reference_rows(body)


def test_the_official_table_matches_the_reference_on_random_bodies() -> None:
    rng = random.Random(13)
    for _ in range(2_000):
        body = _random_list_body(rng)
        assert official_table(body).to_pylist() == _reference_rows(body), body


def test_the_official_table_matches_the_reference_on_random_latin1_bytes() -> None:
    rng = random.Random(193)
    for _ in range(300):
        body = bytes(rng.randrange(256) for _ in range(rng.randrange(300)))
        assert official_table(body).to_pylist() == _reference_rows(body), body


def test_the_column_normaliser_matches_the_reference() -> None:
    rng = random.Random(7)
    alphabet = ["a", "B", "7", " ", "-", "\t", "é", "ß", " ", " "]
    values: list[str | None] = [None, "", " 0378-3310 0 "]
    values += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(12)))
               for _ in range(3_000)]
    printable = [v for v in values if v is None or all(" " <= c <= "~" for c in v)]
    for sample in (values, printable):
        got = normalise_column(pa.array(sample, pa.string())).to_pylist()
        assert got == [normalise(v) for v in sample]


def _reference_csv(table: pa.Table) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(table.column_names)
    writer.writerows(zip(*(column.to_pylist() for column in table.columns), strict=True))
    return stream.getvalue().encode("utf-8")


def test_the_csv_renderer_matches_the_csv_module() -> None:
    rng = random.Random(3)
    alphabet = ["a", " ", ",", '"', "\r", "\n", "\t", "é", "'", "*"]

    def text() -> str | None:
        if rng.random() < 0.1:
            return None
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(6)))

    for rows in (0, 1, 50, 500):
        table = pa.table({
            "plain": pa.array([f"r{i}" for i in range(rows)], pa.string()),
            "text": pa.array([text() for _ in range(rows)], pa.string()),
            "flag": pa.array([rng.random() < 0.5 for _ in range(rows)], pa.bool_()),
            "count": pa.array([rng.randrange(-5, 10**6) for _ in range(rows)], pa.int64()),
        })
        assert csv_bytes(table) == _reference_csv(table)



@pytest.mark.parametrize("table", [
    pa.table({"flag": pa.array([None, True, False], pa.bool_()), "n": [1, 2, 3]}),
    pa.table({"n": pa.array([None, 1], pa.int64()), "s": ["x", "y"]}),
    pa.table({"s": ["", None]}),  # one column: csv writes an empty field as ""
    pa.table({'header,"\n': ["a", "b"]}),
    pa.table({"float": [0.0, -0.0, 1e20, 1e-10, None]}),  # the csv module renders floats
])
def test_the_csv_renderer_matches_on_nulls_headers_and_floats(table: pa.Table) -> None:
    assert csv_bytes(table) == _reference_csv(table)
