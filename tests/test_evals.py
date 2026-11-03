"""The eval set and its grader.

The committed expected values must come out of the committed reference SQL,
so a wrong reference query or a stale answer fails here, without a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.grade import citation, grade, matches, normalise, reference

ROOT = Path(__file__).resolve().parents[1]
ITEMS = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
HOLDOUT = ROOT / "evals" / "holdout.json"
HAVE_TABLES = (ROOT / "output" / "holdings.parquet").exists()
KINDS = {"number", "name", "names", "labels", "yesno", "date", "null"}


def _all_items() -> list[dict]:
    items = list(ITEMS)
    if HOLDOUT.exists():
        items += json.loads(HOLDOUT.read_text(encoding="utf-8"))
    return items


def test_the_eval_files_are_well_formed() -> None:
    items = _all_items()
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
    for item in items:
        assert item["expect"]["kind"] in KINDS, item["id"]
        assert ("reference_sql" in item) == (item["expect"]["kind"] != "null"), item["id"]
        for option in item.get("accept", []):
            assert option["kind"] in KINDS and option.get("why"), item["id"]


@pytest.mark.integration
@pytest.mark.skipif(not HAVE_TABLES, reason="output/*.parquet is absent")
def test_each_reference_sql_gives_the_committed_answer() -> None:
    from agents.db import Engine

    with Engine() as engine:
        for item in _all_items():
            if "reference_sql" not in item:
                continue
            value, sources = reference(item, engine.query(item["reference_sql"]).rows)
            assert value == item["expect"]["value"], item["id"]
            assert sources == item["sources"], item["id"]


def test_names_compare_after_normalisation() -> None:
    assert normalise("D. E. Shaw & Co. Inc.") == "d e shaw co inc"
    assert matches("name", "AQR Capital Management LLC", "AQR CAPITAL MANAGEMENT LLC")
    assert matches("name", "AMAZON COM INC", ["AMAZON COM INC", 404043800])  # a pair passes
    assert not matches("name", "AQR Capital Management LLC", ["AQR Capital Management LLC", "x"])


def test_numbers_accept_equal_forms_only() -> None:
    assert matches("number", 2932, 2932.0)
    assert matches("number", 2932, "2,932")
    assert not matches("number", 2932, 2933)
    assert not matches("number", 1, True)  # a boolean is not a count


def test_large_integer_answers_are_exact_without_float_rounding() -> None:
    assert not matches("number", 10**12, 10**12 + 1)
    assert not matches("number", 2**53 + 1, 2**53)
    assert matches("number", 2**53 + 1, str(2**53 + 1))


def test_nonfinite_numbers_never_pass() -> None:
    for value in [float("inf"), float("nan"), "NaN", "sNaN", "Infinity"]:
        assert not matches("number", 1, value)


def test_name_sets_ignore_order_but_not_extras() -> None:
    expected = ["Citadel Advisors LLC", "Man Group plc"]
    assert matches("names", expected, ["Man Group plc", "Citadel Advisors LLC"])
    assert not matches("names", expected, ["Man Group plc"])
    assert not matches("names", expected, [*expected, "Tudor Investment Corp"])


def test_labels_need_the_right_label_for_every_name() -> None:
    expected = {"AQR Capital Management LLC": "grew", "Citadel Advisors LLC": "shrank"}
    assert matches("labels", expected, ["AQR Capital Management LLC (grew)",
                                        "Citadel Advisors LLC (shrank)"])
    assert matches("labels", expected, [["Citadel Advisors LLC", "shrank"],
                                        ["AQR Capital Management LLC", "grew"]])
    assert not matches("labels", expected, ["AQR Capital Management LLC (shrank)",
                                            "Citadel Advisors LLC (shrank)"])
    assert not matches("labels", expected, ["AQR Capital Management LLC",
                                            "Citadel Advisors LLC"])


def test_yes_no_null_and_dates() -> None:
    assert matches("yesno", "no", "No") and matches("yesno", "yes", True)
    assert matches("null", None, None) and not matches("null", None, 0)
    assert matches("date", "2026-08-14", "2026-08-14T00:00:00")


def test_an_accepted_alternative_passes_and_says_why() -> None:
    item = next(i for i in ITEMS if i["id"] == "renaissance-issuers-q2")
    assert grade(item, item["expect"]["value"])
    assert grade(item, 2898)
    assert not grade(item, 3140)


def test_citation_recall_and_precision() -> None:
    assert citation(["a", "b"], ["a", "b"]) == (1.0, 1.0)
    assert citation(["a", "b"], ["a"]) == (0.5, 1.0)
    assert citation(["a"], ["a", "z"]) == (1.0, 0.5)
    assert citation([], []) == (1.0, 1.0)
