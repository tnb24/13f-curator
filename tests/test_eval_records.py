"""Check that the recovered historical records agree with their answer key."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

from evals.grade import citation, grade

ROOT = Path(__file__).resolve().parents[1] / "evals"
MODELS = {
    "google-gemma-4-31b-it": "google/gemma-4-31b-it",
    "meta-llama-llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "qwen-qwen3-30b-a3b-instruct-2507": "qwen/qwen3-30b-a3b-instruct-2507",
}
SETS = {"dev": ("questions.json", 23), "holdout": ("holdout.json", 12)}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("dataset", SETS)
@pytest.mark.parametrize("slug", MODELS)
def test_historical_record_grades_and_totals(dataset, slug):
    filename, expected_questions = SETS[dataset]
    key = {item["id"]: item for item in _read(ROOT / filename)}
    report = _read(ROOT / "results" / f"{dataset}-{slug}.json")
    summary, rows = report["summary"], report["records"]
    attempts = defaultdict(set)

    assert summary["model"] == MODELS[slug]
    assert summary["set"] == dataset
    assert len(rows) == summary["answers"] == expected_questions * 3
    for row in rows:
        assert row["id"] in key
        item = key[row["id"]]
        assert row["attempt"] not in attempts[row["id"]]
        attempts[row["id"]].add(row["attempt"])
        assert row["trace"]["question"] == item["question"]
        assert row["expected"] == item["expect"].get("value")
        assert row["passed"] == grade(item, row["answer"])
        assert (row["citation_recall"], row["citation_precision"]) == citation(
            item.get("sources", []), row["sources"]
        )

    assert len(attempts) == summary["questions"] == expected_questions
    assert all(numbers == {0, 1, 2} for numbers in attempts.values())
    assert summary["passed"] == sum(row["passed"] for row in rows)
    assert summary["accuracy"] == round(summary["passed"] / len(rows), 4)
    for metric in ("citation_recall", "citation_precision"):
        assert summary[metric] == round(statistics.mean(row[metric] for row in rows), 4)
    for metric in ("prompt_tokens", "completion_tokens"):
        assert summary[metric] == sum(row[metric] for row in rows)
    assert summary["model_calls"] == sum(row["trace"]["model_calls"] for row in rows)


@pytest.mark.parametrize("dataset", SETS)
def test_historical_models_ran_the_same_questions(dataset):
    question_ids = []
    for slug in MODELS:
        rows = _read(ROOT / "results" / f"{dataset}-{slug}.json")["records"]
        question_ids.append({row["id"] for row in rows})
    assert all(ids == question_ids[0] for ids in question_ids[1:])
