"""Run the eval set through the agent and grade every answer.

    python -m evals.run --out dev-run.json
    python -m evals.run --set holdout --out holdout-run.json
    python -m evals.run --only apple-largest-q2 --repeat 3 --out sample.json

Two sets exist. questions.json is the development set: its failures shaped the
prompt. holdout.json was written before any run against it, and no result on
it changes the prompt. Report the two scores apart.

One Agent answers every question, so the timings measure a question, not a
process start. The report goes to stdout. The full record, every trace
included, goes to evals/results/<set>-<model>.json unless --out is set. Existing
records are protected unless --force is passed.

Two scores come out of a run:

- accuracy: the share of answers that meet their expectation (evals/grade.py)
- citation recall and precision: how well `sources` matches the filings that
  the reference SQL reads
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from agents import llm
from agents.answer import Agent
from evals.grade import citation, grade

ROOT = Path(__file__).resolve().parent
SETS = {"dev": ROOT / "questions.json", "holdout": ROOT / "holdout.json"}
RESULTS = ROOT / "results"


def load(name: str = "dev", only: list[str] | None = None) -> list[dict[str, Any]]:
    items = json.loads(SETS[name].read_text(encoding="utf-8"))
    if only:
        items = [item for item in items if item["id"] in set(only)]
    return items


def _percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(share * (len(ordered) - 1)))]


def run(items: list[dict[str, Any]], repeat: int) -> dict[str, Any]:
    started = time.perf_counter()
    agent = Agent()
    records: list[dict[str, Any]] = []
    try:
        for item in items:
            for attempt in range(repeat):
                before = llm.usage()
                answer = agent.ask(item["question"])
                after = llm.usage()
                passed = grade(item, answer.answer)
                recall, precision = citation(item.get("sources", []), answer.sources)
                records.append({
                    "id": item["id"],
                    "attempt": attempt,
                    "passed": passed,
                    "expected": item["expect"].get("value"),
                    "answer": answer.answer,
                    "unit": answer.unit,
                    "sources": answer.sources,
                    "citation_recall": recall,
                    "citation_precision": precision,
                    "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
                    "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
                    "trace": answer.to_dict()["trace"],
                })
                mark = "PASS" if passed else "FAIL"
                total = answer.trace.timings_ms.get("total", 0.0)
                print(f"  {mark}  {total / 1000:5.2f}s  {item['id']:28s} "
                      f"{json.dumps(answer.answer, default=str)[:70]}", file=sys.stderr)
    finally:
        agent.close()
    return summarise(records, agent.startup_ms, time.perf_counter() - started)


def summarise(records: list[dict[str, Any]], startup_ms: float, wall: float) -> dict[str, Any]:
    timed = [r for r in records if r["trace"]["model_calls"]]
    totals = [r["trace"]["timings_ms"]["total"] for r in timed]
    local = [
        r["trace"]["timings_ms"]["total"]
        - sum(v for k, v in r["trace"]["timings_ms"].items() if k.startswith("model_"))
        for r in timed
    ]
    summary = {
        "model": llm.model_id(),
        "questions": len({r["id"] for r in records}),
        "answers": len(records),
        "passed": sum(r["passed"] for r in records),
        "accuracy": round(sum(r["passed"] for r in records) / max(len(records), 1), 4),
        "citation_recall": round(statistics.mean(r["citation_recall"] for r in records), 4),
        "citation_precision": round(statistics.mean(r["citation_precision"] for r in records), 4),
        "model_calls": sum(r["trace"]["model_calls"] for r in records),
        "lookup_calls": sum(bool(r["trace"].get("lookups")) for r in records),
        "repair_calls": sum(
            max(r["trace"]["model_calls"] - 1 - bool(r["trace"].get("lookups")), 0)
            for r in records
        ),
        "prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "completion_tokens": sum(r["completion_tokens"] for r in records),
        "agent_startup_ms": startup_ms,
        "latency_ms_p50": round(_percentile(totals, 0.5), 1) if totals else None,
        "latency_ms_p95": round(_percentile(totals, 0.95), 1) if totals else None,
        "local_ms_p50": round(_percentile(local, 0.5), 2) if local else None,
        "local_ms_max": round(max(local), 2) if local else None,
        "wall_seconds": round(wall, 1),
    }
    return {"summary": summary, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("--set", choices=sorted(SETS), default="dev", help="which question set")
    parser.add_argument("--only", nargs="+", help="question ids to run")
    parser.add_argument("--repeat", type=int, default=1, help="ask each question this many times")
    parser.add_argument("--out", type=Path, help="where to write the JSON record")
    parser.add_argument("--force", action="store_true", help="overwrite an existing record")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    items = load(args.set)
    if args.only:
        missing = set(args.only) - {item["id"] for item in items}
        if missing:
            parser.error(f"unknown question IDs: {', '.join(sorted(missing))}")
        selected = set(args.only)
        items = [item for item in items if item["id"] in selected]
    if not items:
        parser.error("the question set is empty")

    if args.out:
        out = args.out
    else:
        slug = re.sub(r"[^A-Za-z0-9.-]+", "-", llm.model_id()).strip("-")
        out = RESULTS / f"{args.set}-{slug}.json"
    if out.exists() and not args.force:
        parser.error(f"{out} already exists; choose --out or pass --force to overwrite it")
    if not out.parent.exists():
        parser.error(f"output directory {out.parent} does not exist")

    result = run(items, args.repeat)
    summary = {"set": args.set, **result["summary"]}
    result["summary"] = summary
    with out.open("w" if args.force else "x", encoding="utf-8") as record:
        record.write(json.dumps(result, indent=2, default=str) + "\n")

    width = max(len(key) for key in summary)
    for key, value in summary.items():
        print(f"{key:<{width}}  {value}")
    print(f"record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
