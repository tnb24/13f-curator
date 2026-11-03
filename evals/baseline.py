"""Grade the original agent (commit d262efd) on an eval set, with today's grader.

The original agent is a separate checkout with its own pinned dependencies
(DuckDB 1.1.3 and the OpenAI SDK), so it runs in its own virtual environment:

    git archive d262efd | tar -x -C /tmp/baseline
    python -m venv /tmp/venv-baseline
    /tmp/venv-baseline/bin/pip install -r /tmp/baseline/requirements.txt
    /tmp/venv-baseline/bin/python evals/baseline.py /tmp/baseline --set dev

The checkout's `agents` package goes first on sys.path, and this repository
goes last, so `evals.grade` resolves here. The endpoint settings load from
this repository's .env, so no secret is copied into the checkout.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SETS = {"dev": "questions.json", "holdout": "holdout.json"}


def main() -> int:
    parser = argparse.ArgumentParser(prog="python evals/baseline.py")
    parser.add_argument("checkout", type=Path, help="a checkout of commit d262efd")
    parser.add_argument("--set", choices=sorted(SETS), default="dev")
    parser.add_argument("--repeat", type=int, default=1, help="ask each question this many times")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    sys.path.insert(0, str(args.checkout.resolve()))
    sys.path.append(str(REPO))
    from agents import llm  # the checkout's
    from agents.answer import main as ask  # the checkout's
    from evals.grade import citation, grade  # this repository's

    items = json.loads((REPO / "evals" / SETS[args.set]).read_text(encoding="utf-8"))
    records = []
    for item in [item for item in items for _ in range(args.repeat)]:
        before = llm.usage()
        started = time.perf_counter()
        with redirect_stderr(io.StringIO()):
            result = ask(item["question"])
        elapsed = time.perf_counter() - started
        calls = llm.usage()["calls"] - before["calls"]
        passed = grade(item, result["answer"])
        recall, precision = citation(item.get("sources", []), result["sources"])
        records.append({
            "id": item["id"], "passed": passed, "answer": result["answer"],
            "sources": result["sources"], "seconds": round(elapsed, 3), "calls": calls,
            "citation_recall": recall, "citation_precision": precision,
        })
        print(f"  {'PASS' if passed else 'FAIL'} {elapsed:5.2f}s {item['id']:28s} "
              f"{json.dumps(result['answer'], default=str)[:70]}", file=sys.stderr)

    timed = [r["seconds"] for r in records if r["calls"]]
    summary = {
        "set": args.set,
        "agent": "d262efd",
        "questions": len(items),
        "answers": len(records),
        "passed": sum(r["passed"] for r in records),
        "accuracy": round(sum(r["passed"] for r in records) / len(records), 4),
        "citation_recall": round(statistics.mean(r["citation_recall"] for r in records), 4),
        "citation_precision": round(statistics.mean(r["citation_precision"] for r in records), 4),
        "model_calls": sum(r["calls"] for r in records),
        "latency_s_p50": round(statistics.median(timed), 2),
        "latency_s_max": round(max(timed), 2),
    }
    out = args.out or REPO / "evals" / "results" / f"baseline-d262efd-{args.set}.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=2, default=str)
                   + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
