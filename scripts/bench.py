#!/usr/bin/env python3
"""Measure the pipeline and the agent. No network and no model.

    python scripts/bench.py                 # both, 5 pipeline runs, 20 agent rounds
    python scripts/bench.py --pipeline 3 --agent 0

The pipeline part rebuilds the dataset from the local EDGAR cache (.cache/http)
into a temporary directory, as a subprocess per run, and reports the median
time of each stage. Run main.py once first, so every response is cached.

The agent part replaces the model with each eval question's reference SQL, so
it times the agent's own work only: name resolution, validation, and the
query. A model call adds network and provider time on top; evals/run.py
measures that end to end.

docs/PERFORMANCE.md quotes this script's output. Run it in a virtual
environment built from requirements.txt alone: pandas, a dev dependency,
adds about 0.5 s to the first Arrow or DuckDB call in a process.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_STAGE = re.compile(r"main: (\w+) took ([0-9.]+)s")


def bench_pipeline(runs: int) -> None:
    stages: dict[str, list[float]] = {}
    totals: list[float] = []
    with tempfile.TemporaryDirectory(prefix="bench-") as scratch:
        for _ in range(runs):
            started = time.perf_counter()
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--pipeline-child", scratch]
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
            totals.append(time.perf_counter() - started)
            manifest = json.loads((Path(scratch) / "manifest.json").read_text(encoding="utf-8"))
            if manifest["run"]["cache_misses"]:
                raise SystemExit("a request missed the cache; run main.py once first")
            for stage, seconds in _STAGE.findall(done.stderr):
                stages.setdefault(stage, []).append(float(seconds))
    print(f"\npipeline, warm, median of {runs} runs")
    print("| stage | seconds |\n|---|---|")
    for stage, values in stages.items():
        print(f"| {stage} | {statistics.median(values):.2f} |")
    print(f"| whole process (wall) | {statistics.median(totals):.2f} |")


def bench_agent(rounds: int) -> None:
    from agents.answer import Agent

    items = [i for i in json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
             if "reference_sql" in i]
    shapes = {"names": "list", "labels": "list"}
    replies = {
        i["question"]: json.dumps({
            "sql": i["reference_sql"], "unit": "NONE",
            "result_shape": shapes.get(i["expect"]["kind"], "scalar"),
        })
        for i in items
    }

    def model(messages, schema=None, max_tokens=0) -> str:
        question = messages[1]["content"].split("\n", 1)[0].removeprefix("Question: ")
        return replies[question]

    startups = []
    for _ in range(5):
        with Agent(complete=model) as agent:
            startups.append(agent.startup_ms)
    steps: dict[str, list[float]] = {}
    with Agent(complete=model) as agent:
        for _ in range(rounds):
            for item in items:
                answer = agent.ask(item["question"])
                if answer.trace.error or answer.answer is None:
                    raise RuntimeError(
                        f"reference SQL failed for {item['id']}: {answer.trace.error}")
                timings = answer.trace.timings_ms
                for step in ("resolve", "validate", "query", "total"):
                    steps.setdefault(step, []).append(timings.get(step, 0.0))
    print(f"\nagent, local work only, {rounds} rounds x {len(items)} questions")
    print(f"| step | median ms | p95 ms |\n|---|---|---|\n"
          f"| startup (engine + indexes) | {statistics.median(startups):.1f} | |")
    for step, values in steps.items():
        ordered = sorted(values)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        print(f"| {step} | {statistics.median(values):.2f} | {p95:.2f} |")

    imports = []
    processes = []
    for _ in range(5):
        started = time.perf_counter()
        subprocess.run([sys.executable, "-c", "import agents.answer, agents.llm"],
                       cwd=ROOT, check=True)
        imports.append(time.perf_counter() - started)
        started = time.perf_counter()
        subprocess.run([sys.executable, "-m", "agents.answer", "What did Citadel hold in 2026 Q3?"],
                       cwd=ROOT, check=True, capture_output=True)
        processes.append(time.perf_counter() - started)
    print(f"| interpreter + imports (wall) | {1000 * statistics.median(imports):.0f} | |")
    print(f"| CLI process, out-of-scope question (wall) | "
          f"{1000 * statistics.median(processes):.0f} | |")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python scripts/bench.py")
    parser.add_argument("--pipeline", type=int, default=5, help="pipeline runs (0 to skip)")
    parser.add_argument("--agent", type=int, default=20, help="agent rounds (0 to skip)")
    parser.add_argument("--pipeline-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pipeline_child:
        from curator.http import Fetcher
        from curator.pipeline import run

        def offline(*_args, **_kwargs):
            raise RuntimeError("benchmark forbids network: populate the raw response store first")

        Fetcher.get = offline
        run("Bench Run bench@example.com", args.pipeline_child)
        return 0
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    if args.pipeline:
        bench_pipeline(args.pipeline)
    if args.agent:
        bench_agent(args.agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
