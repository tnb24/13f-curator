"""Run every example question in one process and record what each one cost.

All questions run in one process. `agents/llm.py` counts calls and tokens in a
module-level budget, so a subprocess per question reports zero for every question.

The questions come out of `docs/questions-examples.md`. A copy pasted into this
file goes stale as soon as the document changes.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import llm  # noqa: E402
from agents.answer import main  # noqa: E402

QUESTIONS_DOC = ROOT / "docs" / "questions-examples.md"
OUTPUT = ROOT / "output" / "agent_usage.json"

_FENCE = re.compile(r"```(.*?)```", re.DOTALL)


def read_questions(path: Path) -> list[str]:
    """Return the questions from the first fenced block, in document order."""
    text = path.read_text(encoding="utf-8")
    block = _FENCE.search(text)
    if not block:
        raise SystemExit(f"no fenced block in {path}")
    questions = [line.strip() for line in block.group(1).splitlines() if line.strip()]
    if not questions:
        raise SystemExit(f"the fenced block in {path} holds no question")
    return questions


def run() -> None:
    questions = read_questions(QUESTIONS_DOC)
    print(f"running {len(questions)} question(s)", file=sys.stderr)

    entries: list[dict] = []
    for number, question in enumerate(questions, start=1):
        before = llm.usage()
        started = time.monotonic()
        result = main(question)
        elapsed = time.monotonic() - started
        after = llm.usage()

        entry = {
            "question": question,
            "calls": after["calls"] - before["calls"],
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
            "elapsed_seconds": round(elapsed, 2),
            "result": result,
        }
        entries.append(entry)
        print(
            f"  [{number}/{len(questions)}] calls={entry['calls']} "
            f"{elapsed:5.1f}s  {result['unit']:7s} {str(result['answer'])[:60]}",
            file=sys.stderr,
        )

    totals = {
        "calls": sum(entry["calls"] for entry in entries),
        "prompt_tokens": sum(entry["prompt_tokens"] for entry in entries),
        "completion_tokens": sum(entry["completion_tokens"] for entry in entries),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"questions": entries, "totals": totals}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} -- totals {totals}", file=sys.stderr)


if __name__ == "__main__":
    run()
