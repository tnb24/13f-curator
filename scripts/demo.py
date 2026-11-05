"""Run one reproducible question without a model endpoint or network access."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.answer import Agent

QUESTION = "Which manager held the largest Apple position in 2026 Q2?"
SQL = (
    "SELECT f.fund_name AS answer, list(DISTINCT h.accession_number) AS accession_number "
    "FROM holdings h JOIN filings f USING (accession_number) "
    "WHERE h.cusip = '037833100' AND h.report_quarter = '2026Q2' AND h.put_call IS NULL "
    "GROUP BY f.fund_name ORDER BY sum(h.value) DESC LIMIT 1"
)


def scripted_model(_messages, **_kwargs) -> str:
    """Supply known SQL; the rest of the agent still runs normally."""
    return json.dumps({"sql": SQL, "unit": "NAME", "result_shape": "scalar"})


def main() -> int:
    print("Offline walkthrough: scripted SQL, no model or network call")
    print(f"Question: {QUESTION}")
    with Agent(complete=scripted_model) as agent:
        answer = agent.ask(QUESTION)
    if answer.trace.error:
        print(f"Demo failed: {answer.trace.error}", file=sys.stderr)
        return 1
    print(f"SQL: {answer.trace.sql}")
    print(f"Result: {json.dumps(answer.contract())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
