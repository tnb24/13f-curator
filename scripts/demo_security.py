#!/usr/bin/env python3
"""Show the two independent guards on the agent's database access.

The static validator refuses a statement before the database sees it. The
engine accepts one SELECT statement only. Behind that check, the locked DuckDB
settings refuse a write, a setting change, or a file read. Neither guard
depends on the prompt or on the model.

    python scripts/demo_security.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.db import Engine
from agents.validate import SQLRejected, check

VALIDATOR_CASES = [
    "DROP TABLE holdings",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT accession_number FROM holdings; DROP TABLE holdings",
    "SELECT * FROM query_table('holdings')",
    "SELECT accession_number FROM holdings, range(10)",
    "SELECT value FROM holdings",
    "SELECT accession_number FROM holdings WHERE upper(put_call) = 'CALL'",
]

ENGINE_CASES = [
    "DELETE FROM holdings",
    "CREATE TABLE x (a INT)",
    "DROP VIEW holdings",
    "SET enable_external_access = true",
    "SELECT * FROM read_text('verify.py')",
]


def _attempt(run, statement: str) -> None:
    try:
        run(statement)
        print(f"  RAN     {statement}")
    except Exception as refused:  # Any refusal will do; its message is the point.
        first_line = str(refused).splitlines()[0]
        print(f"  REFUSE  {statement}\n            {first_line}")


def main() -> int:
    with Engine() as engine:
        cursor = engine.cursor()
        print("guard 1: the static validator")
        for statement in VALIDATOR_CASES:
            try:
                check(statement, cursor)
                print(f"  ACCEPT  {statement}")
            except SQLRejected as rejected:
                print(f"  REJECT  {statement}\n            reason: {rejected}")
        cursor.close()

        print("\nguard 2: the engine, with no validator in front")
        for statement in ENGINE_CASES:
            _attempt(engine.query, statement)

        print("\nbehind guard 2: the DuckDB settings alone, past the SELECT check")
        raw = engine.cursor()
        for statement in ENGINE_CASES:
            _attempt(raw.execute, statement)
        raw.close()

        rows = engine.query("SELECT count(*) FROM holdings").rows[0][0]
        print(f"\nholdings still holds {rows} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
