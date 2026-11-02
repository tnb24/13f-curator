#!/usr/bin/env python3
"""Show the two independent guards on the agent's database access.

The static validator refuses a statement before the database sees it. The
read-only connection refuses a write even when a statement gets past the
validator. Neither guard depends on the prompt or on the model.

    python scripts/demo_security.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import db  # noqa: E402
from agents.validate import SQLRejected, check  # noqa: E402

CASES = [
    "DROP TABLE holdings",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT accession_number FROM holdings; DROP TABLE holdings",
    "SELECT value FROM holdings",
    "SELECT accession_number FROM holdings WHERE upper(put_call) = 'CALL'",
]


def main() -> int:
    print("guard 1: the static validator")
    for statement in CASES:
        try:
            check(statement)
            print(f"  ACCEPT  {statement}")
        except SQLRejected as rejected:
            print(f"  REJECT  {statement}\n            reason: {rejected}")

    print("\nguard 2: the read-only connection")
    connection = db.connect()
    for statement in ("DELETE FROM holdings", "CREATE TABLE x (a INT)"):
        try:
            connection.execute(statement)
            print(f"  RAN     {statement}")
        except Exception as refused:  # noqa: BLE001 - the message is the point.
            first_line = str(refused).splitlines()[0]
            print(f"  REFUSE  {statement}\n            {first_line}")

    rows = connection.execute("SELECT count(*) FROM holdings").fetchone()
    print(f"\nholdings still holds {rows[0]} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
