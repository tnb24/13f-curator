"""The two command lines, as a user runs them. No network, no model."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "LLM_MODE": "mock", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


def test_the_agent_prints_one_json_object_for_an_out_of_scope_question() -> None:
    done = _run("-m", "agents.answer", "What did Citadel hold in 2026 Q3?")
    assert done.returncode == 0
    assert json.loads(done.stdout) == {"answer": None, "unit": "NONE", "sources": []}
    assert "out of scope" in done.stderr


def test_the_agent_needs_a_question() -> None:
    assert _run("-m", "agents.answer").returncode == 2


def test_the_pipeline_refuses_a_user_agent_without_an_address() -> None:
    done = _run("main.py", "--user-agent", "nobody")
    assert done.returncode != 0
    assert "contact address" in done.stderr


def test_the_pipeline_prints_help() -> None:
    done = _run("main.py", "--help")
    assert done.returncode == 0
    assert "--cache-dir" in done.stdout
