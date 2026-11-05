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


def test_offline_demo_queries_the_bundled_dataset() -> None:
    done = _run("scripts/demo.py")
    assert done.returncode == 0
    assert "scripted SQL, no model or network call" in done.stdout
    assert "AQR Capital Management LLC" in done.stdout
    assert "0001167557-26-000226" in done.stdout


def test_the_agent_reports_a_model_failure_in_its_exit_status() -> None:
    # Mock mode returns a fixed reply that is not an SQL generation response.
    done = _run("-m", "agents.answer", "Which manager held Apple in 2026 Q2?")
    assert done.returncode == 1
    assert json.loads(done.stdout) == {"answer": None, "unit": "NONE", "sources": []}
    assert "returning a null answer" in done.stderr


def test_the_pipeline_refuses_a_user_agent_without_an_address() -> None:
    done = _run("main.py", "--user-agent", "nobody")
    assert done.returncode != 0
    assert "contact address" in done.stderr


def test_the_pipeline_explains_an_invalid_roster(tmp_path) -> None:
    roster = tmp_path / "bad.csv"
    roster.write_text("name\nAlpha\n", encoding="utf-8")
    done = _run("main.py", "--user-agent", "Test User test@example.com",
                "--filers", str(roster), "--output", str(tmp_path / "output"))
    assert done.returncode == 2
    assert "fund_name column" in done.stderr
    assert "Traceback" not in done.stderr


def test_the_pipeline_prints_help() -> None:
    done = _run("main.py", "--help")
    assert done.returncode == 0
    assert "--cache-dir" in done.stdout
    assert "--filers" in done.stdout
    assert "--quarters" in done.stdout
    assert "--cutoff" in done.stdout


def test_the_pipeline_exits_nonzero_if_an_optional_report_fails(tmp_path, monkeypatch) -> None:
    from curator import pipeline

    monkeypatch.setattr(sys, "argv", ["main.py", "--user-agent", "Test User test@example.com",
                                     "--output", str(tmp_path)])
    monkeypatch.setattr(pipeline, "run", lambda *_a, **_kw: {
        "attribution": True, "cusip_check": False,
    })
    assert pipeline.main() == 1


def test_the_pipeline_refuses_a_quarter_it_cannot_read() -> None:
    done = _run("main.py", "--user-agent", "Test User test@example.com", "--quarters", "26Q2")
    assert done.returncode == 2
    assert "not a quarter" in done.stderr
