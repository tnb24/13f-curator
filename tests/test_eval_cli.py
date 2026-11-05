"""The evaluation CLI must validate its inputs before making model calls."""

import sys

import pytest

from evals import run


@pytest.mark.parametrize("arguments", [
    ["--only", "not-a-question"],
    ["--repeat", "0"],
    ["--only"],
])
def test_invalid_arguments_do_not_start_a_run(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["evals.run", *arguments])
    monkeypatch.setattr(run, "run", lambda *_: pytest.fail("model run started"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_existing_results_require_explicit_overwrite(monkeypatch, tmp_path):
    record = tmp_path / "results.json"
    record.write_text("original", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evals.run", "--out", str(record)])
    monkeypatch.setattr(run, "run", lambda *_: pytest.fail("model run started"))
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert record.read_text(encoding="utf-8") == "original"
