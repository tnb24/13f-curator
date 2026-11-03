"""Attribution rules. tests/test_pipeline.py runs the whole stage."""

from __future__ import annotations

import pytest

from curator.attribution import _tokens


@pytest.mark.parametrize("value, tokens", [
    ("1", ["1"]),
    ("1, 11", ["1", "11"]),
    ("2,5", ["2", "5"]),
    (" 3 ,", ["3"]),
    (None, []),
    ("", []),
])
def test_other_manager_tokens_are_whole(value, tokens) -> None:
    assert _tokens(value) == tokens
    if value and "11" in value:
        assert "1" in _tokens(value) and _tokens(value).count("1") == 1
