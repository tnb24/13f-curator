"""Validate custom rosters before a rebuild touches the network or output."""

import pytest

from curator.pipeline import load_filers, run
from tests import world


def test_roster_accepts_utf8_bom_and_trims_names(tmp_path):
    roster = tmp_path / "managers.csv"
    roster.write_text("\ufefffund_name,cik\n  Alpha Capital LLC  ,001\n", encoding="utf-8")
    assert load_filers(roster) == [{"fund_name": "Alpha Capital LLC", "cik": "001"}]


@pytest.mark.parametrize(("body", "message"), [
    ("name\nAlpha\n", "fund_name column"),
    ("fund_name\n", "no managers"),
    ("fund_name,cik\n ,123\n", "empty fund_name"),
    ("fund_name\nAlpha\n alpha \n", "repeats fund_name"),
    ("fund_name\nAlpha,unexpected\n", "more fields"),
    ("fund_name,fund_name\nAlpha,Beta\n", "unique fund_name"),
])
def test_invalid_roster_does_not_touch_existing_output_or_fetch(tmp_path, body, message):
    roster = tmp_path / "managers.csv"
    roster.write_text(body, encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text("previous run", encoding="utf-8")
    source = world.FakeSource(world.bodies())
    with pytest.raises(ValueError, match=message):
        run("Test User test@example.com", output, tmp_path / "cache", fetcher=source,
            roster=roster, workers=1)
    assert manifest.read_text(encoding="utf-8") == "previous run"
    assert source.requested == []
