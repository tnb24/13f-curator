"""Where the official 13F list lives for each quarter, and what a 404 says."""

from __future__ import annotations

import pytest

from curator.cache import FetchError
from curator.cusip_check import fetch_official, official_list_urls

BASE = "https://www.sec.gov/files/investment/"


class FakeCache:
    """Serve canned bodies. A missing URL fails with the status in `failures` or 404."""

    def __init__(self, bodies: dict[str, bytes], failures: dict[str, int] | None = None) -> None:
        self._bodies = bodies
        self._failures = failures or {}
        self.requested: list[str] = []

    def get(self, url: str) -> bytes:
        self.requested.append(url)
        if url in self._bodies:
            return self._bodies[url]
        status = self._failures.get(url, 404)
        raise FetchError(f"GET {url} failed after 1 attempts (status {status})", status=status)


def test_the_two_verified_quarters_keep_their_exact_urls() -> None:
    assert official_list_urls("2026Q2") == (BASE + "13flist2026q2-txt.txt",)
    # 13flist2026q1-txt.txt returns 404, so the Q1 file must not take the Q2 name.
    assert official_list_urls("2026Q1") == (BASE + "13flist2026q1.txt",)


@pytest.mark.parametrize("quarter, stem", [
    ("2026Q3", "13flist2026q3"),
    ("2025Q4", "13flist2025q4"),
    ("2019Q1", "13flist2019q1"),
])
def test_any_other_quarter_tries_both_file_names(quarter: str, stem: str) -> None:
    assert official_list_urls(quarter) == (BASE + stem + ".txt", BASE + stem + "-txt.txt")


def test_a_404_moves_on_to_the_next_name() -> None:
    urls = official_list_urls("2026Q3")
    cache = FakeCache({urls[1]: b"body"})
    assert fetch_official(cache, "2026Q3") == b"body"
    assert cache.requested == list(urls)


def test_the_first_name_that_answers_ends_the_search() -> None:
    urls = official_list_urls("2026Q3")
    cache = FakeCache({urls[0]: b"first", urls[1]: b"second"})
    assert fetch_official(cache, "2026Q3") == b"first"
    assert cache.requested == [urls[0]]


def test_a_quarter_with_no_list_gets_a_clear_message() -> None:
    cache = FakeCache({})
    with pytest.raises(FetchError) as caught:
        fetch_official(cache, "2026Q3")
    message = str(caught.value)
    assert message.startswith("no official 13F list for 2026Q3: the SEC returned 404 for ")
    for url in official_list_urls("2026Q3"):
        assert url in message
    assert "--quarters" in message
    assert caught.value.status == 404


def test_a_known_quarter_with_a_404_names_its_one_url() -> None:
    with pytest.raises(FetchError, match=r"no official 13F list for 2026Q2.*13flist2026q2-txt"):
        fetch_official(FakeCache({}), "2026Q2")


def test_a_failure_other_than_a_404_is_not_hidden() -> None:
    urls = official_list_urls("2026Q3")
    cache = FakeCache({}, failures={urls[0]: 503})
    with pytest.raises(FetchError, match="status 503"):
        fetch_official(cache, "2026Q3")
    assert cache.requested == [urls[0]]
