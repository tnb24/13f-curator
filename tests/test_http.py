"""The EDGAR fetcher and the response cache, over a mock transport. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from curator import http
from curator.cache import Cache, FetchError, Manifest


class Server:
    """Answer each request with the next scripted (status, headers, body), or raise."""

    def __init__(self, *responses) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, headers, body = reply
        return httpx.Response(status, headers=headers, content=body)


@pytest.fixture()
def sleeps(monkeypatch) -> list[float]:
    """Record each backoff sleep instead of sleeping."""
    slept: list[float] = []
    monkeypatch.setattr(http.time, "sleep", slept.append)
    return slept


class NoLimit:
    """A limiter that never waits, so every recorded sleep is a backoff."""

    def acquire(self) -> None:
        pass


def _fetcher(server: Server) -> http.Fetcher:
    return http.Fetcher("Test User test@example.com", NoLimit(),
                        transport=httpx.MockTransport(server))


def test_a_200_returns_the_body_and_sends_the_contact_header(sleeps) -> None:
    server = Server((200, {}, b"body"))
    assert _fetcher(server).get("https://example.test/a") == b"body"
    assert server.requests[0].headers["user-agent"] == "Test User test@example.com"
    assert sleeps == []


def test_a_404_fails_at_once(sleeps) -> None:
    server = Server((404, {}, b""))
    with pytest.raises(FetchError, match=r"after 1 attempts .status 404") as caught:
        _fetcher(server).get("https://example.test/missing")
    assert caught.value.status == 404  # a caller can tell a missing file from a failure
    assert sleeps == []


def test_a_503_is_retried_with_backoff(sleeps) -> None:
    server = Server((503, {}, b""), (503, {}, b""), (200, {}, b"ok"))
    assert _fetcher(server).get("https://example.test/a") == b"ok"
    assert sleeps == [1.0, 2.0]


def test_retry_after_is_honoured_and_capped(sleeps) -> None:
    server = Server((429, {"Retry-After": "3"}, b""), (429, {"Retry-After": "600"}, b""),
                    (200, {}, b"ok"))
    assert _fetcher(server).get("https://example.test/a") == b"ok"
    assert sleeps == [3.0, 60.0]


def test_a_transport_error_is_retried(sleeps) -> None:
    server = Server(httpx.ConnectError("refused"), (200, {}, b"ok"))
    assert _fetcher(server).get("https://example.test/a") == b"ok"
    assert len(sleeps) == 1


def test_every_attempt_failing_raises_with_the_last_status(sleeps) -> None:
    server = Server(*[(500, {}, b"")] * 5)
    with pytest.raises(FetchError, match=r"after 5 attempts .status 500") as caught:
        _fetcher(server).get("https://example.test/a")
    assert caught.value.status == 500
    assert len(server.requests) == 5


def test_no_client_opens_until_the_first_request() -> None:
    fetcher = http.Fetcher("Test User test@example.com", http.RateLimiter(5.0))
    assert fetcher._client is None
    fetcher.close()  # closing an unopened fetcher is harmless


def test_the_rate_limiter_spaces_requests(monkeypatch) -> None:
    """At 5 per second, the second of two back-to-back requests waits 0.2 s."""
    clock = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(http.time, "sleep", sleep)
    limiter = http.RateLimiter(5.0)
    limiter.acquire()
    limiter.acquire()
    assert sum(slept) == pytest.approx(0.2)


class CountingSource:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str) -> bytes:
        self.calls += 1
        return f"body of {url}".encode()


def test_the_cache_fetches_once_then_serves_from_disk(tmp_path) -> None:
    source, manifest = CountingSource(), Manifest()
    cache = Cache(tmp_path, source, manifest)
    first = cache.get("https://example.test/a")
    second = Cache(tmp_path, source, manifest).get("https://example.test/a")
    assert first == second == b"body of https://example.test/a"
    assert source.calls == 1
    assert (manifest.cache_hits, manifest.cache_misses) == (1, 1)
    assert manifest.bytes_downloaded == len(first)


def test_a_body_without_its_sidecar_is_fetched_again(tmp_path) -> None:
    source = CountingSource()
    cache = Cache(tmp_path, source, Manifest())
    cache.get("https://example.test/a")
    for sidecar in (tmp_path / "http").glob("*.json"):
        sidecar.unlink()
    cache.get("https://example.test/a")
    assert source.calls == 2


def test_the_manifest_is_sorted_and_has_fixed_fields(tmp_path) -> None:
    manifest = Manifest()
    manifest.record("https://b", True, 10, 0.0)
    manifest.record("https://a", False, 20, 0.1234)
    manifest.write(tmp_path / "manifest.json")
    document = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [e["url"] for e in document["entries"]] == ["https://a", "https://b"]
    assert document["entries"][0] == {"url": "https://a", "hit": False, "bytes": 20,
                                      "elapsed_seconds": 0.123}
    assert set(document["run"]) == {"started_at", "elapsed_seconds", "requests",
                                    "cache_hits", "cache_misses", "bytes_downloaded"}
    assert "scope" not in document  # recorded only when the caller passes a scope


def test_the_manifest_records_the_scope_between_the_run_and_the_entries(tmp_path) -> None:
    manifest = Manifest()
    manifest.record("https://a", True, 1, 0.0)
    manifest.write(tmp_path / "manifest.json", {"quarters": ["2026Q1"]})
    document = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert list(document) == ["run", "scope", "entries"]
    assert document["scope"] == {"quarters": ["2026Q1"]}
