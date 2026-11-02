"""Sequential EDGAR fetcher with rate limiting and fixed backoff.

SEC blocks clients that send requests too fast or omit a contact address.
This module holds one shared ``httpx.Client`` and never sends two requests
at the same time.
"""

from __future__ import annotations

import sys
import threading
import time

import httpx


class FetchError(RuntimeError):
    """A GET did not return 200 after every retry."""


class RateLimiter:
    """Token bucket. Default rate is 5 requests per second."""

    def __init__(self, rate_per_second: float = 5.0) -> None:
        self._rate = float(rate_per_second)
        self._tokens = 1.0
        self._updated_at = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until the caller may send one request."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._tokens = min(1.0, self._tokens + elapsed * self._rate)
                self._updated_at = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait_seconds = (1.0 - self._tokens) / self._rate
            time.sleep(wait_seconds)


_MAX_ATTEMPTS = 5
_RETRY_STATUSES = frozenset({429, 503})
_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)
_TIMEOUT_SECONDS = httpx.Timeout(30.0, connect=10.0)


class Fetcher:
    """One sequential client. Every request carries the contact User-Agent."""

    def __init__(self, user_agent: str, limiter: RateLimiter) -> None:
        self._limiter = limiter
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=_TIMEOUT_SECONDS,
            follow_redirects=True,
        )

    def get(self, url: str) -> bytes:
        """Send one GET. Return the body. Raise FetchError after the last retry."""
        attempts_used = 0
        last_status: int | None = None
        last_detail = "unknown error"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            attempts_used = attempt
            self._limiter.acquire()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                last_status = None
                last_detail = f"transport error: {exc}"
                if attempt == _MAX_ATTEMPTS:
                    break
                delay = _retry_delay(attempt, None)
                _log_retry(url, attempt, last_detail, delay)
                time.sleep(delay)
                continue
            if response.status_code == 200:
                return response.content
            last_status = response.status_code
            last_detail = f"status {response.status_code}"
            if 400 <= response.status_code < 500 and response.status_code != 429:
                break
            if attempt == _MAX_ATTEMPTS:
                break
            delay = _retry_delay(attempt, response.headers.get("Retry-After"))
            _log_retry(url, attempt, last_detail, delay)
            time.sleep(delay)
        reason = f"status {last_status}" if last_status is not None else last_detail
        raise FetchError(f"GET {url} failed after {attempts_used} attempts ({reason})")


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Return the wait before the next attempt in seconds."""
    if retry_after is not None:
        try:
            return max(0.0, float(int(retry_after)))
        except ValueError:
            pass
    return _BACKOFF_SECONDS[min(attempt - 1, len(_BACKOFF_SECONDS) - 1)]


def _log_retry(url: str, attempt: int, detail: str, delay: float) -> None:
    print(
        f"curator.http: retry {attempt}/{_MAX_ATTEMPTS} for {url} "
        f"after {detail}; sleeping {delay:.0f}s",
        file=sys.stderr,
    )
