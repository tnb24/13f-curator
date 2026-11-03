"""LLM access for the agent: one OpenAI-compatible chat endpoint, over httpx.

Configuration comes from the environment. Copy `.env.example` to `.env`:

    LLM_BASE_URL      OpenAI-compatible endpoint, for example http://localhost:11434/v1
    LLM_API_KEY       bearer token
    LLM_MODEL         model id; leave blank to use the first model the endpoint lists
    LLM_MODE          "live" or "mock"
    LLM_TOKEN_BUDGET  hard cap on tokens for one process

Any OpenAI-compatible server works: Ollama, LM Studio, vLLM, llama.cpp, OpenRouter.
LLM_MODE=mock returns fixed replies and contacts no endpoint.

Check the setup with:  python -m agents.llm

The challenge organisers supplied the first version of this module, built on
the OpenAI SDK. This version keeps its interface, its settings, and its
budget, and sends the one POST it needs through httpx. Importing the SDK took
about 1.7 s per process; httpx takes about 0.3 s.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Iterable
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 4
_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
_MAX_RETRY_AFTER_SECONDS = 30.0


class LLMError(RuntimeError):
    """The endpoint could not be reached, or returned something unusable."""


class TokenBudgetExceeded(LLMError):
    """The run exceeded its token allowance."""


class _Budget:
    """Cumulative token usage for a run, with a hard cap."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def charge(self, usage: dict[str, Any] | None) -> None:
        with self._lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
            if self.total > self.limit:
                raise TokenBudgetExceeded(
                    f"token budget exhausted: {self.total} > {self.limit} after {self.calls} calls"
                )

    def snapshot(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total,
            "limit": self.limit,
        }


BUDGET = _Budget(limit=int(os.environ.get("LLM_TOKEN_BUDGET", 2_000_000)))


class _Endpoint:
    """The configured endpoint: one connection pool and the model id.

    The pool keeps the TLS connection open between questions, so only the
    first question in a process pays for a handshake.
    """

    def __init__(self) -> None:
        base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
        if not base_url:
            raise LLMError("LLM_BASE_URL is not set. Copy .env.example to .env and fill it in.")
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(
            timeout=REQUEST_TIMEOUT,
            headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'not-required')}"},
        )
        self.model = (os.environ.get("LLM_MODEL") or "").strip() or self._first_model()

    def _first_model(self) -> str:
        """Ask the endpoint what it serves and take the first entry."""
        try:
            listing = self.post_or_get("GET", "/models")
        except LLMError as exc:
            raise LLMError(
                f"LLM_MODEL is unset and the endpoint could not be queried: {exc}"
            ) from exc
        models = listing.get("data") or []
        if not models:
            raise LLMError("endpoint reports no available models; set LLM_MODEL explicitly")
        return str(models[0]["id"])

    def post_or_get(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict:
        """Send one request, with retries on throttling and server errors."""
        url = self.base_url + path
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.http.request(method, url, json=body)
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES:
                    raise LLMError(f"{method} {url} failed: {exc}") from exc
                time.sleep(_backoff(attempt, None))
                continue
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as exc:
                    raise LLMError(f"{method} {url} returned a body that is not JSON") from exc
            if response.status_code not in _RETRY_STATUSES or attempt == MAX_RETRIES:
                raise LLMError(
                    f"{method} {url} returned {response.status_code}: {response.text[:300]}"
                )
            time.sleep(_backoff(attempt, response.headers.get("retry-after")))
        raise AssertionError("unreachable")


def _backoff(attempt: int, retry_after: str | None) -> float:
    """Seconds before the next attempt: Retry-After when given, else jittered doubling."""
    if retry_after is not None:
        try:
            return min(max(float(retry_after), 0.0), _MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass
    delay = min(_BACKOFF_BASE_SECONDS * 2 ** attempt, _BACKOFF_MAX_SECONDS)
    return delay * (0.75 + random.random() / 2)


_ENDPOINT: _Endpoint | None = None
_ENDPOINT_LOCK = threading.Lock()


def endpoint() -> _Endpoint:
    """The process-wide endpoint, opened on first use."""
    global _ENDPOINT
    with _ENDPOINT_LOCK:
        if _ENDPOINT is None:
            _ENDPOINT = _Endpoint()
        return _ENDPOINT


def model_id() -> str:
    return endpoint().model


def complete(
    messages: Iterable[dict[str, str]],
    *,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
    stop: list[str] | None = None,
) -> str:
    """One completion. Returns the assistant message content.

    `schema` goes to the endpoint as vLLM's guided_json, which constrains the
    output to the JSON Schema during decoding. An endpoint without guided
    decoding ignores the field, so the caller still checks what comes back.
    """
    msgs = list(messages)
    if os.environ.get("LLM_MODE", "live").lower() == "mock":
        return _mock(msgs, schema)

    api = endpoint()
    body: dict[str, Any] = {
        "model": api.model,
        "messages": msgs,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if stop:
        body["stop"] = stop
    if schema is not None:
        body["guided_json"] = schema

    reply = api.post_or_get("POST", "/chat/completions", body)
    BUDGET.charge(reply.get("usage"))
    try:
        return reply["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"completion reply has no message: {str(reply)[:300]}") from exc


def complete_json(
    messages: Iterable[dict[str, str]],
    schema: dict[str, Any],
    *,
    max_tokens: int = 1024,
) -> Any:
    """Schema-constrained completion, parsed. A reply cut off at max_tokens raises."""
    raw = complete(messages, schema=schema, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"guided output did not parse (truncated at max_tokens={max_tokens}?): {raw[:200]!r}"
        ) from exc


# The contract agents/answer.py must satisfy.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "description": "Scalar or list. Numbers as numbers, not strings.",
            "anyOf": [
                {"type": "number"},
                {"type": "string"},
                {"type": "array", "items": {"type": ["number", "string"]}},
                {"type": "null"},
            ],
        },
        "unit": {
            "type": "string",
            "enum": ["USD", "SHARES", "COUNT", "PERCENT", "NAME", "DATE", "NONE"],
        },
        "sources": {
            "type": "array",
            "items": {"type": "string", "pattern": r"^\d{10}-\d{2}-\d{6}$"},
            "description": "Accession numbers supporting the answer.",
        },
    },
    "required": ["answer", "unit", "sources"],
}


def _mock(messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
    """A fixed reply keyed on the conversation. Exercises wiring, never reasoning."""
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if schema is None:
        return f"[mock response {digest[:8]}]"
    return json.dumps({"answer": None, "unit": "NONE", "sources": []})


def usage() -> dict[str, int]:
    """Token usage for this process."""
    return BUDGET.snapshot()


def check() -> None:
    """Confirm the setup works.  python -m agents.llm"""
    if os.environ.get("LLM_MODE", "live").lower() == "mock":
        print("LLM_MODE=mock -- no endpoint contacted")
        print(complete([{"role": "user", "content": "Confirm connection."}]))
        return
    print(f"endpoint: {os.environ.get('LLM_BASE_URL')}")
    print(f"model:    {model_id()}")
    print(complete([{"role": "user", "content": "Confirm connection."}], max_tokens=64))
    print(f"usage:    {usage()}")


if __name__ == "__main__":
    check()
