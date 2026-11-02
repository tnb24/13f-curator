"""LLM access for the agent.

The challenge organisers supplied this module. Configuration comes from the
environment. Copy `.env.example` to `.env` and fill it in:

    LLM_BASE_URL      OpenAI-compatible endpoint
    LLM_API_KEY       bearer token
    LLM_MODEL         model id; leave blank to use the first model the endpoint lists
    LLM_MODE          "live" or "mock"
    LLM_TOKEN_BUDGET  hard cap on tokens for one process

Any OpenAI-compatible server works: Ollama, LM Studio, vLLM, llama.cpp.
LLM_MODE=mock runs the test suite with no model.

Check the setup with:  python -m agents.llm

The agent was tested against google/gemma-4-31B-it on vLLM, at temperature 0.0,
with guided decoding through extra_body={"guided_json": ...}.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEMPERATURE = 0.0
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 4  # handled inside the SDK, including 429 backoff


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

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def charge(self, usage: Any) -> None:
        self.calls += 1
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        if self.total > self.limit:
            raise TokenBudgetExceeded(
                f"token budget exhausted: {self.total} > {self.limit} "
                f"after {self.calls} calls"
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


@functools.lru_cache(maxsize=1)
def client() -> OpenAI:
    """The shared client. Cached so connection pooling actually happens.

    Retries are delegated to the SDK, which already understands 429s and Retry-After.
    Hand-rolling that on top would double the backoff and make throttling worse.
    """
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise LLMError(
            "LLM_BASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(
        base_url=base_url,
        api_key=os.environ.get("LLM_API_KEY", "not-required"),
        timeout=REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


@functools.lru_cache(maxsize=1)
def model_id() -> str:
    """Resolve the model to call.

    Prefers LLM_MODEL. If unset, asks the endpoint what it serves and takes the first
    entry. A vLLM deployment usually serves exactly one model.
    """
    explicit = (os.environ.get("LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    try:
        models = client().models.list()
    except Exception as exc:
        raise LLMError(
            f"LLM_MODEL is unset and the endpoint could not be queried: {exc}"
        ) from exc
    if not models.data:
        raise LLMError("endpoint reports no available models; set LLM_MODEL explicitly")
    return models.data[0].id


def complete(
    messages: Iterable[dict[str, str]],
    *,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1024,
    stop: list[str] | None = None,
) -> str:
    """One completion. Returns the assistant message content.

    Pass `schema` to constrain output to a JSON Schema. vLLM's guided decoding masks
    tokens that would violate it during generation, so the result is guaranteed
    parseable rather than merely likely to be. Prefer this over prompting for JSON and
    hoping.

    The guarantee is structural, not semantic: a schema-valid {"answer": null} is still
    wrong. Keep schemas flat -- deeply nested constraints measurably degrade answer
    quality, because the model spends capacity satisfying structure.
    """
    msgs = list(messages)
    if os.environ.get("LLM_MODE", "live").lower() == "mock":
        return _mock(msgs, schema)

    # Resolve the client before the model: if LLM_BASE_URL is unset, the direct
    # "copy .env.example" message is more useful than one nested inside model lookup.
    api = client()
    kwargs: dict[str, Any] = {
        "model": model_id(),
        "messages": msgs,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if stop:
        kwargs["stop"] = stop
    if schema is not None:
        # vLLM reads guided_json off the request body on the OpenAI-compatible route.
        kwargs["extra_body"] = {"guided_json": schema}

    try:
        resp = api.chat.completions.create(**kwargs)
    except TokenBudgetExceeded:
        raise
    except Exception as exc:
        raise LLMError(f"completion failed: {exc}") from exc

    BUDGET.charge(getattr(resp, "usage", None))
    return resp.choices[0].message.content or ""


def complete_json(
    messages: Iterable[dict[str, str]],
    schema: dict[str, Any],
    *,
    max_tokens: int = 1024,
) -> Any:
    """Schema-constrained completion, parsed.

    Guided decoding makes the parse safe, but this still wraps it: a response truncated
    at max_tokens is a valid prefix and invalid JSON, and that is worth surfacing
    loudly rather than swallowing.
    """
    raw = complete(messages, schema=schema, max_tokens=max_tokens)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"guided output did not parse (truncated at max_tokens={max_tokens}?): "
            f"{raw[:200]!r}"
        ) from exc


# The contract agents/answer.py must satisfy. Extra fields are allowed but ignored; do not
# remove required ones.
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
    """Deterministic canned responses for offline development and CI.

    Keyed on a hash of the conversation, so responses are stable across runs but differ
    across prompts. Never returns a correct answer: mock mode exercises wiring, not
    reasoning.
    """
    import hashlib

    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    if schema is None:
        return f"[mock response {digest[:8]}]"
    return json.dumps({"answer": None, "unit": "NONE", "sources": []})


def usage() -> dict[str, int]:
    """Token usage for this process."""
    return BUDGET.snapshot()


def check() -> None:
    """Confirm the setup works.  python -m agents.llm"""
    mode = os.environ.get("LLM_MODE", "live").lower()
    if mode == "mock":
        print("LLM_MODE=mock -- no endpoint contacted")
        print(complete([{"role": "user", "content": "Confirm connection."}]))
        return
    print(f"endpoint: {os.environ.get('LLM_BASE_URL')}")
    print(f"model:    {model_id()}")
    print(complete([{"role": "user", "content": "Confirm connection."}], max_tokens=64))
    print(f"usage:    {usage()}")


if __name__ == "__main__":
    check()
