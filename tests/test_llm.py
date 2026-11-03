"""The LLM client and the reply parser, over a mock transport. No network."""

from __future__ import annotations

import json

import httpx
import pytest

from agents import llm
from agents import sql as sqlgen


@pytest.fixture()
def endpoint(monkeypatch):
    """Point the module at a mock server. Return the list of request bodies."""
    bodies: list[dict] = []
    replies: list[httpx.Response] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content) if request.content else {})
        return replies.pop(0)

    monkeypatch.setenv("LLM_BASE_URL", "https://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    built = llm._Endpoint()
    built.http = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(llm, "_ENDPOINT", built)
    return bodies, replies


def _ok(content: str, prompt: int = 10, completion: int = 5) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    })


def test_complete_sends_one_post_with_the_schema(endpoint) -> None:
    bodies, replies = endpoint
    replies.append(_ok("hello"))
    before = llm.usage()
    reply = llm.complete([{"role": "user", "content": "hi"}], schema={"type": "object"},
                         max_tokens=77)
    assert reply == "hello"
    assert bodies[0]["model"] == "test-model"
    assert bodies[0]["max_tokens"] == 77
    assert bodies[0]["temperature"] == 0.0
    assert bodies[0]["guided_json"] == {"type": "object"}
    after = llm.usage()
    assert after["prompt_tokens"] - before["prompt_tokens"] == 10
    assert after["calls"] - before["calls"] == 1


def test_a_429_is_retried(endpoint) -> None:
    _, replies = endpoint
    replies.extend([httpx.Response(429, headers={"retry-after": "1"}), _ok("done")])
    assert llm.complete([{"role": "user", "content": "hi"}]) == "done"


def test_a_400_fails_with_the_body(endpoint) -> None:
    _, replies = endpoint
    replies.append(httpx.Response(400, text="bad model"))
    with pytest.raises(llm.LLMError, match=r"400: bad model"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_a_reply_with_no_message_fails(endpoint) -> None:
    _, replies = endpoint
    replies.append(httpx.Response(200, json={"choices": []}))
    with pytest.raises(llm.LLMError, match=r"no message"):
        llm.complete([{"role": "user", "content": "hi"}])


def test_the_budget_is_a_hard_cap(monkeypatch) -> None:
    budget = llm._Budget(limit=100)
    budget.charge({"prompt_tokens": 60, "completion_tokens": 30})
    with pytest.raises(llm.TokenBudgetExceeded):
        budget.charge({"prompt_tokens": 20, "completion_tokens": 0})


def test_mock_mode_contacts_nothing(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "mock")
    monkeypatch.setattr(llm, "_ENDPOINT", None)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert llm.complete([{"role": "user", "content": "x"}]).startswith("[mock response")


def test_a_missing_base_url_names_the_fix(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setattr(llm, "_ENDPOINT", None)
    monkeypatch.setenv("LLM_BASE_URL", "")
    with pytest.raises(llm.LLMError, match=r".env.example"):
        llm.complete([{"role": "user", "content": "x"}])


@pytest.mark.parametrize("raw", [
    '{"sql": "SELECT 1", "unit": "COUNT", "result_shape": "scalar"}',
    '```json\n{"sql": "SELECT 1", "unit": "COUNT", "result_shape": "scalar"}\n```',
    'Here you go: {"sql": "SELECT \'}\'", "unit": "COUNT", "result_shape": "scalar"} done',
])
def test_the_reply_parser_finds_the_object(raw: str) -> None:
    parsed = sqlgen._parse_object(raw)
    sqlgen._check_shape(parsed)
    assert parsed["unit"] == "COUNT"


@pytest.mark.parametrize("reply, reason", [
    ('{"sql": "SELECT 1", "unit": "COUNT"}', "lacks key"),
    ('{"sql": "SELECT 1", "unit": "MILES", "result_shape": "scalar"}', "unknown unit"),
    ('{"sql": 5, "unit": "COUNT", "result_shape": "scalar"}', "expected a string"),
    ('[1, 2]', "expected an object"),
])
def test_the_reply_parser_rejects_a_bad_shape(reply: str, reason: str) -> None:
    with pytest.raises(sqlgen.AgentError, match=reason):
        sqlgen._check_shape(sqlgen._parse_object(reply))


def test_a_reply_with_no_object_fails() -> None:
    with pytest.raises(sqlgen.AgentError, match=r"no JSON object"):
        sqlgen._parse_object("I cannot answer that.")
