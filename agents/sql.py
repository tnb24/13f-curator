"""The model call: question in, candidate SQL out.

`generate` asks the model for a flat JSON object -- `agents/llm.py` warns that
nesting lowers quality -- and re-checks the parsed object in Python, because
guided decoding may be inactive on a development endpoint. The system prompt is
identical on every call, so a repair call keeps the prefix cacheable.

A reply is one of two objects:

    {"sql": "...", "unit": "...", "result_shape": "..."}    the statement
    {"lookup": ["Alphabet Inc", ...]}                       names to resolve first

The agent resolves each name in the lookup with the same index, and calls the
model again with `lookups`. A call with `allow_lookup=False` offers the
statement schema only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from agents.prompts import build_user_prompt, system_prompt
from agents.resolve import MAX_LOOKUP_CHARS, MAX_LOOKUP_NAMES, Issuer, Lookup

# The model call: messages in, the reply text out. agents.llm.complete is the
# live one; the tests pass a function that returns a fixed reply.
Complete = Callable[..., str]

# A SQL statement for this schema runs to about 200 tokens. The cap stops a
# reply that never closes its JSON object from running on for seconds.
MAX_TOKENS = 512

GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "sql":          {"type": "string"},
        "unit":         {"type": "string",
                         "enum": ["USD", "SHARES", "COUNT", "PERCENT", "NAME", "DATE", "NONE"]},
        "result_shape": {"type": "string", "enum": ["scalar", "list", "pair"]},
    },
    "required": ["sql", "unit", "result_shape"],
}

LOOKUP_SCHEMA = {
    "type": "object",
    "properties": {
        "lookup": {"type": "array", "items": {"type": "string"},
                   "minItems": 1, "maxItems": MAX_LOOKUP_NAMES},
    },
    "required": ["lookup"],
}

# The first call may take either reply. A call after a lookup takes a statement only.
REPLY_SCHEMA = {"anyOf": [GENERATION_SCHEMA, LOOKUP_SCHEMA]}

_UNITS = frozenset(GENERATION_SCHEMA["properties"]["unit"]["enum"])
_SHAPES = frozenset(GENERATION_SCHEMA["properties"]["result_shape"]["enum"])
_REQUIRED_KEYS = ("sql", "unit", "result_shape")


class AgentError(Exception):
    """The model response does not have the shape the agent requires."""


# The development model answers with the JSON object inside a markdown fence.
# `agents.llm.complete_json` runs `json.loads` on the raw text, so it raises on
# every live call. So this module calls `complete` and parses the text itself.
# The schema still goes to the endpoint, so guided decoding keeps working where
# the endpoint supports it.
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _first_json_object(text: str) -> str:
    """Return the first balanced JSON object in the text.

    A brace inside a string literal does not change the depth, so the scan
    tracks whether it sits inside a string.
    """
    start = text.find("{")
    if start == -1:
        raise AgentError(f"model response holds no JSON object: {text[:200]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise AgentError(f"model response holds no complete JSON object: {text[:200]!r}")


def _parse_object(raw: str) -> Any:
    """Parse the model text into an object, tolerating a markdown fence."""
    text = raw.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_first_json_object(text))
    except json.JSONDecodeError as exc:
        raise AgentError(f"model response did not parse as JSON: {text[:200]!r}") from exc


def _check_shape(response: Any) -> None:
    """Raise AgentError unless the response is a well-formed generation."""
    if not isinstance(response, dict):
        raise AgentError(f"model returned {type(response).__name__}, expected an object")
    missing = [key for key in _REQUIRED_KEYS if key not in response]
    if missing:
        raise AgentError(f"model response lacks key(s): {', '.join(missing)}")
    for key in _REQUIRED_KEYS:
        if not isinstance(response[key], str):
            raise AgentError(
                f"model field {key!r} holds {type(response[key]).__name__}, expected a string"
            )
    if response["unit"] not in _UNITS:
        raise AgentError(
            f"unknown unit {response['unit']!r}; expected one of {sorted(_UNITS)}"
        )
    if response["result_shape"] not in _SHAPES:
        raise AgentError(
            f"unknown result_shape {response['result_shape']!r}; "
            f"expected one of {sorted(_SHAPES)}"
        )


def _check_lookup(names: Any) -> list[str]:
    """Return the names of a lookup reply, cleaned. Raise AgentError for a bad list.

    A name is cut at MAX_LOOKUP_CHARS and a list at MAX_LOOKUP_NAMES. The agent
    echoes a name into the next prompt, so the limits keep that prompt small.
    An angle bracket becomes a space: no legal name has one, and a name must
    not close the <looked_up_names> block.
    """
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise AgentError(f"model field 'lookup' holds {names!r}, expected a list of strings")
    cleaned = list(dict.fromkeys(
        " ".join(name.replace("<", " ").replace(">", " ").split())[:MAX_LOOKUP_CHARS]
        for name in names
    ))
    cleaned = [name for name in cleaned if name][:MAX_LOOKUP_NAMES]
    if not cleaned:
        raise AgentError("model field 'lookup' holds no name")
    return cleaned


def generate(
    question: str,
    funds: list[str],
    issuers: list[Issuer],
    complete: Complete,
    error: str | None = None,
    lookups: Sequence[Lookup] = (),
    allow_lookup: bool = False,
    previous: str | None = None,
    *,
    quarters: Sequence[str],
) -> dict:
    """Return {"sql", "unit", "result_shape"}, or {"lookup": [name, ...]}, for one question.

    `error` carries the rejection message on a repair call, and `previous` the
    statement that it rejected: the model cannot fix a statement that it does
    not see. `lookups` are the resolved names that the model asked for earlier.
    All three go into the user message only; the system prompt never changes
    between calls. A reply with a "sql" key is a statement, even when it also
    holds a "lookup" key. `quarters` are the report quarters of the dataset,
    for the system prompt.
    """
    user = build_user_prompt(question, funds, issuers, lookups)
    if error:
        if previous:
            user += f"\n\nYour previous statement:\n{previous.strip()}\n\nIt was rejected: {error}"
        else:
            user += f"\n\nYour previous statement was rejected: {error}"
        if allow_lookup:
            user += "\nIf a company or fund that you need is missing above, ask for a lookup."
        user += "\nReturn one corrected JSON object."
    messages = [
        {"role": "system", "content": system_prompt(quarters)},
        {"role": "user", "content": user},
    ]
    schema = REPLY_SCHEMA if allow_lookup else GENERATION_SCHEMA
    raw = complete(messages, schema=schema, max_tokens=MAX_TOKENS)
    response = _parse_object(raw)
    if isinstance(response, dict) and "lookup" in response and "sql" not in response:
        return {"lookup": _check_lookup(response["lookup"])}
    _check_shape(response)
    return response
