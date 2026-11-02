"""The one model call: question in, candidate SQL out.

`generate` asks the model for a flat JSON object -- `agents/llm.py` warns that
nesting lowers quality -- and re-checks the parsed object in Python, because
guided decoding may be inactive on a development endpoint. The system prompt is
identical on every call, so a repair call keeps the prefix cacheable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.llm import complete
from agents.prompts import SYSTEM_PROMPT, build_user_prompt

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

_UNITS = frozenset(GENERATION_SCHEMA["properties"]["unit"]["enum"])
_SHAPES = frozenset(GENERATION_SCHEMA["properties"]["result_shape"]["enum"])
_REQUIRED_KEYS = ("sql", "unit", "result_shape")


class AgentError(Exception):
    """The model response does not have the shape the agent requires."""


# The development model answers with the JSON object inside a markdown fence.
# `agents.llm.complete_json` runs `json.loads` on the raw text, so it raises on
# every live call. That file is frozen, so this module calls `complete` and
# parses the text itself. The schema still goes to the endpoint, so guided
# decoding keeps working where the endpoint supports it.
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


def generate(
    question: str,
    funds: list[str],
    issuers: list[str],
    error: str | None = None,
) -> dict:
    """Return {"sql", "unit", "result_shape"} for one question.

    `error` carries the validator message on a repair call. It goes into the
    user message only; the system prompt never changes between calls.
    """
    user = build_user_prompt(question, funds, issuers)
    if error:
        user += f"\n\nYour previous statement was rejected: {error}"
        user += "\nReturn one corrected JSON object."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    raw = complete(messages, schema=GENERATION_SCHEMA, max_tokens=1024)
    response = _parse_object(raw)
    _check_shape(response)
    return response
