"""Grade one agent answer against one expectation from evals/questions.json.

An expectation has a kind. Each kind compares one shape of answer:

    number   an integer, or a float equal to it
    name     one name; a [name, number] pair also passes
    names    a set of names, in any order
    labels   a set of (name, label) pairs, given as strings or as pairs
    yesno    "yes" or "no"; True or False also passes
    date     an ISO date, YYYY-MM-DD
    null     no answer

Names compare after normalisation: lower case, and every run of characters
that are neither letters nor digits becomes one space. "D. E. Shaw & Co. Inc."
and "d e shaw co inc" are then equal. An `accept` list holds alternative
expectations, each with the reason it is defensible.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from typing import Any

_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalise(text: Any) -> str:
    return _NON_WORD.sub(" ", str(text).lower()).strip()


def _number(answer: Any) -> Decimal | None:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return Decimal(str(answer))
    if isinstance(answer, str):
        try:
            return Decimal(answer.replace(",", "").replace("$", "").strip())
        except InvalidOperation:
            return None
    if isinstance(answer, list) and len(answer) == 1:
        return _number(answer[0])
    return None


def _one_name(answer: Any) -> str | None:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        if len(answer) == 1 and isinstance(answer[0], str):
            return answer[0]
        if len(answer) == 2 and isinstance(answer[0], str) and _number(answer[1]) is not None:
            return answer[0]
    return None


def _items(answer: Any) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    if isinstance(answer, list):
        return [
            " ".join(map(str, item)) if isinstance(item, list) else str(item) for item in answer
        ]
    return []


def _labels_match(answer: Any, expected: dict[str, str]) -> bool:
    items = [normalise(item) for item in _items(answer)]
    if len(items) != len(expected):
        return False
    labels = {normalise(label) for label in expected.values()}
    for name, label in expected.items():
        key, wanted = normalise(name), normalise(label)
        hits = [item for item in items if key in item]
        if len(hits) != 1:
            return False
        others = labels - {wanted}
        words = set(hits[0].split())
        if wanted not in words or words & others:
            return False
    return True


def matches(kind: str, expected: Any, answer: Any) -> bool:
    """True when the answer has the expected value, read the way `kind` names."""
    if kind == "null":
        return answer is None
    if answer is None:
        return False
    if kind == "number":
        got = _number(answer)
        if got is None or not got.is_finite():
            return False
        wanted = Decimal(str(expected))
        # Integer counts and dollar totals are exact. A relative tolerance on
        # a trillion-dollar total used to accept an error of a thousand dollars;
        # conversion through float also lost integers above 2**53.
        if wanted == wanted.to_integral_value():
            return got == wanted
        return abs(got - wanted) <= Decimal("1e-9") * max(Decimal(1), abs(wanted))
    if kind == "name":
        name = _one_name(answer)
        return name is not None and normalise(name) == normalise(expected)
    if kind == "names":
        return sorted(map(normalise, _items(answer))) == sorted(map(normalise, expected))
    if kind == "labels":
        return _labels_match(answer, expected)
    if kind == "yesno":
        if isinstance(answer, bool):
            answer = "yes" if answer else "no"
        return normalise(answer) == normalise(expected)
    if kind == "date":
        return str(answer)[:10] == str(expected)
    raise ValueError(f"unknown expectation kind {kind!r}")


def grade(item: dict[str, Any], answer: Any) -> bool:
    """True when the answer meets the main expectation or an accepted one."""
    options = [item["expect"], *item.get("accept", [])]
    return any(matches(option["kind"], option.get("value"), answer) for option in options)


def citation(expected: Iterable[str], cited: Iterable[str]) -> tuple[float, float]:
    """Return (recall, precision) of the cited accession numbers."""
    expected, cited = set(expected), set(cited)
    if not expected:
        return (1.0, 1.0 if not cited else 0.0)
    hit = len(expected & cited)
    return hit / len(expected), (hit / len(cited) if cited else 0.0)


def reference(item: dict[str, Any], rows: list[tuple]) -> tuple[Any, list[str]]:
    """Read (expected value, sources) out of the rows of an item's reference SQL.

    The last column holds accession numbers, as one value or a list. Every
    other column is the answer, read the way the item's kind names.
    """
    kind = item["expect"]["kind"]
    sources: set[str] = set()
    for row in rows:
        cell = row[-1]
        sources.update(cell if isinstance(cell, list) else [cell])
    if kind == "names":
        value: Any = [row[0] for row in rows]
    elif kind == "labels":
        value = {row[0]: row[1] for row in rows}
    else:
        value = rows[0][0] if rows else None
        if isinstance(value, datetime.date):
            value = value.isoformat()
    return value, sorted(source for source in sources if source)
