"""Risky-action guard: decide whether a browser action may run, needs a y/n, or is handed off.

The rules are general and site-independent: they only look at the target element's own
text/retained attributes and at the form metadata computed by the observer (`form_has_password`,
`form_has_card_field`, `form_has_search`). No site names or site recipes live here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from jevpilot.browser.port import BrowserAction, ElementRef, Observation
from jevpilot.engine.candidates import is_password_input

LOGIN_PHRASES: Final = ("로그인", "log in", "login", "sign in", "signin")
PAYMENT_PHRASES: Final = ("결제", "pay", "payment", "checkout", "카드번호", "카드 번호", "card number")
MUTATION_PHRASES: Final = (
    "예매", "예약", "구매", "주문", "신청", "제출", "확정",
    "book", "reserve", "purchase", "order", "submit", "confirm", "place order",
)

_TARGETED_KINDS: Final = frozenset({"click", "type_text", "select_option"})

Verdict = Literal["allow", "confirm", "handoff"]


@dataclass(frozen=True, slots=True)
class Safety:
    verdict: Verdict
    reason: str


def _compile(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """ASCII phrases match on word edges (so 'border' is not 'order'); Hangul matches as substring."""
    parts: list[str] = []
    for phrase in phrases:
        body = re.escape(phrase).replace(r"\ ", r"\s+")
        # "Book" is a booking imperative; its plural "books" is a catalog noun.
        suffix = "" if phrase == "book" else "s?"
        parts.append(body if not phrase.isascii() else rf"(?<![a-z]){body}{suffix}(?![a-z])")
    return re.compile("|".join(parts), re.IGNORECASE)


_LOGIN: Final = _compile(LOGIN_PHRASES)
_PAYMENT: Final = _compile(PAYMENT_PHRASES)
_MUTATION: Final = _compile(MUTATION_PHRASES)


def _labels(element: ElementRef) -> list[str]:
    # The browser boundary already filters attributes; input value lengths are
    # not semantic labels. Inspect every other retained attribute, including href/type.
    return [element.text, *(value for name, value in element.attributes.items() if name != "value")]


def _first(pattern: re.Pattern[str], labels: list[str]) -> str | None:
    for label in labels:
        found = pattern.search(label)
        if found:
            return found.group(0)
    return None


def _is_card_field(element: ElementRef) -> bool:
    autocomplete = element.attributes.get("autocomplete", "").lower().split()
    return any(token.startswith("cc-") for token in autocomplete)


def assess(action: BrowserAction, observation: Observation) -> Safety:
    """Classify `action`; handoff (login, payment) outranks confirm (mutation)."""
    target = action.target
    if action.kind not in _TARGETED_KINDS or target is None:
        return Safety("allow", "no_target")

    if is_password_input(target):
        return Safety("handoff", "login: password field")
    if _is_card_field(target):
        return Safety("handoff", "payment: card field")
    labels = _labels(target)
    if action.kind == "click" and target.form_has_password:
        return Safety("handoff", "login: control inside a form with a password field")
    if (hit := _first(_LOGIN, labels)) is not None:
        return Safety("handoff", f"login: '{hit}'")
    if action.kind == "click" and target.form_has_card_field:
        return Safety("handoff", "payment: control inside a form with a card field")
    if (hit := _first(_PAYMENT, labels)) is not None:
        return Safety("handoff", f"payment: '{hit}'")
    if action.kind != "click":
        return Safety("allow", "non_click")
    if not target.form_has_search and (hit := _first(_MUTATION, labels)) is not None:
        return Safety("confirm", f"mutation: '{hit}'")
    return Safety("allow", "no_risk")
