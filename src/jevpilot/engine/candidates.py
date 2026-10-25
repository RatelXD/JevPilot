"""Rule-based action candidates built from an Observation (no site-specific rules)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from jevpilot.browser.port import ElementRef, Observation

CandidateOp = Literal["click", "type_text", "select_option"]

_CLICK_TAGS: Final = frozenset({"a", "button"})
_CLICK_ROLES: Final = frozenset({"link", "button", "tab", "option", "checkbox", "radio"})
_CLICK_INPUT_TYPES: Final = frozenset({"button", "submit", "reset", "checkbox", "radio", "image"})
_TEXT_INPUT_TYPES: Final = frozenset({"text", "search", "email", "tel", "number", "url"})
_TEXT_ROLES: Final = frozenset({"textbox", "searchbox"})
_SELECT_ROLES: Final = frozenset({"combobox", "listbox"})
_PASSWORD_AUTOCOMPLETE: Final = frozenset({"current-password", "new-password"})


@dataclass(frozen=True, slots=True)
class SafetyFlag:
    """An element withheld from candidates, with the reason."""

    index: int
    backend_node_id: int
    reason: Literal["password_input"]


@dataclass(frozen=True, slots=True)
class ActionCandidates:
    click: tuple[ElementRef, ...]
    type_text: tuple[ElementRef, ...]
    select_option: tuple[ElementRef, ...]
    safety_flags: tuple[SafetyFlag, ...]

    def for_operation(self, operation: str) -> tuple[ElementRef, ...]:
        if operation == "click":
            return self.click
        if operation == "type_text":
            return self.type_text
        if operation == "select_option":
            return self.select_option
        return ()

    def all_elements(self) -> list[ElementRef]:
        """Union of the candidate sets in document (index) order."""
        merged = {e.index: e for e in (*self.click, *self.type_text, *self.select_option)}
        return [merged[i] for i in sorted(merged)]


def is_password_input(element: ElementRef) -> bool:
    attrs = element.attributes
    autocomplete = set(attrs.get("autocomplete", "").lower().split())
    return attrs.get("type", "").lower() == "password" or bool(autocomplete & _PASSWORD_AUTOCOMPLETE)


def _is_click(element: ElementRef) -> bool:
    attrs = element.attributes
    if element.tag in _CLICK_TAGS or element.role in _CLICK_ROLES or element.has_onclick:
        return True
    return element.tag == "input" and attrs.get("type", "text").lower() in _CLICK_INPUT_TYPES


def _is_type_text(element: ElementRef) -> bool:
    attrs = element.attributes
    if element.tag == "textarea" or element.role in _TEXT_ROLES or element.contenteditable:
        return True
    return element.tag == "input" and attrs.get("type", "text").lower() in _TEXT_INPUT_TYPES


def _is_select(element: ElementRef) -> bool:
    return element.tag == "select" or element.role in _SELECT_ROLES


def build_candidates(observation: Observation) -> ActionCandidates:
    """Group elements by the operation they can take; password fields are only flagged."""
    click: list[ElementRef] = []
    type_text: list[ElementRef] = []
    select: list[ElementRef] = []
    flags: list[SafetyFlag] = []
    for element in observation.elements:
        if is_password_input(element):
            flags.append(SafetyFlag(element.index, element.backend_node_id, "password_input"))
            continue
        if _is_click(element):
            click.append(element)
        if _is_type_text(element):
            type_text.append(element)
        if _is_select(element):
            select.append(element)
    return ActionCandidates(tuple(click), tuple(type_text), tuple(select), tuple(flags))
