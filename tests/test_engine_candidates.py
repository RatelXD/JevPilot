import pytest

from jevpilot.browser.port import ElementRef, Observation, Option
from jevpilot.engine.candidates import build_candidates


def el(index: int, tag: str, role: str | None = None, **attributes: str) -> ElementRef:
    return ElementRef(
        snapshot_id="s", index=index, backend_node_id=100 + index, tag=tag, role=role,
        text=f"e{index}", attributes=attributes, in_viewport=True,
        options=[Option(value="a", label="A")] if tag == "select" else [],
    )


def obs(*elements: ElementRef) -> Observation:
    return Observation(
        snapshot_id="s", url="https://example.com", title="t", tabs=[],
        elements=list(elements), page_text="", fingerprint="f", observed_at_monotonic=0.0,
    )


def indexes(elements: tuple[ElementRef, ...]) -> list[int]:
    return [e.index for e in elements]


def test_click_candidates_cover_links_buttons_roles_and_submit_inputs() -> None:
    result = build_candidates(obs(
        el(0, "a", "link", href="/x"), el(1, "button"), el(2, "div", "tab"),
        el(3, "div", "option"), el(4, "div", "checkbox"), el(5, "div", "radio"),
        el(6, "div", "button"), el(7, "div"), el(8, "input", type="submit"),
    ))
    assert indexes(result.click) == [0, 1, 2, 3, 4, 5, 6, 8]


def test_onclick_only_is_click_and_contenteditable_is_type_text() -> None:
    result = build_candidates(obs(
        el(0, "div").model_copy(update={"has_onclick": True}),
        el(1, "div").model_copy(update={"contenteditable": True}),
        el(2, "div"),
    ))
    assert indexes(result.click) == [0]
    assert indexes(result.type_text) == [1]
    assert indexes(result.select_option) == []


@pytest.mark.parametrize("input_type", ["text", "search", "email", "tel", "number", "url"])
def test_text_input_types_are_type_text_candidates(input_type: str) -> None:
    assert indexes(build_candidates(obs(el(0, "input", type=input_type))).type_text) == [0]


def test_type_text_covers_textarea_default_input_and_textbox_roles() -> None:
    result = build_candidates(obs(
        el(0, "textarea"), el(1, "input"), el(2, "input", type="checkbox"),
        el(3, "div"), el(4, "div", "textbox"), el(5, "div", "searchbox"),
    ))
    assert indexes(result.type_text) == [0, 1, 4, 5]


def test_select_option_covers_select_combobox_and_listbox() -> None:
    result = build_candidates(obs(
        el(0, "select"), el(1, "div", "combobox"), el(2, "ul", "listbox"), el(3, "div"),
    ))
    assert indexes(result.select_option) == [0, 1, 2]


@pytest.mark.parametrize(
    "attributes",
    [
        {"type": "password"},
        {"autocomplete": "current-password"},
        {"autocomplete": "new-password"},
        {"type": "text", "autocomplete": "username new-password"},
    ],
)
def test_password_input_is_excluded_from_type_text_and_flagged(
    attributes: dict[str, str],
) -> None:
    result = build_candidates(obs(el(0, "input", type="text"), el(1, "input", **attributes)))
    assert indexes(result.type_text) == [0]
    assert [(f.index, f.backend_node_id, f.reason) for f in result.safety_flags] == [
        (1, 101, "password_input"),
    ]
    assert 1 not in [e.index for e in result.all_elements()]
    print("type_text candidates:", indexes(result.type_text), "flags:", result.safety_flags)
