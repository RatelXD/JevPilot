import pytest

from jevpilot.browser.port import ActionKind, BrowserAction, ElementRef, Observation
from jevpilot.engine.candidates import build_candidates
from jevpilot.engine.safety import assess
from tests.test_browser_port import boundary, node, snapshot, state


def el(
    text: str = "",
    tag: str = "button",
    attributes: dict[str, str] | None = None,
    *,
    form_has_password: bool = False,
    form_has_card_field: bool = False,
    form_has_search: bool = False,
) -> ElementRef:
    return ElementRef(
        snapshot_id="s", index=0, backend_node_id=1, tag=tag, text=text,
        attributes=attributes or {}, in_viewport=True,
        form_has_password=form_has_password,
        form_has_card_field=form_has_card_field,
        form_has_search=form_has_search,
    )


OBS = Observation(
    snapshot_id="s", url="https://example.com", title="t", tabs=[], elements=[],
    page_text="", fingerprint="f", observed_at_monotonic=0.0,
)


def verdict(target: ElementRef, kind: ActionKind = "click") -> str:
    return assess(BrowserAction(kind=kind, target=target), OBS).verdict


@pytest.mark.parametrize("text", ["로그인", "Log In", "LOGIN", "Sign in", "signin"])
def test_login_phrases_hand_off(text: str) -> None:
    assert verdict(el(text)) == "handoff"


@pytest.mark.parametrize("text", ["결제하기", "Pay now", "Payment", "Checkout", "카드번호 입력", "Card Number"])
def test_payment_phrases_hand_off(text: str) -> None:
    assert verdict(el(text)) == "handoff"


@pytest.mark.parametrize(
    "text",
    ["예매하기", "예약", "구매", "주문", "신청", "제출", "확정", "Book", "Reserve",
     "Purchase", "Order", "Submit", "Confirm", "Place order"],
)
def test_mutation_phrases_need_confirmation(text: str) -> None:
    assert verdict(el(text)) == "confirm"


@pytest.mark.parametrize("text", ["검색", "Search", "다음", "Border", "Display", "Bookmark"])
def test_neutral_or_lookalike_text_is_allowed(text: str) -> None:
    assert verdict(el(text)) == "allow"


@pytest.mark.parametrize("label,href", [
    ("Books", "/catalog/books/"),
    ("Travel", "/catalog/category/books/travel/index.html"),
])
def test_book_noun_when_browsing_catalog_is_not_booking(label: str, href: str) -> None:
    # Given
    target = el(label, tag="a", attributes={"href": href})
    # When
    result = assess(BrowserAction(kind="click", target=target), OBS)
    # Then
    assert result.verdict == "allow"


def test_search_form_button_is_not_a_mutation() -> None:
    assert verdict(el("예약 조회", form_has_search=True)) == "allow"


def test_neutral_button_in_password_form_hands_off() -> None:
    assert verdict(el("계속", form_has_password=True)) == "handoff"


def test_neutral_button_in_card_form_hands_off() -> None:
    assert verdict(el("계속", form_has_card_field=True)) == "handoff"


def test_password_form_beats_search_flag() -> None:
    assert verdict(el("검색", form_has_password=True, form_has_search=True)) == "handoff"


def test_password_field_hands_off_for_typing_and_clicking() -> None:
    field = el(tag="input", attributes={"type": "password"})
    assert verdict(field, "type_text") == "handoff"
    assert verdict(field, "click") == "handoff"
    autofill = el(tag="input", attributes={"autocomplete": "current-password"})
    assert verdict(autofill, "type_text") == "handoff"


def test_card_autocomplete_field_hands_off() -> None:
    assert verdict(el(tag="input", attributes={"autocomplete": "cc-number"}), "type_text") == "handoff"


def test_aria_label_value_name_and_id_are_checked() -> None:
    assert verdict(el("", attributes={"aria-label": "Sign in"})) == "handoff"
    assert verdict(el("", tag="input", attributes={"type": "submit", "value": "제출"})) == "confirm"
    assert verdict(el("", attributes={"name": "checkout"})) == "handoff"
    assert verdict(el("", attributes={"id": "place-order"})) == "confirm"


def test_login_outranks_payment_and_mutation() -> None:
    assert verdict(el("Sign in to pay and submit")) == "handoff"
    assert verdict(el("Submit payment")) == "handoff"


def test_typing_ordinary_text_and_untargeted_actions_are_allowed() -> None:
    assert verdict(el("주문", tag="input"), "type_text") == "allow"
    assert assess(BrowserAction(kind="scroll_down"), OBS).verdict == "allow"
    assert assess(BrowserAction(kind="navigate", url="https://example.com"), OBS).verdict == "allow"


@pytest.mark.parametrize("kind", ["click", "type_text"])
@pytest.mark.parametrize(
    ("attribute", "label", "expected"),
    [
        ("aria-label", "Card Number", "handoff"),
        ("aria-label", "카드번호", "handoff"),
        ("aria-label", "카드 번호", "handoff"),
        ("text", "Payment", "handoff"),
        ("text", "카드 번호", "handoff"),
        ("placeholder", "Card Number", "handoff"),
        ("placeholder", "Enter card number", "handoff"),
        ("placeholder", "카드 번호 입력", "handoff"),
        ("placeholder", "PAYMENT", "handoff"),
        ("name", "payment", "handoff"),
        ("name", "card number", "handoff"),
        ("name", "카드번호", "handoff"),
        ("aria-label", "Search catalog", "allow"),
        ("aria-label", "Display name", "allow"),
        ("placeholder", "Phone number", "allow"),
        ("placeholder", "검색어 입력", "allow"),
        ("name", "display", "allow"),
        ("name", "payee", "allow"),
        ("text", "다음", "allow"),
    ],
)
def test_real_candidates_apply_payment_labels_to_click_and_typing(
    kind: ActionKind, attribute: str, label: str, expected: str,
) -> None:
    # Label text is represented by ElementRef.text; there is no raw label
    # attribute in the browser port's retained attribute contract.
    field = el(
        text=label if attribute == "text" else "",
        tag="input",
        attributes={"type": "text", **({attribute: label} if attribute != "text" else {})},
    ).model_copy(update={"has_onclick": True})
    observation = OBS.model_copy(update={"elements": [field]})
    candidates = build_candidates(observation).for_operation(kind)
    assert candidates == (field,)
    result = assess(
        BrowserAction(kind=kind, target=candidates[0], value="fixture"),
        observation,
    )
    assert result.verdict == expected
    print(f"{kind} {attribute}={label!r}: {result.verdict} ({result.reason})")


def test_real_select_candidate_with_payment_label_hands_off() -> None:
    field = el(tag="select", attributes={"aria-label": "Payment method"})
    observation = OBS.model_copy(update={"elements": [field]})
    candidate, = build_candidates(observation).select_option
    assert assess(
        BrowserAction(kind="select_option", target=candidate, value="fixture"),
        observation,
    ).verdict == "handoff"


@pytest.mark.parametrize(
    ("tag", "attributes", "search_form", "expected"),
    [
        ("a", {"href": "/checkout"}, False, "handoff"),
        ("a", {"href": "/login"}, False, "handoff"),
        ("a", {"href": "/order"}, False, "confirm"),
        ("input", {"type": "submit", "value": "Continue"}, False, "confirm"),
        ("a", {"href": "/CHECKOUT"}, False, "handoff"),
        ("a", {"href": "/LOGIN"}, False, "handoff"),
        ("a", {"href": "/ORDER"}, False, "confirm"),
        ("input", {"type": "SUBMIT", "value": "Continue"}, False, "confirm"),
        ("a", {"href": "/catalog"}, False, "allow"),
        ("a", {"href": "/border"}, False, "allow"),
        ("a", {"href": "/display"}, False, "allow"),
        ("a", {"href": "/bookmark"}, False, "allow"),
        ("input", {"type": "button", "value": "Continue"}, False, "allow"),
        ("input", {"type": "submit", "value": "Continue"}, True, "allow"),
        ("a", {"href": "/order"}, True, "allow"),
        ("a", {"href": "/checkout"}, True, "handoff"),
        ("a", {"href": "/login"}, True, "handoff"),
    ],
)
def test_adapter_candidates_guard_retained_attributes(
    tag: str, attributes: dict[str, str], search_form: bool, expected: str,
) -> None:
    # Given: real adapter normalization, including input value lengths.
    target = node(tag, attrs={"aria-label": "Continue", **attributes})
    form = node("form", children=[
        target,
        node("input", 11, attrs={"type": "search" if search_form else "text"}),
    ])
    observation = boundary._observation(state({1: target}, form), (1280, 800))
    candidate, = build_candidates(observation).click
    if tag == "input":
        assert candidate.attributes["value"] == str(len(attributes["value"]))
    # When
    result = assess(BrowserAction(kind="click", target=candidate), observation)
    # Then
    assert result.verdict == expected
    print(f"adapter {attributes!r} search={search_form}: {result.verdict}")


def test_adapter_private_input_value_is_not_a_risk_label() -> None:
    # Given
    target = node("input", attrs={"type": "text", "placeholder": "Search catalog"})
    target.snapshot_node = snapshot(input_value="login payment submit")
    observation = boundary._observation(state({1: target}), (1280, 800))
    candidate, = build_candidates(observation).type_text
    # When
    result = assess(BrowserAction(kind="type_text", target=candidate), observation)
    # Then
    assert result.verdict == "allow"
    assert candidate.attributes["value"] == str(len("login payment submit"))
    assert "login payment submit" not in observation.model_dump_json()
