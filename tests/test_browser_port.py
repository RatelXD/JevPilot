from types import SimpleNamespace

import pytest

from jevpilot.browser import browser_use_adapter as boundary
from jevpilot.browser.port import (
    BrowserAction, BrowserConfig, BrowserPort, ElementRef, Observation, Option,
)
from browser_use.browser.views import BrowserStateSummary, TabInfo
from browser_use.dom.views import (
    DOMRect, EnhancedDOMTreeNode, EnhancedSnapshotNode, NodeType,
    SerializedDOMState, SimplifiedNode,
)


def node(
    tag: str = "button", backend_id: int = 10, text: str = "",
    attrs: dict[str, str] | None = None,
    children: list[EnhancedDOMTreeNode] | None = None,
) -> EnhancedDOMTreeNode:
    children = children or []
    result = EnhancedDOMTreeNode(
        node_id=backend_id, node_name=tag.upper(), backend_node_id=backend_id, attributes=attrs or {},
        ax_node=None, parent_node=None, children_nodes=children,
        snapshot_node=None, is_visible=True, has_js_click_listener=False,
        node_type=NodeType.TEXT_NODE if tag == "#text" else NodeType.ELEMENT_NODE, node_value=text,
        is_scrollable=False, absolute_position=None, target_id="test", frame_id=None,
        session_id=None, content_document=None, shadow_root_type=None, shadow_roots=None,
    )
    for child in children:
        child.parent_node = result
    return result


def state(
    selectors: dict[int, EnhancedDOMTreeNode], root: EnhancedDOMTreeNode | None = None,
    url: str = "https://example.test/catalog?q=one",
) -> BrowserStateSummary:
    return BrowserStateSummary(
        url=url, title="Catalog", tabs=[TabInfo(url=url, title="Catalog", target_id="test")],
        dom_state=SerializedDOMState(
            selector_map=selectors,
            _root=SimplifiedNode(original_node=root, children=[]) if root else None,
        ),
    )


def ref(
    target: EnhancedDOMTreeNode, index: int = 1, snapshot: str = "snapshot",
) -> ElementRef:
    return boundary._element(target, index, snapshot, "https://example.test/page", (1280, 800))


def snapshot(
    input_value: str | None = None, rect: DOMRect | None = None,
) -> EnhancedSnapshotNode:
    return EnhancedSnapshotNode(
        is_clickable=None, cursor_style=None, bounds=None, clientRects=rect,
        scrollRects=None, computed_styles=None, paint_order=None, stacking_contexts=None,
        input_value=input_value,
    )


def test_conversion_when_attributes_and_text_are_untrusted() -> None:
    # Given
    target = node(
        children=[node("#text", text="x" * 400)],
        attrs={"aria-label": "a" * 250, "data-secret": "sentinel", "onclick": "run()"},
    )
    # When
    result = ref(target)
    # Then
    assert result.backend_node_id == 10
    assert result.snapshot_id == "snapshot"
    assert result.index == 1
    assert result.tag == "button"
    assert result.text == "x" * 300
    assert result.attributes == {"aria-label": "a" * 200}
    assert result.in_viewport
    assert result.options == []


def test_input_values_when_snapshot_contains_live_private_text() -> None:
    # Given
    target = node("input", attrs={"type": "password", "value": "PRIVATE"})
    target.snapshot_node = snapshot(input_value="PRIVATE-LIVE")
    # When
    result = ref(target)
    # Then
    assert result.attributes["value"] == "12"
    assert "PRIVATE" not in result.model_dump_json()


@pytest.mark.parametrize("tag,role", [("form", None), ("section", "form")])
def test_form_metadata_when_fields_share_a_form(tag: str, role: str | None) -> None:
    # Given
    target = node("input", 21, attrs={"type": "search", "value": "PRIVATE"})
    root = node(
        tag, 100,
        attrs={"action": "//checkout.test/submit", **({"role": role} if role else {})},
        children=[
            node("div", children=[target]),
            node("input", attrs={"type": "password"}),
            node("input", attrs={"autocomplete": "section-payment cc-number", "value": "PRIVATE-CARD"}),
        ],
    )
    # When
    result = ref(target)
    # Then
    assert root.backend_node_id == 100
    assert result.form_key == "100"
    assert result.form_has_password
    assert result.form_has_search
    assert result.form_has_card_field
    assert result.form_action_host == "checkout.test"
    assert "PRIVATE" not in result.model_dump_json()


def test_nearest_form_and_no_form_when_ancestors_differ() -> None:
    # Given
    target = node()
    inner = node("section", 200, attrs={"role": "form"}, children=[target])
    node("form", 100, children=[inner, node("input", attrs={"type": "password"})])
    # When
    inner_ref, orphan_ref = ref(target), ref(node())
    # Then
    assert inner_ref.form_key == "200"
    assert inner_ref.form_action_host == "example.test"
    assert not inner_ref.form_has_password
    assert orphan_ref.form_key is None
    assert orphan_ref.form_action_host is None


@pytest.mark.parametrize("tag,role", [("select", None), ("div", "listbox")])
def test_options_when_select_or_listbox_contains_many_entries(tag: str, role: str | None) -> None:
    # Given
    target = node(
        tag, attrs={"role": role} if role else {},
        children=[
            node("option", attrs={"value": str(index)}, children=[node("#text", text="L" * 150)])
            for index in range(60)
        ],
    )
    # When
    result = ref(target)
    # Then
    assert len(result.options) == 50
    assert result.options[0] == Option(value="0", label="L" * 100)
    assert ref(node("button", children=target.children_and_shadow_roots)).options == []


def test_viewport_when_element_bounds_are_outside_view() -> None:
    # Given
    target = node()
    target.snapshot_node = snapshot(rect=DOMRect(x=0, y=900, width=100, height=40))
    # When / Then
    assert not ref(target).in_viewport


def test_option_labels_when_native_options_are_not_visible() -> None:
    # Given: Chrome reports closed native dropdown options as invisible.
    option = node("option", attrs={"value": "north"}, children=[node("#text", text="North")])
    option.is_visible = False
    target = node("select", children=[option])
    # When / Then
    assert ref(target).options == [Option(value="north", label="North")]


def test_fingerprint_when_snapshot_indices_and_query_change() -> None:
    # Given
    first, second = node("button", 10), node("a", 11)
    root = node("body", children=[node("#text", text="P" * 21000)])
    # When
    a = boundary._observation(state({1: first, 2: second}, root), (1280, 800))
    b = boundary._observation(
        state({40: second, 50: first}, root, "https://other.test/catalog?q=two"), (1280, 800),
    )
    c = boundary._observation(state({1: first}, root), (1280, 800))
    # Then
    assert a.snapshot_id != b.snapshot_id
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint
    assert len(a.page_text) == 20000
    assert a.observed_at_monotonic > 0
    assert a.tabs == ["https://example.test/catalog?q=one"]


def test_text_when_editable_and_script_nodes_contain_private_values() -> None:
    # Given
    root = node("body", children=[
        node("textarea", children=[node("#text", text="PRIVATE")]),
        node("div", attrs={"contenteditable": "true"}, children=[node("#text", text="PRIVATE")]),
        node("script", children=[node("#text", text="PRIVATE")]),
        node("p", children=[node("#text", text="Visible")]),
    ])
    # When / Then
    assert boundary._text(root) == "Visible"


class FakeSession:
    """Mutable fake browser state used to simulate redraws."""

    def __init__(self, current: BrowserStateSummary) -> None:
        self.current = current

    async def get_browser_state_summary(
        self, *, cached: bool, include_screenshot: bool,
    ) -> BrowserStateSummary:
        assert not cached
        assert not include_screenshot
        return self.current

    async def get_current_page(self) -> "FakeSession":
        return self

    async def evaluate(self, expression: str) -> str:
        assert expression == "() => document.title"
        return self.current.title


async def setup_adapter(
    monkeypatch: pytest.MonkeyPatch, target: EnhancedDOMTreeNode,
) -> tuple[boundary.BrowserUseAdapter, FakeSession, Observation, list[str]]:
    adapter = boundary.BrowserUseAdapter()
    session = FakeSession(state({1: target}))
    calls: list[str] = []

    async def execute_action(*args: str, **kwargs: str) -> SimpleNamespace:
        calls.append(args[0])
        return SimpleNamespace(error=None)

    monkeypatch.setattr(adapter, "_session", session)
    monkeypatch.setattr(adapter._tools.registry, "execute_action", execute_action)
    observation = await adapter.observe()
    return adapter, session, observation, calls


@pytest.mark.parametrize("change", ["index", "backend", "tag", "kind", "url"])
async def test_stale_target_when_current_identity_changes(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    # Given
    adapter, session, observation, calls = await setup_adapter(
        monkeypatch, node("input", attrs={"type": "text"}),
    )
    current = node("input", attrs={"type": "text"})
    if change == "backend":
        current.backend_node_id += 1
    if change == "tag":
        current.node_name = "TEXTAREA"
    if change == "kind":
        current.attributes["type"] = "checkbox"
    session.current = state(
        {2 if change == "index" else 1: current},
        url="https://example.test/other" if change == "url" else observation.url,
    )
    # When
    result = await adapter.execute(
        BrowserAction(kind="type_text", target=observation.elements[0], value="text"), observation,
    )
    # Then
    assert result.outcome == "failed_not_executed"
    assert result.error_kind == "stale_target"
    assert calls == []


async def test_selection_when_value_is_not_an_observed_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    target = node("select", children=[
        node("option", attrs={"value": "north"}, children=[node("#text", text="North")]),
    ])
    adapter, _, observation, calls = await setup_adapter(monkeypatch, target)
    # When
    rejected = await adapter.execute(
        BrowserAction(kind="select_option", target=observation.elements[0], value="south"), observation,
    )
    accepted = await adapter.execute(
        BrowserAction(kind="select_option", target=observation.elements[0], value="north"), observation,
    )
    # Then
    assert rejected.error_kind == "invalid_option"
    assert accepted.outcome == "executed"
    assert calls == ["select_dropdown"]


@pytest.mark.parametrize("binding", ["missing", "snapshot", "forged", "superseded"])
async def test_target_binding_when_action_has_no_matching_observation(
    monkeypatch: pytest.MonkeyPatch, binding: str,
) -> None:
    # Given
    adapter, _, observation, calls = await setup_adapter(monkeypatch, node())
    target = observation.elements[0]
    if binding == "snapshot":
        target = target.model_copy(update={"snapshot_id": "other"})
    if binding == "forged":
        target = target.model_copy(update={"backend_node_id": 999})
    if binding == "superseded":
        await adapter.observe()
    # When
    result = await adapter.execute(
        BrowserAction(kind="click", target=None if binding == "missing" else target), observation,
    )
    # Then
    assert result.error_kind == "stale_target"
    assert result.outcome == "failed_not_executed"
    assert calls == []


@pytest.mark.parametrize("kind", list(boundary.ACTION_REGISTRY))
async def test_registry_mapping_when_action_is_executed(
    monkeypatch: pytest.MonkeyPatch, kind: boundary.ActionKind,
) -> None:
    # Given
    target = node("select", children=[node("option", attrs={"value": "north"})])
    if kind == "type_text":
        target = node("input", attrs={"type": "text"})
    adapter, _, observation, calls = await setup_adapter(monkeypatch, target)
    # When
    result = await adapter.execute(
        BrowserAction(kind=kind, target=observation.elements[0], value="north", url="https://example.test"),
        observation,
    )
    # Then
    assert result.outcome == "executed"
    assert calls == [boundary.ACTION_REGISTRY[kind]]


@pytest.mark.parametrize("kind", ["click", "type_text", "select_option", "navigate", "scroll_down"])
@pytest.mark.parametrize("failure", ["timeout", "exception", "returned_error"])
async def test_execution_outcome_when_registry_fails(
    monkeypatch: pytest.MonkeyPatch, kind: boundary.ActionKind, failure: str,
) -> None:
    # Given
    target = node("input", attrs={"type": "text"})
    if kind == "select_option":
        target = node("select", children=[node("option", attrs={"value": "north"})])
    adapter, _, observation, _ = await setup_adapter(monkeypatch, target)

    async def fail_action(*args: str, **kwargs: str) -> SimpleNamespace:
        if failure == "timeout":
            raise TimeoutError
        if failure == "exception":
            raise RuntimeError("PRIVATE")
        return SimpleNamespace(error="PRIVATE")

    monkeypatch.setattr(adapter._tools.registry, "execute_action", fail_action)
    # When
    result = await adapter.execute(
        BrowserAction(kind=kind, target=observation.elements[0], value="north", url="https://example.test"),
        observation,
    )
    # Then
    assert result.outcome == (
        "uncertain" if kind in {"click", "type_text", "select_option"} else "failed_not_executed"
    )
    assert "PRIVATE" not in result.model_dump_json()


def test_port_when_adapter_is_constructed() -> None:
    # Given / When
    adapter = boundary.BrowserUseAdapter()
    config = BrowserConfig()
    # Then
    assert isinstance(adapter, BrowserPort)
    assert config.headless is False
    assert config.viewport == (1280, 800)
    assert config.locale == "ko-KR"
