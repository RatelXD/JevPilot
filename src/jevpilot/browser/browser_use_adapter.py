"""browser-use 0.13.11 boundary; no Agent or model calls are used.

The registry calls are single actions. An error after dispatch cannot prove
that a click, input, or selection did not happen, so those errors are uncertain.
The pinned BrowserSession has no locale argument; use Chrome's --lang switch.
"""

import hashlib
import json
import sys
import time
from collections.abc import Iterator
from typing import Final, assert_never
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import anyio
from cdp_use import CDPClient
from cdp_use.cdp.page.events import FrameStartedNavigatingEvent, LifecycleEventEvent
from pydantic import TypeAdapter

from jevpilot.browser.env import prepare_browser_use_environment
from jevpilot.browser.port import (
    ActionKind, BrowserAction, BrowserConfig, BrowserPort, ElementRef,
    ExecutionResult, Observation, Option,
)

prepare_browser_use_environment()

from browser_use.browser.session import BrowserSession
from browser_use.browser.views import BrowserStateSummary
from browser_use.dom.views import EnhancedDOMTreeNode
from browser_use.tools.service import Tools

ACTION_REGISTRY: Final[dict[ActionKind, str]] = {
    "click": "click", "type_text": "input", "select_option": "select_dropdown",
    "scroll_down": "scroll", "scroll_up": "scroll", "go_back": "go_back",
    "wait": "wait", "navigate": "navigate",
}
_TARGETED: Final = frozenset({"click", "type_text", "select_option"})


def _walk(node: EnhancedDOMTreeNode) -> Iterator[EnhancedDOMTreeNode]:
    yield node
    for child in node.children_and_shadow_roots:
        yield from _walk(child)


def _text(node: EnhancedDOMTreeNode) -> str:
    """Read rendered text, never input values, scripts, or editable contents."""
    if node.tag_name in {"input", "textarea", "script", "style"}:
        return ""
    if node.attributes.get("contenteditable", "false") != "false":
        return ""
    if node.is_visible is False:
        return ""
    if node.node_type == 3:
        return node.node_value
    return " ".join(filter(None, (_text(child) for child in node.children_and_shadow_roots)))


def _role(node: EnhancedDOMTreeNode) -> str | None:
    return node.attributes.get("role") or (node.ax_node.role if node.ax_node else None)


def _kinds(node: EnhancedDOMTreeNode) -> frozenset[str]:
    tag, role, attrs = node.tag_name, _role(node), node.attributes
    kinds: set[str] = set()
    if (
        tag in {"a", "button", "input", "select", "summary", "label"}
        or role in {"button", "link", "tab", "option", "checkbox", "radio"}
        or node.has_js_click_listener or "onclick" in attrs
    ):
        kinds.add("click")
    if (
        tag == "textarea"
        or (tag == "input" and attrs.get("type", "text") in {
            "text", "search", "email", "tel", "number", "url", "password",
        })
        or attrs.get("contenteditable", "false") != "false"
        or role in {"textbox", "searchbox"}
    ):
        kinds.add("type_text")
    if tag == "select" or role in {"combobox", "listbox"}:
        kinds.add("select_option")
    return frozenset(kinds)


def _element(
    node: EnhancedDOMTreeNode, index: int, snapshot_id: str, url: str,
    viewport: tuple[int, int],
) -> ElementRef:
    attrs = dict(node.attributes)
    if node.tag_name in {"input", "textarea"} or attrs.get("contenteditable", "false") != "false":
        live_value = node.snapshot_node.input_value if node.snapshot_node else None
        attrs["value"] = str(len(live_value if live_value is not None else attrs.get("value", "")))
    form = node.parent_node
    while form is not None and form.tag_name != "form" and _role(form) != "form":
        form = form.parent_node
    fields = list(_walk(form)) if form is not None else []
    options = []
    if node.tag_name == "select" or _role(node) == "listbox":
        options = [
            Option(
                value=child.attributes.get("value", child.get_all_children_text()),
                label=child.attributes.get("label") or child.get_all_children_text(),
            )
            for child in _walk(node)
            if child.tag_name == "option" or _role(child) == "option"
        ][:50]
    rect = node.snapshot_node.clientRects if node.snapshot_node else None
    in_viewport = bool(node.is_visible)
    if rect is not None:
        in_viewport = in_viewport and (
            rect.width > 0 and rect.height > 0 and rect.x < viewport[0]
            and rect.y < viewport[1] and rect.x + rect.width > 0 and rect.y + rect.height > 0
        )
    return ElementRef(
        snapshot_id=snapshot_id, index=index, backend_node_id=node.backend_node_id,
        tag=node.tag_name, role=_role(node),
        text=_text(node) or attrs.get("aria-label", "") or (node.ax_node.name or "" if node.ax_node else ""),
        attributes=attrs, in_viewport=in_viewport,
        has_onclick="onclick" in node.attributes,
        contenteditable=node.attributes.get("contenteditable", "false") != "false",
        form_key=str(form.backend_node_id) if form is not None else None,
        form_has_password=any(
            field.attributes.get("type") == "password"
            or field.attributes.get("autocomplete") in {"current-password", "new-password"}
            for field in fields
        ),
        form_has_search=any(
            field.attributes.get("type") == "search" or _role(field) == "searchbox"
            for field in fields
        ),
        form_has_card_field=any(
            any(token.startswith("cc-") for token in field.attributes.get("autocomplete", "").split())
            for field in fields
        ),
        form_action_host=(
            urlsplit(urljoin(url, form.attributes.get("action", ""))).hostname
            if form is not None else None
        ),
        options=options,
    )


def _observation(
    state: BrowserStateSummary, viewport: tuple[int, int],
    selected: dict[int, list[str]] | None = None,
) -> Observation:
    snapshot_id = uuid4().hex
    elements = [
        _element(node, index, snapshot_id, state.url, viewport)
        for index, node in state.dom_state.selector_map.items()
    ]
    if selected is not None:
        elements = [
            item.model_copy(update={"selected_values": [
                value for value in selected.get(item.index, [])
                if value in {option.value for option in item.options}
            ]})
            for item in elements
        ]
    root = state.dom_state._root
    page_text = _text(root.original_node)[:20000] if root is not None else ""
    material = [
        urlsplit(state.url).path,
        sorted((item.tag, item.role or "", item.text[:80], item.selected_values) for item in elements),
        page_text[:2000],
    ]
    return Observation(
        snapshot_id=snapshot_id, url=state.url, title=state.title,
        tabs=[tab.url for tab in state.tabs], elements=elements, page_text=page_text,
        fingerprint=hashlib.sha256(json.dumps(material, ensure_ascii=False).encode()).hexdigest(),
        observed_at_monotonic=time.monotonic(),
    )


class BrowserUseAdapter(BrowserPort):
    """Own the mutable session and the latest snapshot's target capabilities."""

    def __init__(self) -> None:
        self._session: BrowserSession | None = None
        self._tools: Tools[None] = Tools()
        self._config = BrowserConfig()
        self._snapshot_id: str | None = None
        self._capabilities: dict[int, frozenset[str]] = {}
        # A separate CDP connection owns these subscriptions: cdp-use's registry
        # has one callback per event, so sharing would replace browser-use's.
        self._navigation_client: CDPClient | None = None
        self._navigation_session: str | None = None
        self._navigation_frame: str | None = None
        self._navigation_id: str | None = None
        self._navigation_loaded: anyio.Event = anyio.Event()
        self._dispatching_click: bool = False

    def _navigation_started(
        self, event: FrameStartedNavigatingEvent, session_id: str | None = None,
    ) -> None:
        if (
            self._dispatching_click and self._navigation_id is None
            and session_id == self._navigation_session
            and event["frameId"] == self._navigation_frame
        ):
            self._navigation_id = event["loaderId"]

    def _navigation_finished(
        self, event: LifecycleEventEvent, session_id: str | None = None,
    ) -> None:
        if (
            session_id == self._navigation_session
            and event["frameId"] == self._navigation_frame
            and event["loaderId"] == self._navigation_id
            and event["name"] == "load"
        ):
            self._navigation_loaded.set()

    async def _attach_navigation(self) -> None:
        assert self._session is not None and self._navigation_client is not None
        current = await self._session.get_or_create_cdp_session()
        if self._navigation_session is not None:
            _ = await self._navigation_client.send.Target.detachFromTarget(
                params={"sessionId": self._navigation_session},
            )
        attached = await self._navigation_client.send.Target.attachToTarget(
            params={"targetId": current.target_id, "flatten": True},
        )
        self._navigation_session = attached["sessionId"]
        _ = await self._navigation_client.send.Page.enable(session_id=self._navigation_session)
        _ = await self._navigation_client.send.Page.setLifecycleEventsEnabled(
            params={"enabled": True}, session_id=self._navigation_session,
        )
        tree = await self._navigation_client.send.Page.getFrameTree(session_id=self._navigation_session)
        self._navigation_frame = tree["frameTree"]["frame"]["id"]
        self._navigation_id = None
        self._navigation_loaded = anyio.Event()

    async def wait_for_navigation(self, navigation_id: str) -> None:
        if navigation_id == self._navigation_id:
            with anyio.move_on_after(5):
                await self._navigation_loaded.wait()

    async def start(self, config: BrowserConfig) -> None:
        self._config = config
        self._session = BrowserSession(
            executable_path=config.executable_path, user_data_dir=config.user_data_dir,
            headless=config.headless,
            viewport={"width": config.viewport[0], "height": config.viewport[1]},
            args=[f"--lang={config.locale}"], enable_default_extensions=False,
        )
        await self._session.start()
        self._navigation_client = CDPClient(self._session.cdp_client.url)
        await self._navigation_client.start()
        self._navigation_client.register.Page.frameStartedNavigating(self._navigation_started)
        self._navigation_client.register.Page.lifecycleEvent(self._navigation_finished)
        if config.start_url is not None:
            result = await self._tools.registry.execute_action(
                "navigate", {"url": config.start_url}, browser_session=self._session,
            )
            if result.error:
                raise RuntimeError("Browser start navigation failed")

    async def observe(self) -> Observation:
        assert self._session is not None
        document_id: str | None = None
        if self._navigation_client is not None:
            cdp = await self._session.get_or_create_cdp_session()
            tree = await cdp.cdp_client.send.Page.getFrameTree(session_id=cdp.session_id)
            document_id = tree["frameTree"]["frame"]["loaderId"]
        state = await self._session.get_browser_state_summary(cached=False, include_screenshot=False)
        # 0.13.11's summary title comes from TargetInfo and can still be the URL.
        page = await self._session.get_current_page()
        assert page is not None
        state.title = await page.evaluate("() => document.title")
        selected: dict[int, list[str]] = {}
        for index, node in state.dom_state.selector_map.items():
            if node.tag_name != "select":
                continue
            cdp = await self._session.get_or_create_cdp_session(node.target_id, focus=False)
            resolved = await cdp.cdp_client.send.DOM.resolveNode(
                params={"backendNodeId": node.backend_node_id}, session_id=cdp.session_id,
            )
            object_id = resolved["object"].get("objectId")
            if object_id is None:
                raise RuntimeError("Native select node is no longer available")
            try:
                result = await cdp.cdp_client.send.Runtime.callFunctionOn(
                    params={
                        "objectId": object_id,
                        "functionDeclaration": "function() { return Array.from(this.selectedOptions, o => o.value); }",
                        "returnByValue": True,
                    },
                    session_id=cdp.session_id,
                )
                selected[index] = TypeAdapter(list[str]).validate_python(result["result"].get("value"))
            finally:
                await cdp.cdp_client.send.Runtime.releaseObject(
                    params={"objectId": object_id}, session_id=cdp.session_id,
                )
        observation = _observation(state, self._config.viewport, selected)
        if self._navigation_client is not None:
            cdp = await self._session.get_or_create_cdp_session()
            tree = await cdp.cdp_client.send.Page.getFrameTree(session_id=cdp.session_id)
            frame = tree["frameTree"]["frame"]
            # Never pair DOM from one document with a later document's identity.
            if frame["url"] == observation.url and frame["loaderId"] == document_id:
                observation = observation.model_copy(update={"document_id": frame["loaderId"]})
        self._snapshot_id = observation.snapshot_id
        self._capabilities = {
            index: _kinds(node) for index, node in state.dom_state.selector_map.items()
        }
        return observation

    async def execute(
        self, action: BrowserAction, observation: Observation,
    ) -> ExecutionResult:
        assert self._session is not None
        params: dict[str, str | int | bool] = {}
        if action.kind in _TARGETED:
            target = action.target
            if (
                target is None or target.snapshot_id != observation.snapshot_id
                or observation.snapshot_id != self._snapshot_id
                or target not in observation.elements
            ):
                return ExecutionResult(outcome="failed_not_executed", error_kind="stale_target")
            try:
                current = await self._session.get_browser_state_summary(cached=False, include_screenshot=False)
            except Exception:  # Browser boundary: no action has been dispatched yet.
                return ExecutionResult(outcome="failed_not_executed", error_kind="observation_failed")
            node = current.dom_state.selector_map.get(target.index)
            if (
                current.url != observation.url or node is None
                or node.backend_node_id != target.backend_node_id or node.tag_name != target.tag
                or _kinds(node) != self._capabilities.get(target.index)
                or action.kind not in _kinds(node)
            ):
                return ExecutionResult(outcome="failed_not_executed", error_kind="stale_target")
            params["index"] = target.index
        match action.kind:
            case "click":
                pass
            case "type_text":
                if action.value is None:
                    return ExecutionResult(outcome="failed_not_executed", error_kind="invalid_value")
                params["text"] = action.value
            case "select_option":
                assert action.target is not None
                if action.value not in {option.value for option in action.target.options}:
                    return ExecutionResult(outcome="failed_not_executed", error_kind="invalid_option")
                params["text"] = action.value
            case "scroll_down" | "scroll_up":
                params["down"] = action.kind == "scroll_down"
            case "go_back":
                pass
            case "wait":
                params["seconds"] = 2
            case "navigate":
                if action.url is None:
                    return ExecutionResult(outcome="failed_not_executed", error_kind="invalid_url")
                params["url"] = action.url
            case unreachable:
                assert_never(unreachable)
        if action.kind == "click" and self._navigation_client is not None:
            await self._attach_navigation()
        self._dispatching_click = action.kind == "click"
        try:
            with anyio.fail_after(30):
                result = await self._tools.registry.execute_action(
                    ACTION_REGISTRY[action.kind], params, browser_session=self._session,
                )
            if result.error:
                return ExecutionResult(
                    outcome="uncertain" if action.kind in _TARGETED else "failed_not_executed",
                    error_kind="execution_failed",
                )
        except Exception:  # External registry boundary: dispatch may already have happened.
            return ExecutionResult(
                outcome="uncertain" if action.kind in _TARGETED else "failed_not_executed",
                error_kind="execution_failed",
            )
        finally:
            self._dispatching_click = False
        return ExecutionResult(
            outcome="executed",
            navigation_id=self._navigation_id if action.kind == "click" else None,
        )

    async def close(self) -> None:
        if self._navigation_client is not None:
            await self._navigation_client.stop()
            self._navigation_client = None
        if self._session is not None:
            await self._session.kill()
            self._session = None
        self._snapshot_id = None
        self._capabilities.clear()


async def _smoke(url: str) -> None:
    adapter = BrowserUseAdapter()
    try:
        await adapter.start(BrowserConfig(start_url=url))
        observation = await adapter.observe()
        print(f"elements={len(observation.elements)} fingerprint={observation.fingerprint}")
        target = next(
            item for item in observation.elements
            if "click" in adapter._capabilities[item.index]
        )
        result = await adapter.execute(BrowserAction(kind="click", target=target), observation)
        print(f"first_click={result.model_dump_json()}")
        print(f"after_click_elements={len((await adapter.observe()).elements)}")
    finally:
        await adapter.close()
        print("cleanup=browser_closed")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--smoke":
        raise SystemExit("usage: python -m jevpilot.browser.browser_use_adapter --smoke URL")
    anyio.run(_smoke, sys.argv[2])
