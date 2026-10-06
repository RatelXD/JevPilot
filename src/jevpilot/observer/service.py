from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, TypedDict, cast, final
from urllib.parse import urlsplit
from uuid import uuid4

from playwright.async_api import ElementHandle, Error as BrowserError, Frame, Page
from pydantic import TypeAdapter

from jevpilot.privacy import PrivacyPolicy, redact_text
from jevpilot.task import ActionType, TaskSpec

from .models import (
    ActionCandidate,
    ControlObservation,
    Observation,
    OptionObservation,
    OmittedCounts,
    WaitCondition,
)

_SNAPSHOT_SCRIPT = Path(__file__).with_name("snapshot.js").read_text(encoding="utf-8")
T = TypeVar("T")


class _ScrollState(TypedDict):
    url: str
    scrollY: int
    scrollHeight: int
    viewportHeight: int


class _WaitState(TypedDict):
    url: str
    title: str
    text: str


_OPERATION_LIST: TypeAdapter[list[str]] = TypeAdapter[list[str]](list[str])
_OPTION_LIST: TypeAdapter[list[OptionObservation]] = TypeAdapter[list[OptionObservation]](
    list[OptionObservation]
)
_SCROLL_STATE: TypeAdapter[_ScrollState] = TypeAdapter[_ScrollState](_ScrollState)
_WAIT_STATE: TypeAdapter[_WaitState] = TypeAdapter[_WaitState](_WaitState)
_WAIT_RESULT: TypeAdapter[bool] = TypeAdapter[bool](bool)
_SUPPORTED_OPERATIONS = frozenset(
    {
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SELECT,
        ActionType.SCROLL_UP,
        ActionType.SCROLL_DOWN,
        ActionType.WAIT,
        ActionType.DONE,
        ActionType.BLOCKED,
    }
)
@dataclass(frozen=True, slots=True)
class _RegisteredTarget:
    candidate: ActionCandidate
    element: ElementHandle


@dataclass(frozen=True, slots=True)
class _WaitBaseline:
    url: str
    title: str
    text: str
    scroll_y: int
    scroll_height: int
    viewport_height: int


@dataclass(slots=True)
class _ActiveRegistry:
    snapshot_id: str
    document_id: str
    navigation_id: str
    targets: dict[str, _RegisteredTarget]
    candidates: tuple[ActionCandidate, ...]
    wait_baseline: _WaitBaseline


@dataclass(frozen=True, slots=True)
class _PreparedClick:
    element: ElementHandle | None
    error_kind: str | None
    evidence: tuple[str, ...]


@final
class Observer:
    def __init__(
        self,
        page: Page,
        *,
        task: TaskSpec,
        privacy: PrivacyPolicy,
    ) -> None:
        unsupported = set(task.allowed_operations) - _SUPPORTED_OPERATIONS
        if unsupported:
            names = ", ".join(sorted(operation.value for operation in unsupported))
            raise ValueError(f"unsupported first-slice operations: {names}")

        self._page = page
        self._task = task
        self._privacy = privacy
        self._active: _ActiveRegistry | None = None
        self._navigation_index = 0
        self._closed = False
        self._registry_lock = asyncio.Lock()
        self._page.on("framenavigated", self._on_frame_navigated)

    @property
    def active_snapshot_id(self) -> str | None:
        active = self._active
        return active.snapshot_id if active is not None else None

    async def observe(
        self,
        *,
        deadline_monotonic: float,
    ) -> tuple[Observation, tuple[ActionCandidate, ...]]:
        self._ensure_open()
        self._ensure_deadline(deadline_monotonic)
        self._validate_current_origin()

        snapshot_id = f"snapshot-{uuid4().hex}"
        navigation_id = f"navigation-{self._navigation_index}"
        raw_value = cast(
            object,
            await self._with_deadline(
            self._page.evaluate(
                _SNAPSHOT_SCRIPT,
                {
                    "snapshotId": snapshot_id,
                    "navigationId": navigation_id,
                    "allowedOrigins": list(self._task.allowed_origins),
                    "candidateLimit": 128,
                    "textLimit": 6000,
                    "allowedOperations": [
                        operation.value
                        for operation in self._task.allowed_operations
                        if (
                            operation is not ActionType.WAIT
                            or self._task.budgets.max_waits > 0
                        )
                    ],
                },
            ),
            deadline_monotonic,
            ),
        )
        if not isinstance(raw_value, dict):
            raise RuntimeError("snapshot script returned an invalid payload")
        raw = cast(dict[str, object], raw_value)
        self._validate_current_origin()

        document_id = self._required_string(raw, "documentId")
        controls_raw = raw.get("controls")
        omitted_raw = raw.get("omittedCounts")
        if not isinstance(controls_raw, list) or not isinstance(omitted_raw, dict):
            raise RuntimeError("snapshot script omitted required fields")
        control_items = cast(list[object], controls_raw)
        omitted_counts = cast(dict[str, object], omitted_raw)

        controls: list[ControlObservation] = []
        candidates: list[ActionCandidate] = []
        targets: dict[str, _RegisteredTarget] = {}
        handles: list[ElementHandle] = []
        handles_by_ref: dict[str, ElementHandle] = {}
        try:
            for item_value in control_items:
                if not isinstance(item_value, dict):
                    raise RuntimeError("snapshot control was not an object")
                item = cast(dict[str, object], item_value)
                target_ref = self._required_string(item, "targetRef")
                operations_raw = item.get("operations")
                options_raw = item.get("options", [])
                if not isinstance(operations_raw, list) or not isinstance(options_raw, list):
                    raise RuntimeError("snapshot operation data was invalid")
                raw_operations = _OPERATION_LIST.validate_python(operations_raw)
                option_list: list[OptionObservation] = []
                for option in _OPTION_LIST.validate_python(options_raw):
                    option_list.append(
                        OptionObservation(
                            option_ref=option.option_ref,
                            name=redact_text(option.name, policy=self._privacy),
                            value=redact_text(option.value, policy=self._privacy),
                            selected=option.selected,
                            disabled=option.disabled,
                        )
                    )
                options = tuple(option_list)
                control = ControlObservation(
                    target_ref=target_ref,
                    role=self._redacted_string(item, "role"),
                    name=self._redacted_string(item, "name"),
                    value=self._redacted_optional_string(item, "value"),
                    enabled=bool(item.get("enabled")),
                    readonly=bool(item.get("readonly")),
                    checked=self._optional_bool(item.get("checked")),
                    selected=self._optional_bool(item.get("selected")),
                    expanded=self._optional_bool(item.get("expanded")),
                    href=self._redacted_optional_string(item, "href"),
                    max_length=self._optional_non_negative_int(item.get("maxLength")),
                    options=options,
                )
                handle = handles_by_ref.get(target_ref)
                if handle is None:
                    handle = await self._element_handle(target_ref, deadline_monotonic)
                    handles.append(handle)
                    handles_by_ref[target_ref] = handle
                controls.append(control)
                for operation_raw in raw_operations:
                    operation = ActionType(operation_raw)
                    selected_options = options if operation is ActionType.SELECT else (None,)
                    for option in selected_options:
                        candidate_id = f"candidate-{len(candidates) + 1}"
                        candidate = ActionCandidate(
                            candidate_id=candidate_id,
                            snapshot_id=snapshot_id,
                            operation=operation,
                            target_ref=target_ref,
                            option_ref=option.option_ref if option is not None else None,
                            role=control.role,
                            name=control.name,
                            value=option.value if option is not None else control.value,
                            enabled=control.enabled,
                            readonly=control.readonly,
                            max_length=control.max_length,
                        )
                        candidates.append(candidate)
                        targets[candidate_id] = _RegisteredTarget(
                            candidate=candidate,
                            element=handle,
                        )
        except BaseException:
            await self._dispose_handles(handles)
            await self._retire_registry()
            raise

        redacted_url = redact_text(
            self._required_string(raw, "url"),
            policy=self._privacy,
        )
        redacted_title = redact_text(
            str(raw.get("title", "")),
            policy=self._privacy,
        )
        redacted_text = redact_text(
            str(raw.get("text", "")),
            policy=self._privacy,
        )
        fingerprint_payload = {
            "url": redacted_url,
            "title": redacted_title,
            "text": redacted_text,
            "controls": [control.model_dump(mode="json") for control in controls],
            "scroll_y": self._non_negative_int(raw, "scrollY"),
            "scroll_height": self._non_negative_int(raw, "scrollHeight"),
            "viewport_height": self._non_negative_int(raw, "viewportHeight"),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        scroll_y = self._non_negative_int(raw, "scrollY")
        scroll_height = max(1, self._non_negative_int(raw, "scrollHeight"))
        viewport_height = max(1, self._non_negative_int(raw, "viewportHeight"))
        wait_state = _WAIT_STATE.validate_python(raw.get("waitState"))
        wait_baseline = _WaitBaseline(
            url=wait_state["url"],
            title=wait_state["title"],
            text=wait_state["text"],
            scroll_y=scroll_y,
            scroll_height=scroll_height,
            viewport_height=viewport_height,
        )
        action_overflow = 0
        for operation, available in (
            (ActionType.SCROLL_UP, scroll_y > 0),
            (
                ActionType.SCROLL_DOWN,
                scroll_y + viewport_height < scroll_height - 2,
            ),
            (
                ActionType.WAIT,
                self._task.budgets.max_waits > 0
                and self._task.budgets.max_wait_ms > 0,
            ),
        ):
            if not available or operation not in self._task.allowed_operations:
                continue
            if len(candidates) >= 128:
                action_overflow += 1
                continue
            candidate_id = f"candidate-{len(candidates) + 1}"
            candidates.append(
                ActionCandidate(
                    candidate_id=candidate_id,
                    snapshot_id=snapshot_id,
                    operation=operation,
                    wait_condition=(
                        WaitCondition(
                            kind="state_change",
                            timeout_ms=self._task.budgets.max_wait_ms,
                        )
                        if operation is ActionType.WAIT
                        else None
                    ),
                )
            )
        observation = Observation(
            snapshot_id=snapshot_id,
            document_id=document_id,
            navigation_id=navigation_id,
            url=redacted_url,
            title=redacted_title,
            text=redacted_text,
            controls=tuple(controls),
            scroll_y=scroll_y,
            scroll_height=scroll_height,
            viewport_height=viewport_height,
            fingerprint=fingerprint,
            observed_at=datetime.now(UTC),
            omitted_counts=OmittedCounts(
                hidden=self._non_negative_int(omitted_counts, "hidden"),
                disabled=self._non_negative_int(omitted_counts, "disabled"),
                readonly=self._non_negative_int(omitted_counts, "readonly"),
                unnamed=self._non_negative_int(omitted_counts, "unnamed"),
                over_limit=(
                    self._non_negative_int(omitted_counts, "overLimit")
                    + action_overflow
                ),
            ),
        )

        async with self._registry_lock:
            previous = self._active
            self._active = _ActiveRegistry(
                snapshot_id=snapshot_id,
                document_id=document_id,
                navigation_id=navigation_id,
                targets=targets,
                candidates=tuple(candidates),
                wait_baseline=wait_baseline,
            )
        if previous is not None:
            await self._dispose_handles(
                self._unique_handles(previous.targets)
            )
        return observation, tuple(candidates)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._page.remove_listener("framenavigated", self._on_frame_navigated)
        cleanup = asyncio.create_task(self._retire_registry())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    async def prepare_action(
        self,
        *,
        snapshot_id: str,
        candidate_id: str,
        target_ref: str,
        operation: ActionType,
        option_ref: str | None,
        deadline_monotonic: float,
    ) -> _PreparedClick:
        self._ensure_open()
        self._ensure_deadline(deadline_monotonic)
        if operation not in self._task.allowed_operations:
            return _PreparedClick(None, "operation_not_allowed", ("operation not allowed",))
        try:
            self._validate_current_origin()
        except ValueError:
            return _PreparedClick(None, "origin_not_allowed", ("origin changed",))

        async with self._registry_lock:
            active = self._active
            if active is None:
                return _PreparedClick(None, "no_active_snapshot", ("registry retired",))
            if active.snapshot_id != snapshot_id:
                return _PreparedClick(
                    None,
                    "snapshot_mismatch",
                    ("request does not match active snapshot",),
                )
            target = active.targets.get(candidate_id)
            if target is None:
                return _PreparedClick(
                    None,
                    "candidate_mismatch",
                    ("candidate is not in active registry",),
                )
            candidate = target.candidate
            if (
                candidate.operation is not operation
                or candidate.target_ref != target_ref
                or candidate.option_ref != option_ref
            ):
                return _PreparedClick(
                    None,
                    "target_mismatch",
                    ("candidate target does not match request",),
                )
            element = target.element

        try:
            await self._with_deadline(
                element.scroll_into_view_if_needed(),
                deadline_monotonic,
            )
            result_value = cast(
                object,
                await self._with_deadline(
                element.evaluate(
                    """(node, request) => {
                        const {allowedOrigins, operation, optionRef} = request;
                        const state = globalThis.__jevpilotObserverState;
                        if (!state || typeof state.preflight !== "function") {
                            return {ok: false, reason: "registry_retired"};
                        }
                        return state.preflight(node, request);
                    }""",
                    {
                        "allowedOrigins": list(self._task.allowed_origins),
                        "operation": operation.value,
                        "optionRef": option_ref,
                    },
                ),
                deadline_monotonic,
                ),
            )
        except (BrowserError, TimeoutError):
            reason = "document_unavailable" if self._page.is_closed() else "node_disconnected"
            return _PreparedClick(
                None,
                reason,
                (reason,),
            )
        if not isinstance(result_value, dict):
            return _PreparedClick(None, "preflight_rejected", ("invalid preflight",))
        result = cast(dict[str, object], result_value)
        if result.get("ok") is not True:
            reason = str(result.get("reason", "preflight_rejected"))
            error_kind = (
                "origin_not_allowed"
                if reason == "href_origin_not_allowed"
                else reason
            )
            return _PreparedClick(None, error_kind, (reason,))
        return _PreparedClick(element, None, ("freshness and hit-test passed",))

    def matches_active_snapshot(self, snapshot_id: str) -> bool:
        active = self._active
        return active is not None and active.snapshot_id == snapshot_id

    def has_candidate(
        self,
        snapshot_id: str,
        candidate_id: str,
        operation: ActionType,
    ) -> bool:
        active = self._active
        return active is not None and active.snapshot_id == snapshot_id and any(
            candidate.candidate_id == candidate_id
            and candidate.operation is operation
            for candidate in active.candidates
        )

    def scroll_delta(
        self, *, snapshot_id: str, operation: ActionType
    ) -> int | None:
        active = self._active
        if active is None or active.snapshot_id != snapshot_id:
            return None
        if not any(candidate.operation is operation for candidate in active.candidates):
            return None
        distance = max(1, int(active.wait_baseline.viewport_height * 0.8))
        if operation is ActionType.SCROLL_UP:
            return -distance
        if operation is ActionType.SCROLL_DOWN:
            return distance
        return None

    def scroll_position(self, *, snapshot_id: str) -> int | None:
        active = self._active
        if active is None or active.snapshot_id != snapshot_id:
            return None
        return active.wait_baseline.scroll_y

    async def scroll_is_fresh(
        self, *, snapshot_id: str, operation: ActionType, deadline_monotonic: float
    ) -> bool:
        self._ensure_deadline(deadline_monotonic)
        active = self._active
        if (
            active is None
            or active.snapshot_id != snapshot_id
            or operation not in {ActionType.SCROLL_UP, ActionType.SCROLL_DOWN}
            or not any(
                candidate.operation is operation for candidate in active.candidates
            )
        ):
            return False
        self._validate_current_origin()
        baseline = active.wait_baseline
        current = _SCROLL_STATE.validate_python(
            await self._with_deadline(
                self._page.evaluate(
                    """() => ({
                      url: location.href,
                      scrollY: Math.max(0, Math.trunc(window.scrollY)),
                      scrollHeight: Math.max(1, Math.trunc(document.documentElement.scrollHeight)),
                      viewportHeight: Math.max(1, Math.trunc(window.innerHeight))
                    })"""
                ),
                deadline_monotonic,
            )
        )
        return (
            current["url"] == baseline.url
            and current["scrollY"] == baseline.scroll_y
            and current["scrollHeight"] == baseline.scroll_height
            and current["viewportHeight"] == baseline.viewport_height
        )

    async def wait_for_state_change(
        self,
        *,
        snapshot_id: str,
        candidate_id: str,
        wait_condition: WaitCondition,
        deadline_monotonic: float,
    ) -> bool:
        self._ensure_deadline(deadline_monotonic)
        active = self._active
        if active is None or active.snapshot_id != snapshot_id:
            raise ValueError("wait snapshot is stale")
        candidate = next(
            (
                action
                for action in active.candidates
                if action.candidate_id == candidate_id
                and action.operation is ActionType.WAIT
            ),
            None,
        )
        if (
            candidate is None
            or candidate.wait_condition is None
            or candidate.wait_condition != wait_condition
        ):
            raise ValueError("wait candidate is stale or unavailable")
        baseline = active.wait_baseline
        remaining_ms = min(
            candidate.wait_condition.timeout_ms,
            max(1, int((deadline_monotonic - time.monotonic()) * 1000)),
        )
        result = _WAIT_RESULT.validate_python(
            await self._with_deadline(
                self._page.evaluate(
                """({baseline, timeoutMs}) => new Promise(resolve => {
                  const normalized = value => String(value ?? "").replace(/\\s+/g, " ").trim();
                  const changed = () =>
                    location.href !== baseline.url ||
                    document.title !== baseline.title ||
                    normalized(document.body?.innerText) !== baseline.text;
                  if (changed()) {
                    resolve(true);
                    return;
                  }
                  let finished = false;
                  globalThis.__jevpilotWaitActive = true;
                  const finish = result => {
                    if (finished) return;
                    finished = true;
                    globalThis.__jevpilotWaitActive = false;
                    observer.disconnect();
                    clearTimeout(timer);
                    resolve(result);
                  };
                  const observer = new MutationObserver(() => {
                    if (changed()) finish(true);
                  });
                  observer.observe(document.documentElement, {
                    subtree: true,
                    childList: true,
                    attributes: true,
                    characterData: true
                  });
                  const timer = setTimeout(() => finish(false), timeoutMs);
                  if (changed()) finish(true);
                })""",
                    {
                        "baseline": {
                            "url": baseline.url,
                            "title": baseline.title,
                            "text": baseline.text,
                        },
                        "timeoutMs": remaining_ms,
                    },
                ),
                deadline_monotonic,
            )
        )
        return result

    def _on_frame_navigated(self, frame: Frame) -> None:
        if frame != self._page.main_frame:
            return
        self._navigation_index += 1
        active = self._active
        self._active = None
        if active is not None:
            _ = asyncio.create_task(
                self._dispose_handles(
                    self._unique_handles(active.targets)
                )
            )

    async def _retire_registry(self) -> None:
        async with self._registry_lock:
            active = self._active
            self._active = None
        if active is not None:
            await self._dispose_handles(
                self._unique_handles(active.targets)
            )

    async def _element_handle(
        self,
        target_ref: str,
        deadline_monotonic: float,
    ) -> ElementHandle:
        handle = await self._with_deadline(
            self._page.evaluate_handle(
                """targetRef => {
                    const state = globalThis.__jevpilotObserverState;
                    return state ? state.nodes.get(targetRef) ?? null : null;
                }""",
                target_ref,
            ),
            deadline_monotonic,
        )
        element = handle.as_element()
        if element is None:
            await handle.dispose()
            raise RuntimeError("snapshot node handle is unavailable")
        return element

    async def _dispose_handles(self, handles: list[ElementHandle]) -> None:
        _ = await asyncio.gather(
            *(handle.dispose() for handle in handles),
            return_exceptions=True,
        )

    @staticmethod
    def _unique_handles(
        targets: dict[str, _RegisteredTarget],
    ) -> list[ElementHandle]:
        unique: dict[str, ElementHandle] = {}
        for target in targets.values():
            unique[target.candidate.target_ref or ""] = target.element
        return list(unique.values())

    async def _with_deadline(
        self,
        awaitable: Awaitable[T],
        deadline_monotonic: float,
    ) -> T:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("browser observation deadline expired")
        return await asyncio.wait_for(awaitable, timeout=remaining)

    def _validate_current_origin(self) -> None:
        origin = self._origin(self._page.url)
        if origin not in self._task.allowed_origins:
            raise ValueError(f"page origin is not allowed: {origin}")

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return ""
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"

    def _redacted_string(self, item: dict[str, object], key: str) -> str:
        return redact_text(self._required_string(item, key), policy=self._privacy)

    def _redacted_optional_string(
        self,
        item: dict[str, object],
        key: str,
    ) -> str | None:
        value = item.get(key)
        if value is None:
            return None
        return redact_text(str(value), policy=self._privacy)

    @staticmethod
    def _required_string(item: dict[str, object], key: str) -> str:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"snapshot field {key} is invalid")
        return value

    @staticmethod
    def _optional_bool(value: object) -> bool | None:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _optional_non_negative_int(value: object) -> int | None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @staticmethod
    def _non_negative_int(item: dict[str, object], key: str) -> int:
        value = item.get(key)
        if not isinstance(value, int) or value < 0:
            raise RuntimeError(f"snapshot count {key} is invalid")
        return value

    @staticmethod
    def _ensure_deadline(deadline_monotonic: float) -> None:
        if deadline_monotonic <= time.monotonic():
            raise TimeoutError("browser observation deadline expired")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("observer is closed")
