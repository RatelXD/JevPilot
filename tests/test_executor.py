from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import anyio
import pytest
from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Error as BrowserError,
    Page,
    Route,
    async_playwright,
)

from jevpilot.executor import ActionExecution, ActionRequest, PlaywrightExecutor
from jevpilot.observer import ActionCandidate, Observer
from jevpilot.privacy import PrivacyPolicy
from jevpilot.task import ActionType, Budget, TaskSpec

ORIGIN = "http://fixture.test"


@pytest.fixture
async def browser() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        yield browser
        await browser.close()


def task_spec(
    *,
    max_waits: int = 0,
    operations: tuple[ActionType, ...] = (
        ActionType.CLICK,
        ActionType.DONE,
        ActionType.BLOCKED,
    ),
) -> TaskSpec:
    return TaskSpec(
        task_id="T-executor",
        goal="Click the requested synthetic control",
        mode="jev_hybrid",
        initial_url=f"{ORIGIN}/start",
        allowed_origins=(ORIGIN,),
        allowed_operations=operations,
        initializer_id="fixture",
        evaluator_id="fixture",
        budgets=Budget(
            max_steps=8,
            max_mutations=4,
            max_replans=1,
            max_model_attempts=4,
            max_attempts_per_call=1,
            max_stale_observations=3,
            max_waits=max_waits,
            max_wait_ms=500,
            wall_time_ms=10_000,
        ),
        locale="en-US",
        fixture_version="1",
    )


async def fixture_page(
    browser: Browser,
    html: str,
    *,
    route_handler: Callable[[Route], object] | None = None,
) -> tuple[BrowserContext, Page]:
    context = await browser.new_context()
    page = await context.new_page()

    async def fulfill(route: Route) -> None:
        if route_handler is not None:
            result = route_handler(route)
            if asyncio.iscoroutine(result):
                await result
            return
        await route.fulfill(status=200, content_type="text/html", body=html)

    _ = await page.route(f"{ORIGIN}/**", fulfill)
    _ = await page.goto(f"{ORIGIN}/start")
    return context, page


def click_request(
    candidate: ActionCandidate,
    *,
    decision_id: str = "decision-1",
) -> ActionRequest:
    return ActionRequest(
        action_id=f"action-{decision_id}",
        decision_id=decision_id,
        snapshot_id=candidate.snapshot_id,
        operation=ActionType.CLICK,
        candidate_id=candidate.candidate_id,
        target_ref=candidate.target_ref,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            """() => {
                const target = document.querySelector("#target");
                target.outerHTML = target.outerHTML;
            }""",
            "node_disconnected",
        ),
        (
            "() => document.querySelector('#target').textContent = 'Changed'",
            "semantic_guard_changed",
        ),
        (
            """() => {
                const overlay = document.createElement("div");
                Object.assign(overlay.style, {
                  position: "fixed", inset: "0", zIndex: "9999", background: "white"
                });
                document.body.append(overlay);
            }""",
            "target_occluded",
        ),
    ],
)
async def test_executor_rejects_stale_identity_semantics_and_overlay(
    browser: Browser,
    mutation: str,
    expected_reason: str,
) -> None:
    context, page = await fixture_page(
        browser,
        """
        <button id="target" onclick="window.clicks += 1">Open item</button>
        <script>window.clicks = 0</script>
        """,
    )
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page,
        observer,
        task=task_spec(),
        record_execution=receipts.append,
    )
    try:
        _, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        await page.evaluate(mutation)
        execution = await executor.execute(
            click_request(candidates[0]),
            deadline_monotonic=time.monotonic() + 5,
        )
        assert execution.status == "failed_not_executed"
        assert execution.dispatched is False
        assert expected_reason in execution.evidence
        assert execution.error_kind == expected_reason
        assert await page.evaluate("window.clicks") == 0
        assert receipts == [execution]
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_records_uncertain_receipt_before_propagating_cancellation(
    browser: Browser,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, page = await fixture_page(browser, "<button>Open item</button>")
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page,
        observer,
        task=task_spec(),
        record_execution=receipts.append,
    )
    dispatch_started = asyncio.Event()
    never_finishes = asyncio.Event()
    try:
        _, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        async def cancelled_click(
            self: object,
            *,
            timeout: float | None = None,
            **kwargs: object,
        ) -> None:
            del self, timeout, kwargs
            _ = dispatch_started.set()
            _ = await never_finishes.wait()

        monkeypatch.setattr(ElementHandle, "click", cancelled_click)
        execution_task = asyncio.create_task(
            executor.execute(
                click_request(candidates[0]),
                deadline_monotonic=time.monotonic() + 5,
            )
        )
        _ = await asyncio.wait_for(dispatch_started.wait(), timeout=2)
        _ = execution_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await execution_task
        assert len(receipts) == 1
        assert receipts[0].status == "uncertain"
        assert receipts[0].error_kind == "dispatch_cancelled"
    finally:
        await asyncio.shield(observer.close())
        await context.close()


@pytest.mark.asyncio
async def test_executor_clicks_actual_node_once_and_consumes_decision(
    browser: Browser,
) -> None:
    context, page = await fixture_page(
        browser,
        """
        <div id="unrelated">before</div>
        <button id="target" onclick="window.clicks += 1">Open item</button>
        <script>window.clicks = 0</script>
        """,
    )
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page,
        observer,
        task=task_spec(),
        record_execution=receipts.append,
    )
    try:
        _, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        await page.evaluate(
            "() => document.querySelector('#unrelated').textContent = 'after'"
        )
        request = click_request(candidates[0])
        first = await executor.execute(
            request,
            deadline_monotonic=time.monotonic() + 5,
        )
        second = await executor.execute(
            request,
            deadline_monotonic=time.monotonic() + 5,
        )
        assert first.status == "executed"
        assert second.status == "failed_not_executed"
        assert second.error_kind == "decision_already_consumed"
        assert await page.evaluate("window.clicks") == 1
        assert receipts == [first, second]
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_supports_done_and_blocked_for_active_snapshot(
    browser: Browser,
) -> None:
    context, page = await fixture_page(browser, "<button>Open item</button>")
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page,
        observer,
        task=task_spec(),
        record_execution=receipts.append,
    )
    try:
        observation, _ = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        for operation in (ActionType.DONE, ActionType.BLOCKED):
            execution = await executor.execute(
                ActionRequest(
                    action_id=f"action-{operation.value}",
                    decision_id=f"decision-{operation.value}",
                    snapshot_id=observation.snapshot_id,
                    operation=operation,
                ),
                deadline_monotonic=time.monotonic() + 5,
            )
            assert execution.status == "executed"
            assert execution.dispatched is True
        assert [receipt.operation for receipt in receipts] == [
            ActionType.DONE,
            ActionType.BLOCKED,
        ]
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_marks_interrupted_post_dispatch_click_uncertain(
    browser: Browser,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, page = await fixture_page(
        browser,
        """
        <button id="target" onclick="window.clicks += 1">Open item</button>
        <script>window.clicks = 0</script>
        """,
    )
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page,
        observer,
        task=task_spec(),
        record_execution=receipts.append,
    )
    original_click = ElementHandle.click

    async def interrupted_click(
        self: ElementHandle,
        *,
        timeout: float | None = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        await original_click(self, timeout=timeout)
        raise BrowserError("simulated transport interruption")

    monkeypatch.setattr(ElementHandle, "click", interrupted_click)
    try:
        _, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        execution = await executor.execute(
            click_request(candidates[0]),
            deadline_monotonic=time.monotonic() + 5,
        )
        assert execution.status == "uncertain"
        assert execution.dispatched is True
        assert execution.error_kind == "dispatch_interrupted"
        assert await page.evaluate("window.clicks") == 1
        assert receipts == [execution]
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_selects_only_the_observed_native_option(browser: Browser) -> None:
    # Given
    context, page = await fixture_page(
        browser,
        """
        <label for="region">Region</label>
        <select id="region">
          <option value="north">North</option>
          <option value="south">South</option>
        </select>
        """,
    )
    task = task_spec(
        operations=(
            ActionType.SELECT,
            ActionType.DONE,
            ActionType.BLOCKED,
        )
    )
    observer = Observer(page, task=task, privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page, observer, task=task, record_execution=receipts.append
    )
    try:
        _, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        north = next(
            candidate
            for candidate in candidates
            if candidate.operation is ActionType.SELECT
            and candidate.value == "north"
        )

        # When
        execution = await executor.execute(
            ActionRequest(
                action_id="select-north",
                decision_id="decision-select-north",
                snapshot_id=north.snapshot_id,
                operation=ActionType.SELECT,
                candidate_id=north.candidate_id,
                target_ref=north.target_ref,
                option_ref=north.option_ref,
            ),
            deadline_monotonic=time.monotonic() + 5,
        )

        # Then
        assert execution.status == "executed", execution
        assert execution.operation is ActionType.SELECT
        assert await page.locator("#region").input_value() == "north"
        assert receipts == [execution]
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_scrolls_only_when_current_snapshot_allows_direction(
    browser: Browser,
) -> None:
    # Given
    context, page = await fixture_page(
        browser,
        "<main style='height:2400px'><h1>Long local page</h1></main>",
    )
    task = task_spec(
        operations=(
            ActionType.SCROLL_UP,
            ActionType.SCROLL_DOWN,
            ActionType.DONE,
            ActionType.BLOCKED,
        )
    )
    observer = Observer(page, task=task, privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page, observer, task=task, record_execution=receipts.append
    )
    try:
        observation, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        assert any(candidate.operation is ActionType.SCROLL_DOWN for candidate in candidates)

        # When
        execution = await executor.execute(
            ActionRequest(
                action_id="scroll-down",
                decision_id="decision-scroll-down",
                snapshot_id=observation.snapshot_id,
                operation=ActionType.SCROLL_DOWN,
            ),
            deadline_monotonic=time.monotonic() + 5,
        )

        # Then
        assert execution.status == "executed"
        assert await page.evaluate("window.scrollY") > 0
        next_observation, next_candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        assert next_observation.snapshot_id != observation.snapshot_id
        assert any(candidate.operation is ActionType.SCROLL_UP for candidate in next_candidates)
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_executor_wait_subscribes_before_dom_change_and_times_out_by_condition(
    browser: Browser,
) -> None:
    # Given
    context, page = await fixture_page(
        browser,
        """<main><p id="status">waiting</p>
        <button onclick="document.querySelector('#status').textContent='ready'">
          Update
        </button></main>""",
    )
    task = task_spec(
        max_waits=1,
        operations=(ActionType.WAIT, ActionType.DONE, ActionType.BLOCKED)
    )
    observer = Observer(page, task=task, privacy=PrivacyPolicy())
    receipts: list[ActionExecution] = []
    executor = PlaywrightExecutor(
        page, observer, task=task, record_execution=receipts.append
    )
    try:
        observation, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        wait = next(
            candidate for candidate in candidates
            if candidate.operation is ActionType.WAIT
        )
        assert wait.wait_condition is not None

        async def run_wait(*, task_status: anyio.abc.TaskStatus[None]) -> None:
            task_status.started()
            execution = await executor.execute(
                ActionRequest(
                    action_id="wait-state-change",
                    decision_id="decision-wait",
                    snapshot_id=observation.snapshot_id,
                    operation=ActionType.WAIT,
                    candidate_id=wait.candidate_id,
                    wait_condition=wait.wait_condition,
                ),
                deadline_monotonic=time.monotonic() + 5,
            )
            receipts.append(execution)

        async with anyio.create_task_group() as group:
            await group.start(run_wait)
            _ = await page.wait_for_function(
                "window.__jevpilotWaitActive === true", timeout=3000
            )
            await page.get_by_role("button", name="Update").click()

        # Then
        assert receipts[0].status == "executed"
        assert receipts[0].error_kind is None
        assert "observed page state changed" in receipts[0].evidence
    finally:
        await observer.close()
        await context.close()
