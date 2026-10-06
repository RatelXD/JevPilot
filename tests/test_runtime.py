from typing import Literal

import pytest
from playwright.async_api import async_playwright

from jevpilot.privacy import PrivacyPolicy
from jevpilot.runtime import HybridBrowserAgent
from jevpilot.task import ActionType
from jevpilot.telemetry.recorder import RunRecorder
from tests.fixture_tasks import catalog_task, filter_task, fixture_server, search_task
from tests.runtime_support import ScriptedPlanner, ScriptedSelector, ScriptedText, budgets


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_completes_local_browser_task_once_per_plan(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        task, binding = catalog_task(origin, mode, budgets=budgets())
        planner = ScriptedPlanner()
        selector = ScriptedSelector(confidence=0.99 if mode == "jev_hybrid" else None)
        recorder = RunRecorder(budgets())
        agent = HybridBrowserAgent(
            planner,
            selector,
            text=ScriptedText(),
            privacy=PrivacyPolicy(),
            operation_threshold=0.5,
            target_threshold=0.5,
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                # When
                trace = await agent.run(
                    browser, task, binding=binding, recorder=recorder
                )
            finally:
                await browser.close()

    # Then
    assert trace.terminal.status == "succeeded", (trace.terminal, trace.executions)
    assert planner.calls == 1
    assert selector.calls == 1
    assert trace.metrics.mutations == 1
    assert trace.metrics.network_attempts == 0
    assert trace.metrics.total_cost_usd == 0
    assert trace.verifications[-1].task_evaluation.status == "passed"


@pytest.mark.asyncio
async def test_runtime_false_done_replans_with_failure_evidence() -> None:
    # Given
    with fixture_server() as origin:
        task, binding = catalog_task(origin, "llm_only", budgets=budgets())
        planner = ScriptedPlanner()
        selector = ScriptedSelector(false_first_done=True)
        recorder = RunRecorder(budgets())
        agent = HybridBrowserAgent(
            planner,
            selector,
            text=ScriptedText(),
            privacy=PrivacyPolicy(),
            operation_threshold=0.5,
            target_threshold=0.5,
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                # When
                trace = await agent.run(
                    browser, task, binding=binding, recorder=recorder
                )
            finally:
                await browser.close()

    # Then
    assert trace.terminal.status == "succeeded", trace.terminal
    assert planner.calls == 2
    assert planner.failures == [None, "false_done"]
    assert trace.metrics.replans == 1
    assert trace.verifications[0].task_evaluation.status == "not_yet"
    assert trace.verifications[-1].task_evaluation.status == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_search_generates_text_then_clicks_result(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        task, binding = search_task(origin, mode, budgets=budgets())
        planner = ScriptedPlanner()
        selector = ScriptedSelector(
            confidence=0.99 if mode == "jev_hybrid" else None,
            actions=(ActionType.TYPE_TEXT, ActionType.CLICK, ActionType.CLICK),
            target_names=("Search catalog", "Search", "Aurora"),
        )
        text = ScriptedText()
        recorder = RunRecorder(budgets())
        agent = HybridBrowserAgent(
            planner,
            selector,
            text=text,
            privacy=PrivacyPolicy(),
            operation_threshold=0.5,
            target_threshold=0.5,
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                # When
                trace = await agent.run(
                    browser, task, binding=binding, recorder=recorder
                )
            finally:
                await browser.close()

    # Then
    assert trace.terminal.status == "succeeded", trace.terminal
    assert planner.calls == 1
    assert selector.calls == 3
    assert text.calls == 1
    assert trace.metrics.mutations == 3
    assert trace.verifications[-1].task_evaluation.status == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_selects_current_native_option_before_filtering(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        task, binding = filter_task(origin, mode, budgets=budgets())
        planner = ScriptedPlanner()
        selector = ScriptedSelector(
            confidence=0.99 if mode == "jev_hybrid" else None,
            actions=(ActionType.SELECT, ActionType.CLICK, ActionType.CLICK),
            target_names=("Region", "Apply filters", "Aurora"),
            target_values=("north", None, None),
        )
        recorder = RunRecorder(budgets())
        agent = HybridBrowserAgent(
            planner,
            selector,
            text=ScriptedText(),
            privacy=PrivacyPolicy(),
            operation_threshold=0.5,
            target_threshold=0.5,
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                # When
                trace = await agent.run(
                    browser, task, binding=binding, recorder=recorder
                )
            finally:
                await browser.close()

    # Then
    assert trace.terminal.status == "succeeded", trace.terminal
    assert planner.calls == 1
    assert selector.calls == 3
    assert recorder.executions[0].operation is ActionType.SELECT
    assert recorder.executions[0].option_ref == "1"
    assert trace.verifications[-1].task_evaluation.status == "passed"
