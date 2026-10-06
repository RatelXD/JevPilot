from typing import Literal

import pytest
from playwright.async_api import async_playwright

from jevpilot.privacy import PrivacyPolicy
from jevpilot.runtime import HybridBrowserAgent
from jevpilot.task import ActionType
from jevpilot.telemetry.recorder import RunRecorder
from tests.fixture_tasks import autocomplete_task, fixture_server, form_task
from tests.runtime_support import ScriptedPlanner, ScriptedSelector, ScriptedText, budgets


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_completes_multistep_form_from_observed_controls(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        form_budgets = budgets().model_copy(update={"max_mutations": 6})
        task, binding = form_task(origin, mode, budgets=form_budgets)
        planner = ScriptedPlanner()
        selector = ScriptedSelector(
            confidence=0.99 if mode == "jev_hybrid" else None,
            actions=(
                ActionType.TYPE_TEXT,
                ActionType.TYPE_TEXT,
                ActionType.CLICK,
                ActionType.SELECT,
                ActionType.CLICK,
            ),
            target_names=("Full name", "City", "Continue", "Region", "Submit profile"),
            target_values=(None, None, None, "north", None),
        )
        text = ScriptedText()
        recorder = RunRecorder(form_budgets)
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
    assert trace.terminal.status == "succeeded", (
        trace.terminal,
        trace.executions,
    )
    assert planner.calls == 1
    assert selector.calls == 5
    assert text.calls == 2
    assert tuple(item.operation for item in trace.executions) == (
        ActionType.TYPE_TEXT,
        ActionType.TYPE_TEXT,
        ActionType.CLICK,
        ActionType.SELECT,
        ActionType.CLICK,
    )
    assert trace.verifications[-1].task_evaluation.status == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_handles_delayed_autocomplete_with_bounded_wait(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        task, binding = autocomplete_task(origin, mode, budgets=budgets())
        planner = ScriptedPlanner()
        selector = ScriptedSelector(
            confidence=0.99 if mode == "jev_hybrid" else None,
            autocomplete=True,
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
    assert text.calls == 1
    assert recorder.waits <= 1
    assert trace.verifications[-1].task_evaluation.status == "passed"
