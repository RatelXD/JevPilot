from typing import Literal

import pytest
from playwright.async_api import async_playwright

from jevpilot.cli_config import UserRunConfig
from jevpilot.privacy import PrivacyPolicy
from jevpilot.runtime import HybridBrowserAgent
from jevpilot.task import ActionType
from jevpilot.telemetry.recorder import RunRecorder
from jevpilot.user_task import build_user_task
from tests.fixture_tasks import fixture_server, multi_filter_task
from tests.runtime_support import ScriptedPlanner, ScriptedSelector, ScriptedText, budgets


@pytest.mark.asyncio
async def test_generic_user_url_goal_and_completion_conditions_run_in_browser() -> None:
    # Given
    with fixture_server() as origin:
        config = UserRunConfig(
            mode="llm_only",
            url=f"{origin}/list.html",
            goal="Open Aurora",
            allowed_origins=(origin,),
            verify_url_path="/items/aurora.html",
            verify_text="Aurora",
            max_steps=8,
            wall_time_ms=15_000,
        )
        task, binding = build_user_task(config)
        planner = ScriptedPlanner()
        selector = ScriptedSelector()
        recorder = RunRecorder(task.budgets)
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
    assert trace.verifications[-1].task_evaluation.evaluator_id == "user_condition"
    assert trace.verifications[-1].task_evaluation.status == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["llm_only", "jev_hybrid"])
async def test_runtime_combines_select_and_checkbox_filters(
    mode: Literal["llm_only", "jev_hybrid"],
) -> None:
    # Given
    with fixture_server() as origin:
        filter_budgets = budgets().model_copy(update={"max_mutations": 6})
        task, binding = multi_filter_task(origin, mode, budgets=filter_budgets)
        planner = ScriptedPlanner()
        selector = ScriptedSelector(
            confidence=0.99 if mode == "jev_hybrid" else None,
            actions=(
                ActionType.SELECT,
                ActionType.CLICK,
                ActionType.CLICK,
                ActionType.CLICK,
            ),
            target_names=(
                "Region",
                "Free cancellation",
                "Apply filters",
                "Aurora",
            ),
            target_values=("north", None, None, None),
        )
        recorder = RunRecorder(filter_budgets)
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
    assert tuple(item.operation for item in trace.executions) == (
        ActionType.SELECT,
        ActionType.CLICK,
        ActionType.CLICK,
        ActionType.CLICK,
    )
    assert trace.verifications[-1].task_evaluation.status == "passed"
