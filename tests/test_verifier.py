from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from playwright.async_api import Page, async_playwright

from jevpilot.observer.models import Observation, OmittedCounts
from jevpilot.planner.models import ProgressHint, Subgoal
from jevpilot.task import TaskSpec
from jevpilot.verifier.models import Check, TaskEvaluation
from jevpilot.verifier.verifiers import (
    EvaluationError,
    IndependentVerifier,
    TaskBinding,
)


@pytest.fixture
async def page() -> AsyncIterator[Page]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        context = await browser.new_context()
        try:
            yield await context.new_page()
        finally:
            await context.close()
            await browser.close()


async def initialize(page: Page, task: TaskSpec) -> None:
    _ = await page.goto(task.initial_url)


async def evaluate_status(page: Page) -> TaskEvaluation:
    value = await page.locator("#status").inner_text()
    status = "passed" if value == "done" else "not_yet"
    return TaskEvaluation(
        evaluator_id="status",
        status=status,
        checks=(Check(check_id="status", scope="task", status=status, evidence=(value,)),),
    )


def observation() -> Observation:
    return Observation(
        snapshot_id="s1", document_id="d1", navigation_id="n1",
        url="https://example.test/complete", title="", text="", controls=(),
        fingerprint="f1", observed_at=datetime.now(UTC),
        omitted_counts=OmittedCounts(hidden=0, disabled=0, readonly=0, unnamed=0, over_limit=0),
    )


@pytest.mark.asyncio
async def test_subgoal_completion_does_not_override_task_evaluator(page: Page) -> None:
    # Given
    await page.set_content('<p id="status">not done</p>')
    subgoal = Subgoal(
        subgoal_id="sg1", description="Open the page",
        progress_hint=ProgressHint(kind="url_path", expected="/complete"),
    )
    binding = TaskBinding("init", "status", initialize, evaluate_status)

    # When
    result = await IndependentVerifier().verify(
        page, observation=observation(), subgoal=subgoal, binding=binding,
    )

    # Then
    assert result.subgoal_status == "passed"
    assert result.task_evaluation.status == "not_yet"


@pytest.mark.asyncio
async def test_task_passes_from_actual_page_state(page: Page) -> None:
    # Given
    await page.set_content('<p id="status">done</p>')
    binding = TaskBinding("init", "status", initialize, evaluate_status)

    # When
    result = await IndependentVerifier().verify(
        page, observation=observation(),
        subgoal=Subgoal(subgoal_id="sg1", description="Run"), binding=binding,
    )

    # Then
    assert result.task_evaluation.status == "passed"
    assert result.task_evaluation.checks[0].evidence == ("done",)


@pytest.mark.asyncio
async def test_evaluator_cannot_claim_success_without_checks(page: Page) -> None:
    # Given
    async def invalid_evaluator(_page: Page) -> TaskEvaluation:
        return TaskEvaluation(evaluator_id="invalid", status="passed", checks=())

    binding = TaskBinding("init", "invalid", initialize, invalid_evaluator)

    # When / Then
    with pytest.raises(EvaluationError, match="success_without_passing_checks"):
        _ = await IndependentVerifier().verify(
            page, observation=observation(),
            subgoal=Subgoal(subgoal_id="sg1", description="Run"), binding=binding,
        )
