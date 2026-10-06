"""Build a user task and its independent completion evaluator."""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import uuid4

from playwright.async_api import Page

from jevpilot.cli_config import UserRunConfig
from jevpilot.task import ActionType, Budget, TaskSpec
from jevpilot.verifier.models import Check, TaskEvaluation
from jevpilot.verifier.verifiers import TaskBinding


def build_user_task(
    config: UserRunConfig, *, task_id: str | None = None
) -> tuple[TaskSpec, TaskBinding]:
    """Bind URL initialization and user-declared final checks to a task."""
    run_id = task_id or f"user-{uuid4().hex}"
    budget = Budget(
        max_steps=config.max_steps,
        max_mutations=config.max_steps,
        max_replans=3,
        max_model_attempts=max(4, config.max_steps * 2),
        max_attempts_per_call=2,
        max_stale_observations=3,
        max_waits=3,
        max_wait_ms=5_000,
        wall_time_ms=config.wall_time_ms,
    )

    async def initialize(page: Page, task: TaskSpec) -> None:
        _ = await page.goto(task.initial_url, wait_until="domcontentloaded")

    async def evaluate(page: Page) -> TaskEvaluation:
        checks: list[Check] = []
        if config.verify_url_path is not None:
            actual_path = urlsplit(page.url).path
            passed = actual_path == config.verify_url_path
            checks.append(
                Check(
                    check_id="user_url_path",
                    scope="task",
                    status="passed" if passed else "not_yet",
                    evidence=(
                        "URL condition matched"
                        if passed
                        else "URL condition pending",
                    ),
                )
            )
        if config.verify_text is not None:
            visible_text = await page.locator("body").inner_text()
            passed = config.verify_text in visible_text
            checks.append(
                Check(
                    check_id="user_visible_text",
                    scope="task",
                    status="passed" if passed else "not_yet",
                    evidence=(
                        "visible text condition matched"
                        if passed
                        else "visible text condition pending",
                    ),
                )
            )
        status = (
            "passed"
            if checks and all(check.status == "passed" for check in checks)
            else "not_yet"
        )
        return TaskEvaluation(
            evaluator_id="user_condition",
            status=status,
            checks=tuple(checks),
        )

    task = TaskSpec(
        task_id=run_id,
        goal=config.goal,
        mode=config.mode,
        initial_url=config.url,
        allowed_origins=config.allowed_origins,
        allowed_operations=(
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
            ActionType.SCROLL_UP,
            ActionType.SCROLL_DOWN,
            ActionType.WAIT,
            ActionType.DONE,
            ActionType.BLOCKED,
        ),
        initializer_id="user_url",
        evaluator_id="user_condition",
        budgets=budget,
        locale="en-US",
        fixture_version="user-run-1",
    )
    return task, TaskBinding("user_url", "user_condition", initialize, evaluate)
