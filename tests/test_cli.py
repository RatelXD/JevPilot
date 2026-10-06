from collections.abc import AsyncIterator
import os
from pathlib import Path
import subprocess
import sys

import pytest
from playwright.async_api import Browser, Route, async_playwright
from pydantic import ValidationError

from jevpilot.cli_config import UserRunConfig, parse_options
from jevpilot.user_task import build_user_task

ROOT = Path(__file__).parents[1]


@pytest.fixture
async def browser() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            yield browser
        finally:
            await browser.close()


def test_cli_requires_live_mode_and_explicit_completion_condition() -> None:
    # Given
    options = parse_options(
        [
            "run",
            "--mode",
            "llm_only",
            "--url",
            "https://example.test/start",
            "--goal",
            "Open the task result",
            "--verify-url-path",
            "/done",
            "--live",
        ]
    )

    # When / Then
    assert options.mode == "llm_only"
    assert options.url is not None
    assert options.goal is not None
    config = UserRunConfig(
        mode=options.mode,
        url=options.url,
        goal=options.goal,
        allowed_origins=("https://example.test",),
        verify_url_path=options.verify_url_path,
        verify_text=options.verify_text,
        max_steps=options.max_steps,
        wall_time_ms=options.wall_time_ms,
    )
    task, _ = build_user_task(config)
    assert task.mode == "llm_only"
    assert task.initial_url == "https://example.test/start"
    assert "verify_url_path" not in task.model_dump()
    assert "verify_text" not in task.model_dump()


def test_cli_rejects_missing_completion_condition_and_embedded_credentials() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = UserRunConfig(
            mode="llm_only",
            url="https://example.test/start",
            goal="Do something",
            allowed_origins=("https://example.test",),
            max_steps=20,
            wall_time_ms=30_000,
        )
    with pytest.raises(ValidationError):
        _ = UserRunConfig(
            mode="llm_only",
            url="https://user:secret@example.test/start",
            goal="Do something",
            allowed_origins=("https://example.test",),
            verify_url_path="/done",
            max_steps=20,
            wall_time_ms=30_000,
        )


@pytest.mark.asyncio
async def test_cli_completion_condition_checks_real_page_state(browser: Browser) -> None:
    # Given
    config = UserRunConfig(
        mode="llm_only",
        url="https://example.test/start",
        goal="Open the result",
        allowed_origins=("https://example.test",),
        verify_url_path="/done",
        verify_text="Task complete",
        max_steps=20,
        wall_time_ms=30_000,
    )
    _, binding = build_user_task(config)
    context = await browser.new_context()
    page = await context.new_page()

    async def fulfill(route: Route) -> None:
        await route.fulfill(body="<main><h1>Task complete</h1></main>")

    _ = await page.route("https://example.test/**", fulfill)
    try:
        _ = await page.goto("https://example.test/done")

        # When
        result = await binding.evaluator(page)

        # Then
        assert result.status == "passed"
        assert {check.check_id for check in result.checks} == {
            "user_url_path",
            "user_visible_text",
        }
        assert all("Task complete" not in " ".join(check.evidence) for check in result.checks)
    finally:
        await context.close()


def test_cli_rejects_product_mock_mode() -> None:
    # Given / When / Then
    with pytest.raises(ValueError, match="usage:"):
        _ = parse_options(
            ["run", "--mode", "mock", "--url", "https://example.test", "--goal", "run", "--live"]
        )


def test_live_user_cli_fails_before_invocation_when_model_is_unset() -> None:
    # Given
    environment = os.environ.copy()
    for name in (
        "JEVPILOT_LLM_MODEL",
        "JEVPILOT_LLM_PROVIDER",
        "JEVPILOT_LLM_API_KEY",
        "JEVPILOT_JEV_API_KEY",
        "JEVPILOT_JEV_MODEL",
        "JEVPILOT_JEV_OPERATION_THRESHOLD",
        "JEVPILOT_JEV_TARGET_THRESHOLD",
        "JEVPILOT_TEST_LLM_CREDIT_LIMIT",
        "JEVPILOT_TEST_JEV_SPEND_LIMIT_USD",
        "JEVPILOT_TEST_JEV_RESERVATION_USD",
    ):
        _ = environment.pop(name, None)

    # When
    result = subprocess.run(
        (
            sys.executable,
            "-m",
            "jevpilot",
            "run",
            "--mode",
            "llm_only",
            "--url",
            "https://example.test/start",
            "--goal",
            "Open the task result",
            "--verify-url-path",
            "/done",
            "--live",
        ),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert "JEVPILOT_LLM_MODEL must be set" in result.stderr
    assert "Run artifacts written" not in result.stdout
