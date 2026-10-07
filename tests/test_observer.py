from __future__ import annotations

import time
from collections.abc import AsyncIterator

import anyio
import pytest
from playwright.async_api import Browser, BrowserContext, Page, Route, async_playwright

from jevpilot.observer import Observer
from jevpilot.privacy import PrivacyPolicy
from jevpilot.task import ActionType, Budget, TaskSpec

ORIGIN = "http://fixture.test"


@pytest.fixture
async def browser() -> AsyncIterator[Browser]:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        yield browser
        await browser.close()


@pytest.mark.asyncio
async def test_observer_waits_for_document_parsing(
    browser: Browser,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_requested = anyio.Event()
    release_script = anyio.Event()
    waiting_for_document = anyio.Event()
    context = await browser.new_context()
    page = await context.new_page()
    original_wait = page.wait_for_load_state

    async def wait_for_document(*args, **kwargs):
        waiting_for_document.set()
        return await original_wait(*args, **kwargs)

    async def serve(route: Route) -> None:
        if route.request.url.endswith("/parser.js"):
            script_requested.set()
            await release_script.wait()
            await route.fulfill(body="", content_type="application/javascript")
        else:
            await route.fulfill(
                body='<html><head><script src="/parser.js"></script></head>'
                '<body><button>Ready control</button></body></html>',
                content_type="text/html",
            )

    await page.route("**/*", serve)
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    candidates = ()

    async def observe() -> None:
        nonlocal candidates
        _, candidates = await observer.observe(deadline_monotonic=time.monotonic() + 10)

    try:
        await page.goto(ORIGIN, wait_until="commit")
        with anyio.fail_after(10):
            await script_requested.wait()
            assert await page.evaluate("document.body === null")
            monkeypatch.setattr(page, "wait_for_load_state", wait_for_document)
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(observe)
                await waiting_for_document.wait()
                release_script.set()
        assert "Ready control" in [candidate.name for candidate in candidates]
    finally:
        release_script.set()
        await observer.close()
        await context.close()


def task_spec(
    *,
    operations: tuple[ActionType, ...] = (
        ActionType.CLICK,
        ActionType.DONE,
        ActionType.BLOCKED,
    ),
) -> TaskSpec:
    return TaskSpec(
        task_id="T-observer",
        goal="Click the requested synthetic control",
        mode="llm_only",
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
            max_waits=0,
            max_wait_ms=500,
            wall_time_ms=10_000,
        ),
        locale="en-US",
        fixture_version="1",
    )


async def fixture_page(
    browser: Browser,
    html: str,
) -> tuple[BrowserContext, Page]:
    context = await browser.new_context()
    page = await context.new_page()

    async def fulfill(route: Route) -> None:
        await route.fulfill(status=200, content_type="text/html", body=html)

    _ = await page.route(f"{ORIGIN}/**", fulfill)
    _ = await page.goto(f"{ORIGIN}/start")
    return context, page


@pytest.mark.asyncio
async def test_observer_discovers_dynamic_controls_and_redacts_public_projection(
    browser: Browser,
) -> None:
    context, page = await fixture_page(
        browser,
        """
        <title>synthetic-secret catalog</title>
        <main>
          <p>synthetic-secret visible content</p>
          <button id="first">Open synthetic-secret item</button>
          <button disabled>Disabled</button>
          <button hidden>Hidden</button>
          <button aria-label=""></button>
          <a href="https://forbidden.test/item">Forbidden destination</a>
        </main>
        """,
    )
    observer = Observer(
        page,
        task=task_spec(),
        privacy=PrivacyPolicy(secret_values=("synthetic-secret",)),
    )
    try:
        first, first_candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        assert [candidate.name for candidate in first_candidates] == [
            "Open [REDACTED] item"
        ]
        assert "[REDACTED]" in first.title
        assert "[REDACTED]" in first.text
        assert "synthetic-secret" not in first.model_dump_json()
        assert first.omitted_counts.disabled == 1
        assert first.omitted_counts.hidden == 1
        assert first.omitted_counts.unnamed == 1

        await page.evaluate(
            """() => {
                const button = document.createElement("button");
                button.textContent = "Dynamic item";
                document.querySelector("main").append(button);
            }"""
        )
        second, second_candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        assert second.snapshot_id != first.snapshot_id
        assert [candidate.name for candidate in second_candidates] == [
            "Open [REDACTED] item",
            "Dynamic item",
        ]
        assert all(
            candidate.snapshot_id == second.snapshot_id
            for candidate in second_candidates
        )
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_observer_caps_candidates_and_visible_text(browser: Browser) -> None:
    buttons = "".join(f"<button>Item {index}</button>" for index in range(130))
    context, page = await fixture_page(
        browser,
        f"<main>{'x' * 7000}{buttons}</main>",
    )
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    try:
        observation, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
        assert len(candidates) == 128
        assert len(observation.controls) == 128
        assert len(observation.text) == 6000
        assert observation.omitted_counts.over_limit == 2
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_observer_rejects_disallowed_current_origin(browser: Browser) -> None:
    context = await browser.new_context()
    page = await context.new_page()

    async def fulfill(route: Route) -> None:
        await route.fulfill(body="<button>Wrong origin</button>")

    _ = await page.route("http://other.test/**", fulfill)
    _ = await page.goto("http://other.test/start")
    observer = Observer(page, task=task_spec(), privacy=PrivacyPolicy())
    try:
        with pytest.raises(ValueError, match="origin is not allowed"):
            _ = await observer.observe(deadline_monotonic=time.monotonic() + 5)
    finally:
        await observer.close()
        await context.close()


@pytest.mark.asyncio
async def test_observer_discovers_editable_text_and_native_select_options(
    browser: Browser,
) -> None:
    # Given
    context, page = await fixture_page(
        browser,
        """
        <label for="query">Search catalog</label>
        <input id="query" type="search" maxlength="24" value="old">
        <label for="region">Region</label>
        <select id="region">
          <option value="north">North</option>
          <option value="south" selected>South</option>
          <option value="blocked" disabled>Blocked</option>
        </select>
        <input type="password" aria-label="Password" value="synthetic-secret">
        <input type="text" aria-label="Readonly" readonly value="locked">
        """,
    )
    task = task_spec(
        operations=(
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
            ActionType.DONE,
            ActionType.BLOCKED,
        )
    )
    observer = Observer(page, task=task, privacy=PrivacyPolicy())

    # When
    try:
        observation, candidates = await observer.observe(
            deadline_monotonic=time.monotonic() + 5
        )
    finally:
        await observer.close()
        await context.close()

    # Then
    text_target = next(
        candidate
        for candidate in candidates
        if candidate.operation is ActionType.TYPE_TEXT
    )
    select_candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.operation is ActionType.SELECT
    )
    assert text_target.name == "Search catalog"
    assert text_target.value == "old"
    assert text_target.max_length == 24
    assert {candidate.value for candidate in select_candidates} == {"north", "south"}
    assert all(candidate.option_ref is not None for candidate in select_candidates)
    assert all(candidate.snapshot_id == observation.snapshot_id for candidate in candidates)
    assert all(control.name != "Password" for control in observation.controls)
    assert all(control.name != "Readonly" for control in observation.controls)
    assert "synthetic-secret" not in observation.model_dump_json()
