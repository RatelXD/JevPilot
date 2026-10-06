"""Test-owned task bindings for the local synthetic catalog."""

from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Literal
from urllib.parse import urlsplit

from playwright.async_api import Page

from jevpilot.task import ActionType, Budget, TaskSpec
from jevpilot.verifier.models import Check, TaskEvaluation
from jevpilot.verifier.verifiers import TaskBinding


@contextmanager
def fixture_server() -> Generator[str, None, None]:
    directory = Path(__file__).parents[1] / "benchmarks" / "tasks" / "fixtures"
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            if thread.is_alive():
                raise RuntimeError("Fixture server did not stop")


async def initialize_catalog(page: Page, task: TaskSpec) -> None:
    _ = await page.goto(task.initial_url, wait_until="domcontentloaded")


async def evaluate_catalog(page: Page) -> TaskEvaluation:
    actual_path = urlsplit(page.url).path
    item = page.locator("#item-id")
    value = await item.inner_text() if await item.count() == 1 else ""
    url_status = "passed" if actual_path == "/items/aurora.html" else "not_yet"
    id_status = "passed" if value == "aurora" else "not_yet"
    return TaskEvaluation(
        evaluator_id="catalog",
        status="passed" if url_status == id_status == "passed" else "not_yet",
        checks=(
            Check(check_id="detail_url", scope="task", status=url_status, evidence=(actual_path,)),
            Check(check_id="item_id", scope="task", status=id_status, evidence=(value,)),
        ),
    )


async def evaluate_filter(page: Page) -> TaskEvaluation:
    base = await evaluate_catalog(page)
    query = dict(
        item.split("=", maxsplit=1)
        for item in urlsplit(page.url).query.split("&")
        if "=" in item
    )
    region_status = "passed" if query.get("region") == "north" else "not_yet"
    return TaskEvaluation(
        evaluator_id="filter",
        status=(
            "passed"
            if base.status == "passed" and region_status == "passed"
            else "not_yet"
        ),
        checks=(
            *base.checks,
            Check(
                check_id="selected_region",
                scope="task",
                status=region_status,
                evidence=(query.get("region", ""),),
            ),
        ),
    )


async def evaluate_form(page: Page) -> TaskEvaluation:
    query = dict(
        item.split("=", maxsplit=1)
        for item in urlsplit(page.url).query.split("&")
        if "=" in item
    )
    expected = {"name": "Riley", "city": "Reykjavik", "region": "north"}
    checks = tuple(
        Check(
            check_id=f"form_{name}",
            scope="task",
            status="passed" if query.get(name) == value else "not_yet",
            evidence=(f"{name}: {'matched' if query.get(name) == value else 'pending'}",),
        )
        for name, value in expected.items()
    )
    return TaskEvaluation(
        evaluator_id="form",
        status="passed" if all(check.status == "passed" for check in checks) else "not_yet",
        checks=checks,
    )


def catalog_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-00",
            goal="Open the details page for Aurora.",
            mode=mode,
            initial_url=f"{origin}/list.html",
            allowed_origins=(origin,),
            allowed_operations=(
                ActionType.CLICK,
                ActionType.SCROLL_UP,
                ActionType.SCROLL_DOWN,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            initializer_id="catalog",
            evaluator_id="catalog",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding("catalog", "catalog", initialize_catalog, evaluate_catalog),
    )


def search_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-01",
            goal="Search for Aurora and open its details page.",
            mode=mode,
            initial_url=f"{origin}/search.html",
            allowed_origins=(origin,),
            allowed_operations=(
                ActionType.CLICK,
                ActionType.TYPE_TEXT,
                ActionType.SCROLL_UP,
                ActionType.SCROLL_DOWN,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            initializer_id="catalog",
            evaluator_id="catalog",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding("catalog", "catalog", initialize_catalog, evaluate_catalog),
    )


def filter_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-02",
            goal="Select the North region, apply the filter, and open Aurora.",
            mode=mode,
            initial_url=f"{origin}/filter.html",
            allowed_origins=(origin,),
            allowed_operations=(
                ActionType.CLICK,
                ActionType.SELECT,
                ActionType.SCROLL_UP,
                ActionType.SCROLL_DOWN,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            initializer_id="catalog",
            evaluator_id="filter",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding("catalog", "filter", initialize_catalog, evaluate_filter),
    )


def form_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-03",
            goal=(
                "Complete the local profile form. Enter name Riley and city Reykjavik. "
                "Continue, select the North region, and submit the profile."
            ),
            mode=mode,
            initial_url=f"{origin}/form-step-one.html",
            allowed_origins=(origin,),
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
            initializer_id="catalog",
            evaluator_id="form",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding("catalog", "form", initialize_catalog, evaluate_form),
    )


def autocomplete_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-04",
            goal="Search the catalog for Aurora and open its suggestion.",
            mode=mode,
            initial_url=f"{origin}/autocomplete.html",
            allowed_origins=(origin,),
            allowed_operations=(
                ActionType.CLICK,
                ActionType.TYPE_TEXT,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            initializer_id="catalog",
            evaluator_id="catalog",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding("catalog", "catalog", initialize_catalog, evaluate_catalog),
    )


async def evaluate_multi_filter(page: Page) -> TaskEvaluation:
    query = dict(
        item.split("=", maxsplit=1)
        for item in urlsplit(page.url).query.split("&")
        if "=" in item
    )
    conditions = {
        "detail_path": urlsplit(page.url).path == "/items/aurora.html",
        "region": query.get("region") == "north",
        "free_cancellation": query.get("free") == "true",
    }
    checks = tuple(
        Check(
            check_id=name,
            scope="task",
            status="passed" if passed else "not_yet",
            evidence=(f"{name}: {'matched' if passed else 'pending'}",),
        )
        for name, passed in conditions.items()
    )
    return TaskEvaluation(
        evaluator_id="multi_filter",
        status="passed" if all(conditions.values()) else "not_yet",
        checks=checks,
    )


def multi_filter_task(
    origin: str, mode: Literal["llm_only", "jev_hybrid"], *, budgets: Budget
) -> tuple[TaskSpec, TaskBinding]:
    return (
        TaskSpec(
            task_id="T-05",
            goal=(
                "Choose the North region, select Free cancellation, apply both filters, "
                "and open Aurora."
            ),
            mode=mode,
            initial_url=f"{origin}/multi-filter.html",
            allowed_origins=(origin,),
            allowed_operations=(
                ActionType.CLICK,
                ActionType.SELECT,
                ActionType.SCROLL_UP,
                ActionType.SCROLL_DOWN,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            initializer_id="catalog",
            evaluator_id="multi_filter",
            budgets=budgets,
            locale="en-US",
            fixture_version="catalog-1",
        ),
        TaskBinding(
            "catalog",
            "multi_filter",
            initialize_catalog,
            evaluate_multi_filter,
        ),
    )
