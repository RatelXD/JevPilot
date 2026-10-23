import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from jevpilot.browser.browser_use_adapter import ACTION_REGISTRY, BrowserUseAdapter
from jevpilot.browser.port import BrowserAction, BrowserConfig

pytestmark = pytest.mark.browser


@pytest.fixture(scope="module")
def fixture_server() -> Iterator[str]:
    directory = Path(__file__).resolve().parents[1] / "benchmarks/tasks/fixtures"
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "http.server", "0", "--bind", "127.0.0.1", "--directory", str(directory)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        assert process.stdout is not None
        ready = process.stdout.readline()
        match = re.search(r"port (\d+)", ready)
        assert match is not None, ready
        print(f"server_started pid={process.pid} port={match[1]}")
        yield f"http://127.0.0.1:{match[1]}"
    finally:
        process.terminate()
        process.wait(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
        print(f"cleanup=server_terminated pid={process.pid} exit={process.returncode}")


async def test_observe_click_observe_when_search_page_is_live(
    fixture_server: str, tmp_path: Path,
) -> None:
    # Given
    adapter = BrowserUseAdapter()
    try:
        await adapter.start(BrowserConfig(
            headless=True, user_data_dir=tmp_path / "profile", start_url=f"{fixture_server}/search.html",
        ))
        # When
        before = await adapter.observe()
        target = next(item for item in before.elements if item.tag == "button")
        result = await adapter.execute(BrowserAction(kind="click", target=target), before)
        after = await adapter.observe()
        # Then
        assert before.title == "Catalog search"
        assert "Catalog search" in before.page_text
        assert len(before.elements) >= 5
        assert result.outcome == "executed"
        assert after.title == "Catalog search"
        assert before.snapshot_id != after.snapshot_id
        print(f"observe_click_observe={result.outcome} elements={len(after.elements)}")
    finally:
        await adapter.close()
        print("cleanup=search_browser_closed")


async def test_select_north_when_filter_page_is_live(
    fixture_server: str, tmp_path: Path,
) -> None:
    # Given
    adapter = BrowserUseAdapter()
    try:
        await adapter.start(BrowserConfig(
            headless=True, user_data_dir=tmp_path / "profile", start_url=f"{fixture_server}/filter.html",
        ))
        observation = await adapter.observe()
        target = next(item for item in observation.elements if item.tag == "select")
        assert {(option.value, option.label) for option in target.options} == {
            ("", "Select a region"), ("north", "North"), ("south", "South"),
        }
        # When
        result = await adapter.execute(
            BrowserAction(kind="select_option", target=target, value="north"), observation,
        )
        after = await adapter.observe()
        apply = next(item for item in after.elements if item.tag == "button")
        applied = await adapter.execute(BrowserAction(kind="click", target=apply), after)
        filtered = await adapter.observe()
        # Then
        assert result.outcome == "executed"
        assert applied.outcome == "executed"
        links = [item.text for item in filtered.elements if item.tag == "a"]
        assert links == ["Aurora"]
        print(f"select_north={result.outcome} filtered_links={links}")
    finally:
        await adapter.close()
        print("cleanup=filter_browser_closed")


async def test_stale_target_when_page_is_redrawn(
    fixture_server: str, tmp_path: Path,
) -> None:
    # Given
    adapter = BrowserUseAdapter()
    try:
        await adapter.start(BrowserConfig(
            headless=True, user_data_dir=tmp_path / "profile", start_url=f"{fixture_server}/search.html",
        ))
        before = await adapter.observe()
        target = next(item for item in before.elements if item.tag == "button")
        assert adapter._session is not None
        page = await adapter._session.get_current_page()
        assert page is not None
        # When: replace the entire form, preserving tags and selector positions.
        await page.evaluate(
            "() => { const form = document.querySelector('form'); form.replaceWith(form.cloneNode(true)); }",
        )
        result = await adapter.execute(BrowserAction(kind="click", target=target), before)
        # Then
        assert result.outcome == "failed_not_executed"
        assert result.error_kind == "stale_target"
        print(f"redraw_result={result.model_dump_json()}")
    finally:
        await adapter.close()
        print("cleanup=redraw_browser_closed")


def test_registry_when_pinned_browser_use_is_installed() -> None:
    # Given
    adapter = BrowserUseAdapter()
    registry = adapter._tools.registry.registry.actions
    # When / Then
    for kind, action_name in ACTION_REGISTRY.items():
        assert action_name in registry, (kind, action_name)
        print(f"registry {kind} -> {action_name}: present")
