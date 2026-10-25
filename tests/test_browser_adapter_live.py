import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import anyio
from anyio.abc import SocketAttribute, SocketStream
from cdp_use.cdp.page.events import FrameNavigatedEvent

from jevpilot.browser.browser_use_adapter import ACTION_REGISTRY, BrowserUseAdapter
from jevpilot.browser.port import BrowserAction, BrowserConfig, ExecutionResult
from jevpilot.engine.brief import OpenedTarget, TaskBrief
from jevpilot.engine.verify import OpenEvidence, verify

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
        selected = next(item for item in after.elements if item.tag == "select")
        assert target.selected_values == [""]
        assert selected.selected_values == ["north"]
        assert after.fingerprint != observation.fingerprint
        unchanged = await adapter.observe()
        assert unchanged.fingerprint == after.fingerprint
        apply = next(item for item in unchanged.elements if item.tag == "button")
        applied = await adapter.execute(BrowserAction(kind="click", target=apply), unchanged)
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


async def test_delayed_navigation_is_bound_to_real_document_not_later_visit(tmp_path: Path) -> None:
    # Given: the server cannot return the destination until the old observation
    # has been asserted. The ordering is an event handshake, not a timed delay.
    requested, release, clicked = anyio.Event(), anyio.Event(), anyio.Event()
    adapter = BrowserUseAdapter()
    results: list[ExecutionResult] = []

    async def serve(stream: SocketStream) -> None:
        async with stream:
            request = b""
            while b"\r\n\r\n" not in request:
                try:
                    request += await stream.receive()
                except anyio.EndOfStream:
                    # Chrome may close a speculative preconnection without a request.
                    return
            path = request.split(b" ")[1]
            if path == b"/detail":
                requested.set()
                await release.wait()
            body = (
                b'<h1>Catalog</h1><button onclick="location.href=\'/detail\'">Widget</button>'
                if path == b"/list" else b"<h1>Widget details</h1>"
            )
            await stream.send(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\nContent-Length: "
                + str(len(body)).encode() + b"\r\n\r\n" + body,
            )

    async with await anyio.create_tcp_listener(local_host="127.0.0.1", local_port=0) as listener:
        port = listener.extra(SocketAttribute.local_port)
        url = f"http://127.0.0.1:{port}"
        async with anyio.create_task_group() as group:
            group.start_soon(listener.serve, serve)
            try:
                await adapter.start(BrowserConfig(
                    headless=False, user_data_dir=tmp_path / "delayed-profile", start_url=url + "/list",
                ))
                before = await adapter.observe()
                commits: list[str] = []

                def committed(event: FrameNavigatedEvent, session_id: str | None = None) -> None:
                    if session_id == adapter._navigation_session:
                        commits.append(event["frame"]["loaderId"])

                assert adapter._navigation_client is not None
                adapter._navigation_client.register.Page.frameNavigated(committed)
                target = next(item for item in before.elements if item.tag == "button")
                action = BrowserAction(kind="click", target=target)
                task = TaskBrief(
                    goal_restated="Open Widget", start_url=before.url, input_values={},
                    required_outputs=[], success_conditions=[OpenedTarget(text="Widget")],
                    missing_inputs=[], mutating_intent=False, multi_step=False, comparison=None,
                )

                async def click() -> None:
                    results.append(await adapter.execute(action, before))
                    clicked.set()

                # When
                with anyio.fail_after(30):
                    group.start_soon(click)
                    await requested.wait()
                    # Chrome queues document commands during provisional loading.
                    # The pre-dispatch subscription independently proves that no
                    # destination document has committed while the response is gated.
                    assert commits == []
                    assert verify(before, task).status == "FAIL"
                    release.set()
                    await clicked.wait()
                    result, = results
                    assert result.outcome == "executed" and result.navigation_id is not None
                    await adapter.wait_for_navigation(result.navigation_id)
                    after = await adapter.observe()
                    evidence = OpenEvidence(before, action, before, result)
                    assert result.navigation_id in commits
                    assert after.document_id == result.navigation_id
                    assert verify(after, task, opened=(evidence,)).status == "PASS"
                    # A new visit to the SAME URL must not inherit old click proof.
                    assert (await adapter.execute(
                        BrowserAction(kind="navigate", url=url + "/list"), after,
                    )).outcome == "executed"
                    listing = await adapter.observe()
                    assert (await adapter.execute(
                        BrowserAction(kind="navigate", url=url + "/detail"), listing,
                    )).outcome == "executed"
                    later = await adapter.observe()
                    assert later.url == after.url and later.document_id != after.document_id
                    assert verify(later, task, opened=(evidence,)).status == "FAIL"
                    print("delayed_open=PASS first_observation=source later_same_url=FAIL")
            finally:
                release.set()
                with anyio.CancelScope(shield=True):
                    await adapter.close()
                group.cancel_scope.cancel()
                print(f"cleanup=delayed_browser_closed server_port={port}")
