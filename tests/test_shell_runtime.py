import io
import json
import os
from pathlib import Path
import pty
import sys
import termios
from typing import Literal, assert_never

import anyio
import httpx2
import pytest
from pydantic import JsonValue

from jevpilot.browser.browser_use_adapter import BrowserUseAdapter
from jevpilot.browser.port import BrowserConfig, NavigationWaiter
from jevpilot.engine.brief import StructuredTask
from jevpilot.engine.control import InteractiveControl, RunControl, ScriptedControl
from jevpilot.engine.jev import JevSelector, JevUnavailable
from jevpilot.engine.loop import RunOutcome
from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.profile import Profile, load_profile
from jevpilot.llm.types import Capabilities, ModelRef
from jevpilot.shell import app_shell as shell
from jevpilot.shell.prompt import PromptInput
from jevpilot.shell.render import ShellSink
from tests.test_engine_loop import BRIEF, TASK, Browser, Pick, Provider, page, wire
from tests.test_shell_prompt import Terminal


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Given: no ambient credential or actual HOME changes.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("JEVPILOT_JEV_API_KEY", "synthetic-jev")
    return tmp_path


def profile() -> Profile:
    return Profile("api", ModelRef("openai", "api", "test-model", "low"))


@pytest.mark.parametrize("natural", [False, True])
async def test_execute_uses_real_engine_and_emits_each_call_once(
    monkeypatch: pytest.MonkeyPatch, home: Path, natural: bool,
) -> None:
    # Given
    browser, provider = Browser([page(), page("Complete")]), Provider([BRIEF.model_dump_json()])
    def browser_factory() -> Browser:
        return browser

    def provider_factory(_profile: Profile) -> Provider:
        return provider

    monkeypatch.setattr(shell, "_ShellBrowser", browser_factory)
    monkeypatch.setattr(shell, "build_provider", provider_factory)
    lines: list[str] = []
    monkeypatch.setattr(shell, "write_line", lines.append)
    seen: list[dict[str, JsonValue]] = []
    async with httpx2.AsyncClient(transport=wire([Pick()], seen)) as client:
        def selector_factory(*, api_key: str) -> JevSelector:
            return JevSelector(api_key=api_key, http_client=client)

        monkeypatch.setattr(shell, "JevSelector", selector_factory)
        monkeypatch.setattr(sys, "stdin", io.StringIO())
        # When
        outcome = await shell.execute_task("Finish" if natural else TASK, profile(), headless=True)
    # Then
    assert outcome.status == "verified"
    assert provider.closed and browser.closed
    assert provider.purposes == (["brief"] if natural else [])
    assert len([line for line in lines if line.startswith("Jev #")]) == 1
    assert outcome.receipt.by_kind["jev"].calls == 1
    receipt, = (home / ".jevpilot" / "runs").glob("*/receipt.json")
    assert json.loads(receipt.read_text())["run_id"] == outcome.receipt.run_id
    assert browser.actions[0].kind == "click"


@pytest.mark.parametrize("headless", [False, True])
async def test_browser_configuration_is_private_and_inherits_navigation_wait(
    monkeypatch: pytest.MonkeyPatch, home: Path, headless: bool,
) -> None:
    # Given
    configs: list[BrowserConfig] = []

    async def started(_self: BrowserUseAdapter, config: BrowserConfig) -> None:
        configs.append(config)

    monkeypatch.setattr(BrowserUseAdapter, "start", started)
    browser = shell._ShellBrowser()
    browser.headless = headless
    # When
    await browser.start(BrowserConfig(locale="en-US"))
    # Then
    assert configs == [BrowserConfig(
        user_data_dir=home / ".jevpilot" / "chrome-profile",
        headless=headless, locale="en-US",
    )]
    assert isinstance(browser, NavigationWaiter)
    assert (home / ".jevpilot" / "chrome-profile").stat().st_mode & 0o777 == 0o700


async def test_missing_jev_refuses_before_provider_or_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.delenv("JEVPILOT_JEV_API_KEY")
    def forbidden(_profile: Profile) -> Provider:
        pytest.fail("provider started")

    monkeypatch.setattr(shell, "build_provider", forbidden)
    # When / Then
    with pytest.raises(JevUnavailable) as raised:
        _ = await shell.execute_task("Goal", profile())
    assert raised.value.kind == "auth_missing"


@pytest.mark.parametrize("approve", [False, True])
async def test_non_tty_task_uses_scripted_confirmation(
    monkeypatch: pytest.MonkeyPatch, approve: bool,
) -> None:
    # Given
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    answers: list[bool] = []

    async def execute(
        _goal: str | StructuredTask, _selected: Profile, control: RunControl, *,
        start_url: str | None = None, headless: bool = False,
    ) -> RunOutcome:
        assert isinstance(control, ScriptedControl)
        assert start_url is None and not headless
        answers.append(await control.confirm("submit"))
        return RunOutcome("verified", "ok", None, 0, None, RunLedger().finish())

    monkeypatch.setattr(shell, "execute_controlled", execute)
    # When
    _ = await shell.execute_task("Goal", profile(), approve_submit=approve)
    # Then
    assert answers == [approve]


@pytest.mark.parametrize("action", ["stop", "interrupt", "exit", "eof", "confirm", "ask", "handoff"])
async def test_active_control_and_cleanup_are_event_driven(
    monkeypatch: pytest.MonkeyPatch,
    action: Literal["stop", "interrupt", "exit", "eof", "confirm", "ask", "handoff"],
) -> None:
    # Given: the shell consumes actual InteractiveControl prompts and signals.
    started, question_visible, reader_cancelled = anyio.Event(), anyio.Event(), anyio.Event()
    reader_waiting = anyio.Event()
    finished: list[bool | str] = []
    controls: list[InteractiveControl] = []
    used = False

    def emit(_line: str) -> None:
        question_visible.set()

    async def execute(
        _goal: str | StructuredTask, _selected: Profile, control: InteractiveControl, *,
        start_url: str | None = None, headless: bool = False,
    ) -> RunOutcome:
        assert start_url is None and not headless
        controls.append(control)
        started.set()
        match action:
            case "confirm":
                finished.append(await control.confirm("submit"))
            case "ask":
                finished.append(await control.ask("value"))
            case "handoff":
                await control.handoff("login")
                finished.append(not control.recording_paused)
            case _:
                await control.stop_event.wait()
                finished.append(control.stop_requested)
        if action not in {"exit", "eof"}:
            await reader_waiting.wait()
        return RunOutcome("cancelled", "stopped", None, 0, None, RunLedger().finish())

    async def read(
        _selected: Profile | None, *, initial_text: str = "", initial_cursor: int = 0,
    ) -> PromptInput | None:
        nonlocal used
        assert initial_text == "" and initial_cursor == 0
        await started.wait()
        if used:
            try:
                reader_waiting.set()
                await anyio.Event().wait()
            finally:
                reader_cancelled.set()
        used = True
        match action:
            case "confirm" | "ask" | "handoff":
                await question_visible.wait()
                return PromptInput({"confirm": "y", "ask": "answer", "handoff": "/resume"}[action], False)
            case "stop":
                return PromptInput("/stop", False)
            case "interrupt":
                return PromptInput("", False, interrupted=True)
            case "exit":
                return PromptInput("/exit", False)
            case "eof":
                return None
            case unreachable:
                assert_never(unreachable)

    monkeypatch.setattr(shell, "execute_controlled", execute)
    monkeypatch.setattr(shell, "read_prompt", read)
    monkeypatch.setattr(shell, "write_line", emit)
    # When
    with anyio.fail_after(3):
        outcome, exited = await shell._interactive("Goal", profile())
    # Then
    assert outcome.status == "cancelled"
    assert exited == (action in {"exit", "eof"})
    assert finished == (["answer"] if action == "ask" else [True])
    assert reader_cancelled.is_set() == (action not in {"exit", "eof"})
    assert not controls[0].recording_paused


async def test_idle_shift_tab_preserves_draft_and_uses_supported_efforts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    selected = profile()
    monkeypatch.setattr(shell, "suggest_profile", lambda: selected)
    events = iter([
        PromptInput("first\nsecond", True, 3),
        PromptInput("first\nsecond", True, 3),
        PromptInput("/exit", False),
    ])
    reads: list[tuple[str, int, str | None]] = []

    async def capabilities(_profile: Profile) -> Capabilities:
        return Capabilities("native", False, ("low", "high"), False)

    async def read(
        current: Profile | None, *, initial_text: str = "", initial_cursor: int = 0,
    ) -> PromptInput:
        assert current is not None
        reads.append((initial_text, initial_cursor, current.model.reasoning_effort))
        return next(events)

    monkeypatch.setattr(shell, "read_prompt", read)
    monkeypatch.setattr(shell, "model_capabilities", capabilities)
    # When
    assert await shell._shell() == 0
    # Then
    assert reads == [("", 0, "low"), ("first\nsecond", 3, "high"), ("first\nsecond", 3, "low")]
    assert load_profile() == selected


async def test_multiline_slash_goal_is_forwarded_once_without_command_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(shell, "suggest_profile", profile)
    entries = iter([PromptInput("/stop\nsecond\nthird", False), PromptInput("/exit", False)])
    seen: list[str] = []

    async def read(
        _current: Profile | None, *, initial_text: str = "", initial_cursor: int = 0,
    ) -> PromptInput:
        assert initial_text == "" and initial_cursor == 0
        return next(entries)

    async def execute(goal: str, _selected: Profile) -> tuple[RunOutcome, bool]:
        seen.append(goal)
        return RunOutcome("verified", "ok", None, 0, None, RunLedger().finish()), False

    monkeypatch.setattr(shell, "read_prompt", read)
    monkeypatch.setattr(shell, "_interactive", execute)
    # When
    assert await shell._shell() == 0
    # Then
    assert seen == ["/stop\nsecond\nthird"]


def test_sink_persists_before_allowlisted_progress_without_payload_leak(home: Path) -> None:
    # Given
    lines: list[str] = []
    path = home / "run" / "events.jsonl"

    def emitted(line: str) -> None:
        assert path.is_file()
        assert len(path.read_text().splitlines()) == 1
        lines.append(line)

    sink = ShellSink("run", root=home, emit=emitted)
    # When
    sink.event("span", {"kind": "jev", "end_s": None, "label": "synthetic-private"})
    sink.event("route", {"text": "synthetic-private"})
    # Then
    assert len(lines) == 1
    assert "synthetic-private" not in lines[0]


async def test_actual_bracketed_paste_reaches_shell_as_one_unchanged_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: real terminal input and real producer editor, not prebuilt PromptInput.
    master, slave = pty.openpty()
    goals: list[str] = []
    received = anyio.Event()
    monkeypatch.setattr(shell, "suggest_profile", profile)

    async def execute(goal: str, _selected: Profile) -> tuple[RunOutcome, bool]:
        goals.append(goal)
        received.set()
        return RunOutcome("verified", "ok", None, 0, None, RunLedger().finish()), False

    monkeypatch.setattr(shell, "_interactive", execute)
    with os.fdopen(master, "rb", buffering=0), os.fdopen(
        slave, "r", encoding="utf-8",
    ) as input_stream, os.fdopen(os.dup(slave), "w", encoding="utf-8") as output:
        terminal = Terminal(master, input_stream, output, bytearray(), monkeypatch)
        terminal.activate()
        before = termios.tcgetattr(slave)

        async def type_goal() -> None:
            _ = await terminal.until(b"\x1b[?2004h")
            _ = os.write(master, b"\x1b[200~/stop\nsecond\nthird\x1b[201~\r")
            await received.wait()
            _ = await terminal.until(b"\x1b[?2004h")
            _ = os.write(master, b"/exit\r")

        # When
        with anyio.fail_after(3):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(type_goal)
                assert await shell._shell() == 0
        # Then
        assert goals == ["/stop\nsecond\nthird"]
        assert termios.tcgetattr(slave) == before
