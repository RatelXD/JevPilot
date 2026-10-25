import io
import os
import pty
import sys
import termios
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import anyio
import pytest

from jevpilot.llm.profile import Profile
from jevpilot.llm.types import ModelRef
from jevpilot.shell import prompt


@dataclass
class Terminal:
    master: int
    input: TextIO
    output: TextIO
    transcript: bytearray
    patch: pytest.MonkeyPatch

    def activate(self) -> None:
        # Pytest installs per-phase capture after fixture setup, so bind the
        # real PTY streams in the test phase, not the fixture setup phase.
        self.patch.setattr(sys, "stdin", self.input)
        self.patch.setattr(sys, "stdout", self.output)

    async def until(self, marker: bytes) -> bytes:
        chunks = bytearray()
        with anyio.fail_after(3):
            while marker not in chunks:
                await anyio.wait_readable(self.master)
                chunk = os.read(self.master, 65536)
                assert chunk
                chunks.extend(chunk)
                self.transcript.extend(chunk)
        return bytes(chunks)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> Iterator[Terminal]:
    master, slave = pty.openpty()
    with os.fdopen(master, "rb", buffering=0), os.fdopen(
        slave, "r", encoding="utf-8"
    ) as input_stream, os.fdopen(os.dup(slave), "w", encoding="utf-8") as output:
        yield Terminal(master, input_stream, output, bytearray(), monkeypatch)


async def submit(
    terminal: Terminal, payload: bytes, *, profile: Profile | None = None,
    initial_text: str = "", initial_cursor: int = 0,
) -> prompt.PromptInput | None:
    terminal.activate()
    before = termios.tcgetattr(terminal.input.fileno())
    results: list[prompt.PromptInput | None] = []
    done = anyio.Event()

    async def read() -> None:
        results.append(await prompt.read_prompt(
            profile, initial_text=initial_text, initial_cursor=initial_cursor,
        ))
        done.set()

    with anyio.fail_after(3):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(read)
            _ = await terminal.until(b"\x1b[?2004h")
            _ = os.write(terminal.master, payload)
            await done.wait()
            _ = await terminal.until(b"\x1b[?2004l")
    assert termios.tcgetattr(terminal.input.fileno()) == before
    return results[0]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"first\nsecond\x1b\rthird\r", "first\nsecond\nthird"),
        (b"\x1b[200~/stop\nnot a command\r\n\x1b[201~\r",
         "/stop\nnot a command\r\n"),
        (b"/stop\r", "/stop"),
        ("한글🙂\x1b[D!\r".encode(), "한글!🙂"),
        (b"abc\x1b[D\x1b[3~\x7fZ\r", "aZ"),
        (b"abc\x1b[D\x1b[D\x1b[CZ\r", "abZc"),
        (b"\x1b[D\x7f\x1b[C\r", ""),
        (b"ab\x1b[999~c\x1bx\r", "abcx"),
        (b"ab\x1b[12\r", "ab"),
        (b"\x1b[200~x\x1b[201Xy\x1b[201~\r", "x\x1b[201Xy"),
        (b"\x1b[200~\x1b[Z\n\x1b[201~\r", "\x1b[Z\n"),
        (b"x\xffy\r", "x�y"),
    ],
)
async def test_tty_submission_preserves_goal_bytes(
    terminal: Terminal, payload: bytes, expected: str,
) -> None:
    # Given: actual raw PTY input, with no authentication profile required.
    # When
    result = await submit(terminal, payload)
    # Then: the shell receives goal bytes, not continuation/paste delimiters.
    assert result is not None
    assert result.text == expected
    assert not result.effort_changed
    assert not result.interrupted
    if "\n" in expected:
        assert b"... " in terminal.transcript


async def test_shift_tab_returns_and_resumes_multiline_character_cursor(
    terminal: Terminal,
) -> None:
    # Given: profile changes belong to the caller and supported capability tuple.
    profile = Profile("api", ModelRef("openai", "api", "test-model", "low"))
    draft = "한\ncd"
    # When: repeated Shift+Tab events do not submit or mutate the draft.
    first = await submit(terminal, b"\x1b[Z", profile=profile,
                         initial_text=draft, initial_cursor=2)
    assert first is not None
    assert first == prompt.PromptInput(draft, True, 2)
    profile = replace(profile, model=replace(profile.model, reasoning_effort="high"))
    second = await submit(terminal, b"\x1b[Z", profile=profile,
                          initial_text=first.text, initial_cursor=first.cursor)
    assert second is not None
    assert second == first
    final = await submit(terminal, b"!\r", profile=profile,
                         initial_text=second.text, initial_cursor=second.cursor)
    # Then
    assert final == prompt.PromptInput("한\n!cd", False, 3)
    assert profile.model.reasoning_effort == "high"


@pytest.mark.parametrize("prefix", [b"", b"\x1b", b"\x1b[20", b"\x1b[200~draft", b"\xe3"])
@pytest.mark.parametrize("key", [b"\x03", b"\x04"])
async def test_interrupt_and_eof_restore_terminal_during_partial_sequences(
    terminal: Terminal, prefix: bytes, key: bytes,
) -> None:
    # Given / When
    result = await submit(terminal, prefix + key)
    # Then
    assert result == (prompt.PromptInput("", False, interrupted=True) if key == b"\x03" else None)


@pytest.mark.parametrize("prefix", [b"\x1b[20", b"\xe3", b"\x1b[200~\x1b[201"])
async def test_cancellation_during_incomplete_input_does_not_block_other_tasks(
    terminal: Terminal, monkeypatch: pytest.MonkeyPatch, prefix: bytes,
) -> None:
    # Given: observe the next descriptor wait after every input byte was read.
    terminal.activate()
    before = termios.tcgetattr(terminal.input.fileno())
    original_wait = anyio.wait_readable
    consumed = anyio.Event()
    waits = 0
    finished = anyio.Event()

    async def observe_wait(descriptor: int) -> None:
        nonlocal waits
        if descriptor == terminal.input.fileno():
            waits += 1
            if waits == len(prefix) + 1:
                consumed.set()
        await original_wait(descriptor)

    monkeypatch.setattr(anyio, "wait_readable", observe_wait)

    async def read() -> None:
        try:
            _ = await prompt.read_prompt(None)
        finally:
            finished.set()

    # When: wait for exact partial-input consumption, then cancel the reader.
    with anyio.fail_after(3):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(read)
            _ = await terminal.until(b"\x1b[?2004h")
            _ = os.write(terminal.master, prefix)
            await consumed.wait()
            tasks.cancel_scope.cancel()
    # Then: cancellation joined the task and restored the complete terminal state.
    assert finished.is_set()
    assert termios.tcgetattr(terminal.input.fileno()) == before
    _ = await terminal.until(b"\x1b[?2004l")


async def test_progress_line_preserves_active_draft_and_cursor(terminal: Terminal) -> None:
    # Given
    terminal.activate()
    done = anyio.Event()
    results: list[prompt.PromptInput | None] = []
    before = termios.tcgetattr(terminal.input.fileno())

    async def read() -> None:
        results.append(await prompt.read_prompt(None, initial_text="a\nbc", initial_cursor=3))
        done.set()

    # When
    with anyio.fail_after(3):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(read)
            _ = await terminal.until(b"\x1b[?2004h")
            prompt.write_line("stage-event")
            _ = await terminal.until(b"stage-event")
            _ = os.write(terminal.master, b"!\r")
            await done.wait()
            _ = await terminal.until(b"\x1b[?2004l")
    # Then
    assert results == [prompt.PromptInput("a\nb!c", False, 4)]
    assert termios.tcgetattr(terminal.input.fileno()) == before


async def test_scripted_input_is_line_oriented(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(sys, "stdin", io.StringIO("/stop\nnext\nlast"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    # When
    results = [await prompt.read_prompt(None) for _ in range(4)]
    # Then
    assert results == [prompt.PromptInput(text, False) for text in ("/stop", "next", "last")] + [None]


async def test_regular_file_input_never_registers_disk_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    path = tmp_path / "input"
    _ = path.write_text("first\nlast", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    async def forbidden_wait(descriptor: int) -> None:
        pytest.fail(f"disk descriptor registered: {descriptor}")

    monkeypatch.setattr(anyio, "wait_readable", forbidden_wait)
    with path.open() as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        # When
        results = [await prompt.read_prompt(None) for _ in range(3)]
    # Then
    assert results == [prompt.PromptInput("first", False), prompt.PromptInput("last", False), None]


async def test_pipe_preserves_pending_lines_and_final_unterminated_utf8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    reader, writer = os.pipe()
    with os.fdopen(reader, "r", encoding="utf-8") as stream, os.fdopen(writer, "wb") as out:
        monkeypatch.setattr(sys, "stdin", stream)
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        _ = out.write("/stop\r\n다음\n끝".encode())
        out.flush()
        out.close()
        # When
        with anyio.fail_after(3):
            results = [await prompt.read_prompt(None) for _ in range(4)]
    # Then
    assert results == [prompt.PromptInput(text, False) for text in ("/stop", "다음", "끝")] + [None]


async def test_pipe_read_cancels_without_an_abandoned_input_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    reader, writer = os.pipe()
    original_wait = anyio.wait_readable
    waiting = anyio.Event()

    async def observe_wait(descriptor: int) -> None:
        waiting.set()
        await original_wait(descriptor)

    monkeypatch.setattr(anyio, "wait_readable", observe_wait)
    with os.fdopen(reader, "r") as stream, os.fdopen(writer, "wb"):
        monkeypatch.setattr(sys, "stdin", stream)
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        # When: descriptor readiness is registered before cancellation.
        with anyio.fail_after(3):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(prompt.read_prompt, None)
                await waiting.wait()
                tasks.cancel_scope.cancel()
        # Then: a subsequent owner can still read the descriptor.
        _ = os.write(writer, b"next\n")
        assert await prompt.read_prompt(None) == prompt.PromptInput("next", False)


def test_write_line_without_active_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    # When
    prompt.write_line("event")
    # Then: machine-consumed line framing, not pinned UI prose.
    assert output.getvalue().splitlines() == ["event"]
