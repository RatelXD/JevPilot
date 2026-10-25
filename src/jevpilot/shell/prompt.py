"""Cancellable, single-owner terminal input for the new shell."""

import codecs
import io
import os
import stat
import sys
import termios
import threading
import tty
import unicodedata
from dataclasses import dataclass
from typing import Final, TextIO

import anyio
from anyio.lowlevel import checkpoint

from jevpilot.llm.profile import Profile

_PASTE_ON: Final = "\x1b[?2004h"
_PASTE_OFF: Final = "\x1b[?2004l"
_PASTE_END: Final = "\x1b[201~"
_OUTPUT_LOCK: Final = threading.RLock()
_active: "_Draft | None" = None


@dataclass(frozen=True, slots=True)
class PromptInput:
    text: str
    effort_changed: bool
    cursor: int = 0
    interrupted: bool = False


def prompt_label(profile: Profile | None) -> str:
    if profile is None:
        return "jevpilot> "
    model = profile.model
    effort = f":{model.reasoning_effort}" if model.reasoning_effort else ""
    fast = " ⚡" if model.fast else ""
    mode = "구독" if profile.auth_mode == "subscription" else "API"
    return f"{model.provider}/{model.model_id}{effort}{fast} ({mode})> "


def _display(text: str) -> str:
    """Never replay pasted control bytes as terminal instructions."""
    return "".join(
        char if char == "\n" or char.isprintable() else "    " if char == "\t" else "�"
        for char in text
    )


def _position(text: str, width: int) -> tuple[int, int]:
    row, column = 0, 0
    for char in text:
        if char == "\n":
            row, column = row + 1, 0
        else:
            cells = 0 if unicodedata.combining(char) else (
                2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
            )
            if column + cells > width:
                row, column = row + 1, 0
            column += cells
    return row, column


@dataclass
class _Draft:
    """Mutable editor buffer and cursor; only the stdin owner edits it."""

    output: TextIO
    label: str
    text: str
    cursor: int
    row: int = 0

    def clear(self) -> None:
        _ = self.output.write("\r" + (f"\x1b[{self.row}A" if self.row else "") + "\x1b[J")
        self.row = 0

    def draw(self) -> None:
        self.clear()
        body = self.label + _display(self.text).replace("\n", "\n... ")
        before = self.label + _display(self.text[:self.cursor]).replace("\n", "\n... ")
        try:
            width = os.get_terminal_size(self.output.fileno()).columns
        except (OSError, io.UnsupportedOperation):
            width = 80
        end_row, _ = _position(body, max(1, width))
        self.row, column = _position(before, max(1, width))
        _ = self.output.write(body.replace("\n", "\r\n"))
        if end_row > self.row:
            _ = self.output.write(f"\x1b[{end_row - self.row}A")
        _ = self.output.write("\r" + (f"\x1b[{column}C" if column else ""))
        self.output.flush()

    def insert(self, text: str) -> None:
        self.text = self.text[:self.cursor] + text + self.text[self.cursor:]
        self.cursor += len(text)


def write_line(text: str) -> None:
    """Write an atomic output line, then redraw the active draft unchanged."""
    with _OUTPUT_LOCK:
        if _active is None:
            _ = sys.stdout.write(text + "\n")
            _ = sys.stdout.flush()
        else:
            _active.clear()
            _ = _active.output.write(text.replace("\n", "\r\n") + "\r\n")
            _active.draw()


async def _character(
    descriptor: int, decoder: codecs.IncrementalDecoder, *, raw: bool = False,
) -> str:
    """Read one decoded character without blocking on an incomplete UTF-8 sequence."""
    while True:
        await anyio.wait_readable(descriptor)
        chunk = os.read(descriptor, 1)
        if not chunk:
            return decoder.decode(b"", final=True)
        if raw and (chunk[0] < 32 or chunk[0] == 127):
            # A broken UTF-8 prefix must not absorb Enter, interrupt or EOF.
            decoder.reset()
            return chunk.decode("ascii")
        decoded = decoder.decode(chunk)
        if decoded:
            return decoded


async def _line() -> PromptInput | None:
    """Read pipes with readiness; disk and scripted text inputs need no selector."""
    try:
        descriptor = sys.stdin.fileno()
    except (AttributeError, io.UnsupportedOperation):
        descriptor = None
    if descriptor is None or stat.S_ISREG(os.fstat(descriptor).st_mode):
        await checkpoint()
        line = sys.stdin.readline()
        return PromptInput(line.rstrip("\r\n"), False) if line else None
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
    chars: list[str] = []
    while True:
        char = await _character(descriptor, decoder)
        if not char or char == "\n":
            text = "".join(chars)
            if char and text.endswith("\r"):
                text = text[:-1]
            return PromptInput(text, False) if chars or char else None
        chars.append(char)


async def read_prompt(
    profile: Profile | None, *, initial_text: str = "", initial_cursor: int = 0,
) -> PromptInput | None:
    """Read a draft, yielding it unsubmitted on Shift+Tab for capability cycling."""
    global _active
    if not sys.stdin.isatty():
        _ = sys.stdout.write(prompt_label(profile))
        _ = sys.stdout.flush()
        return await _line()

    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    draft = _Draft(sys.stdout, prompt_label(profile), initial_text,
                   max(0, min(initial_cursor, len(initial_text))))
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    escape = ""
    paste = False
    paste_end = ""
    try:
        _ = tty.setraw(descriptor, termios.TCSANOW)
        with _OUTPUT_LOCK:
            _active = draft
            _ = draft.output.write(_PASTE_ON)
            draft.draw()
        while True:
            char = await _character(descriptor, decoder, raw=True)
            # Interrupt/EOF remain usable even inside malformed escapes or paste.
            if not char or char == "\x04":
                return None
            if char == "\x03":
                return PromptInput("", False, interrupted=True)
            with _OUTPUT_LOCK:
                if paste:
                    paste_end += char
                    while paste_end and not _PASTE_END.startswith(paste_end):
                        draft.insert(paste_end[0])
                        paste_end = paste_end[1:]
                    if paste_end == _PASTE_END:
                        paste, paste_end = False, ""
                    draft.draw()
                    continue
                if char in {"\r", "\n"}:
                    if char == "\r" and escape != "\x1b":
                        return PromptInput(draft.text, False, draft.cursor)
                    draft.insert("\n")
                    escape = ""
                elif escape:
                    escape += char
                    match escape:
                        case "\x1b[Z":
                            return PromptInput(draft.text, True, draft.cursor)
                        case "\x1b[D":
                            draft.cursor = max(0, draft.cursor - 1)
                            escape = ""
                        case "\x1b[C":
                            draft.cursor = min(len(draft.text), draft.cursor + 1)
                            escape = ""
                        case "\x1b[3~":
                            draft.text = draft.text[:draft.cursor] + draft.text[draft.cursor + 1:]
                            escape = ""
                        case "\x1b[200~":
                            paste, escape = True, ""
                        case other:
                            # Unknown CSI is consumed through its final byte, with
                            # a hard cap. It never emits terminal control bytes.
                            if not other.startswith("\x1b["):
                                escape = ""
                                if char.isprintable():
                                    draft.insert(char)
                            elif len(other) > 2 and (
                                "@" <= char <= "~" or len(other) >= 64
                            ):
                                escape = ""
                elif char == "\x1b":
                    escape = char
                elif char in {"\x7f", "\b"}:
                    if draft.cursor:
                        draft.text = draft.text[:draft.cursor - 1] + draft.text[draft.cursor:]
                        draft.cursor -= 1
                elif char.isprintable() or char == "\t":
                    draft.insert(char)
                draft.draw()
    finally:
        with _OUTPUT_LOCK:
            _active = None
            try:
                _ = draft.output.write(_PASTE_OFF + "\r\n")
                draft.output.flush()
            finally:
                termios.tcsetattr(descriptor, termios.TCSANOW, previous)
