"""Run control channel between the shell (or a script) and the engine loop.

The engine checks `stop_requested` at the start of every step and right before executing an
action, asks `confirm` before mutations, and calls `handoff` for login/payment. While a handoff
is active `recording_paused` is True: the engine must neither observe nor log until `resume`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream


class NeedsUser(Exception):
    """The run needs a human but this control cannot wait for one; end as `needs_user`."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RunStopped(Exception):
    """The user asked to stop while the engine was waiting on them."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """Something the shell must show the user and (except `handoff`) answer."""

    kind: Literal["confirm", "ask", "handoff"]
    text: str


@runtime_checkable
class RunControl(Protocol):
    @property
    def stop_requested(self) -> bool: ...

    @property
    def handoff_requested(self) -> bool: ...

    @property
    def recording_paused(self) -> bool: ...

    async def confirm(self, question: str) -> bool: ...

    async def ask(self, question: str) -> str: ...

    async def handoff(self, reason: str) -> None:
        """Block until the user resumes; the caller must re-observe afterwards."""
        ...


class ScriptedControl:
    """Non-interactive control: confirm answers immediately, anything else ends as needs_user."""

    def __init__(self, approve_confirm: bool = False) -> None:
        self.approve_confirm = approve_confirm

    @property
    def stop_requested(self) -> bool:
        return False

    @property
    def handoff_requested(self) -> bool:
        return False

    @property
    def recording_paused(self) -> bool:
        return False

    async def confirm(self, question: str) -> bool:
        return self.approve_confirm

    async def ask(self, question: str) -> str:
        raise NeedsUser(question)

    async def handoff(self, reason: str) -> None:
        raise NeedsUser(reason)


class InteractiveControl:
    """In-memory channel: the shell reads `prompts` and calls the request_/answer/resume methods."""

    def __init__(self) -> None:
        self._stop = anyio.Event()
        self._resume = anyio.Event()
        self._handoff_active = False
        self._prompt_send: MemoryObjectSendStream[Prompt]
        self.prompts: MemoryObjectReceiveStream[Prompt]
        self._prompt_send, self.prompts = anyio.create_memory_object_stream[Prompt](16)
        self._answer_send: MemoryObjectSendStream[str]
        self._answer_recv: MemoryObjectReceiveStream[str]
        self._answer_send, self._answer_recv = anyio.create_memory_object_stream[str](16)

    @property
    def stop_event(self) -> anyio.Event:
        return self._stop

    @property
    def resume_event(self) -> anyio.Event:
        """Set when the current handoff ends (a fresh event per handoff)."""
        return self._resume

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def handoff_requested(self) -> bool:
        return self._handoff_active

    @property
    def recording_paused(self) -> bool:
        return self._handoff_active

    def request_stop(self) -> None:
        self._stop.set()

    def request_handoff(self) -> None:
        if not self._handoff_active:
            self._handoff_active = True
            self._resume = anyio.Event()

    def resume(self) -> None:
        if self._handoff_active:
            self._handoff_active = False
            self._resume.set()

    def answer(self, text: str) -> None:
        self._answer_send.send_nowait(text)

    async def _next_answer(self) -> str:
        result: list[str] = []
        async with anyio.create_task_group() as group:

            async def receive() -> None:
                result.append(await self._answer_recv.receive())
                group.cancel_scope.cancel()

            async def stopped() -> None:
                await self._stop.wait()
                group.cancel_scope.cancel()

            group.start_soon(receive)
            group.start_soon(stopped)
        if not result:
            raise RunStopped
        return result[0]

    async def confirm(self, question: str) -> bool:
        self._prompt_send.send_nowait(Prompt("confirm", question))
        try:
            reply = await self._next_answer()
        except RunStopped:
            return False
        return reply.strip().lower() in {"y", "yes", "예", "네", "ㅇ"}

    async def ask(self, question: str) -> str:
        self._prompt_send.send_nowait(Prompt("ask", question))
        return await self._next_answer()

    async def handoff(self, reason: str) -> None:
        self.request_handoff()
        self._prompt_send.send_nowait(Prompt("handoff", reason))
        resumed = self._resume
        async with anyio.create_task_group() as group:

            async def wait_resume() -> None:
                await resumed.wait()
                group.cancel_scope.cancel()

            async def stopped() -> None:
                await self._stop.wait()
                group.cancel_scope.cancel()

            group.start_soon(wait_resume)
            group.start_soon(stopped)
        if self._stop.is_set() and not resumed.is_set():
            self._handoff_active = False
            raise RunStopped
