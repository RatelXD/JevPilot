import anyio
import pytest

from jevpilot.engine.control import (
    InteractiveControl,
    NeedsUser,
    Prompt,
    RunControl,
    RunStopped,
    ScriptedControl,
)

TIMEOUT = 5.0


def test_both_controls_satisfy_protocol() -> None:
    assert isinstance(ScriptedControl(), RunControl)
    assert isinstance(InteractiveControl(), RunControl)


async def test_scripted_confirm_returns_setting_immediately() -> None:
    assert await ScriptedControl().confirm("제출할까요?") is False
    assert await ScriptedControl(approve_confirm=True).confirm("제출할까요?") is True


async def test_scripted_ask_and_handoff_raise_needs_user() -> None:
    control = ScriptedControl()
    with pytest.raises(NeedsUser) as asked:
        await control.ask("날짜는?")
    assert asked.value.reason == "날짜는?"
    with pytest.raises(NeedsUser):
        await control.handoff("login")
    assert not control.stop_requested
    assert not control.recording_paused


@pytest.mark.parametrize(("reply", "expected"), [("y", True), ("Y", True), ("yes", True), ("n", False), ("", False)])
async def test_interactive_confirm_follows_shell_reply(reply: str, expected: bool) -> None:
    control = InteractiveControl()
    result: list[bool] = []

    async def engine() -> None:
        result.append(await control.confirm("제출할까요?"))

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            prompt = await control.prompts.receive()
            assert prompt == Prompt("confirm", "제출할까요?")
            control.answer(reply)
    assert result == [expected]


async def test_interactive_ask_returns_answer_text() -> None:
    control = InteractiveControl()
    result: list[str] = []

    async def engine() -> None:
        result.append(await control.ask("날짜는?"))

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            assert (await control.prompts.receive()).kind == "ask"
            control.answer("내일")
    assert result == ["내일"]


async def test_stop_while_waiting_denies_confirm_and_aborts_ask() -> None:
    control = InteractiveControl()
    confirmed: list[bool] = []
    stopped: list[bool] = []

    async def confirm() -> None:
        confirmed.append(await control.confirm("q"))

    async def ask() -> None:
        try:
            await control.ask("q2")
        except RunStopped:
            stopped.append(True)

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(confirm)
            group.start_soon(ask)
            await control.prompts.receive()
            await control.prompts.receive()
            assert not control.stop_requested
            control.request_stop()
    assert control.stop_requested
    assert confirmed == [False]
    assert stopped == [True]


async def test_handoff_pauses_recording_until_resume() -> None:
    control = InteractiveControl()
    order: list[str] = []

    async def engine() -> None:
        order.append("handoff_start")
        await control.handoff("login")
        order.append("resumed")

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            assert await control.prompts.receive() == Prompt("handoff", "login")
            assert control.handoff_requested
            assert control.recording_paused
            assert order == ["handoff_start"]
            control.resume()
    assert order == ["handoff_start", "resumed"]
    assert not control.handoff_requested
    assert not control.recording_paused


async def test_user_requested_handoff_blocks_next_handoff_wait_and_second_cycle_works() -> None:
    control = InteractiveControl()
    control.request_handoff()
    assert control.handoff_requested
    first = control.resume_event
    control.resume()
    assert first.is_set()
    assert not control.handoff_requested

    done: list[str] = []

    async def engine() -> None:
        await control.handoff("again")
        done.append("ok")

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            await control.prompts.receive()
            assert control.resume_event is not first
            assert not control.resume_event.is_set()
            control.resume()
    assert done == ["ok"]


async def test_stop_during_handoff_raises_run_stopped() -> None:
    control = InteractiveControl()
    raised: list[bool] = []

    async def engine() -> None:
        try:
            await control.handoff("login")
        except RunStopped:
            raised.append(True)

    with anyio.fail_after(TIMEOUT):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            await control.prompts.receive()
            control.request_stop()
    assert raised == [True]
    assert not control.recording_paused
