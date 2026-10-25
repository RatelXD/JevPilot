"""One event loop and one stdin owner for tasks and interactive run control."""

import sys
from typing import assert_never, override
from uuid import uuid4

import anyio

from jevpilot.browser.browser_use_adapter import BrowserUseAdapter
from jevpilot.browser.port import BrowserConfig
from jevpilot.codex_auth import ensure_jevpilot_home, ensure_private_directory
from jevpilot.engine.brief import StructuredTask
from jevpilot.engine.control import InteractiveControl, Prompt, RunControl, ScriptedControl
from jevpilot.engine.jev import JevSelector, JevUnavailable
from jevpilot.engine.loop import RunOutcome, run_task
from jevpilot.engine.routing import RoutingPolicy
from jevpilot.ledger.models import CallRecord
from jevpilot.ledger.receipt import render_call_line, render_receipt
from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.credentials import get_api_key
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import Profile, suggest_profile
from jevpilot.model_client import ProviderError
from jevpilot.llm.registry import build_provider
from jevpilot.shell.login import login
from jevpilot.shell.model_menu import cycle_effort, model_capabilities, select_model, set_fast
from jevpilot.shell.prompt import read_prompt, write_line
from jevpilot.shell.render import ShellSink

_HELP = (
    "/login 인증 · /model 모델 선택 · /fast [on|off] · /stop 중단 · "
    "/handoff 인계 · /resume 재개 · /help 도움말 · /exit 종료\n"
    "여러 줄 붙여넣기 · Ctrl+J/Alt+Enter 줄바꿈 · Enter 제출 · "
    "Shift+Tab 추론 단계 · Ctrl-C 실행 중단 · 확인 y/n"
)


class _ShellBrowser(BrowserUseAdapter):
    """The engine still owns startup; only its browser configuration changes."""

    headless: bool = False

    @override
    async def start(self, config: BrowserConfig) -> None:
        directory = ensure_private_directory(ensure_jevpilot_home() / "chrome-profile")
        await super().start(config.model_copy(update={
            "user_data_dir": directory, "headless": self.headless,
        }))


async def execute_controlled(
    goal: str | StructuredTask, profile: Profile, control: RunControl, *,
    start_url: str | None = None, headless: bool = False,
) -> RunOutcome:
    key = get_api_key("jev")
    if not key:
        raise JevUnavailable("auth_missing")
    policy = RoutingPolicy.from_profile(profile)
    root = ensure_jevpilot_home() / "runs"
    run_id = uuid4().hex
    sink = ShellSink(run_id, emit=write_line, root=root)
    ordinal = 0

    def called(record: CallRecord) -> None:
        nonlocal ordinal
        if record.kind == "jev":
            ordinal += 1
        write_line(render_call_line(record, ordinal))

    browser = _ShellBrowser()
    browser.headless = headless
    provider = build_provider(profile)
    ledger = RunLedger(run_id, sink=sink, on_call=called)
    outcome = await run_task(
        goal, browser=browser, llm=provider,
        jev=JevSelector(api_key=key), control=control, ledger=ledger,
        policy=policy, start_url=start_url,
    )
    answer = outcome.reason
    quote = None
    if outcome.extraction is not None:
        answer = "; ".join(
            f"{name}: {field.value}" for name, field in outcome.extraction.fields.items()
        ) or outcome.reason
        quote = next((field.quote for field in outcome.extraction.fields.values()), None)
    write_line(render_receipt(
        outcome.receipt, outcome=outcome.status, answer_summary=answer,
        evidence_quote=quote, calls=ledger.calls,
    ))
    return outcome


async def _interactive(
    goal: str | StructuredTask, profile: Profile, *,
    start_url: str | None = None, headless: bool = False,
) -> tuple[RunOutcome, bool]:
    control = InteractiveControl()
    outcomes: list[RunOutcome] = []
    failures: list[LLMError | JevUnavailable | OSError | ValueError] = []
    exit_requested = False
    pending: Prompt | None = None

    async with anyio.create_task_group() as tasks:
        async def engine() -> None:
            try:
                outcomes.append(await execute_controlled(
                    goal, profile, control, start_url=start_url, headless=headless,
                ))
            except (LLMError, JevUnavailable, OSError, ValueError) as error:
                failures.append(error)
            finally:
                tasks.cancel_scope.cancel()

        async def prompts() -> None:
            nonlocal pending
            async for question in control.prompts:
                pending = question
                match question.kind:
                    case "confirm":
                        write_line(f"{question.text} [y/n]")
                    case "ask":
                        write_line(question.text)
                    case "handoff":
                        write_line(f"진행: 인계 대기 · {question.text} · /resume")
                    case unreachable:
                        assert_never(unreachable)

        async def inputs() -> None:
            nonlocal exit_requested, pending
            draft, cursor = "", 0
            while True:
                result = await read_prompt(profile, initial_text=draft, initial_cursor=cursor)
                draft, cursor = "", 0
                if result is None:
                    exit_requested = True
                    control.request_stop()
                    return
                if result.interrupted:
                    control.request_stop()
                    continue
                if result.effort_changed:
                    draft, cursor = result.text, result.cursor
                    continue
                text = result.text
                command = text.strip() if "\n" not in text and "\r" not in text else ""
                match command:
                    case "/exit":
                        exit_requested = True
                        control.request_stop()
                        return
                    case "/stop":
                        control.request_stop()
                    case "/handoff":
                        control.request_handoff()
                    case "/resume":
                        control.resume()
                        pending = None
                    case "/help":
                        write_line(_HELP)
                    case _:
                        if command.startswith("/"):
                            write_line("실행 중에는 /stop · /handoff · /resume을 사용하세요.")
                        elif pending is not None and pending.kind != "handoff":
                            if pending.kind == "confirm" and text.strip().lower() not in {
                                "y", "n", "yes", "no", "예", "네", "아니오", "ㅇ",
                            }:
                                write_line("y/n으로 답하세요.")
                                continue
                            control.answer(text)
                            pending = None
                        elif text.strip():
                            write_line("작업이 실행 중입니다. /stop으로 중단할 수 있습니다.")

        tasks.start_soon(prompts)
        tasks.start_soon(inputs)
        tasks.start_soon(engine)
    if failures:
        raise failures[0]
    return outcomes[0], exit_requested


async def execute_task(
    goal: str | StructuredTask, profile: Profile, *,
    start_url: str | None = None, headless: bool = False, approve_submit: bool = False,
) -> RunOutcome:
    # task '-' consumed stdin to EOF; it can never double as run control.
    if sys.stdin.isatty():
        outcome, _ = await _interactive(
            goal, profile, start_url=start_url, headless=headless,
        )
        return outcome
    return await execute_controlled(
        goal, profile, ScriptedControl(approve_confirm=approve_submit),
        start_url=start_url, headless=headless,
    )


async def _shell() -> int:
    write_line("JevPilot · 자연어 작업 입력 · /login 인증 · /model 모델 · /help · /exit")
    try:
        profile = suggest_profile()
    except (OSError, ValueError):
        profile = None
        write_line("설정을 읽지 못했습니다.")
    if profile is None:
        write_line("먼저 /login으로 API 키 또는 구독 로그인을 선택하세요.")
    draft, cursor = "", 0
    while True:
        result = await read_prompt(profile, initial_text=draft, initial_cursor=cursor)
        draft, cursor = "", 0
        if result is None:
            return 0
        if result.interrupted:
            continue
        try:
            if result.effort_changed:
                draft, cursor = result.text, result.cursor
                if profile is not None:
                    profile = cycle_effort(profile, await model_capabilities(profile))
                continue
            text = result.text
            if not text.strip():
                continue
            command = text.strip() if "\n" not in text and "\r" not in text else ""
            match command.split():
                case ["/exit"]:
                    return 0
                case ["/help"]:
                    write_line(_HELP)
                case ["/login"]:
                    profile = await login(profile, emit=write_line)
                case ["/model"] if profile is not None:
                    profile = await select_model(profile, emit=write_line)
                case ["/fast", *options] if profile is not None:
                    if options not in ([], ["on"], ["off"]):
                        write_line("/fast [on|off]")
                    else:
                        enabled = None if not options else options == ["on"]
                        profile = set_fast(profile, await model_capabilities(profile), enabled)
                case _:
                    if command.startswith("/"):
                        write_line(_HELP)
                    elif profile is None:
                        write_line("먼저 /login으로 API 키 또는 구독 로그인을 선택하세요.")
                    elif not get_api_key("jev"):
                        write_line("Jev 키가 없습니다. /login에서 3번을 선택하세요.")
                    else:
                        _, exit_requested = await _interactive(text, profile)
                        if exit_requested:
                            return 0
        except (LLMError, JevUnavailable) as error:
            write_line(f"인증과 연결을 확인하세요. ({error.kind})")
        except (ProviderError, OSError, ValueError, TimeoutError):
            write_line("인증과 설정, 연결을 확인하세요.")


def run_shell() -> int:
    return anyio.run(_shell)
