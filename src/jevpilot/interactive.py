"""Interactive Codex login, model selection, and prompt composer."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import anyio

from jevpilot.codex_auth import (
    codex_login_status,
    is_chatgpt_login_status,
    launch_chatgpt_login,
)
from jevpilot.model_catalog import ChatGPTModel, list_chatgpt_models
from jevpilot.model_client import ProviderError
from jevpilot.model_selection import (
    ModelSelection,
    ModelSelectionError,
    REASONING_EFFORTS,
    ReasoningEffort,
    load_model_selection,
    next_reasoning_effort,
    save_model_selection,
    select_available_model,
)


def run_shell() -> int:
    """Run the interactive subscription setup and prompt composer."""
    print("JevPilot · /login 인증 · /model 선택 · /exit 종료")
    try:
        selection = load_model_selection()
    except ModelSelectionError as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        return 2
    while True:
        line = _read_prompt(selection)
        if line is None:
            print()
            return 0
        while line.effort_changed:
            if selection is None:
                break
            selection = _change_effort(selection)
            line = _resume_prompt(selection, line)
            if line is None:
                print()
                return 0
        command = line.text.strip()
        if not command:
            continue

        match command:
            case "/login":
                _login()
            case "/model":
                selection = _select_model(selection)
            case "/fast":
                selection = _toggle_fast(selection)
            case "/fast on":
                selection = _toggle_fast(selection, True)
            case "/fast off":
                selection = _toggle_fast(selection, False)
            case _ if command.startswith("/fast "):
                value = command.removeprefix("/fast ").strip()
                if value in {"on", "off"}:
                    selection = _toggle_fast(selection, value == "on")
                else:
                    print("/fast [on|off] 형식으로 입력하세요.")
            case "/help":
                print("/login  공식 Codex ChatGPT OAuth 로그인")
                print("/model  계정의 모델 선택")
                print("/fast [on|off]  현재 모델 Fast 모드 전환")
                print("Shift+Tab  프롬프트에서 추론 강도 순환")
                print("입력 중 Shift+Tab은 프롬프트를 전송하지 않고 추론 수준만 변경합니다.")
                print("/exit   종료")
            case "/exit":
                return 0
            case _:
                print("명령을 확인하세요. /help를 입력하면 명령을 볼 수 있습니다.")


@dataclass(frozen=True, slots=True)
class _PromptInput:
    text: str
    effort_changed: bool
    cursor: int = 0


def _resume_prompt(
    selection: ModelSelection, previous: _PromptInput
) -> _PromptInput | None:
    return _read_prompt(
        selection,
        initial_text=previous.text,
        initial_cursor=previous.cursor,
    )


def _read_prompt(
    selection: ModelSelection | None,
    *,
    initial_text: str = "",
    initial_cursor: int = 0,
) -> _PromptInput | None:
    """Read one edited prompt, consuming Shift+Tab without losing its text."""
    if selection is None or not sys.stdin.isatty():
        try:
            return _PromptInput(input(_prompt_label(selection)), False)
        except EOFError:
            return None

    import termios
    import tty

    prompt = _prompt_label(selection)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    buffer = list(initial_text)
    cursor = min(initial_cursor, len(buffer))
    effort_changed = False
    try:
        tty.setraw(descriptor)
        if buffer:
            sys.stdout.write("".join(buffer))
            for _ in range(len(buffer) - cursor):
                sys.stdout.write("\x1b[D")
            sys.stdout.flush()
        while True:
            key = sys.stdin.read(1)
            if key in {"\x04", "\x03"}:
                if key == "\x04":
                    return None
                return _PromptInput("", False)
            if key in {"\r", "\n"}:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return _PromptInput("".join(buffer), effort_changed, cursor)
            if key in {"\x7f", "\b"}:
                if cursor:
                    buffer.pop(cursor - 1)
                    cursor -= 1
                    sys.stdout.write("\r" + prompt + "".join(buffer) + " \x1b[K\r" + prompt + "".join(buffer))
                    for _ in range(len(buffer) - cursor):
                        sys.stdout.write("\x1b[D")
                    sys.stdout.flush()
                continue
            if key == "\x1b":
                following = sys.stdin.read(2)
                if following == "[Z":
                    effort_changed = True
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return _PromptInput("".join(buffer), effort_changed, cursor)
                if following == "[D" and cursor:
                    cursor -= 1
                    sys.stdout.write("\x1b[D")
                    sys.stdout.flush()
                    continue
                if following == "[C" and cursor < len(buffer):
                    cursor += 1
                    sys.stdout.write("\x1b[C")
                    sys.stdout.flush()
                    continue
                if following == "[3":
                    _ = sys.stdin.read(1)
                    if cursor < len(buffer):
                        buffer.pop(cursor)
                        sys.stdout.write("".join(buffer[cursor:]) + " \x1b[K")
                        for _ in range(len(buffer) - cursor):
                            sys.stdout.write("\x1b[D")
                        sys.stdout.flush()
                    continue
                buffer.extend(("\x1b", *following))
                sys.stdout.write("\x1b" + following)
                sys.stdout.flush()
                continue
            buffer.insert(cursor, key)
            cursor += 1
            sys.stdout.write(key)
            if cursor < len(buffer):
                sys.stdout.write("".join(buffer[cursor:]))
                for _ in range(len(buffer) - cursor):
                    sys.stdout.write("\x1b[D")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _prompt_label(selection: ModelSelection | None) -> str:
    if selection is None:
        return "jevpilot> "
    fast_icon = "⚡ " if selection.fast else ""
    fast_suffix = "-fast" if selection.fast else ""
    return f"{fast_icon}{selection.model_id}{fast_suffix}:{selection.reasoning_effort}> "


def _change_effort(selection: ModelSelection) -> ModelSelection:
    effort = next_reasoning_effort(
        selection.reasoning_effort,
        selection.supported_reasoning_efforts,
    )
    updated = _copy_selection(selection, reasoning_effort=effort)
    try:
        save_model_selection(updated)
    except OSError:
        print("\r\n추론 설정을 저장하지 못했습니다.")
        return selection
    print(f"\r\n추론 강도: {effort}")
    return updated


def _toggle_fast(
    selection: ModelSelection | None,
    enabled: bool | None = None,
) -> ModelSelection | None:
    if selection is None:
        print("먼저 /model에서 구독 모델을 선택하세요.")
        return None
    target = not selection.fast if enabled is None else enabled
    if target and not selection.supports_fast:
        print(f"{selection.model_id}은(는) Fast 모드를 지원하지 않습니다.")
        return selection
    updated = _copy_selection(selection, fast=target)
    try:
        save_model_selection(updated)
    except OSError:
        print("Fast 설정을 저장하지 못했습니다.")
        return selection
    print(f"Fast {'켜짐' if target else '꺼짐'}: {_prompt_label(updated).strip()}")
    return updated


def _copy_selection(
    selection: ModelSelection,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    fast: bool | None = None,
) -> ModelSelection:
    return ModelSelection(
        provider=selection.provider,
        model_id=selection.model_id,
        reasoning_effort=reasoning_effort or selection.reasoning_effort,
        fast=selection.fast if fast is None else fast,
        supports_fast=selection.supports_fast,
        supported_reasoning_efforts=selection.supported_reasoning_efforts,
    )


def _login() -> None:
    """Start or confirm the official Codex-managed ChatGPT login."""
    try:
        status = anyio.run(codex_login_status)
    except (OSError, TimeoutError):
        status = None
    if status is not None and is_chatgpt_login_status(status):
        print("ChatGPT 구독 계정에 로그인되어 있습니다.")
        return

    print("공식 Codex OAuth 주소가 아래에 표시됩니다.")
    print("Windows 브라우저에서 로그인을 마친 뒤 이 터미널로 돌아오세요.")
    try:
        return_code = launch_chatgpt_login()
    except OSError:
        print("Codex CLI를 실행할 수 없습니다.")
        return
    if return_code != 0:
        print("Codex 로그인이 완료되지 않았습니다. 다시 시도하세요.")
        return
    try:
        status = anyio.run(codex_login_status)
    except (OSError, TimeoutError):
        print("Codex 로그인 상태를 확인할 수 없습니다.")
        return
    if is_chatgpt_login_status(status):
        print("ChatGPT 구독 계정에 로그인했습니다.")
    else:
        print("ChatGPT 구독 로그인을 확인하지 못했습니다.")


def _select_model(
    current: ModelSelection | None = None,
) -> ModelSelection | None:
    """List available account models and save one selected model."""
    try:
        status = anyio.run(codex_login_status)
    except (OSError, TimeoutError):
        print("Codex 로그인 상태를 확인할 수 없습니다.")
        return current
    if not is_chatgpt_login_status(status):
        print("먼저 /login으로 ChatGPT 구독 계정에 로그인하세요.")
        return current
    print("Codex 계정 모델 목록을 불러오는 중...", flush=True)
    try:
        models = anyio.run(list_chatgpt_models)
    except ProviderError:
        print("모델 목록을 가져오지 못했습니다.")
        print("Codex CLI와 /login 상태를 확인한 뒤 /model을 다시 실행하세요.")
        return current
    if not models:
        print("이 계정에서 표시 가능한 모델이 없습니다.")
        print("/login으로 계정을 확인한 뒤 /model을 다시 실행하세요.")
        return current

    _print_models(models, current)
    try:
        choice = input("모델 번호 또는 provider/model (Enter 취소): ").strip()
    except EOFError:
        print()
        return current
    if not choice:
        print("모델 선택을 취소했습니다.")
        return current

    try:
        selection = _resolve_model_choice(choice, models, current)
        save_model_selection(selection)
    except ModelSelectionError as exc:
        print(f"선택하지 않았습니다: {exc}")
        return current
    except OSError:
        print("선택하지 않았습니다: 설정을 저장하지 못했습니다")
        return current
    print(f"선택된 모델: {selection.model_ref}")
    return selection


def _print_models(
    models: tuple[ChatGPTModel, ...],
    current: ModelSelection | None = None,
) -> None:
    """Print visible models with their supported effort and Fast options."""
    print("현재 Codex 계정에서 표시 가능한 모델:")
    for index, model in enumerate(models, start=1):
        default = " · 기본" if model.is_default else ""
        selected = " ✓" if current is not None and model.model_id == current.model_id else ""
        fast = " · Fast" if model.supports_fast else ""
        efforts = f" · {', '.join(model.reasoning_efforts)} 추론"
        print(
            f"{index}. {model.model_ref}{selected} — {model.display_name}"
            f"{default}{fast}{efforts}"
        )


def _resolve_model_choice(
    value: str,
    models: tuple[ChatGPTModel, ...],
    current: ModelSelection | None = None,
) -> ModelSelection:
    """Resolve a visible numeric row or exact provider/model reference."""
    if value.isdecimal():
        index = int(value)
        if not 1 <= index <= len(models):
            raise ModelSelectionError("목록에 있는 번호를 선택하세요")
        value = models[index - 1].model_ref
    selected = select_available_model(
        value,
        frozenset(model.model_id for model in models),
    )
    model = next(model for model in models if model.model_id == selected.model_id)
    efforts = (
        model.reasoning_efforts
        or (
            current.supported_reasoning_efforts
            if current is not None and current.model_id == model.model_id
            else ("low", "medium", "high")
        )
    )
    effort = (
        current.reasoning_effort
        if current is not None and current.reasoning_effort in efforts
        else "medium"
        if "medium" in efforts
        else efforts[0]
    )
    supported_efforts: tuple[ReasoningEffort, ...] = tuple(
        candidate for candidate in REASONING_EFFORTS if candidate in efforts
    ) or ("medium",)
    return ModelSelection(
        provider=selected.provider,
        model_id=selected.model_id,
        reasoning_effort=effort,
        fast=(
            current.fast
            if current is not None
            and current.model_id == model.model_id
            and model.supports_fast
            else False
        ),
        supports_fast=model.supports_fast,
        supported_reasoning_efforts=supported_efforts,
    )
