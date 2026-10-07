import subprocess
from pathlib import Path

import pytest

from jevpilot import interactive
from jevpilot.model_catalog import ChatGPTModel
from jevpilot.model_selection import ModelSelection, load_model_selection


def test_shell_model_command_persists_only_a_catalog_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    inputs = iter(("/model", "1", "/exit"))

    def scripted_input(_prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr("builtins.input", scripted_input)

    async def logged_in_status() -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=("codex", "login", "status"),
            returncode=0,
            stdout=b"",
            stderr=b"Logged in using ChatGPT\n",
        )

    async def account_models() -> tuple[ChatGPTModel, ...]:
        return (
            ChatGPTModel(
                model_id="gpt-6.1-sol",
                display_name="GPT-6.1 Sol",
                is_default=True,
            ),
        )

    monkeypatch.setattr(interactive, "codex_login_status", logged_in_status)
    monkeypatch.setattr(interactive, "list_chatgpt_models", account_models)

    # When
    result = interactive.run_shell()
    output = capsys.readouterr().out

    # Then
    selection = load_model_selection()
    assert result == 0
    assert selection is not None
    assert selection.provider == "chatgpt-subscription"
    assert selection.model_id == "gpt-6.1-sol"
    assert "chatgpt-subscription/gpt-6.1-sol" in output


def test_shell_cycles_repeated_effort_without_losing_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    selection = ModelSelection(
        provider="chatgpt-subscription",
        model_id="gpt-6.1-sol",
        reasoning_effort="low",
        supported_reasoning_efforts=("low", "medium", "high"),
    )
    monkeypatch.setattr(interactive, "load_model_selection", lambda: selection)
    saved: list[ModelSelection] = []
    monkeypatch.setattr(interactive, "save_model_selection", saved.append)
    prompts = iter(
        (
            interactive._PromptInput("draft", True, 2),
            interactive._PromptInput("draft", True, 2),
            interactive._PromptInput("draft", False, 2),
            interactive._PromptInput("/exit", False),
        )
    )
    reads: list[tuple[str, int]] = []

    def read_prompt(
        _selection: ModelSelection | None,
        *,
        initial_text: str = "",
        initial_cursor: int = 0,
    ) -> interactive._PromptInput:
        reads.append((initial_text, initial_cursor))
        return next(prompts)

    monkeypatch.setattr(interactive, "_read_prompt", read_prompt)

    # When
    result = interactive.run_shell()

    # Then
    assert result == 0
    assert [model.reasoning_effort for model in saved] == ["medium", "high"]
    assert reads == [("", 0), ("draft", 2), ("draft", 2), ("", 0)]


def test_shell_login_starts_codex_without_a_browser_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    inputs = iter(("/login", "/exit"))

    def scripted_input(_prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr("builtins.input", scripted_input)
    statuses = iter(
        (
            subprocess.CompletedProcess(("codex",), 1, b"", b"Not logged in\n"),
            subprocess.CompletedProcess(
                ("codex",),
                0,
                b"",
                b"Logged in using ChatGPT\n",
            ),
        )
    )
    launches: list[bool] = []

    async def login_status() -> subprocess.CompletedProcess[bytes]:
        return next(statuses)

    def launch_login() -> int:
        launches.append(True)
        return 0

    monkeypatch.setattr(interactive, "codex_login_status", login_status)
    monkeypatch.setattr(interactive, "launch_chatgpt_login", launch_login)

    # When
    result = interactive.run_shell()

    # Then
    assert result == 0
    assert launches == [True]
