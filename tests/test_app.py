import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jevpilot import app, cli
from jevpilot.engine.brief import AnswerFields, StructuredTask, UrlContains, VisibleText
from jevpilot.engine.control import ScriptedControl
from jevpilot.engine.loop import RunOutcome
from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.profile import Profile, save_profile
from jevpilot.llm.types import ModelRef, ProviderId


@pytest.mark.parametrize("arguments", [[], ["shell"]])
def test_entrypoint_routes_new_shell(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str],
) -> None:
    # Given
    invoked: list[bool] = []
    monkeypatch.setattr(sys, "argv", ["jevpilot", *arguments])

    def shell() -> int:
        invoked.append(True)
        return 0

    monkeypatch.setattr(app, "run_shell", shell)
    # When / Then
    with pytest.raises(SystemExit) as result:
        app.entrypoint()
    assert result.value.code == 0
    assert invoked == [True]


@pytest.mark.parametrize("arguments", [
    ["run", "--help"], ["--help"], ["unknown", "untouched"],
])
def test_entrypoint_preserves_all_legacy_arguments(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str],
) -> None:
    # Given
    seen: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", ["jevpilot", *arguments])
    monkeypatch.setattr(cli, "entrypoint", lambda: seen.append(sys.argv[1:]))
    # When
    app.entrypoint()
    # Then
    assert seen == [arguments]


@pytest.mark.parametrize("installed", [False, True])
def test_new_shell_exits_through_real_entrypoints(
    tmp_path: Path, installed: bool,
) -> None:
    # Given
    env = {**os.environ, "HOME": str(tmp_path)}
    command = [str(Path(sys.executable).parent / "jevpilot")] if installed else [
        sys.executable, "-m", "jevpilot",
    ]
    # When
    result = subprocess.run(
        [*command, "shell"], input="/exit\n", capture_output=True,
        text=True, env=env, timeout=20, check=False,
    )
    # Then: prompt is shipped CLI compatibility, not arbitrary prose.
    assert result.returncode == 0, result.stderr
    assert "jevpilot>" in result.stdout


def test_installed_legacy_run_help_remains_available(tmp_path: Path) -> None:
    # Given
    command = Path(sys.executable).parent / "jevpilot"
    # When
    result = subprocess.run(
        [str(command), "run", "--help"], capture_output=True, text=True,
        env={**os.environ, "HOME": str(tmp_path)}, timeout=20, check=False,
    )
    # Then: machine-consumed legacy flags remain present.
    assert result.returncode == 0
    assert "--mode" in result.stdout + result.stderr
    assert "--verify-url-path" in result.stdout + result.stderr


@pytest.mark.parametrize("stdin_goal", [False, True])
def test_task_forwards_complete_natural_goal_once(
    monkeypatch: pytest.MonkeyPatch, stdin_goal: bool,
) -> None:
    # Given
    text = "/stop\nsecond line\nthird line\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    monkeypatch.setattr(app, "suggest_profile", lambda: Profile(
        "api", ModelRef("openai", "api", "test"),
    ))
    seen: list[str | StructuredTask] = []
    controls: list[ScriptedControl] = []

    async def execute(
        goal: str | StructuredTask, profile: Profile,
        control: ScriptedControl | None = None, *,
        start_url: str | None = None, headless: bool = False,
        approve_submit: bool = False,
    ) -> RunOutcome:
        assert profile.model.model_id == "test"
        assert start_url is None and not headless and not approve_submit
        seen.append(goal)
        if control is not None:
            controls.append(control)
        return RunOutcome("verified", "verified", None, 0, None, RunLedger().finish())

    monkeypatch.setattr(app, "execute_task", execute)
    monkeypatch.setattr(app, "execute_controlled", execute)
    monkeypatch.setattr(sys, "argv", ["jevpilot", "task", "-" if stdin_goal else text])
    # When / Then
    with pytest.raises(SystemExit) as result:
        app.entrypoint()
    assert result.value.code == 0
    assert seen == [text]
    assert len(controls) == int(stdin_goal)


def test_task_builds_explicit_structured_conditions_and_first_equals_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(app, "suggest_profile", lambda: Profile(
        "api", ModelRef("openai", "api", "test"),
    ))
    seen: list[tuple[str | StructuredTask, str | None, bool, bool]] = []

    async def execute(
        goal: str | StructuredTask, profile: Profile, *,
        start_url: str | None = None, headless: bool = False, approve_submit: bool = False,
    ) -> RunOutcome:
        assert profile.model.model_id == "test"
        seen.append((goal, start_url, headless, approve_submit))
        return RunOutcome("verified", "verified", None, 0, None, RunLedger().finish())

    monkeypatch.setattr(app, "execute_task", execute)
    monkeypatch.setattr(sys, "argv", [
        "jevpilot", "task", "Finish", "--start-url", "https://example.test",
        "--verify-url-contains", "/done", "--verify-text", "Complete",
        "--value", "query=a=b", "--output", "answer", "--approve-submit", "--headless",
    ])
    # When / Then
    with pytest.raises(SystemExit) as result:
        app.entrypoint()
    assert result.value.code == 0
    task, url, headless, approve = seen[0]
    assert isinstance(task, StructuredTask)
    assert task.input_values == {"query": "a=b"}
    assert task.success_conditions == [
        UrlContains(text="/done"), VisibleText(text="Complete"), AnswerFields(names=["answer"]),
    ]
    assert [field.name for field in task.required_outputs] == ["answer"]
    assert (url, headless, approve) == ("https://example.test", True, True)


@pytest.mark.parametrize("value", ["bad", "=x", "name=", " =value"])
def test_task_rejects_malformed_values_before_auth(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    # Given
    monkeypatch.setattr(sys, "argv", ["jevpilot", "task", "Goal", "--value", value])

    def forbidden() -> None:
        raise AssertionError("must reject before auth")

    monkeypatch.setattr(app, "suggest_profile", forbidden)
    # When / Then
    with pytest.raises(SystemExit) as result:
        app.entrypoint()
    assert result.value.code == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("stdin_goal", [False, True])
@pytest.mark.parametrize("missing_jev", [False, True])
def test_task_identifies_the_missing_credential_at_its_actual_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    provider: ProviderId, stdin_goal: bool, missing_jev: bool,
) -> None:
    # Given: real saved profile and provider factory, but no API credential.
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("JEVPILOT_OPENAI_API_KEY", "JEVPILOT_ANTHROPIC_API_KEY", "JEVPILOT_JEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if not missing_jev:
        monkeypatch.setenv("JEVPILOT_JEV_API_KEY", "synthetic-present-jev")
    save_profile(Profile("api", ModelRef(provider, "api", "test-model")))
    monkeypatch.setattr(sys, "stdin", io.StringIO("Goal\n"))
    monkeypatch.setattr(sys, "argv", ["jevpilot", "task", "-" if stdin_goal else "Goal"])
    # When
    with pytest.raises(SystemExit) as result:
        app.entrypoint()
    # Then: diagnostic subjects, not natural-language wording, distinguish sources.
    output = capsys.readouterr().out
    assert result.value.code == 2
    assert ("Jev" in output) == missing_jev
    assert ("LLM" in output) == (not missing_jev)
    assert "/login" in output
    assert "synthetic-present-jev" not in output
