import sys
import time
from pathlib import Path

import pytest
from pydantic import JsonValue

from jevpilot.codex_client import CodexClient, CodexModelClientAdapter
from jevpilot.model_client import ProviderError
from jevpilot.privacy import PrivacyPolicy
from jevpilot.telemetry.models import CallContext, CallEvent


def context(events: list[CallEvent]) -> CallContext:
    return CallContext(
        run_id="run", logical_call_id="call", role="selector", purpose="normal",
        deadline=time.monotonic() + 10, max_attempts=1,
        reserve_attempt=lambda: None, record_call=events.append,
    )


def schema() -> JsonValue:
    return {
        "type": "object", "properties": {"choice": {"type": "string", "enum": ["c1"]}},
        "required": ["choice"], "additionalProperties": False,
    }


def fake_cli(
    tmp_path: Path,
    final: str,
    *,
    tool_event: bool = False,
    non_fatal_error_event: bool = False,
    expected_home: Path,
    expected_reasoning_effort: str | None = None,
    expected_fast: bool = False,
) -> tuple[str, ...]:
    """Launch a real local process with the documented Codex JSONL wire shape."""
    script = tmp_path / "fake_cli.py"
    lines = [
        "import json, os, sys",
        "prompt = sys.stdin.read()",
        "assert 'JEVPILOT_JEV_API_KEY' not in os.environ",
        "assert 'JEVPILOT_LLM_API_KEY' not in os.environ",
        "assert 'synthetic-secret' not in prompt",
        f"assert os.environ['CODEX_HOME'] == {str(expected_home)!r}",
        "assert '--ignore-user-config' in sys.argv",
        "assert '--output-schema' in sys.argv",
        "assert 'read-only' in sys.argv",
        "print(json.dumps({'type':'thread.started','thread_id':'test'}))",
    ]
    if expected_reasoning_effort is not None:
        configured_effort = f'model_reasoning_effort="{expected_reasoning_effort}"'
        lines.append(f"assert {configured_effort!r} in sys.argv")
    if expected_fast:
        lines.extend(
            (
                "assert '--enable' in sys.argv",
                "assert 'fast_mode' in sys.argv",
                "assert 'service_tier=\"fast\"' in sys.argv",
            )
        )
    if non_fatal_error_event:
        event = {
            "type": "item.completed",
            "item": {
                "type": "error",
                "message": "in-process app-server event stream lagged; dropped 12 events",
            },
        }
        lines.append(f"print(json.dumps({event!r}))")
    if tool_event:
        event = {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "unrequested"},
        }
        lines.append(f"print(json.dumps({event!r}))")
    else:
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": final},
        }
        lines.append(f"print(json.dumps({event!r}))")
    usage_event = {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "cached_input_tokens": 2,
        },
    }
    lines.append(f"print(json.dumps({usage_event!r}))")
    _ = script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sys.executable, str(script)


@pytest.mark.asyncio
async def test_subscription_process_returns_validated_output_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_JEV_API_KEY", "not-for-codex")
    monkeypatch.setenv("JEVPILOT_LLM_API_KEY", "gateway-key-not-for-codex")
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(secret_values=("synthetic-secret",)),
        command=fake_cli(
            tmp_path,
            '{"choice":"c1"}',
            expected_home=tmp_path / "codex",
        ),
        codex_home=tmp_path / "codex",
    )

    # When
    result = await client.generate_json(
        instructions="Choose the observed candidate.",
        payload={"name": "synthetic-secret"}, schema=schema(),
        request_model="codex-default", context=context(events),
    )

    # Then
    assert result.data == {"choice": "c1"}
    assert len(events) == 1
    assert events[0].source == "subscription_cli"
    assert events[0].sent is False
    assert events[0].invoked is True
    assert events[0].usage is not None
    assert events[0].usage.input_tokens == 12
    assert events[0].response_model is None
    assert events[0].billed_cost_usd is None


@pytest.mark.asyncio
async def test_configured_codex_adapter_passes_effort_and_fast_mode(
    tmp_path: Path,
) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=fake_cli(
            tmp_path,
            '{"choice":"c1"}',
            expected_home=tmp_path / "codex",
            expected_reasoning_effort="xhigh",
            expected_fast=True,
        ),
        codex_home=tmp_path / "codex",
    )
    configured = CodexModelClientAdapter(
        client,
        reasoning_effort="xhigh",
        fast=True,
    )

    # When
    result = await configured.generate_json(
        instructions="Choose the observed candidate.",
        payload={},
        schema=schema(),
        request_model="codex-default",
        context=context(events),
    )

    # Then
    assert result.data == {"choice": "c1"}
    assert events[0].outcome == "succeeded"


@pytest.mark.asyncio
async def test_subscription_invalid_choice_is_rejected_with_usage(tmp_path: Path) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=fake_cli(
            tmp_path,
            '{"choice":"missing"}',
            expected_home=tmp_path / "codex",
        ),
        codex_home=tmp_path / "codex",
    )

    # When / Then
    with pytest.raises(ProviderError, match="invalid_response"):
        _ = await client.generate_json(
            instructions="Choose.", payload={}, schema=schema(),
            request_model="codex-default", context=context(events),
        )
    assert events[0].outcome == "failed"
    assert events[0].usage is not None
    assert events[0].usage.output_tokens == 3


@pytest.mark.asyncio
async def test_subscription_missing_cli_is_not_a_network_request(tmp_path: Path) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=(str(tmp_path / "absent"),),
        codex_home=tmp_path / "codex",
    )

    # When / Then
    with pytest.raises(ProviderError, match="configuration"):
        _ = await client.generate_json(
            instructions="Choose.", payload={}, schema=schema(),
            request_model="codex-default", context=context(events),
        )
    assert events[0].sent is False
    assert events[0].outcome == "failed"


@pytest.mark.asyncio
async def test_subscription_tool_event_is_not_accepted_as_a_decision(
    tmp_path: Path,
) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=fake_cli(
            tmp_path,
            "{}",
            tool_event=True,
            expected_home=tmp_path / "codex",
        ),
        codex_home=tmp_path / "codex",
    )

    # When / Then
    with pytest.raises(ProviderError, match="unexpected_tool_use"):
        _ = await client.generate_json(
            instructions="Choose.",
            payload={},
            schema=schema(),
            request_model="codex-default",
            context=context(events),
        )
    assert events[0].outcome == "failed"
    assert events[0].error_kind == "unexpected_tool_use"
    assert events[0].error_message == "Codex CLI returned unsupported item type: command_execution"


@pytest.mark.asyncio
async def test_subscription_nonfatal_error_item_does_not_fail_completed_turn(
    tmp_path: Path,
) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=fake_cli(
            tmp_path,
            '{"choice":"c1"}',
            non_fatal_error_event=True,
            expected_home=tmp_path / "codex",
        ),
        codex_home=tmp_path / "codex",
    )

    # When
    result = await client.generate_json(
        instructions="Choose.",
        payload={},
        schema=schema(),
        request_model="codex-default",
        context=context(events),
    )

    # Then
    assert result.data == {"choice": "c1"}
    assert events[0].outcome == "succeeded"
    assert events[0].error_kind is None
    assert events[0].error_message is None


@pytest.mark.asyncio
async def test_subscription_login_status_accepts_authenticated_stderr(
    tmp_path: Path,
) -> None:
    # Given
    script = tmp_path / "fake_codex.py"
    script_source = "\n".join(
        (
            "import os, sys",
            f"assert os.environ['CODEX_HOME'] == {str(tmp_path / 'codex')!r}",
            "assert '-c' in sys.argv",
            "print('Logged in using ChatGPT', file=sys.stderr)",
        )
    ) + "\n"
    _ = script.write_text(
        script_source,
        encoding="utf-8",
    )
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=(sys.executable, str(script)),
        codex_home=tmp_path / "codex",
    )

    # When / Then
    await client.verify_subscription_login()


@pytest.mark.asyncio
async def test_subscription_login_status_rejects_api_key_authentication(
    tmp_path: Path,
) -> None:
    # Given
    script = tmp_path / "fake_codex.py"
    _ = script.write_text(
        "import sys\nprint('Logged in using API key', file=sys.stderr)\n",
        encoding="utf-8",
    )
    client = CodexClient(
        privacy=PrivacyPolicy(),
        command=(sys.executable, str(script)),
        codex_home=tmp_path / "codex",
    )

    # When / Then
    with pytest.raises(ProviderError, match="ChatGPT subscription authentication"):
        await client.verify_subscription_login()
