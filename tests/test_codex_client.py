from pathlib import Path
import sys
import time

import pytest
from pydantic import JsonValue

from jevpilot.codex_client import CodexClient
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
) -> tuple[str, ...]:
    """Launch a real local process with the documented Codex JSONL wire shape."""
    script = tmp_path / "fake_cli.py"
    lines = [
        "import json, os, sys",
        "prompt = sys.stdin.read()",
        "assert 'JEVPILOT_JEV_API_KEY' not in os.environ",
        "assert 'JEVPILOT_LLM_API_KEY' not in os.environ",
        "assert 'synthetic-secret' not in prompt",
        "assert '--ignore-user-config' in sys.argv",
        "assert '--output-schema' in sys.argv",
        "assert 'read-only' in sys.argv",
        "print(json.dumps({'type':'thread.started','thread_id':'test'}))",
    ]
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
        command=fake_cli(tmp_path, '{"choice":"c1"}'),
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
async def test_subscription_invalid_choice_is_rejected_with_usage(tmp_path: Path) -> None:
    # Given
    events: list[CallEvent] = []
    client = CodexClient(privacy=PrivacyPolicy(), command=fake_cli(tmp_path, '{"choice":"missing"}'))

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
    client = CodexClient(privacy=PrivacyPolicy(), command=(str(tmp_path / "absent"),))

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
        privacy=PrivacyPolicy(), command=fake_cli(tmp_path, "{}", tool_event=True)
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
