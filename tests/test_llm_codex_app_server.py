import os
import sys
import time
from pathlib import Path
from typing import ClassVar, override

import anyio
import pytest
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.codex_app_server import CodexAppServerProvider
from jevpilot.llm.codex_exec_adapter import CodexExecProvider
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import LLMProvider, LLMRequest, LLMUsage, ModelRef


class Answer(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    answer: str


REF = ModelRef(
    provider="chatgpt_subscription", auth_mode="subscription",
    model_id="gpt-5.3-codex", reasoning_effort="low",
)


def command(scenario: str) -> tuple[str, ...]:
    return (
        sys.executable, str(Path(__file__).with_name("fixtures_fake_codex_app_server.py")),
        scenario,
    )


def request() -> LLMRequest[Answer]:
    return LLMRequest(
        purpose="choose", system="Return the answer.",
        user_payload={"question": "1+1"}, output_type=Answer,
    )


def wire(home: Path) -> list[dict[str, JsonValue]]:
    adapter = TypeAdapter(dict[str, JsonValue])
    return [adapter.validate_json(line) for line in (home / "wire.jsonl").read_text().splitlines()]


def assert_reaped(home: Path) -> None:
    for pid in {int(str(entry["pid"])) for entry in wire(home)}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


async def test_server_reuses_process_but_not_threads_and_records_subscription_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "must-not-reach-child")
    provider = CodexAppServerProvider(REF, command=command("happy"), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        await provider.start()
        result = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
        for _ in range(2):
            result = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
            assert result.value == Answer(answer="2")
    finally:
        await provider.aclose()
    # Then
    assert result.usage == LLMUsage(100, 5, 10, 2)
    messages = wire(tmp_path)
    assert len({str(entry["pid"]) for entry in messages}) == 1
    assert sum(entry["method"] == "initialize" for entry in messages) == 1
    assert sum(entry["method"] == "thread/start" for entry in messages) == 3
    assert sum(entry["method"] == "turn/start" for entry in messages) == 3
    assert len(ledger.calls) == 3
    assert all(call.auth_mode == "subscription" and call.billed_usd is None for call in ledger.calls)
    assert all(call.api_estimate_usd is None and call.api_equivalent_usd is not None for call in ledger.calls)
    assert provider.rate_limits == {
        "limitId": "codex", "primary": {"usedPercent": 21, "resetsAt": 1234},
    }
    assert_reaped(tmp_path)
    assert isinstance(provider, LLMProvider)


@pytest.mark.parametrize("transport", ["app_server", "exec"])
@pytest.mark.parametrize("scenario,attempts", [("repair", 2), ("happy", 1)])
async def test_shared_validation_and_usage_accounting(
    tmp_path: Path, transport: str, scenario: str, attempts: int,
) -> None:
    # Given
    cls = CodexAppServerProvider if transport == "app_server" else CodexExecProvider
    provider = cls(REF, command=command(scenario), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        result = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
    finally:
        await provider.aclose()
    # Then
    assert result.value.answer == "2"
    assert result.attempts == attempts
    assert result.usage is not None
    assert result.usage.input_tokens == 100 * attempts
    assert len(ledger.calls) == attempts
    assert [call.outcome for call in ledger.calls] == (
        ["failed", "ok"] if attempts == 2 else ["ok"]
    )
    assert [call.call_id for call in ledger.calls] == result.call_ids


@pytest.mark.parametrize("transport", ["app_server", "exec"])
@pytest.mark.parametrize("scenario", ["tool", "invalid"])
async def test_invalid_or_tool_output_never_succeeds(
    tmp_path: Path, transport: str, scenario: str,
) -> None:
    # Given
    cls = CodexAppServerProvider if transport == "app_server" else CodexExecProvider
    provider = cls(REF, command=command(scenario), codex_home=tmp_path)
    ledger = RunLedger()
    # When / Then
    try:
        with pytest.raises(LLMError) as caught:
            _ = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
        assert caught.value.kind == "invalid_output"
        assert len(ledger.calls) == (2 if scenario == "invalid" else 1)
    finally:
        await provider.aclose()


async def test_crashed_server_is_restarted_on_next_call_only(tmp_path: Path) -> None:
    # Given
    provider = CodexAppServerProvider(REF, command=command("crash_once"), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        with pytest.raises(LLMError) as caught:
            _ = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
        assert caught.value.kind == "provider_unavailable"
        assert sum(m["method"] == "initialize" for m in wire(tmp_path)) == 1
        result = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
    finally:
        await provider.aclose()
    # Then
    assert result.value.answer == "2"
    assert len({str(entry["pid"]) for entry in wire(tmp_path)}) == 2
    assert [call.outcome for call in ledger.calls] == ["failed", "ok"]


async def test_cancellation_sends_interrupt_and_reaps_process(tmp_path: Path) -> None:
    # Given
    ready = anyio.Event()

    class ObservedProvider(CodexAppServerProvider):
        @override
        async def _receive(self) -> dict[str, JsonValue]:
            message = await super()._receive()
            if message.get("method") == "turn/started":
                ready.set()
            return message

    provider = ObservedProvider(REF, command=command("cancel"), codex_home=tmp_path)
    ledger = RunLedger()
    cancellation = anyio.CancelScope()

    async def call() -> None:
        with cancellation:
            _ = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))

    # When
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call)
            await ready.wait()
            cancellation.cancel()
    # Then
    assert sum(entry["method"] == "turn/interrupt" for entry in wire(tmp_path)) == 1
    assert_reaped(tmp_path)
    assert [call.outcome for call in ledger.calls] == ["cancelled"]


@pytest.mark.parametrize("scenario,kind", [
    ("rpc_rate", "rate_limited"), ("notification_rate", "rate_limited"),
    ("quota", "quota_exhausted"),
])
async def test_limit_errors_preserve_account_snapshot_without_leaking_details(
    tmp_path: Path, scenario: str, kind: str,
) -> None:
    # Given
    provider = CodexAppServerProvider(REF, command=command(scenario), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        with pytest.raises(LLMError) as caught:
            _ = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
    finally:
        await provider.aclose()
    # Then
    assert caught.value.kind == kind
    assert provider.rate_limits["limitId"] == "codex"
    assert "private upstream details" not in str(caught.value)
    assert ledger.calls[-1].error_kind == kind


@pytest.mark.parametrize("transport", ["app_server", "exec"])
async def test_missing_auth_does_not_start_a_server(tmp_path: Path, transport: str) -> None:
    # Given
    cls = CodexAppServerProvider if transport == "app_server" else CodexExecProvider
    provider = cls(REF, command=command("auth_missing"), codex_home=tmp_path)
    ledger = RunLedger()
    # When / Then
    try:
        with pytest.raises(LLMError) as caught:
            _ = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
        assert caught.value.kind == "auth_missing"
        assert not (tmp_path / "wire.jsonl").exists()
    finally:
        await provider.aclose()


async def test_missing_usage_stays_unknown(tmp_path: Path) -> None:
    # Given
    provider = CodexAppServerProvider(REF, command=command("no_usage"), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        result = await provider.generate(request(), ledger.scope(deadline=time.monotonic() + 10))
    finally:
        await provider.aclose()
    # Then
    assert result.usage is None
    assert ledger.calls[0].api_equivalent_usd is None
    assert ledger.calls[0].billed_usd is None


@pytest.mark.parametrize("transport", ["app_server", "exec"])
async def test_text_output_round_trip(tmp_path: Path, transport: str) -> None:
    # Given
    cls = CodexAppServerProvider if transport == "app_server" else CodexExecProvider
    provider = cls(REF, command=command("text"), codex_home=tmp_path)
    ledger = RunLedger()
    # When
    try:
        result = await provider.generate(
            LLMRequest(purpose="text", system="s", user_payload="1+1", output_type=str),
            ledger.scope(deadline=time.monotonic() + 10),
        )
    finally:
        await provider.aclose()
    # Then
    assert result.value == "2"
    assert result.attempts == 1
    assert len(ledger.calls) == 1


async def test_exec_invalid_text_is_not_successful_empty_output(tmp_path: Path) -> None:
    # Given
    provider = CodexExecProvider(REF, command=command("text_invalid"), codex_home=tmp_path)
    ledger = RunLedger()
    # When / Then
    with pytest.raises(LLMError) as caught:
        _ = await provider.generate(
            LLMRequest(purpose="text", system="s", user_payload="1+1", output_type=str),
            ledger.scope(deadline=time.monotonic() + 10),
        )
    assert caught.value.kind == "invalid_output"
    assert ledger.calls[0].error_kind == "invalid_output"
