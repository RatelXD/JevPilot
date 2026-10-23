import asyncio
import json
import time

import pytest
from pydantic import BaseModel

from jevpilot.llm import (
    LLMCallRecord,
    LLMError,
    LLMRequest,
    LLMUsage,
    ModelRef,
    RawCompletion,
    StructuredOutputError,
    StructuredPrompt,
    generate_structured,
    parse_structured,
)

PROMPT_SENTINEL = "PROMPT-SENTINEL-7f3a"
OUTPUT_SENTINEL = "OUTPUT-SENTINEL-91bc"
REF = ModelRef(provider="openai_compatible", auth_mode="api", model_id="test-model")


class Choice(BaseModel):
    action: str
    index: int


class _Scope:
    def __init__(self, deadline: float | None = None) -> None:
        self.records: list[LLMCallRecord] = []
        self._deadline: float = time.monotonic() + 30 if deadline is None else deadline

    @property
    def run_id(self) -> str:
        return "run-1"

    @property
    def parent_span_id(self) -> str | None:
        return "span-1"

    @property
    def deadline(self) -> float:
        return self._deadline

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class _ScriptedSender:
    def __init__(self, replies: list[str], usage: list[LLMUsage | None] | None = None) -> None:
        self.replies: list[str] = replies
        self.usage: list[LLMUsage | None] = usage or [_usage(10, 4)] * len(replies)
        self.prompts: list[StructuredPrompt] = []

    async def __call__(self, prompt: StructuredPrompt) -> RawCompletion:
        index = len(self.prompts)
        self.prompts.append(prompt)
        return RawCompletion(
            text=self.replies[index], resolved_model="test-model-2026", usage=self.usage[index]
        )


def _usage(inputs: int, outputs: int) -> LLMUsage:
    return LLMUsage(
        input_tokens=inputs, output_tokens=outputs, cached_input_tokens=0, reasoning_tokens=0
    )


def _request() -> LLMRequest[Choice]:
    return LLMRequest(
        purpose="choose",
        system=f"Pick an action. {PROMPT_SENTINEL}",
        user_payload={"goal": f"click {PROMPT_SENTINEL}", "candidates": ["a", "b"]},
        output_type=Choice,
    )


def test_parse_structured_accepts_plain_and_fenced_json() -> None:
    assert parse_structured('{"action": "click", "index": 2}', Choice) == Choice(
        action="click", index=2
    )
    assert parse_structured('```json\n{"action": "type", "index": 0}\n```', Choice) == Choice(
        action="type", index=0
    )
    assert parse_structured("free text", str) == "free text"


def test_parse_errors_do_not_echo_model_output() -> None:
    with pytest.raises(StructuredOutputError) as caught:
        _ = parse_structured(f'{{"action": "{OUTPUT_SENTINEL}", "index": "{OUTPUT_SENTINEL}"}}', Choice)

    assert caught.value.errors
    assert OUTPUT_SENTINEL not in str(caught.value)


async def test_valid_first_reply_uses_one_attempt() -> None:
    scope = _Scope()
    sender = _ScriptedSender(['{"action": "click", "index": 1}'])

    result = await generate_structured(
        _request(), scope, ref=REF, mode="prompted", send=sender
    )

    assert result.value == Choice(action="click", index=1)
    assert result.attempts == 1
    assert result.usage == _usage(10, 4)
    assert result.resolved_model == "test-model-2026"
    assert [record.outcome for record in scope.records] == ["ok"]
    assert result.call_ids == [scope.records[0].call_id]
    assert scope.records[0].purpose == "choose"
    prompt = sender.prompts[0]
    assert '"index"' in prompt.system
    assert json.loads(prompt.user_text)["candidates"] == ["a", "b"]


async def test_native_mode_passes_schema_without_prompt_injection() -> None:
    sender = _ScriptedSender(['{"action": "click", "index": 1}'])

    _ = await generate_structured(_request(), _Scope(), ref=REF, mode="native", send=sender)

    prompt = sender.prompts[0]
    assert prompt.system == _request().system
    assert isinstance(prompt.json_schema, dict)


async def test_one_repair_then_success_records_both_usages() -> None:
    scope = _Scope()
    sender = _ScriptedSender(
        [f'{{"action": "{OUTPUT_SENTINEL}"}}', '{"action": "click", "index": 3}'],
        usage=[_usage(10, 4), _usage(20, 6)],
    )

    result = await generate_structured(
        _request(), scope, ref=REF, mode="prompted", send=sender
    )

    assert result.value == Choice(action="click", index=3)
    assert result.attempts == 2
    assert result.usage == _usage(30, 10)
    assert [record.outcome for record in scope.records] == ["invalid_output", "ok"]
    assert [record.usage for record in scope.records] == [_usage(10, 4), _usage(20, 6)]
    assert result.call_ids == [record.call_id for record in scope.records]
    repair = sender.prompts[1]
    assert repair.attempt == 2
    assert any("index" in error for error in repair.repair_errors)
    assert OUTPUT_SENTINEL not in repair.user_text
    assert OUTPUT_SENTINEL not in repair.system


async def test_two_invalid_replies_raise_invalid_output_without_leaks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scope = _Scope()
    sender = _ScriptedSender([f"{{broken {OUTPUT_SENTINEL}", f"not json {OUTPUT_SENTINEL}"])

    with pytest.raises(LLMError) as caught:
        _ = await generate_structured(_request(), scope, ref=REF, mode="prompted", send=sender)

    error = caught.value
    assert error.kind == "invalid_output"
    assert error.retryable is False
    assert len(sender.prompts) == 2
    assert [record.outcome for record in scope.records] == ["invalid_output", "invalid_output"]
    assert [record.usage for record in scope.records] == [_usage(10, 4), _usage(10, 4)]
    rendered = " ".join([str(error), repr(error), error.message, *map(str, error.args)])
    for sentinel in (PROMPT_SENTINEL, OUTPUT_SENTINEL):
        assert sentinel not in rendered
    print(
        "QA invalid_output:",
        json.dumps(
            {
                "kind": error.kind,
                "message": error.message,
                "records": [
                    {"attempt": r.attempt, "outcome": r.outcome, "usage": r.usage.__repr__()}
                    for r in scope.records
                ],
            }
        ),
    )
    assert "QA invalid_output" in capsys.readouterr().out


async def test_missing_usage_on_any_attempt_makes_total_unknown() -> None:
    sender = _ScriptedSender(
        ['{"action": 1}', '{"action": "click", "index": 0}'], usage=[None, _usage(5, 5)]
    )

    result = await generate_structured(_request(), _Scope(), ref=REF, mode="prompted", send=sender)

    assert result.usage is None


async def test_text_output_is_single_attempt_without_schema() -> None:
    sender = _ScriptedSender(["just words"])
    request = LLMRequest(purpose="text", system="s", user_payload="hello", output_type=str)

    result = await generate_structured(request, _Scope(), ref=REF, mode="prompted", send=sender)

    assert result.value == "just words"
    assert sender.prompts[0].json_schema is None
    assert sender.prompts[0].user_text == "hello"


async def test_provider_error_is_recorded_and_not_retried() -> None:
    scope = _Scope()
    calls: list[StructuredPrompt] = []

    async def failing(prompt: StructuredPrompt) -> RawCompletion:
        calls.append(prompt)
        raise LLMError("rate_limited", retryable=True, retry_after_s=1.0, message="rate limited")

    with pytest.raises(LLMError) as caught:
        _ = await generate_structured(_request(), scope, ref=REF, mode="prompted", send=failing)

    assert caught.value.kind == "rate_limited"
    assert len(calls) == 1
    assert [(r.outcome, r.usage) for r in scope.records] == [("rate_limited", None)]


async def test_cancellation_is_recorded_and_propagated() -> None:
    scope = _Scope()

    async def cancelled(_prompt: StructuredPrompt) -> RawCompletion:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _ = await generate_structured(_request(), scope, ref=REF, mode="prompted", send=cancelled)

    assert [r.outcome for r in scope.records] == ["cancelled"]


async def test_expired_deadline_sends_nothing() -> None:
    scope = _Scope(deadline=time.monotonic() - 1)
    sender = _ScriptedSender(['{"action": "click", "index": 1}'])

    with pytest.raises(LLMError) as caught:
        _ = await generate_structured(_request(), scope, ref=REF, mode="prompted", send=sender)

    assert caught.value.kind == "timeout"
    assert sender.prompts == []
    assert scope.records == []
