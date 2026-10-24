import time
from typing import cast, get_args

from jevpilot.llm import (
    Capabilities,
    LLMCallRecord,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResult,
    LLMUsage,
    ModelRef,
    ProviderId,
)


class _Scope:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    @property
    def run_id(self) -> str:
        return "run-1"

    @property
    def parent_span_id(self) -> str | None:
        return None

    @property
    def deadline(self) -> float:
        return time.monotonic() + 10

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class _EchoProvider:
    def __init__(self) -> None:
        self._ref: ModelRef = ModelRef(provider="openai", auth_mode="api", model_id="m")

    @property
    def ref(self) -> ModelRef:
        return self._ref

    def capabilities(self) -> Capabilities:
        return Capabilities(
            structured_output="prompted",
            images=False,
            reasoning_efforts=("low", "high"),
            supports_fast=False,
        )

    async def generate[T](self, request: LLMRequest[T], context: _Scope) -> LLMResult[T]:
        _ = (request, context)
        return LLMResult(
            value=cast(T, "echo"), resolved_model="m-1", usage=None, attempts=1, call_ids=["c"]
        )

    async def aclose(self) -> None:
        return None


def test_model_ref_defaults() -> None:
    ref = ModelRef(provider="chatgpt_subscription", auth_mode="subscription", model_id="gpt")

    assert ref.reasoning_effort is None
    assert ref.fast is False
    assert ref.base_url is None


def test_provider_ids_are_openai_anthropic_and_chatgpt_subscription() -> None:
    assert get_args(ProviderId) == ("openai", "anthropic", "chatgpt_subscription")


def test_request_images_default_is_not_shared() -> None:
    first = LLMRequest(purpose="text", system="s", user_payload={"a": 1}, output_type=str)
    second = LLMRequest(purpose="text", system="s", user_payload=None, output_type=str)

    first.images.append(b"x")

    assert second.images == []


def test_usage_sum_keeps_unknown_fields_unknown() -> None:
    known = LLMUsage(input_tokens=10, output_tokens=5, cached_input_tokens=0, reasoning_tokens=None)
    other = LLMUsage(input_tokens=3, output_tokens=2, cached_input_tokens=1, reasoning_tokens=7)

    total = known.plus(other)

    assert total == LLMUsage(
        input_tokens=13, output_tokens=7, cached_input_tokens=1, reasoning_tokens=None
    )


async def test_provider_protocol_is_structural() -> None:
    provider = _EchoProvider()

    assert isinstance(provider, LLMProvider)
    result = await provider.generate(
        LLMRequest(purpose="text", system="s", user_payload="hi", output_type=str), _Scope()
    )
    assert result.value == "echo"


def test_llm_error_fields_and_safe_rendering() -> None:
    error = LLMError("rate_limited", retryable=True, retry_after_s=2.5, message="slow down")

    assert (error.kind, error.retryable, error.retry_after_s) == ("rate_limited", True, 2.5)
    assert str(error) == "slow down (kind=rate_limited)"
    assert "slow down" in repr(error)
