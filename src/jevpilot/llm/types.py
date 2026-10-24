from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from pydantic import JsonValue

from jevpilot.llm.errors import LLMErrorKind

AuthMode = Literal["api", "subscription"]
ProviderId = Literal["openai", "anthropic", "chatgpt_subscription"]
Purpose = Literal["brief", "text", "choose", "replan", "extract", "agent_step"]
StructuredOutputMode = Literal["native", "prompted"]
CallOutcome = Literal["ok"] | LLMErrorKind


@dataclass(frozen=True, slots=True)
class ModelRef:
    provider: ProviderId
    auth_mode: AuthMode
    model_id: str
    reasoning_effort: str | None = None
    fast: bool = False
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class LLMRequest[T]:
    purpose: Purpose
    system: str
    user_payload: JsonValue
    output_type: type[T]
    images: list[bytes] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Token usage; ``None`` means the provider did not report it, never zero."""

    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None

    def plus(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=_add(self.input_tokens, other.input_tokens),
            output_tokens=_add(self.output_tokens, other.output_tokens),
            cached_input_tokens=_add(
                self.cached_input_tokens, other.cached_input_tokens
            ),
            reasoning_tokens=_add(self.reasoning_tokens, other.reasoning_tokens),
        )


def _add(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


@dataclass(frozen=True, slots=True)
class LLMResult[T]:
    value: T
    resolved_model: str | None
    usage: LLMUsage | None
    attempts: int
    call_ids: list[str]


@dataclass(frozen=True, slots=True)
class Capabilities:
    structured_output: StructuredOutputMode
    images: bool
    reasoning_efforts: tuple[str, ...]
    supports_fast: bool


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    """One transmitted provider call, handed to ``CallScope.record_call``.

    Times are ``time.monotonic()`` seconds; ``usage`` is ``None`` when unknown.
    """

    call_id: str
    purpose: Purpose
    provider: ProviderId
    auth_mode: AuthMode
    model_id: str
    attempt: int
    resolved_model: str | None
    usage: LLMUsage | None
    started_monotonic: float
    ended_monotonic: float
    outcome: CallOutcome


class CallScope(Protocol):
    """Per-call context implemented by the run ledger.

    ``deadline`` is a ``time.monotonic()`` instant; ``parent_span_id`` is the
    ledger span the call belongs to.
    """

    @property
    def run_id(self) -> str: ...

    @property
    def parent_span_id(self) -> str | None: ...

    @property
    def deadline(self) -> float: ...

    def record_call(self, record: LLMCallRecord) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def ref(self) -> ModelRef: ...

    def capabilities(self) -> Capabilities: ...

    async def generate[T](
        self, request: LLMRequest[T], context: CallScope
    ) -> LLMResult[T]: ...

    async def aclose(self) -> None: ...
