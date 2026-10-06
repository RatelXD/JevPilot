from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from jevpilot.executor.models import ActionExecution
from jevpilot.task import ContractViolation
from jevpilot.verifier.models import VerificationResult

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class CallUsage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    input_tokens: NonNegativeInt | None = None
    output_tokens: NonNegativeInt | None = None
    cache_read_tokens: NonNegativeInt | None = None


class CallEvent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    run_id: NonEmptyStr
    call_id: NonEmptyStr
    logical_call_id: NonEmptyStr
    parent_call_id: NonEmptyStr | None = None
    source: Literal["network", "subscription_cli", "test_double", "cache"]
    role: Literal["planner", "selector", "text"]
    purpose: Literal["initial", "normal", "fallback"]
    provider: NonEmptyStr
    request_model: NonEmptyStr
    response_model: NonEmptyStr | None = None
    attempt_index: PositiveInt
    sent: bool
    invoked: bool = False
    started_at: datetime
    ended_at: datetime
    outcome: Literal["succeeded", "failed"]
    usage: CallUsage | None = None
    billed_cost_usd: NonNegativeFloat | None = None
    estimated_cost_usd: NonNegativeFloat | None = None
    price_source: str | None = None
    price_checked_on: str | None = None
    estimated_cost_credits: NonNegativeDecimal | None = None
    credit_price_source: str | None = None
    credit_price_checked_on: str | None = None
    status_code: int | None = None
    error_kind: NonEmptyStr | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_event(self) -> "CallEvent":
        if self.ended_at < self.started_at:
            raise ContractViolation(
                field="ended_at",
                reason="must not precede started_at",
            )
        if self.source == "subscription_cli" and self.sent:
            raise ContractViolation(
                field="sent",
                reason="CLI invocation cannot assert an HTTP request was sent",
            )
        if self.source != "subscription_cli" and self.invoked:
            raise ContractViolation(
                field="invoked",
                reason="only subscription events can mark a CLI invocation",
            )
        if self.source in {"test_double", "cache"} and self.sent:
            raise ContractViolation(
                field="sent",
                reason="non-network call events must not be marked sent",
            )
        if self.outcome == "failed" and self.error_kind is None:
            raise ContractViolation(
                field="error_kind",
                reason="failed calls require it",
            )
        if self.outcome == "succeeded" and (
            self.error_kind is not None or self.error_message is not None
        ):
            raise ContractViolation(
                field="error_kind",
                reason="successful calls must not include errors",
            )
        return self


class TransitionEvent(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    from_state: NonEmptyStr
    to_state: NonEmptyStr
    reason: NonEmptyStr
    occurred_at: datetime


class StageSpan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    span_id: NonEmptyStr
    stage: NonEmptyStr
    started_at: datetime
    ended_at: datetime
    outcome: Literal["succeeded", "failed", "cancelled"]

    @model_validator(mode="after")
    def validate_timestamps(self) -> "StageSpan":
        if self.ended_at < self.started_at:
            raise ContractViolation(
                field="ended_at",
                reason="must not precede started_at",
            )
        return self


class TerminalResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    status: Literal[
        "succeeded",
        "failed",
        "blocked",
        "uncertain_execution",
        "timeout",
    ]
    reason: NonEmptyStr
    ended_at: datetime


class ExecutionMetrics(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    task_success: bool
    startup_ms: NonNegativeFloat
    task_e2e_ms: NonNegativeFloat | None = None
    shutdown_ms: NonNegativeFloat
    full_run_ms: NonNegativeFloat
    model_attempts: NonNegativeInt
    network_attempts: NonNegativeInt | None
    observed_network_attempts: NonNegativeInt = 0
    subscription_invocations: NonNegativeInt = 0
    actions: NonNegativeInt
    mutations: NonNegativeInt
    replans: NonNegativeInt
    waits: NonNegativeInt
    wait_ms: NonNegativeFloat
    stale_observations: NonNegativeInt
    known_cost_usd: NonNegativeFloat
    total_cost_usd: NonNegativeFloat | None = None


class ExecutionTrace(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2]
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    mode: Literal["llm_only", "jev_hybrid"]
    config_sha256: str | None = None
    browser_version: NonEmptyStr | None = None
    fixture_version: NonEmptyStr
    metrics: ExecutionMetrics
    transitions: tuple[TransitionEvent, ...]
    executions: tuple[ActionExecution, ...]
    verifications: tuple[VerificationResult, ...]
    calls: tuple[CallEvent, ...]
    stage_spans: tuple[StageSpan, ...]
    terminal: TerminalResult


ReserveAttempt = Callable[[], None]
RecordCall = Callable[[CallEvent], None]


@dataclass(frozen=True, slots=True)
class CallContext:
    run_id: str
    logical_call_id: str
    role: Literal["planner", "selector", "text"]
    purpose: Literal["initial", "normal", "fallback"]
    deadline: float
    max_attempts: int
    reserve_attempt: ReserveAttempt
    record_call: RecordCall
