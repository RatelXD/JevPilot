"""Validated span trees and independent cost bases for one execution."""

from decimal import Decimal
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevpilot.task import ContractViolation

SpanKind = Literal[
    "run", "brief", "observe", "route", "jev", "llm", "execute", "wait",
    "recovery", "verify", "handoff", "confirm",
]
Outcome = Literal["ok", "failed", "cancelled"]
Count = Annotated[int, Field(ge=0)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class LedgerModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class Span(LedgerModel):
    span_id: str
    parent_id: str | None
    kind: SpanKind
    label: str
    start_s: Seconds
    end_s: Seconds | None = None
    outcome: Outcome | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "Span":
        if self.end_s is not None and self.end_s < self.start_s:
            raise ContractViolation("end_s", "must not precede start_s")
        if (self.end_s is None) != (self.outcome is None):
            raise ContractViolation("outcome", "must be set exactly when the span closes")
        return self


class CallUsage(LedgerModel):
    """Cached input and reasoning are subsets, not extra billable tokens."""

    input_tokens: Count | None = None
    output_tokens: Count | None = None
    cached_input_tokens: Count | None = None
    reasoning_tokens: Count | None = None

    @model_validator(mode="after")
    def validate_cached_input(self) -> "CallUsage":
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ContractViolation("cached_input_tokens", "must not exceed input_tokens")
        return self


class CallRecord(LedgerModel):
    call_id: str
    span_id: str
    kind: Literal["jev", "llm"]
    provider: str
    auth_mode: Literal["api", "subscription"]
    purpose: str
    request_model: str
    resolved_model: str | None = None
    attempts: Annotated[int, Field(gt=0)] = 1
    outcome: Outcome
    error_kind: str | None = None
    usage: CallUsage = Field(default_factory=CallUsage)
    billed_usd: Money | None = None
    api_estimate_usd: Money | None = None
    api_equivalent_usd: Money | None = None
    price_source: str | None = None
    price_checked_on: str | None = None
    latency_s: Seconds
    route_reason: str | None = None

    @model_validator(mode="after")
    def validate_cost_basis(self) -> "CallRecord":
        match self.auth_mode:
            case "subscription":
                if self.billed_usd is not None or self.api_estimate_usd is not None:
                    raise ContractViolation("auth_mode", "subscription only permits API equivalent")
            case "api":
                if self.api_equivalent_usd is not None:
                    raise ContractViolation("api_equivalent_usd", "only applies to subscriptions")
        return self


class Aggregate(LedgerModel):
    known_sum: Decimal = Decimal("0")
    # Older receipts prove at least one known entry only for nonzero sums.
    known_count: Count = Field(default_factory=lambda data: 1 if data["known_sum"] != 0 else 0)
    unknown_count: Count = 0
    complete: bool = True


class KindSummary(LedgerModel):
    calls: Count
    input_tokens: Aggregate
    output_tokens: Aggregate
    cached_input_tokens: Aggregate
    reasoning_tokens: Aggregate
    billed_usd: Aggregate
    api_estimate_usd: Aggregate
    api_equivalent_usd: Aggregate


class RunSummary(LedgerModel):
    run_id: str
    run_started_at_utc: str
    elapsed_s: Seconds
    by_kind: dict[Literal["jev", "llm"], KindSummary]
    duration_s: dict[SpanKind, float]
    overlap_s: Seconds
