"""Translate each frozen v1 transport attempt into one v2 ledger record."""

from decimal import Decimal
from typing import Literal
from uuid import uuid4

from jevpilot.ledger.models import CallRecord, CallUsage
from jevpilot.ledger.recorder import CallScope
from jevpilot.task import ContractViolation
from jevpilot.telemetry.models import CallContext, CallEvent


def legacy_call_context(
    scope: CallScope, *, role: Literal["planner", "selector", "text"],
    max_attempts: int,
) -> CallContext:
    """Use monotonic attempt boundaries, never v1 wall-clock durations."""
    if max_attempts < 1:
        raise ContractViolation("max_attempts", "must be positive")
    started = scope.ledger.clock()
    attempts = 0

    def reserve() -> None:
        nonlocal started, attempts
        if scope.ledger.clock() >= scope.deadline:
            raise ContractViolation("deadline", "call deadline exceeded")
        if attempts >= max_attempts:
            raise ContractViolation("max_attempts", "attempt budget exceeded")
        scope.reserve_attempt()
        attempts += 1
        started = scope.ledger.clock()

    def record(event: CallEvent) -> None:
        if event.run_id != scope.run_id:
            raise ContractViolation("run_id", "foreign run event")
        subscription = event.source == "subscription_cli"
        estimate = (
            Decimal(str(event.estimated_cost_usd))
            if event.estimated_cost_usd is not None else None
        )
        billed = (
            Decimal(str(event.billed_cost_usd))
            if event.billed_cost_usd is not None and not subscription else None
        )
        usage = event.usage
        scope.record_call(CallRecord(
            call_id=event.call_id, span_id=scope.parent_span_id,
            kind="jev" if event.provider == "jev" else "llm",
            provider=event.provider,
            auth_mode="subscription" if subscription else "api",
            purpose=event.purpose, request_model=event.request_model,
            resolved_model=event.response_model, attempts=1,
            outcome="ok" if event.outcome == "succeeded" else "failed",
            error_kind=event.error_kind,
            usage=CallUsage(
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
                cached_input_tokens=usage.cache_read_tokens if usage is not None else None,
            ),
            billed_usd=billed,
            api_estimate_usd=estimate if not subscription else None,
            api_equivalent_usd=estimate if subscription else None,
            price_source=event.price_source, price_checked_on=event.price_checked_on,
            latency_s=scope.ledger.clock() - started,
        ))

    return CallContext(
        run_id=scope.run_id, logical_call_id=uuid4().hex, role=role,
        purpose="normal", deadline=scope.deadline, max_attempts=max_attempts,
        reserve_attempt=reserve, record_call=record,
    )
