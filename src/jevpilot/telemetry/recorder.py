"""Run-local accounting; providers never own cumulative experiment metrics."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal, override
from uuid import uuid4

from jevpilot.executor.models import ActionExecution
from jevpilot.task import Budget
from jevpilot.telemetry.models import (
    CallContext,
    CallEvent,
    ExecutionMetrics,
    ExecutionTrace,
    StageSpan,
    TerminalResult,
    TransitionEvent,
)
from jevpilot.verifier.models import VerificationResult


class BudgetExceeded(RuntimeError):
    resource: str

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(resource)

    @override
    def __str__(self) -> str:
        return f"Run budget exceeded: {self.resource}"


@dataclass
class JevSpendBudget:
    """Serial suite-wide conservative reservation, not an invoice balance."""

    limit_usd: Decimal
    reservation_per_attempt_usd: Decimal
    reserved_usd: Decimal = Decimal("0")
    requests: int = 0

    def __post_init__(self) -> None:
        if (
            not self.limit_usd.is_finite()
            or not self.reservation_per_attempt_usd.is_finite()
            or self.limit_usd <= 0
            or self.reservation_per_attempt_usd <= 0
            or self.reservation_per_attempt_usd > self.limit_usd
        ):
            raise BudgetExceeded("invalid_spend_limit")

    def reserve(self) -> None:
        following = self.reserved_usd + self.reservation_per_attempt_usd
        if following > self.limit_usd:
            raise BudgetExceeded("jev_spend_reservation")
        self.reserved_usd = following
        self.requests += 1


class RunRecorder:
    """Mutable event accumulator owned by exactly one serial browser run."""

    def __init__(
        self, budgets: Budget, *,
        reserve_paid_selector: Callable[[], None] | None = None,
        config_sha256: str | None = None,
        on_call_event: Callable[[CallEvent], None] | None = None,
    ) -> None:
        self.run_id: str = uuid4().hex
        self.budgets: Budget = budgets
        self.started: float = time.monotonic()
        self.task_started: float | None = None
        self.task_ended: float | None = None
        self.deadline: float = self.started + budgets.wall_time_ms / 1000
        self.calls: list[CallEvent] = []
        self.executions: list[ActionExecution] = []
        self.verifications: list[VerificationResult] = []
        self.transitions: list[TransitionEvent] = []
        self.spans: list[StageSpan] = []
        self.attempts: int = 0
        self.steps: int = 0
        self.mutations: int = 0
        self.replans: int = 0
        self.waits: int = 0
        self.wait_ms: float = 0.0
        self.stale: int = 0
        self.state: str = "READY"
        self.reserve_paid_selector: Callable[[], None] | None = reserve_paid_selector
        self.config_sha256: str | None = config_sha256
        self.on_call_event: Callable[[CallEvent], None] | None = on_call_event

    def reserve_attempt(self) -> None:
        """Reserve an actual transmission before the transport sends it."""
        self.check_deadline()
        if self.attempts >= self.budgets.max_model_attempts:
            raise BudgetExceeded("model_attempts")
        self.attempts += 1

    def check_deadline(self) -> None:
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("wall_time")

    def start_task(self) -> None:
        if self.task_started is None:
            self.task_started = time.monotonic()

    def context(
        self,
        role: Literal["planner", "selector", "text"],
        purpose: Literal["initial", "normal", "fallback"],
    ) -> CallContext:
        def reserve() -> None:
            self.check_deadline()
            if self.attempts >= self.budgets.max_model_attempts:
                raise BudgetExceeded("model_attempts")
            if role == "selector" and self.reserve_paid_selector is not None:
                self.reserve_paid_selector()
            elif role == "selector" and self.budgets.cost_limit_usd is not None:
                raise BudgetExceeded("cost_reservation_unavailable")
            self.reserve_attempt()

        return CallContext(
            run_id=self.run_id,
            logical_call_id=uuid4().hex,
            role=role,
            purpose=purpose,
            deadline=self.deadline,
            max_attempts=self.budgets.max_attempts_per_call,
            reserve_attempt=reserve,
            record_call=self.record_call,
        )

    def record_call(self, event: CallEvent) -> None:
        if event.run_id != self.run_id:
            raise BudgetExceeded("foreign_run_event")
        if any(previous.call_id == event.call_id for previous in self.calls):
            raise BudgetExceeded("duplicate_call_event")
        self.calls.append(event)
        if self.on_call_event is not None:
            self.on_call_event(event)

    def transition(self, state: str, reason: str) -> None:
        self.transitions.append(
            TransitionEvent(
                from_state=self.state,
                to_state=state,
                reason=reason,
                occurred_at=datetime.now(UTC),
            )
        )
        self.state = state

    def finish_task(self) -> None:
        self.task_ended = time.monotonic()

    def trace(
        self,
        *,
        task_id: str,
        mode: Literal["llm_only", "jev_hybrid"],
        fixture_version: str,
        terminal: TerminalResult,
        browser_version: str | None = None,
    ) -> ExecutionTrace:
        """Build the final trace after browser cleanup, without cumulative state."""
        end = time.monotonic()
        task_end = self.task_ended if self.task_ended is not None else end
        startup = (
            self.task_started - self.started
            if self.task_started is not None
            else task_end - self.started
        )
        network = tuple(c for c in self.calls if c.source == "network" and c.sent)
        subscription = tuple(
            c for c in self.calls if c.source == "subscription_cli" and c.invoked
        )
        costs = tuple(
            c.billed_cost_usd
            if c.billed_cost_usd is not None
            else c.estimated_cost_usd
            for c in (*network, *subscription)
        )
        known = sum(cost for cost in costs if cost is not None)
        total = known if all(cost is not None for cost in costs) else None
        return ExecutionTrace(
            schema_version=2,
            run_id=self.run_id,
            task_id=task_id,
            mode=mode,
            config_sha256=self.config_sha256,
            browser_version=browser_version,
            fixture_version=fixture_version,
            metrics=ExecutionMetrics(
                task_success=terminal.status == "succeeded",
                startup_ms=startup * 1000,
                task_e2e_ms=(
                    (task_end - self.task_started) * 1000
                    if self.task_started is not None
                    else None
                ),
                shutdown_ms=(end - task_end) * 1000,
                full_run_ms=(end - self.started) * 1000,
                model_attempts=self.attempts,
                network_attempts=None if subscription else len(network),
                observed_network_attempts=len(network),
                subscription_invocations=len(subscription),
                actions=len(self.executions),
                mutations=self.mutations,
                replans=self.replans,
                waits=self.waits,
                wait_ms=self.wait_ms,
                stale_observations=self.stale,
                known_cost_usd=known,
                total_cost_usd=total,
            ),
            transitions=tuple(self.transitions),
            executions=tuple(self.executions),
            verifications=tuple(self.verifications),
            calls=tuple(self.calls),
            stage_spans=tuple(self.spans),
            terminal=terminal,
        )
