from datetime import UTC, datetime
from decimal import Decimal

import pytest

from jevpilot.task import Budget
from jevpilot.telemetry.models import CallEvent, TerminalResult
from jevpilot.telemetry.recorder import BudgetExceeded, JevSpendBudget, RunRecorder


def budget() -> Budget:
    return Budget(
        max_steps=8, max_mutations=4, max_replans=1, max_model_attempts=3,
        max_attempts_per_call=2, max_stale_observations=2, max_waits=1,
        max_wait_ms=500, wall_time_ms=30000,
    )


def test_execution_trace_keeps_unknown_failure_cost() -> None:
    # Given
    recorder = RunRecorder(budget())
    now = datetime.now(UTC)
    recorder.start_task()
    recorder.reserve_attempt()
    recorder.record_call(CallEvent(
        run_id=recorder.run_id, call_id="failed-1", logical_call_id="logical-1",
        source="network", role="planner", purpose="initial", provider="openai",
        request_model="configured-model", attempt_index=1, sent=True,
        started_at=now, ended_at=now, outcome="failed", error_kind="transport_timeout",
    ))
    recorder.finish_task()

    # When
    trace = recorder.trace(
        task_id="T-00", mode="llm_only", fixture_version="1",
        terminal=TerminalResult(status="failed", reason="transport_timeout", ended_at=now),
    )

    # Then
    assert trace.metrics.network_attempts == 1
    assert trace.metrics.total_cost_usd is None
    assert trace.metrics.known_cost_usd == 0
    assert trace.metrics.task_success is False


def test_run_recorders_do_not_share_attempt_counts() -> None:
    # Given
    first = RunRecorder(budget())
    second = RunRecorder(budget())
    first.reserve_attempt()

    # When
    second.reserve_attempt()

    # Then
    assert first.attempts == 1
    assert second.attempts == 1
    assert first.run_id != second.run_id


def test_attempt_budget_rejects_transmission_before_it_happens() -> None:
    # Given
    recorder = RunRecorder(budget())
    for _ in range(3):
        recorder.reserve_attempt()

    # When / Then
    with pytest.raises(BudgetExceeded, match="model_attempts"):
        recorder.reserve_attempt()
    assert recorder.attempts == 3


def test_test_double_does_not_count_as_network_cost() -> None:
    # Given
    recorder = RunRecorder(budget())
    now = datetime.now(UTC)
    recorder.record_call(CallEvent(
        run_id=recorder.run_id, call_id="fake-1", logical_call_id="logical-1",
        source="test_double", role="planner", purpose="initial", provider="test",
        request_model="none", attempt_index=1, sent=False,
        started_at=now, ended_at=now, outcome="succeeded",
    ))
    recorder.finish_task()

    # When
    trace = recorder.trace(
        task_id="T-00", mode="llm_only", fixture_version="1",
        terminal=TerminalResult(status="failed", reason="offline_test", ended_at=now),
    )

    # Then
    assert trace.metrics.network_attempts == 0
    assert trace.metrics.model_attempts == 0
    assert trace.metrics.total_cost_usd == 0


def test_subscription_invocation_does_not_invent_http_count_or_cost() -> None:
    # Given
    recorder = RunRecorder(budget())
    now = datetime.now(UTC)
    recorder.reserve_attempt()
    recorder.record_call(CallEvent(
        run_id=recorder.run_id, call_id="sub-1", logical_call_id="logical-1",
        source="subscription_cli", role="planner", purpose="initial",
        provider="chatgpt-subscription", request_model="configured-model",
        attempt_index=1, sent=False, invoked=True,
        started_at=now, ended_at=now, outcome="succeeded",
    ))
    recorder.finish_task()

    # When
    trace = recorder.trace(
        task_id="T-00", mode="llm_only", fixture_version="1",
        terminal=TerminalResult(status="failed", reason="test_end", ended_at=now),
    )

    # Then
    assert trace.metrics.subscription_invocations == 1
    assert trace.metrics.observed_network_attempts == 0
    assert trace.metrics.network_attempts is None
    assert trace.metrics.total_cost_usd is None


def test_paid_attempt_reservations_are_shared_across_runs() -> None:
    # Given
    spend = JevSpendBudget(Decimal("0.005"), Decimal("0.002"))
    first = RunRecorder(budget(), reserve_paid_selector=spend.reserve)
    second = RunRecorder(budget(), reserve_paid_selector=spend.reserve)
    first.context("selector", "normal").reserve_attempt()
    second.context("selector", "normal").reserve_attempt()

    # When / Then
    with pytest.raises(BudgetExceeded, match="jev_spend_reservation"):
        second.context("selector", "normal").reserve_attempt()
    assert spend.requests == 2
    assert spend.reserved_usd == Decimal("0.004")
    assert second.attempts == 1


def test_subscription_planning_does_not_consume_jev_reservation() -> None:
    # Given
    spend = JevSpendBudget(Decimal("0.005"), Decimal("0.002"))
    recorder = RunRecorder(budget(), reserve_paid_selector=spend.reserve)

    # When
    recorder.context("planner", "initial").reserve_attempt()

    # Then
    assert recorder.attempts == 1
    assert spend.requests == 0


def test_spend_reservation_cannot_exceed_total_authorized_limit() -> None:
    # Given / When / Then
    with pytest.raises(BudgetExceeded, match="invalid_spend_limit"):
        _ = JevSpendBudget(Decimal("0.50"), Decimal("1.00"))
