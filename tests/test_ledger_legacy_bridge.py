from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx2
import pytest
from pydantic import JsonValue

from jevpilot.ledger import RunLedger
from jevpilot.ledger.legacy_bridge import legacy_call_context
from jevpilot.model_client import ModelClient, ModelClientConfig
from jevpilot.task import ContractViolation
from jevpilot.telemetry.models import CallEvent, CallUsage


def test_two_v1_attempt_events_become_two_records_with_separate_costs() -> None:
    # Given
    times = iter([10.0, 20.0, 21.0, 21.0, 23.0, 24.0, 24.0, 27.0])
    ledger = RunLedger(clock=lambda: next(times))
    scope = ledger.scope(deadline=100)
    context = legacy_call_context(scope, role="selector", max_attempts=2)
    now = datetime(2026, 10, 23, tzinfo=UTC)
    # When
    for index in (1, 2):
        context.reserve_attempt()
        context.record_call(CallEvent(
            run_id=ledger.run_id, call_id=f"attempt-{index}",
            logical_call_id=context.logical_call_id, source="network",
            role="selector", purpose="normal", provider="jev",
            request_model="jev-1.13.0", response_model="jev-1.13.0",
            attempt_index=index, sent=True, started_at=now,
            ended_at=now + timedelta(days=1),
            outcome="failed" if index == 1 else "succeeded",
            error_kind="rate_limited" if index == 1 else None,
            usage=None if index == 1 else CallUsage(input_tokens=1000, output_tokens=2),
            billed_cost_usd=None if index == 1 else 0.2,
            estimated_cost_usd=None if index == 1 else 0.1,
            price_source="v1-source", price_checked_on="2026-10-06",
        ))
    # Then
    assert len(ledger.calls) == 2
    first, second = ledger.calls
    assert [c.attempts for c in ledger.calls] == [1, 1]
    assert first.outcome == "failed"
    assert first.api_estimate_usd is None
    assert first.usage.input_tokens is None
    assert second.outcome == "ok"
    assert second.billed_usd == Decimal("0.2")
    assert second.api_estimate_usd == Decimal("0.1")
    assert second.api_equivalent_usd is None
    assert second.usage.output_tokens == 2
    assert second.price_source == "v1-source"
    assert [c.latency_s for c in ledger.calls] == [2, 3]


def test_subscription_v1_estimate_maps_only_to_api_equivalent() -> None:
    # Given
    ledger = RunLedger()
    context = legacy_call_context(ledger.scope(deadline=float("inf")), role="selector", max_attempts=1)
    now = datetime(2026, 10, 23, tzinfo=UTC)
    event = CallEvent(
        run_id=ledger.run_id, call_id="sub", logical_call_id=context.logical_call_id,
        source="subscription_cli", role="selector", purpose="normal",
        provider="chatgpt-subscription", request_model="gpt-5.3-codex",
        attempt_index=1, sent=False, invoked=True,
        started_at=now, ended_at=now, outcome="succeeded",
        usage=CallUsage(input_tokens=100, output_tokens=2, cache_read_tokens=10),
        billed_cost_usd=99, estimated_cost_usd=0.01,
    )
    # When
    context.record_call(event)
    # Then
    record = ledger.calls[0]
    assert record.kind == "llm"
    assert record.billed_usd is None
    assert record.api_estimate_usd is None
    assert record.api_equivalent_usd == Decimal("0.01")
    assert record.usage.cached_input_tokens == 10
    assert record.usage.reasoning_tokens is None


@pytest.mark.asyncio
async def test_frozen_model_transport_can_record_through_bridge_without_changes() -> None:
    # Given
    ledger = RunLedger()
    scope = ledger.scope(deadline=ledger.clock() + 10)
    context = legacy_call_context(scope, role="selector", max_attempts=1)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={
            "model": "jev-1.13.0", "usage": {"input_tokens": 1000, "output_tokens": 0},
        })

    # When
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as transport:
        client = ModelClient[JsonValue](
            ModelClientConfig(
                provider="jev", endpoint="https://api.typesafe.ai/v1/systemone",
                api_key="synthetic", retry_statuses=frozenset(),
            ),
            http_client=transport,
        )
        _ = await client.request_json(
            request_model="jev-1.13.0", payload={}, context=context,
            parser=lambda body, metadata: body,
        )
    # Then
    assert len(ledger.calls) == 1
    assert ledger.calls[0].kind == "jev"
    assert ledger.calls[0].api_estimate_usd == Decimal("0.000042")
    assert ledger.calls[0].billed_usd is None


def test_bridge_reserves_each_attempt_before_transmission() -> None:
    # Given
    reserved: list[None] = []
    ledger = RunLedger(clock=lambda: 1)
    context = legacy_call_context(
        ledger.scope(deadline=2, reserve_attempt=lambda: reserved.append(None)),
        role="planner", max_attempts=1,
    )
    # When
    context.reserve_attempt()
    # Then
    assert len(reserved) == 1
    assert context.purpose == "normal"
    assert context.deadline == 2
    with pytest.raises(ContractViolation, match="attempt budget"):
        context.reserve_attempt()
    assert len(reserved) == 1


def test_expired_bridge_does_not_reserve_or_send() -> None:
    # Given
    reserved: list[None] = []
    ledger = RunLedger(clock=lambda: 2)
    context = legacy_call_context(
        ledger.scope(deadline=1, reserve_attempt=lambda: reserved.append(None)),
        role="text", max_attempts=1,
    )
    # When / Then
    with pytest.raises(ContractViolation, match="deadline"):
        context.reserve_attempt()
    assert not reserved
