from datetime import UTC, datetime

from benchmarks.tasks.run_summary import summarize_jev_usage
from jevpilot.telemetry.models import CallEvent, CallUsage

_NOW = datetime(2026, 10, 6, tzinfo=UTC)


def test_jev_usage_counts_sent_requests_and_keeps_incomplete_totals_unknown() -> None:
    # Given
    calls = (
        CallEvent(
            run_id="run-1",
            call_id="jev-1",
            logical_call_id="logical-1",
            source="network",
            role="selector",
            purpose="initial",
            provider="jev",
            request_model="jev-latest",
            attempt_index=1,
            sent=True,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="succeeded",
            usage=CallUsage(input_tokens=100, output_tokens=20),
            estimated_cost_usd=0.0000042,
        ),
        CallEvent(
            run_id="run-1",
            call_id="jev-2",
            logical_call_id="logical-2",
            source="network",
            role="selector",
            purpose="normal",
            provider="jev",
            request_model="jev-latest",
            attempt_index=1,
            sent=True,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="failed",
            error_kind="transport_timeout",
        ),
        CallEvent(
            run_id="run-1",
            call_id="jev-test-double",
            logical_call_id="logical-3",
            source="test_double",
            role="selector",
            purpose="normal",
            provider="jev",
            request_model="jev-latest",
            attempt_index=1,
            sent=False,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="succeeded",
            usage=CallUsage(input_tokens=900, output_tokens=900),
        ),
        CallEvent(
            run_id="run-1",
            call_id="gateway-1",
            logical_call_id="logical-4",
            source="network",
            role="planner",
            purpose="initial",
            provider="chosun_api_gateway",
            request_model="gpt-6.1-sol",
            attempt_index=1,
            sent=True,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="succeeded",
            usage=CallUsage(input_tokens=500, output_tokens=50),
        ),
    )

    # When
    summary = summarize_jev_usage(calls)

    # Then
    assert summary == {
        "scope": "jevpilot_run",
        "request_count": 2,
        "input_tokens": None,
        "output_tokens": None,
        "observed_input_tokens": 100,
        "observed_output_tokens": 20,
        "incomplete_usage_request_count": 1,
        "estimated_cost_usd": None,
        "observed_estimated_cost_usd": 0.0000042,
        "unknown_estimated_cost_request_count": 1,
    }


def test_jev_usage_reports_complete_token_totals_without_inventing_billing() -> None:
    # Given
    calls = (
        CallEvent(
            run_id="run-1",
            call_id="jev-1",
            logical_call_id="logical-1",
            source="network",
            role="selector",
            purpose="initial",
            provider="jev",
            request_model="jev-latest",
            attempt_index=1,
            sent=True,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="succeeded",
            usage=CallUsage(input_tokens=100, output_tokens=20),
            estimated_cost_usd=0.0000042,
        ),
        CallEvent(
            run_id="run-1",
            call_id="jev-2",
            logical_call_id="logical-2",
            source="network",
            role="selector",
            purpose="normal",
            provider="jev",
            request_model="jev-latest",
            attempt_index=1,
            sent=True,
            started_at=_NOW,
            ended_at=_NOW,
            outcome="succeeded",
            usage=CallUsage(input_tokens=50, output_tokens=10),
            estimated_cost_usd=0.0000021,
        ),
    )

    # When
    summary = summarize_jev_usage(calls)

    # Then
    assert summary == {
        "scope": "jevpilot_run",
        "request_count": 2,
        "input_tokens": 150,
        "output_tokens": 30,
        "observed_input_tokens": 150,
        "observed_output_tokens": 30,
        "incomplete_usage_request_count": 0,
        "estimated_cost_usd": 0.0000063,
        "observed_estimated_cost_usd": 0.0000063,
        "unknown_estimated_cost_request_count": 0,
    }
