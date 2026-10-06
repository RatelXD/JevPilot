from datetime import UTC, datetime

import pytest

from jevpilot.telemetry.call_reporting import report_jev_call
from jevpilot.telemetry.models import CallEvent, CallUsage


def test_jev_call_report_prints_per_request_tokens_and_estimate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given
    now = datetime(2026, 10, 6, tzinfo=UTC)
    event = CallEvent(
        run_id="run-1",
        call_id="jev-call-1",
        logical_call_id="logical-1",
        source="network",
        role="selector",
        purpose="initial",
        provider="jev",
        request_model="jev-latest",
        response_model="jev-1.13.0",
        attempt_index=1,
        sent=True,
        started_at=now,
        ended_at=now,
        outcome="succeeded",
        usage=CallUsage(input_tokens=35_569, output_tokens=3_986),
        estimated_cost_usd=0.001493898,
        price_source="TypeSafe published per-token rate",
        price_checked_on="2026-10-06",
    )

    # When
    report_jev_call(event)

    # Then
    assert capsys.readouterr().out == (
        "jev_request call_id=jev-call-1 model=jev-1.13.0 "
        "input_tokens=35569 output_tokens=3986 "
        "estimated_cost_usd=0.001493898 price_checked_on=2026-10-06\n"
    )
