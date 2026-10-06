from decimal import Decimal

from jevpilot.jev_pricing import estimate_jev_cost_usd
from jevpilot.telemetry.models import CallUsage


def test_jev_estimate_matches_reported_input_usage_cost() -> None:
    # Given
    usage = CallUsage(input_tokens=35_569, output_tokens=3_986)

    # When
    estimate = estimate_jev_cost_usd("jev-1.13.0", usage)

    # Then
    assert estimate == Decimal("0.001493898")


def test_jev_estimate_requires_a_known_response_model_and_input_count() -> None:
    # Given / When
    unknown_model = estimate_jev_cost_usd(
        "jev-next", CallUsage(input_tokens=10, output_tokens=2)
    )
    missing_input = estimate_jev_cost_usd(
        "jev-1.13.0", CallUsage(input_tokens=None, output_tokens=2)
    )

    # Then
    assert unknown_model is None
    assert missing_input is None
