from decimal import Decimal

import pytest
from pydantic import ValidationError

from jevpilot.ledger.models import CallRecord, CallUsage, Span
from jevpilot.llm.pricing import PRICES, estimate_cost_usd


def test_unknown_cost_and_usage_remain_nullable() -> None:
    # Given / When
    record = CallRecord(
        call_id="call", span_id="span", kind="llm", provider="custom",
        auth_mode="api", purpose="choose", request_model="unknown",
        outcome="failed", error_kind="timeout", latency_s=1,
    )
    # Then
    assert record.billed_usd is None
    assert record.api_estimate_usd is None
    assert record.api_equivalent_usd is None
    assert record.usage.model_dump() == {
        "input_tokens": None, "output_tokens": None,
        "cached_input_tokens": None, "reasoning_tokens": None,
    }
    assert record.model_dump(mode="json")["billed_usd"] is None


@pytest.mark.parametrize(
    ("auth_mode", "billed", "estimate", "equivalent"),
    [
        ("subscription", Decimal("1"), None, None),
        ("subscription", None, Decimal("1"), None),
        ("api", None, None, Decimal("1")),
    ],
)
def test_invalid_cost_bases_are_rejected(
    auth_mode: str, billed: Decimal | None, estimate: Decimal | None,
    equivalent: Decimal | None,
) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = CallRecord.model_validate({
            "call_id": "call", "span_id": "span", "kind": "llm", "provider": "test",
            "auth_mode": auth_mode, "purpose": "choose", "request_model": "test",
            "outcome": "ok", "latency_s": 1,
            "billed_usd": billed, "api_estimate_usd": estimate,
            "api_equivalent_usd": equivalent,
        })


@pytest.mark.parametrize(
    ("end_s", "outcome"),
    [(1, "ok"), (None, "ok"), (3, None)],
)
def test_invalid_span_intervals_are_rejected(end_s: float | None, outcome: str | None) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = Span.model_validate({
            "span_id": "s", "parent_id": None, "kind": "run", "label": "",
            "start_s": 2, "end_s": end_s, "outcome": outcome,
        })


def test_pricing_uses_cached_subset_without_double_counting_reasoning() -> None:
    # Given
    usage = CallUsage(
        input_tokens=1000, output_tokens=100, cached_input_tokens=200, reasoning_tokens=50,
    )
    # When
    cost = estimate_cost_usd("openai", "gpt-5.3-codex", usage)
    # Then
    assert cost == Decimal("0.002835")
    assert PRICES[("openai", "gpt-5.3-codex")].checked_on == "2026-10-23"


@pytest.mark.parametrize(
    "usage",
    [
        CallUsage(),
        CallUsage(input_tokens=100, output_tokens=1),
        CallUsage(input_tokens=100, cached_input_tokens=0),
        CallUsage(output_tokens=1, cached_input_tokens=0),
    ],
)
def test_incomplete_billable_usage_stays_unknown(usage: CallUsage) -> None:
    # Given / When
    cost = estimate_cost_usd("openai", "gpt-5.3-codex", usage)
    # Then
    assert cost is None


def test_jev_reuses_v1_estimator_with_free_unknown_output() -> None:
    # Given
    usage = CallUsage(input_tokens=35_569)
    # When
    cost = estimate_cost_usd("jev", "jev-1.13.0", usage)
    # Then
    assert cost == Decimal("0.001493898")
    assert estimate_cost_usd("jev", "jev-latest", usage) is None
    assert estimate_cost_usd("custom", "gpt-5.3-codex", usage) is None


def test_cached_tokens_cannot_exceed_total_input() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = CallUsage(input_tokens=1, cached_input_tokens=2)


def test_known_zero_usage_has_known_zero_estimate() -> None:
    # Given
    usage = CallUsage(input_tokens=0, output_tokens=0)
    # When
    cost = estimate_cost_usd("openai", "gpt-5.3-codex", usage)
    # Then
    assert cost == Decimal("0")
