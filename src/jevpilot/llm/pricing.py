"""Standard token rates verified on official pricing pages, never invoices."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from jevpilot.jev_pricing import estimate_jev_cost_usd
from jevpilot.telemetry.models import CallUsage as LegacyUsage

if TYPE_CHECKING:
    from jevpilot.ledger.models import CallUsage


@dataclass(frozen=True, slots=True)
class Price:
    input_per_m: Decimal
    output_per_m: Decimal
    cached_input_per_m: Decimal | None
    source_url: str
    checked_on: str


# Only exact, verified model IDs are priced. Aliases and other providers stay unknown.
PRICES: Final[dict[tuple[str, str], Price]] = {
    ("jev", "jev-1.13.0"): Price(
        Decimal("0.042"), Decimal("0"), Decimal("0.042"),
        "https://docs.typesafe.ai/models.md", "2026-10-23",
    ),
    ("jev", "jev-1.14.0"): Price(
        Decimal("0.042"), Decimal("0"), Decimal("0.042"),
        "https://docs.typesafe.ai/models.md", "2026-10-24",
    ),
    ("openai", "gpt-5.3-codex"): Price(
        Decimal("1.75"), Decimal("14"), Decimal("0.175"),
        "https://developers.openai.com/api/docs/pricing", "2026-10-23",
    ),
}


def estimate_cost_usd(
    provider_family: str, model_id: str, usage: CallUsage,
) -> Decimal | None:
    """Return an estimate only when all price-relevant usage is known."""
    price = PRICES.get((provider_family, model_id))
    if price is None:
        return None
    if provider_family == "jev":
        return estimate_jev_cost_usd(
            model_id, LegacyUsage(input_tokens=usage.input_tokens),
        )
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = usage.cached_input_tokens
    if cached is None:
        if price.cached_input_per_m != price.input_per_m and usage.input_tokens != 0:
            return None
        cached = 0
    if cached and price.cached_input_per_m is None:
        return None
    cached_rate = price.cached_input_per_m or Decimal("0")
    return (
        Decimal(usage.input_tokens - cached) * price.input_per_m
        + Decimal(cached) * cached_rate
        + Decimal(usage.output_tokens) * price.output_per_m
    ) / Decimal(1_000_000)
