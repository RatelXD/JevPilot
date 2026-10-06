"""Published TypeSafe Jev token prices for per-request cost estimates."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from jevpilot.telemetry.models import CallUsage

_PRICE_CHECKED_ON: Final = "2026-10-06"
_PRICE_SOURCE: Final = (
    "https://docs.typesafe.ai/models.md: $0.042 per million input tokens; output free"
)
_INPUT_COST_PER_TOKEN: Final[dict[str, Decimal]] = {
    "jev-1.13.0": Decimal("0.000000042"),
}


def estimate_jev_cost_usd(model: str, usage: CallUsage | None) -> Decimal | None:
    """Estimate one response cost from its resolved model and input usage."""
    input_cost = _INPUT_COST_PER_TOKEN.get(model)
    if input_cost is None or usage is None or usage.input_tokens is None:
        return None
    return Decimal(usage.input_tokens) * input_cost


def jev_price_source() -> str:
    """Return the official pricing source recorded with each estimate."""
    return _PRICE_SOURCE


def jev_price_checked_on() -> str:
    """Return the date the official rate was checked."""
    return _PRICE_CHECKED_ON
