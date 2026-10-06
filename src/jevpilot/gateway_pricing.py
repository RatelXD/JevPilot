"""Documented Chosun Gateway token-credit estimates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final, final

if TYPE_CHECKING:
    from jevpilot.telemetry.models import CallUsage

_PRICE_CHECKED_ON: Final = "2026-09-30"
_PRICE_SOURCE: Final = (
    "Chosun Gateway model-credit rates; cached input charged at full rate"
)


@final
@dataclass(frozen=True, slots=True)
class GatewayCreditRate:
    """Input/output credits per thousand tokens."""

    input_per_1k: Decimal
    output_per_1k: Decimal


def _rate(input_per_1k: str, output_per_1k: str) -> GatewayCreditRate:
    return GatewayCreditRate(Decimal(input_per_1k), Decimal(output_per_1k))


_MODEL_RATES: Final[dict[str, GatewayCreditRate]] = {
    "gpt-6.1-sol": _rate("2", "10"),
    "gpt-6-astra": _rate("10", "50"),
    "gpt-6-sol": _rate("2", "10"),
    "gpt-6-luna": _rate("0.1", "0.5"),
    "gpt-5.6-sol": _rate("4", "20"),
    "gpt-5.6-terra": _rate("2", "12"),
    "gpt-5.6-luna": _rate("0.2", "1.2"),
    "gpt-5.5": _rate("5", "30"),
    "claude-fable-5-1": _rate("10", "50"),
    "claude-fable-5": _rate("10", "50"),
    "claude-opus-5-5": _rate("4", "20"),
    "claude-opus-5": _rate("5", "25"),
    "claude-opus-4-8": _rate("5", "25"),
    "claude-sonnet-5-5": _rate("2", "10"),
    "claude-sonnet-5": _rate("2", "10"),
    "claude-haiku-4-5-20251001": _rate("1", "5"),
    "gemini-3.1-pro-preview": _rate("2", "12"),
    "gemini-3.8-flash": _rate("0.75", "3.75"),
    "gemini-3.7-flash": _rate("0.75", "3.75"),
    "gemini-3.6-flash": _rate("0.75", "3.75"),
    "gemini-3.5-flash": _rate("1.5", "9"),
    "gemini-3.5-flash-lite": _rate("0.3", "2.5"),
    "google/gemma-4-31B-it": _rate("0.13", "0.38"),
    "muse-spark-1.3": _rate("1.25", "4.25"),
    "grok-4.7": _rate("2", "6"),
    "grok-4.6": _rate("2", "6"),
    "grok-4.5": _rate("2", "6"),
    "grok-4-1-fast": _rate("0.2", "0.5"),
    "solar-pro4": _rate("0.3", "1.2"),
    "solar-mini4": _rate("0.1", "0.4"),
    "qwen3.8-max": _rate("2", "6"),
    "qwen3.8-flash": _rate("0.15", "0.47"),
    "deepseek-v4.1-flash": _rate("0.3", "1.2"),
    "deepseek-v4-pro": _rate("2.4", "4.8"),
    "deepseek-v4-flash": _rate("0.2", "0.4"),
    "kimi-k3": _rate("3", "15"),
    "glm-5.2": _rate("1.4", "4.4"),
    "glm-5.3": _rate("1.4", "4.4"),
    "glm-5.3-flash": _rate("0.15", "0.5"),
    "qwen3.7-max": _rate("2.5", "7.5"),
    "qwen3.7-plus": _rate("0.4", "1.6"),
    "seed-2-0-pro-260328": _rate("0.5", "3"),
    "seed-2-0-lite-260428": _rate("0.25", "2"),
    "sonar-pro": _rate("3", "15"),
    "sonar-reasoning-pro": _rate("2", "8"),
}


def estimate_gateway_credits(
    model: str, usage: CallUsage | None
) -> Decimal | None:
    """Estimate credits from published rates; return unknown for missing data."""
    rate = _MODEL_RATES.get(model)
    if (
        rate is None
        or usage is None
        or usage.input_tokens is None
        or usage.output_tokens is None
    ):
        return None
    input_credits = Decimal(usage.input_tokens) * rate.input_per_1k
    output_credits = Decimal(usage.output_tokens) * rate.output_per_1k
    return (input_credits + output_credits) / Decimal(1000)


def gateway_credit_price_source() -> str:
    """Return the published source label for trace metadata."""
    return _PRICE_SOURCE


def gateway_credit_price_checked_on() -> str:
    """Return the model-credit card date used by the estimate."""
    return _PRICE_CHECKED_ON
