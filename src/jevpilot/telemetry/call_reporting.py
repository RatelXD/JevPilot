"""Live, privacy-safe usage reporting for individual Jev requests."""

from __future__ import annotations

from jevpilot.telemetry.models import CallEvent


def report_jev_call(event: CallEvent) -> None:
    """Print usage and the token-price estimate as each Jev response arrives."""
    if event.provider != "jev" or not event.sent:
        return
    usage = event.usage
    print(
        "jev_request",
        f"call_id={event.call_id}",
        f"model={event.response_model or event.request_model}",
        f"input_tokens={usage.input_tokens if usage and usage.input_tokens is not None else 'unknown'}",
        f"output_tokens={usage.output_tokens if usage and usage.output_tokens is not None else 'unknown'}",
        f"estimated_cost_usd={event.estimated_cost_usd if event.estimated_cost_usd is not None else 'unknown'}",
        f"price_checked_on={event.price_checked_on or 'unknown'}",
        flush=True,
    )
