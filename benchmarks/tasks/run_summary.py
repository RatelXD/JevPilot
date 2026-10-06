"""Aggregate task outcomes by benchmark mode and task."""

from __future__ import annotations

from decimal import Decimal
import statistics

from pydantic import JsonValue, TypeAdapter

from jevpilot.telemetry.models import CallEvent, ExecutionTrace

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def summarize_jev_usage(calls: tuple[CallEvent, ...]) -> JsonValue:
    """Aggregate usage returned for Jev requests in one JevPilot run."""
    jev_calls = tuple(
        call for call in calls if call.provider == "jev" and call.sent
    )
    input_tokens = sum(
        call.usage.input_tokens
        for call in jev_calls
        if call.usage is not None and call.usage.input_tokens is not None
    )
    output_tokens = sum(
        call.usage.output_tokens
        for call in jev_calls
        if call.usage is not None and call.usage.output_tokens is not None
    )
    incomplete_usage_requests = sum(
        call.usage is None
        or call.usage.input_tokens is None
        or call.usage.output_tokens is None
        for call in jev_calls
    )
    estimated_costs = tuple(call.estimated_cost_usd for call in jev_calls)
    known_estimated_costs = tuple(
        cost for cost in estimated_costs if cost is not None
    )
    observed_estimated_cost = sum(
        (Decimal(str(cost)) for cost in known_estimated_costs),
        Decimal(0),
    )
    return _JSON.validate_python(
        {
            "scope": "jevpilot_run",
            "request_count": len(jev_calls),
            "input_tokens": (
                input_tokens if incomplete_usage_requests == 0 else None
            ),
            "output_tokens": (
                output_tokens if incomplete_usage_requests == 0 else None
            ),
            "observed_input_tokens": input_tokens,
            "observed_output_tokens": output_tokens,
            "incomplete_usage_request_count": incomplete_usage_requests,
            "estimated_cost_usd": (
                float(observed_estimated_cost)
                if len(known_estimated_costs) == len(jev_calls)
                else None
            ),
            "observed_estimated_cost_usd": float(observed_estimated_cost),
            "unknown_estimated_cost_request_count": (
                len(jev_calls) - len(known_estimated_costs)
            ),
        }
    )


def summarize_runs(
    traces: tuple[ExecutionTrace, ...],
    task_ids: tuple[str, ...],
) -> tuple[JsonValue, JsonValue]:
    """Build mode and task summaries without converting unknown costs to zero."""
    by_mode: dict[str, JsonValue] = {}
    for mode in ("llm_only", "jev_hybrid"):
        arm = tuple(trace for trace in traces if trace.mode == mode)
        if not arm:
            continue
        times = tuple(
            trace.metrics.task_e2e_ms
            for trace in arm
            if trace.metrics.task_e2e_ms is not None
        )
        success_count = sum(trace.metrics.task_success for trace in arm)
        cost_values = tuple(trace.metrics.total_cost_usd for trace in arm)
        total_cost = (
            sum(value for value in cost_values if value is not None)
            if all(value is not None for value in cost_values)
            else None
        )
        gateway_credit_values = tuple(
            call.estimated_cost_credits
            for trace in arm
            for call in trace.calls
            if call.provider == "chosun_api_gateway" and call.sent
        )
        estimated_gateway_credits = (
            sum(
                (value for value in gateway_credit_values if value is not None),
                Decimal(0),
            )
            if gateway_credit_values
            and all(value is not None for value in gateway_credit_values)
            else None
        )
        by_mode[mode] = _JSON.validate_python(
            {
                "run_count": len(arm),
                "success_count": success_count,
                "failure_count": sum(
                    trace.terminal.status == "failed" for trace in arm
                ),
                "blocked_count": sum(
                    trace.terminal.status == "blocked" for trace in arm
                ),
                "timeout_count": sum(
                    trace.terminal.status == "timeout" for trace in arm
                ),
                "uncertain_count": sum(
                    trace.terminal.status == "uncertain_execution"
                    for trace in arm
                ),
                "task_success_rate": success_count / len(arm),
                "task_e2e_ms": list(times),
                "median_task_e2e_ms": statistics.median(times) if times else None,
                "successful_task_e2e_ms": [
                    trace.metrics.task_e2e_ms
                    for trace in arm
                    if trace.metrics.task_success
                ],
                "failed_task_e2e_ms": [
                    trace.metrics.task_e2e_ms
                    for trace in arm
                    if not trace.metrics.task_success
                ],
                "network_attempts": [
                    trace.metrics.network_attempts for trace in arm
                ],
                "subscription_invocations": [
                    trace.metrics.subscription_invocations for trace in arm
                ],
                "replans": sum(trace.metrics.replans for trace in arm),
                "stale_observations": sum(
                    trace.metrics.stale_observations for trace in arm
                ),
                "fallback_requests": sum(
                    call.purpose == "fallback"
                    for trace in arm
                    for call in trace.calls
                ),
                "attempts_by_role": {
                    role: {
                        "subscription_invocations": sum(
                            call.source == "subscription_cli"
                            and call.invoked
                            and call.role == role
                            for trace in arm
                            for call in trace.calls
                        ),
                        "observed_network_attempts": sum(
                            call.source == "network"
                            and call.sent
                            and call.role == role
                            for trace in arm
                            for call in trace.calls
                        ),
                    }
                    for role in ("planner", "selector", "text")
                },
                "unknown_usd_cost_calls": sum(
                    call.source in {"network", "subscription_cli"}
                    and call.billed_cost_usd is None
                    and call.estimated_cost_usd is None
                    for trace in arm
                    for call in trace.calls
                ),
                "unknown_gateway_credit_calls": sum(
                    call.provider == "chosun_api_gateway"
                    and call.sent
                    and call.estimated_cost_credits is None
                    for trace in arm
                    for call in trace.calls
                ),
                "estimated_gateway_credits": (
                    str(estimated_gateway_credits)
                    if estimated_gateway_credits is not None
                    else None
                ),
                "known_cost_usd": sum(
                    trace.metrics.known_cost_usd for trace in arm
                ),
                "total_cost_usd": total_cost,
                "cost_per_verified_success_usd": (
                    total_cost / success_count
                    if total_cost is not None and success_count > 0
                    else None
                ),
            }
        )

    by_task_and_mode: dict[str, JsonValue] = {}
    for task_id in task_ids:
        for mode in ("llm_only", "jev_hybrid"):
            arm = tuple(
                trace
                for trace in traces
                if trace.task_id == task_id and trace.mode == mode
            )
            if not arm:
                continue
            success_count = sum(trace.metrics.task_success for trace in arm)
            cost_values = tuple(trace.metrics.total_cost_usd for trace in arm)
            total_cost = (
                sum(value for value in cost_values if value is not None)
                if all(value is not None for value in cost_values)
                else None
            )
            gateway_credit_values = tuple(
                call.estimated_cost_credits
                for trace in arm
                for call in trace.calls
                if call.provider == "chosun_api_gateway" and call.sent
            )
            estimated_gateway_credits = (
                sum(
                    (value for value in gateway_credit_values if value is not None),
                    Decimal(0),
                )
                if gateway_credit_values
                and all(value is not None for value in gateway_credit_values)
                else None
            )
            by_task_and_mode[f"{task_id}:{mode}"] = _JSON.validate_python(
                {
                    "run_count": len(arm),
                    "success_count": success_count,
                    "failure_count": sum(
                        trace.terminal.status == "failed" for trace in arm
                    ),
                    "blocked_count": sum(
                        trace.terminal.status == "blocked" for trace in arm
                    ),
                    "timeout_count": sum(
                        trace.terminal.status == "timeout" for trace in arm
                    ),
                    "uncertain_count": sum(
                        trace.terminal.status == "uncertain_execution"
                        for trace in arm
                    ),
                    "task_success_rate": success_count / len(arm),
                    "task_e2e_ms": [
                        trace.metrics.task_e2e_ms for trace in arm
                    ],
                    "total_cost_usd": total_cost,
                    "estimated_gateway_credits": (
                        str(estimated_gateway_credits)
                        if estimated_gateway_credits is not None
                        else None
                    ),
                    "cost_per_verified_success_usd": (
                        total_cost / success_count
                        if total_cost is not None and success_count > 0
                        else None
                    ),
                }
            )
    return _JSON.validate_python(by_mode), _JSON.validate_python(by_task_and_mode)
