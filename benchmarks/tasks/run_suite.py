"""Orchestrate live benchmark trials and write one result artifact."""
# noqa: SIZE_OK — phased smoke/compare state machine shares one run-wide spend budget
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run:
#   uv run python benchmarks/tasks/run_suite.py --mode compare --task all --repeat 5 --live

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import os
import subprocess
import sys

import anyio
from pydantic import JsonValue, TypeAdapter

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.tasks.run_artifact import RunArtifactInput, write_run_artifact
from benchmarks.tasks.run_settings import (
    Options,
    SettingsError,
    enforce_benchmark_gateway_credit_limit,
    jev_spend_budget,
    jev_thresholds,
    llm_credit_limit,
    parse_options,
    required_setting,
)
from benchmarks.tasks.run_plan import build_trial_plan
from benchmarks.tasks.run_summary import summarize_jev_usage, summarize_runs
from benchmarks.tasks.run_trial import TrialSpec, run_budgets, run_trial
from benchmarks.tasks.tasks import fixture_server
from jevpilot.gateway_client import GatewayModelClient
from jevpilot.llm_provider import LLMProvider, select_llm_client
from jevpilot.model_client import ProviderError
from jevpilot.telemetry.models import ExecutionTrace

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def budget_stop_reason(trace: ExecutionTrace) -> str | None:
    """Stop a suite after either provider's test budget is exhausted."""
    if (
        trace.terminal.reason == "jev_spend_reservation"
        and trace.terminal.status == "blocked"
    ):
        return "jev_spend_reservation"
    if any(
        call.provider == "chosun_api_gateway" and call.status_code == 402
        for call in trace.calls
    ):
        return "gateway_credit_balance_exhausted"
    return None


async def execute(options: Options) -> Path:
    """Run selected real-provider trials under benchmark-only budgets."""
    llm_model = required_setting("JEVPILOT_LLM_MODEL")
    spend = (
        jev_spend_budget()
        if options.mode in {"jev_hybrid", "compare"}
        else None
    )
    operation_threshold, target_threshold = (
        jev_thresholds()
        if options.mode in {"jev_hybrid", "compare"}
        else (0.0, 0.0)
    )
    if options.mode in {"jev_hybrid", "compare"}:
        if not os.getenv("JEVPILOT_JEV_API_KEY", "").strip():
            raise SettingsError(
                "JEVPILOT_JEV_API_KEY must be injected in the process environment"
            )
        if not os.getenv("JEVPILOT_JEV_MODEL", "").strip():
            raise SettingsError("JEVPILOT_JEV_MODEL must be set in the process environment")
    llm_selection = await select_llm_client()
    task_ids = (
        ("T-01", "T-02", "T-03", "T-04", "T-05")
        if options.task_id == "all"
        else (options.task_id,)
    )
    trial_plan = build_trial_plan(options)
    smoke_specs = tuple(spec for spec in trial_plan if spec.phase == "smoke")
    comparison_specs = tuple(
        spec for spec in trial_plan if spec.phase == "comparison"
    )
    primary_specs = tuple(spec for spec in trial_plan if spec.phase == "primary")
    smoke_task_ids = ("T-01", "T-02", "T-03") if options.smoke_gate else ()
    scheduled_runs = len(trial_plan)
    runs: list[JsonValue] = []
    smoke_traces: list[ExecutionTrace] = []
    comparison_traces: list[ExecutionTrace] = []
    primary_traces: list[ExecutionTrace] = []
    stop_reason: str | None = None
    gateway_credit_limit: Decimal | None = None
    gateway_credit_balance_start: Decimal | None = None
    gateway_credit_balance_after_smoke: Decimal | None = None
    gateway_credit_balance_end: Decimal | None = None
    gateway_credit_balance_error: str | None = None
    jev_spend_reserved_after_smoke_usd: Decimal | None = None
    jev_spend_remaining_after_smoke_usd: Decimal | None = None

    async def run_one(origin: str, spec: TrialSpec) -> ExecutionTrace:
        trace = await run_trial(
            origin=origin,
            mode=spec.mode,
            task_id=spec.task_id,
            llm_client=llm_selection.client,
            llm_provider=llm_selection.provider,
            llm_model=llm_model,
            spend=spend,
            operation_threshold=operation_threshold,
            target_threshold=target_threshold,
        )
        runs.append(
            _JSON.validate_python(
                {
                    "phase": spec.phase,
                    "repeat_index": spec.repeat_index,
                    "task_id": spec.task_id,
                    "trace": trace.model_dump(mode="json"),
                }
            )
        )
        return trace

    try:
        if isinstance(llm_selection.client, GatewayModelClient):
            gateway_credit_limit = llm_credit_limit()
            balance = await llm_selection.client.credit_balance()
            enforce_benchmark_gateway_credit_limit(
                balance.total.remaining,
                gateway_credit_limit,
            )
            gateway_credit_balance_start = balance.total.remaining
        with fixture_server() as origin:
            for spec in smoke_specs:
                trace = await run_one(origin, spec)
                smoke_traces.append(trace)
                stop_reason = budget_stop_reason(trace)
                if stop_reason is not None:
                    break
            if options.smoke_gate:
                jev_spend_reserved_after_smoke_usd = (
                    spend.reserved_usd if spend is not None else None
                )
                jev_spend_remaining_after_smoke_usd = (
                    spend.limit_usd - spend.reserved_usd
                    if spend is not None
                    else None
                )
                if (
                    isinstance(llm_selection.client, GatewayModelClient)
                    and gateway_credit_balance_start is not None
                ):
                    try:
                        gateway_credit_balance_after_smoke = (
                            await llm_selection.client.credit_balance()
                        ).total.remaining
                    except ProviderError as exc:
                        gateway_credit_balance_error = exc.kind
                        if stop_reason is None:
                            stop_reason = "gateway_balance_unavailable_after_smoke"
                    else:
                        credits_spent = max(
                            Decimal(0),
                            gateway_credit_balance_start
                            - gateway_credit_balance_after_smoke,
                        )
                        remaining_limit = (
                            gateway_credit_limit - credits_spent
                            if gateway_credit_limit is not None
                            else Decimal(0)
                        )
                        if (
                            gateway_credit_balance_after_smoke <= 0
                            or remaining_limit <= 0
                            or gateway_credit_balance_after_smoke > remaining_limit
                        ):
                            if stop_reason is None:
                                stop_reason = "gateway_credit_test_limit_reached"
                if (
                    stop_reason is None
                    and spend is not None
                    and spend.reserved_usd >= spend.limit_usd
                ):
                    stop_reason = "jev_spend_reservation"
                smoke_succeeded = (
                    len(smoke_traces) == len(smoke_specs)
                    and all(trace.terminal.status == "succeeded" for trace in smoke_traces)
                )
                if stop_reason is None and not smoke_succeeded:
                    stop_reason = "smoke_gate_failed"

            if stop_reason is None:
                for spec in comparison_specs:
                    trace = await run_one(origin, spec)
                    comparison_traces.append(trace)
                    stop_reason = budget_stop_reason(trace)
                    if stop_reason is not None:
                        break
            if stop_reason is None:
                for spec in primary_specs:
                    trace = await run_one(origin, spec)
                    primary_traces.append(trace)
                    stop_reason = budget_stop_reason(trace)
                    if stop_reason is not None:
                        break
    finally:
        if (
            isinstance(llm_selection.client, GatewayModelClient)
            and gateway_credit_balance_start is not None
        ):
            try:
                gateway_credit_balance_end = (
                    await llm_selection.client.credit_balance()
                ).total.remaining
            except ProviderError as exc:
                gateway_credit_balance_error = exc.kind
        await llm_selection.aclose()

    main_traces = (
        tuple(comparison_traces)
        if options.mode == "compare"
        else tuple(primary_traces)
    )
    all_traces = (*smoke_traces, *comparison_traces, *primary_traces)
    jev_usage = summarize_jev_usage(
        tuple(call for trace in all_traces for call in trace.calls)
    )
    summary_by_mode, summary_by_task_and_mode = summarize_runs(
        main_traces,
        task_ids,
    )
    smoke_summary: JsonValue | None = None
    if smoke_specs:
        smoke_by_mode, smoke_by_task = summarize_runs(
            tuple(smoke_traces),
            smoke_task_ids,
        )
        smoke_summary = _JSON.validate_python(
            {"by_mode": smoke_by_mode, "by_task_and_mode": smoke_by_task}
        )
    codex_cli_version: str | None = None
    if llm_selection.provider is LLMProvider.CODEX_SUBSCRIPTION:
        codex_version = await anyio.run_process(
            ("codex", "--version"),
            check=False,
            stderr=subprocess.DEVNULL,
        )
        codex_cli_version = (
            codex_version.stdout.decode("utf-8", errors="replace").strip()
            if codex_version.returncode == 0
            else "unknown"
        )
    gateway_credit_balance_net_delta = (
        gateway_credit_balance_start - gateway_credit_balance_end
        if gateway_credit_balance_start is not None
        and gateway_credit_balance_end is not None
        else None
    )
    return await write_run_artifact(
        RunArtifactInput(
            options=options,
            llm_provider=llm_selection.provider,
            llm_model=llm_model,
            jev_model=(
                os.getenv("JEVPILOT_JEV_MODEL")
                if spend is not None
                else None
            ),
            jev_spend_reserved_usd=spend.reserved_usd if spend else None,
            jev_spend_limit_usd=spend.limit_usd if spend else None,
            operation_threshold=operation_threshold,
            target_threshold=target_threshold,
            budgets=run_budgets(),
            task_ids=task_ids,
            smoke_scheduled_runs=len(smoke_specs),
            smoke_completed_runs=len(smoke_traces),
            comparison_scheduled_runs=len(comparison_specs),
            comparison_completed_runs=len(comparison_traces),
            primary_scheduled_runs=len(primary_specs),
            primary_completed_runs=len(primary_traces),
            scheduled_runs=scheduled_runs,
            completed_runs=len(runs),
            stop_reason=stop_reason,
            codex_cli_version=codex_cli_version,
            gateway_credit_limit=gateway_credit_limit,
            gateway_credit_balance_start=gateway_credit_balance_start,
            gateway_credit_balance_after_smoke=gateway_credit_balance_after_smoke,
            gateway_credit_balance_end=gateway_credit_balance_end,
            gateway_credit_balance_net_delta=gateway_credit_balance_net_delta,
            gateway_credit_balance_error=gateway_credit_balance_error,
            jev_spend_reserved_after_smoke_usd=jev_spend_reserved_after_smoke_usd,
            jev_spend_remaining_after_smoke_usd=jev_spend_remaining_after_smoke_usd,
            jev_usage=jev_usage,
            smoke_summary=smoke_summary,
            summary_by_mode=summary_by_mode,
            summary_by_task_and_mode=summary_by_task_and_mode,
            runs=tuple(runs),
        )
    )


async def main() -> int:
    """CLI entrypoint for the benchmark runner."""
    try:
        options = parse_options(sys.argv[1:])
        path = await execute(options)
    except SettingsError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(
            f"provider error: kind={exc.kind} status={exc.status_code}",
            file=sys.stderr,
        )
        return 1
    print(f"Run artifacts written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(anyio.run(main))
