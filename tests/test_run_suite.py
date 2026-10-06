"""Subprocess contract tests for paid-runner safety gates."""

import os
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
import subprocess
import sys
from typing import Literal

import pytest

from benchmarks.tasks.run_settings import parse_options
from benchmarks.tasks.run_trial import run_budgets
from jevpilot.gateway_client import GatewayCreditBalance, GatewayModelClient
from jevpilot.llm_provider import LLMClientSelection, LLMProvider
from jevpilot.model_client import StructuredModelClient
from jevpilot.telemetry.models import ExecutionTrace, TerminalResult
from jevpilot.telemetry.recorder import JevSpendBudget, RunRecorder

from benchmarks.tasks.run_settings import (
    SettingsError,
    enforce_benchmark_gateway_credit_limit,
)
from benchmarks.tasks.run_artifact import RunArtifactInput
from benchmarks.tasks import run_suite

ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "benchmarks" / "tasks" / "run_suite.py"


@pytest.mark.parametrize(
    ("remaining", "should_reject"),
    [
        (Decimal("9999.99"), False),
        (Decimal("10000"), False),
        (Decimal("10000.01"), True),
    ],
)
def test_gateway_credit_ceiling_is_benchmark_only(
    remaining: Decimal,
    should_reject: bool,
) -> None:
    # Given
    limit = Decimal("10000")

    # When / Then
    if should_reject:
        with pytest.raises(SettingsError, match="benchmark-only"):
            enforce_benchmark_gateway_credit_limit(remaining, limit)
    else:
        enforce_benchmark_gateway_credit_limit(remaining, limit)


@pytest.mark.asyncio
async def test_smoke_and_comparison_share_test_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    for name, value in (
        ("JEVPILOT_LLM_MODEL", "gpt-6.1-sol"),
        ("JEVPILOT_JEV_API_KEY", "test-jev-key"),
        ("JEVPILOT_JEV_MODEL", "jev-latest"),
        ("JEVPILOT_JEV_OPERATION_THRESHOLD", "0.5"),
        ("JEVPILOT_JEV_TARGET_THRESHOLD", "0.5"),
        ("JEVPILOT_TEST_LLM_CREDIT_LIMIT", "10000"),
        ("JEVPILOT_TEST_JEV_SPEND_LIMIT_USD", "1"),
        ("JEVPILOT_TEST_JEV_RESERVATION_USD", "0.01"),
    ):
        monkeypatch.setenv(name, value)
    balance_values = iter(
        (Decimal("10000"), Decimal("9997"), Decimal("9972"))
    )
    jev_budgets: list[JevSpendBudget] = []
    artifact_inputs: list[RunArtifactInput] = []
    trial_phases: list[str] = []

    async def selected_provider() -> LLMClientSelection:
        client = GatewayModelClient(api_key="gateway-test-key")
        return LLMClientSelection(LLMProvider.CHOSUN_GATEWAY, client)

    async def account_balance(self: GatewayModelClient) -> GatewayCreditBalance:
        _ = self
        return GatewayCreditBalance.model_validate(
            {"total": {"remaining": next(balance_values)}}
        )

    def offline_fixture_server() -> AbstractContextManager[str]:
        return nullcontext("http://127.0.0.1:8765")

    async def offline_trial(
        *,
        origin: str,
        mode: Literal["llm_only", "jev_hybrid"],
        task_id: str,
        llm_client: StructuredModelClient,
        llm_provider: LLMProvider,
        llm_model: str,
        spend: JevSpendBudget | None,
        operation_threshold: float,
        target_threshold: float,
    ) -> ExecutionTrace:
        _ = (
            origin,
            llm_client,
            llm_provider,
            llm_model,
            operation_threshold,
            target_threshold,
        )
        trial_phases.append(f"{task_id}:{mode}")
        if mode == "jev_hybrid" and spend is not None:
            jev_budgets.append(spend)
            spend.reserve()
        recorder = RunRecorder(run_budgets())
        return recorder.trace(
            task_id=task_id,
            mode=mode,
            fixture_version="offline-smoke-plan",
            terminal=TerminalResult(
                status="succeeded",
                reason="offline_plan_test",
                ended_at=datetime.now(UTC),
            ),
        )

    async def capture_artifact(data: RunArtifactInput) -> Path:
        artifact_inputs.append(data)
        return Path("artifacts/runs/offline-smoke-plan.json")

    monkeypatch.setattr(run_suite, "select_llm_client", selected_provider)
    monkeypatch.setattr(GatewayModelClient, "credit_balance", account_balance)
    monkeypatch.setattr(run_suite, "fixture_server", offline_fixture_server)
    monkeypatch.setattr(run_suite, "run_trial", offline_trial)
    monkeypatch.setattr(run_suite, "write_run_artifact", capture_artifact)
    options = parse_options(
        ["--mode", "compare", "--task", "all", "--repeat", "5", "--live", "--smoke-gate"]
    )

    # When
    artifact = await run_suite.execute(options)

    # Then
    assert artifact == Path("artifacts/runs/offline-smoke-plan.json")
    assert trial_phases[:6] == [
        "T-01:llm_only",
        "T-01:jev_hybrid",
        "T-02:jev_hybrid",
        "T-02:llm_only",
        "T-03:llm_only",
        "T-03:jev_hybrid",
    ]
    assert len(trial_phases) == 56
    assert len(jev_budgets) == 28
    assert len({id(budget) for budget in jev_budgets}) == 1
    assert jev_budgets[0].reserved_usd == Decimal("0.28")
    assert len(artifact_inputs) == 1
    assert artifact_inputs[0].smoke_scheduled_runs == 6
    assert artifact_inputs[0].smoke_completed_runs == 6
    assert artifact_inputs[0].comparison_scheduled_runs == 50
    assert artifact_inputs[0].comparison_completed_runs == 50
    assert artifact_inputs[0].jev_spend_reserved_after_smoke_usd == Decimal("0.03")
    assert artifact_inputs[0].jev_spend_remaining_after_smoke_usd == Decimal("0.97")
    assert artifact_inputs[0].jev_usage == {
        "scope": "jevpilot_run",
        "request_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "observed_input_tokens": 0,
        "observed_output_tokens": 0,
        "incomplete_usage_request_count": 0,
        "estimated_cost_usd": 0.0,
        "observed_estimated_cost_usd": 0.0,
        "unknown_estimated_cost_request_count": 0,
    }
    assert artifact_inputs[0].gateway_credit_balance_after_smoke == Decimal("9997")


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            ("--mode", "llm_only", "--task", "T-00", "--repeat", "1", "--live"),
            "JEVPILOT_LLM_MODEL must be set",
        ),
        (
            ("--mode", "mock", "--task", "T-00", "--repeat", "1", "--live"),
            "usage: run_suite.py",
        ),
        (
            ("--mode", "compare", "--task", "all", "--repeat", "5", "--live"),
            "JEVPILOT_LLM_MODEL must be set",
        ),
    ],
)
def test_live_runner_rejects_missing_or_non_product_configuration_without_call(
    arguments: tuple[str, ...],
    expected: str,
) -> None:
    # Given
    environment = os.environ.copy()
    for name in (
        "JEVPILOT_LLM_MODEL",
        "JEVPILOT_LLM_PROVIDER",
        "JEVPILOT_LLM_API_KEY",
        "JEVPILOT_JEV_API_KEY",
        "JEVPILOT_JEV_MODEL",
        "JEVPILOT_TEST_LLM_CREDIT_LIMIT",
        "JEVPILOT_TEST_JEV_SPEND_LIMIT_USD",
        "JEVPILOT_TEST_JEV_RESERVATION_USD",
    ):
        _ = environment.pop(name, None)

    # When
    result = subprocess.run(
        (sys.executable, str(RUNNER), *arguments),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    # Then
    assert result.returncode == 2
    assert expected in result.stderr
    assert "Run artifacts written" not in result.stdout
