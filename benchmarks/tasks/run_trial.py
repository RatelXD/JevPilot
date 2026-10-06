"""Execute one benchmark task and collect its privacy-safe trace."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Literal, final

from playwright.async_api import async_playwright
from pydantic import JsonValue, TypeAdapter

from benchmarks.tasks.tasks import (
    autocomplete_task,
    catalog_task,
    filter_task,
    form_task,
    multi_filter_task,
    search_task,
)
from jevpilot.decision import JevDecisionProvider, LLMDecisionProvider
from jevpilot.llm_provider import LLMProvider
from jevpilot.model_client import StructuredModelClient
from jevpilot.planner import LLMPlannerProvider
from jevpilot.privacy import PrivacyPolicy
from jevpilot.runtime import HybridBrowserAgent
from jevpilot.task import Budget
from jevpilot.telemetry.models import ExecutionTrace
from jevpilot.telemetry.call_reporting import report_jev_call
from jevpilot.telemetry.recorder import JevSpendBudget, RunRecorder
from jevpilot.text import LLMTextValueProvider

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
ProductMode = Literal["llm_only", "jev_hybrid"]
TrialPhase = Literal["smoke", "comparison", "primary"]


@final
@dataclass(frozen=True, slots=True)
class TrialSpec:
    """One planned trial and its artifact phase label."""

    phase: TrialPhase
    repeat_index: int
    task_id: str
    mode: ProductMode


def run_budgets() -> Budget:
    """Create identical wall, action, and model budgets for each trial."""
    return Budget(
        max_steps=12,
        max_mutations=6,
        max_replans=2,
        max_model_attempts=12,
        max_attempts_per_call=2,
        max_stale_observations=3,
        max_waits=2,
        max_wait_ms=2_000,
        wall_time_ms=60_000,
    )


async def run_trial(
    *,
    origin: str,
    mode: ProductMode,
    task_id: str,
    llm_client: StructuredModelClient,
    llm_provider: LLMProvider,
    llm_model: str,
    spend: JevSpendBudget | None,
    operation_threshold: float,
    target_threshold: float,
) -> ExecutionTrace:
    """Run one fixture/browser trial with the selected product mode."""
    budgets = run_budgets()
    task_factory = {
        "T-00": catalog_task,
        "T-01": search_task,
        "T-02": filter_task,
        "T-03": form_task,
        "T-04": autocomplete_task,
        "T-05": multi_filter_task,
    }[task_id]
    task, binding = task_factory(origin, mode, budgets=budgets)
    planner = LLMPlannerProvider(llm_client, model=llm_model)
    text = LLMTextValueProvider(llm_client, model=llm_model)
    if mode == "jev_hybrid":
        selector = JevDecisionProvider.from_env()
    else:
        selector = LLMDecisionProvider(llm_client, model=llm_model)
    run_configuration = _JSON.validate_python(
        {
            "task": task.model_dump(mode="json"),
            "mode": mode,
            "llm_provider": llm_provider.value,
            "llm_model": llm_model,
            "jev_model": os.getenv("JEVPILOT_JEV_MODEL") if mode == "jev_hybrid" else None,
            "operation_threshold": operation_threshold,
            "target_threshold": target_threshold,
            "budgets": budgets.model_dump(mode="json"),
            "jev_spend_limit_usd": str(spend.limit_usd) if spend else None,
            "jev_reservation_usd": (
                str(spend.reservation_per_attempt_usd) if spend else None
            ),
        }
    )
    config_sha256 = hashlib.sha256(
        json.dumps(
            run_configuration,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    recorder = RunRecorder(
        budgets,
        reserve_paid_selector=spend.reserve if mode == "jev_hybrid" and spend else None,
        config_sha256=config_sha256,
        on_call_event=report_jev_call,
    )
    agent = HybridBrowserAgent(
        planner,
        selector,
        text=text,
        privacy=PrivacyPolicy(),
        operation_threshold=operation_threshold,
        target_threshold=target_threshold,
    )
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch()
        try:
            return await agent.run(browser, task, binding=binding, recorder=recorder)
        finally:
            await browser.close()
            if isinstance(selector, JevDecisionProvider):
                await selector.aclose()
