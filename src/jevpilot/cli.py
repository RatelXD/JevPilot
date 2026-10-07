"""User-facing CLI for bounded same-origin browser tasks."""

import hashlib
import json
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import anyio
from playwright.async_api import async_playwright
from pydantic import ValidationError

from jevpilot.cli_config import (
    USAGE,
    CliError,
    UserRunConfig,
    create_user_config,
    parse_options,
)
from jevpilot.codex_client import CodexClient, CodexModelClientAdapter
from jevpilot.decision import JevDecisionProvider, LLMDecisionProvider
from jevpilot.interactive import run_shell
from jevpilot.llm_provider import LLMProvider, select_llm_client
from jevpilot.model_catalog import list_chatgpt_models
from jevpilot.model_client import ProviderError, StructuredModelClient
from jevpilot.model_selection import (
    ModelSelectionError,
    ModelSelection,
    load_model_selection,
    save_model_selection,
)
from jevpilot.planner import LLMPlannerProvider
from jevpilot.privacy import PrivacyPolicy
from jevpilot.runtime import HybridBrowserAgent
from jevpilot.telemetry.call_reporting import report_jev_call
from jevpilot.telemetry.models import ExecutionTrace
from jevpilot.telemetry.recorder import RunRecorder
from jevpilot.text import LLMTextValueProvider
from jevpilot.user_task import build_user_task


def _jev_thresholds() -> tuple[float, float]:
    thresholds: list[float] = []
    for key in (
        "JEVPILOT_JEV_OPERATION_THRESHOLD",
        "JEVPILOT_JEV_TARGET_THRESHOLD",
    ):
        try:
            value = Decimal(_required_setting(key))
        except InvalidOperation:
            raise CliError(f"{key} must be between 0 and 1") from None
        if not value.is_finite() or not 0 <= value <= 1:
            raise CliError(f"{key} must be between 0 and 1")
        thresholds.append(float(value))
    return thresholds[0], thresholds[1]


def _required_setting(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise CliError(f"{key} must be set in the process environment")
    return value


async def _run(config: UserRunConfig) -> ExecutionTrace:
    task, binding = build_user_task(config)
    configured_provider = os.getenv("JEVPILOT_LLM_PROVIDER", "").strip()
    saved_model = (
        load_model_selection()
        if configured_provider in {"", "chatgpt-subscription"}
        else None
    )
    if saved_model is None:
        llm_model = _required_setting("JEVPILOT_LLM_MODEL")
        provider = None
    else:
        llm_model = saved_model.model_id
        provider = LLMProvider.CODEX_SUBSCRIPTION
    selection = await select_llm_client(provider=provider)
    selector: JevDecisionProvider | LLMDecisionProvider | None = None
    try:
        if saved_model is not None:
            available_models = await list_chatgpt_models()
            if saved_model.model_id not in {model.model_id for model in available_models}:
                raise CliError(
                    "the saved ChatGPT model is no longer available; run jevpilot shell and /model"
                )
            selected_catalog_model = next(
                model for model in available_models
                if model.model_id == saved_model.model_id
            )
            if saved_model.fast and not selected_catalog_model.supports_fast:
                raise CliError("the selected ChatGPT model does not support Fast mode")
            if saved_model.reasoning_effort not in selected_catalog_model.reasoning_efforts:
                raise CliError("the selected ChatGPT model does not support its saved reasoning level")
        thresholds = _jev_thresholds() if config.mode == "jev_hybrid" else (0.0, 0.0)
        if isinstance(selection.client, CodexClient):
            reasoning_effort = saved_model.reasoning_effort if saved_model else "medium"
            fast = saved_model.fast if saved_model else False
            model_client: StructuredModelClient = CodexModelClientAdapter(
                selection.client,
                reasoning_effort=reasoning_effort,
                fast=fast,
            )
        else:
            reasoning_effort = "medium"
            fast = False
            model_client = selection.client
        planner = LLMPlannerProvider(model_client, model=llm_model)
        text = LLMTextValueProvider(model_client, model=llm_model)
        selector = (
            JevDecisionProvider.from_env()
            if config.mode == "jev_hybrid"
            else LLMDecisionProvider(model_client, model=llm_model)
        )
        run_configuration = {
            "user_run": config.model_dump(mode="json"),
            "task": task.model_dump(mode="json"),
            "llm_provider": selection.provider.value,
            "llm_model": llm_model,
            "llm_model_ref": f"{selection.provider.value}/{llm_model}",
            "llm_reasoning_effort": reasoning_effort,
            "llm_fast": fast,
            "llm_service_tier": "fast" if fast else None,
            "llm_supported_reasoning_efforts": (
                saved_model.supported_reasoning_efforts
                if saved_model is not None
                else None
            ),
            "codex_preferences": (
                {
                    "reasoning_effort": reasoning_effort,
                    "fast": fast,
                }
                if saved_model is not None
                else None
            ),
            "jev_model": (
                os.getenv("JEVPILOT_JEV_MODEL")
                if config.mode == "jev_hybrid"
                else None
            ),
            "operation_threshold": thresholds[0],
            "target_threshold": thresholds[1],
        }
        config_sha256 = hashlib.sha256(
            json.dumps(
                run_configuration,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        recorder = RunRecorder(
            task.budgets,
            config_sha256=config_sha256,
            on_call_event=report_jev_call,
        )
        agent = HybridBrowserAgent(
            planner,
            selector,
            text=text,
            privacy=PrivacyPolicy(),
            operation_threshold=thresholds[0],
            target_threshold=thresholds[1],
        )
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                trace = await agent.run(
                    browser,
                    task,
                    binding=binding,
                    recorder=recorder,
                )
            finally:
                await browser.close()
        return trace
    finally:
        if isinstance(selector, JevDecisionProvider):
            await selector.aclose()
        await selection.aclose()


async def async_main() -> int:
    try:
        options = parse_options(sys.argv[1:])
        if options.help:
            print(USAGE)
            return 0
        config = create_user_config(options)
        trace = await _run(config)
    except CliError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ModelSelectionError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ValidationError:
        print("configuration error: URL, origin or completion condition is invalid", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(
            f"provider error: kind={exc.kind} status={exc.status_code}",
            file=sys.stderr,
        )
        return 1
    artifact_dir = Path("artifacts/runs")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact = artifact_dir / f"{trace.run_id}.json"
    summary = {"trace": trace.model_dump(mode="json")}
    _ = artifact.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    llm_calls = tuple(
        call
        for call in trace.calls
        if call.provider in {"chosun_api_gateway", "chatgpt-subscription"}
    )
    llm_providers = tuple(dict.fromkeys(call.provider for call in llm_calls))
    llm_model_refs = tuple(
        dict.fromkeys(
            f"{call.provider}/{call.request_model}"
            for call in llm_calls
        )
    )
    input_tokens = sum(
        call.usage.input_tokens
        for call in llm_calls
        if call.usage is not None and call.usage.input_tokens is not None
    )
    output_tokens = sum(
        call.usage.output_tokens
        for call in llm_calls
        if call.usage is not None and call.usage.output_tokens is not None
    )
    incomplete_usage_calls = sum(
        call.usage is None
        or call.usage.input_tokens is None
        or call.usage.output_tokens is None
        for call in llm_calls
    )
    gateway_calls = tuple(
        call for call in llm_calls if call.provider == "chosun_api_gateway"
    )
    gateway_credit_estimates = tuple(
        call.estimated_cost_credits for call in gateway_calls
    )
    estimated_gateway_credits = (
        sum(
            (value for value in gateway_credit_estimates if value is not None),
            Decimal(0),
        )
        if gateway_credit_estimates
        and all(value is not None for value in gateway_credit_estimates)
        else None
    )
    gateway_credit_display = (
        str(estimated_gateway_credits)
        if estimated_gateway_credits is not None
        else "unknown"
        if gateway_calls
        else "not_applicable"
    )
    print(
        f"status={trace.terminal.status}",
        f"reason={trace.terminal.reason}",
        f"artifact={artifact}",
        f"llm_provider={','.join(llm_providers) or 'none'}",
        f"llm_model_ref={','.join(llm_model_refs) or 'none'}",
        f"llm_requests={len(llm_calls)}",
        f"llm_usage_tokens_known={input_tokens + output_tokens}",
        f"llm_usage_incomplete_calls={incomplete_usage_calls}",
        f"gateway_credits_estimated={gateway_credit_display}",
        "llm_billed_usd=unknown",
    )
    return 0 if trace.terminal.status == "succeeded" else 1


def entrypoint() -> None:
    if sys.argv[1:] == ["shell"]:
        raise SystemExit(run_shell())
    raise SystemExit(anyio.run(async_main))
