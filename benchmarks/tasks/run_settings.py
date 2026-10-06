"""Options and test-only environment settings for the live task runner."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from typing import Final, Literal, final

from jevpilot.telemetry.recorder import JevSpendBudget

Mode = Literal["llm_only", "jev_hybrid", "compare"]
_MODE_BY_VALUE: Final[dict[str, Mode]] = {
    "llm_only": "llm_only",
    "jev_hybrid": "jev_hybrid",
    "compare": "compare",
}
_USAGE: Final = (
    "usage: run_suite.py --mode llm_only|jev_hybrid|compare "
    "[--task T-00|T-01|T-02|T-03|T-04|T-05|all] --repeat N "
    "[--smoke-gate] --live"
)


@final
@dataclass(frozen=True, slots=True)
class Options:
    mode: Mode
    task_id: str
    repeat: int
    live: bool
    smoke_gate: bool = False


class SettingsError(ValueError):
    """Required live configuration is missing or invalid."""


def parse_options(arguments: list[str]) -> Options:
    """Parse the benchmark runner's explicit live-only arguments."""
    values: dict[str, str] = {}
    live = False
    smoke_gate = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--live":
            live = True
            index += 1
            continue
        if argument == "--smoke-gate":
            if smoke_gate:
                raise SettingsError(_USAGE)
            smoke_gate = True
            index += 1
            continue
        if argument not in {"--mode", "--task", "--repeat"}:
            raise SettingsError(_USAGE)
        if index + 1 >= len(arguments):
            raise SettingsError(_USAGE)
        key = argument.removeprefix("--")
        if key in values:
            raise SettingsError(_USAGE)
        values[key] = arguments[index + 1]
        index += 2
    if not live:
        raise SettingsError("live API execution requires the explicit --live option")
    selected_mode = _MODE_BY_VALUE.get(values.get("mode", ""))
    if selected_mode is None:
        raise SettingsError(_USAGE)
    task_id = values.get("task", "all")
    if task_id not in {"T-00", "T-01", "T-02", "T-03", "T-04", "T-05", "all"}:
        raise SettingsError("select T-00 through T-05 or all tasks")
    if smoke_gate and (selected_mode != "compare" or task_id != "all"):
        raise SettingsError("--smoke-gate requires --mode compare --task all")
    raw_repeat = values.get("repeat", "1")
    try:
        repeat = int(raw_repeat)
    except ValueError:
        raise SettingsError("--repeat must be a positive integer") from None
    if repeat < 1:
        raise SettingsError("--repeat must be a positive integer")
    return Options(
        mode=selected_mode,
        task_id=task_id,
        repeat=repeat,
        live=live,
        smoke_gate=smoke_gate,
    )


def required_setting(name: str) -> str:
    """Read one required benchmark environment value without revealing it."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SettingsError(f"{name} must be set in the process environment")
    return value


def jev_spend_budget() -> JevSpendBudget:
    """Create the benchmark-only Jev reservation budget."""
    try:
        limit = Decimal(required_setting("JEVPILOT_TEST_JEV_SPEND_LIMIT_USD"))
        reserve = Decimal(required_setting("JEVPILOT_TEST_JEV_RESERVATION_USD"))
        return JevSpendBudget(limit, reserve)
    except InvalidOperation:
        raise SettingsError("Jev spend limit settings must be decimal amounts") from None


def llm_credit_limit() -> Decimal:
    """Return the benchmark-only Gateway account-balance ceiling."""
    try:
        limit = Decimal(required_setting("JEVPILOT_TEST_LLM_CREDIT_LIMIT"))
    except InvalidOperation:
        raise SettingsError("LLM test credit limit must be a decimal amount") from None
    if not limit.is_finite() or limit <= 0:
        raise SettingsError("LLM test credit limit must be positive")
    return limit


def enforce_benchmark_gateway_credit_limit(
    remaining_credits: Decimal, limit_credits: Decimal
) -> None:
    """Require the account's server-side balance to enforce the test ceiling."""
    if remaining_credits > limit_credits:
        raise SettingsError("Gateway balance exceeds the benchmark-only credit limit")


def jev_thresholds() -> tuple[float, float]:
    """Parse validated Jev thresholds for benchmark comparisons."""
    values: list[float] = []
    for name in (
        "JEVPILOT_JEV_OPERATION_THRESHOLD",
        "JEVPILOT_JEV_TARGET_THRESHOLD",
    ):
        try:
            value = Decimal(required_setting(name))
        except InvalidOperation:
            raise SettingsError(f"{name} must be a decimal between 0 and 1") from None
        if not value.is_finite() or not 0 <= value <= 1:
            raise SettingsError(f"{name} must be between 0 and 1")
        values.append(float(value))
    return values[0], values[1]
