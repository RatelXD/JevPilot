"""Hash run inputs and write benchmark artifacts."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from importlib import metadata
import hashlib
import json
from pathlib import Path
import subprocess
from typing import ClassVar
from uuid import uuid4

import anyio
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from benchmarks.tasks.run_settings import Options
from jevpilot.llm_provider import LLMProvider
from jevpilot.task import Budget

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


class RunArtifactInput(BaseModel):
    """Validated serializable inputs for one benchmark result artifact."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")

    options: Options
    llm_provider: LLMProvider
    llm_model: str
    jev_model: str | None
    jev_spend_reserved_usd: Decimal | None
    jev_spend_limit_usd: Decimal | None
    operation_threshold: float
    target_threshold: float
    budgets: Budget
    task_ids: tuple[str, ...]
    smoke_scheduled_runs: int
    smoke_completed_runs: int
    comparison_scheduled_runs: int
    comparison_completed_runs: int
    primary_scheduled_runs: int
    primary_completed_runs: int
    scheduled_runs: int
    completed_runs: int
    stop_reason: str | None
    codex_cli_version: str | None
    gateway_credit_limit: Decimal | None
    gateway_credit_balance_start: Decimal | None
    gateway_credit_balance_after_smoke: Decimal | None
    gateway_credit_balance_end: Decimal | None
    gateway_credit_balance_net_delta: Decimal | None
    gateway_credit_balance_error: str | None
    jev_spend_reserved_after_smoke_usd: Decimal | None
    jev_spend_remaining_after_smoke_usd: Decimal | None
    jev_usage: JsonValue
    smoke_summary: JsonValue | None
    summary_by_mode: JsonValue
    summary_by_task_and_mode: JsonValue
    runs: tuple[JsonValue, ...]


def dependency_versions() -> dict[str, str]:
    """Return versions of packages used by the benchmark artifact."""
    versions: dict[str, str] = {}
    for package in ("jevpilot", "pydantic", "playwright", "httpx2", "jsonschema"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


async def write_run_artifact(data: RunArtifactInput) -> Path:
    """Hash source/fixture inputs and persist one complete result document."""
    root = Path(__file__).resolve().parents[2]
    source_root = root / "src" / "jevpilot"
    fixture_root = Path(__file__).parent / "fixtures"
    source_hash = hashlib.sha256()
    source_paths = (
        *source_root.rglob("*.py"),
        *source_root.rglob("*.js"),
        *Path(__file__).parent.glob("run_*.py"),
        Path(__file__).with_name("tasks.py"),
    )
    for source in sorted(set(source_paths)):
        source_hash.update(source.relative_to(root).as_posix().encode())
        source_hash.update(source.read_bytes())
    source_fingerprint = source_hash.hexdigest()
    fixture_hash = hashlib.sha256()
    for fixture in sorted(path for path in fixture_root.rglob("*") if path.is_file()):
        fixture_hash.update(fixture.relative_to(root).as_posix().encode())
        fixture_hash.update(fixture.read_bytes())
    fixture_fingerprint = fixture_hash.hexdigest()
    config_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "task_id": data.options.task_id,
                "mode": data.options.mode,
                "repeat": data.options.repeat,
                "smoke_gate": data.options.smoke_gate,
                "llm_provider": data.llm_provider.value,
                "llm_model": data.llm_model,
                "jev_model": data.jev_model,
                "jev_spend_limit_usd": (
                    str(data.jev_spend_limit_usd)
                    if data.jev_spend_limit_usd is not None
                    else None
                ),
                "jev_reservation_usd": (
                    str(data.jev_spend_reserved_usd)
                    if data.jev_spend_reserved_usd is not None
                    else None
                ),
                "operation_threshold": data.operation_threshold,
                "target_threshold": data.target_threshold,
                "budgets": data.budgets.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    revision = await anyio.run_process(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=False,
        stderr=subprocess.DEVNULL,
    )
    code_revision = (
        revision.stdout.decode("ascii", errors="ignore").strip()
        if revision.returncode == 0
        else "unknown"
    )
    document = _JSON.validate_python(
        {
            "schema_version": 3,
            "created_at": datetime.now(UTC).isoformat(),
            "code_revision": code_revision,
            "source_sha256": source_fingerprint,
            "fixture_sha256": fixture_fingerprint,
            "config_sha256": config_fingerprint,
            "task_id": data.options.task_id,
            "task_ids": list(data.task_ids),
            "smoke_gate": data.options.smoke_gate,
            "smoke_scheduled_runs": data.smoke_scheduled_runs,
            "smoke_completed_runs": data.smoke_completed_runs,
            "comparison_scheduled_runs": data.comparison_scheduled_runs,
            "comparison_completed_runs": data.comparison_completed_runs,
            "primary_scheduled_runs": data.primary_scheduled_runs,
            "primary_completed_runs": data.primary_completed_runs,
            "scheduled_runs": data.scheduled_runs,
            "completed_runs": data.completed_runs,
            "not_run_count": data.scheduled_runs - data.completed_runs,
            "stop_reason": data.stop_reason,
            "requested_mode": data.options.mode,
            "requested_llm_provider": data.llm_provider.value,
            "requested_model": data.llm_model,
            "codex_cli_version": data.codex_cli_version,
            "gateway_test_credit_limit": (
                str(data.gateway_credit_limit)
                if data.gateway_credit_limit is not None
                else None
            ),
            "gateway_credit_balance_start": (
                str(data.gateway_credit_balance_start)
                if data.gateway_credit_balance_start is not None
                else None
            ),
            "gateway_credit_balance_after_smoke": (
                str(data.gateway_credit_balance_after_smoke)
                if data.gateway_credit_balance_after_smoke is not None
                else None
            ),
            "gateway_credit_balance_end": (
                str(data.gateway_credit_balance_end)
                if data.gateway_credit_balance_end is not None
                else None
            ),
            "gateway_credit_balance_net_delta": (
                str(data.gateway_credit_balance_net_delta)
                if data.gateway_credit_balance_net_delta is not None
                else None
            ),
            "gateway_credit_balance_error": data.gateway_credit_balance_error,
            "dependency_versions": dependency_versions(),
            "jev_spend_reserved_usd": (
                str(data.jev_spend_reserved_usd)
                if data.jev_spend_reserved_usd is not None
                else None
            ),
            "jev_spend_limit_usd": (
                str(data.jev_spend_limit_usd)
                if data.jev_spend_limit_usd is not None
                else None
            ),
            "jev_spend_reserved_after_smoke_usd": (
                str(data.jev_spend_reserved_after_smoke_usd)
                if data.jev_spend_reserved_after_smoke_usd is not None
                else None
            ),
            "jev_spend_remaining_after_smoke_usd": (
                str(data.jev_spend_remaining_after_smoke_usd)
                if data.jev_spend_remaining_after_smoke_usd is not None
                else None
            ),
            "jev_usage": data.jev_usage,
            "smoke_summary": data.smoke_summary,
            "summary": {
                "by_mode": data.summary_by_mode,
                "by_task_and_mode": data.summary_by_task_and_mode,
            },
            "runs": list(data.runs),
        }
    )
    output = Path("artifacts/runs")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{uuid4().hex}.json"
    _ = destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return destination
