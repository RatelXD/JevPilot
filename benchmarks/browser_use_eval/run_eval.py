"""Serial, resumable four-arm campaigns with hard process-group deadlines."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Literal

# Support the acceptance command as well as python -m.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import anyio
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from benchmarks.browser_use_eval.scenarios import (
    DEFAULT_FIXTURE_URL, HOLDOUT, HoldoutMismatch, Scenario, campaign_date,
    check_holdout, load_scenarios,
)
from benchmarks.browser_use_eval.oracle import PageEvidence
from benchmarks.browser_use_eval.summarize import Measure, PlannedRun, Trial, write_report

Arm = Literal["A", "B", "C", "D"]


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    provider: Literal["chatgpt_subscription"] = "chatgpt_subscription"
    auth_mode: Literal["subscription"] = "subscription"
    model_id: str
    reasoning_effort: str | None = None
    fast: bool = False
    base_url: str | None = None


class Campaign(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    split: Literal["dev", "holdout"]
    repeat: int
    arms: list[Arm]
    scenarios: list[Scenario]
    scheduled: list[PlannedRun]
    holdout_sha256: str
    h4_date: str
    fixture_base_url: str
    model: ModelSpec
    head: str
    diff_sha256: str
    policy_version: str = "bu-1"
    chrome: str
    real_home_before: dict[str, str | None]


class WorkerSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    arm: Arm
    scenario: Scenario
    model: ModelSpec
    output: Path


class WorkerResult(BaseModel):
    status: str
    oracle: Literal["passed", "failed", "blocked", "unverifiable"]
    oracle_reason: str
    answer: str = ""
    page: PageEvidence | None = None
    extraction_verified: bool | None = None
    error_kind: str | None = None


def schedule(scenarios: list[Scenario], arms: list[Arm], repeat: int) -> list[PlannedRun]:
    return [
        PlannedRun(arm=arm, scenario=scenario.scenario_id, repeat=number, split=scenario.split)
        for number in range(repeat)
        for scenario in scenarios
        for arm in (arms[number % len(arms):] + arms[:number % len(arms)])
    ]


def home_hashes(home: Path) -> dict[str, str | None]:
    names = [
        ".codex/auth.json", ".codex/config.toml", ".jevpilot/profile.json",
        ".jevpilot/settings.json", ".jevpilot/credentials.json",
        ".jevpilot/codex/auth.json", ".jevpilot/codex/config.toml",
    ]
    return {
        name: hashlib.sha256((home / name).read_bytes()).hexdigest()
        if (home / name).is_file() else None for name in names
    }


def selected_model(home: Path) -> ModelSpec:
    profile = home / ".jevpilot/profile.json"
    if profile.exists():
        data = json.loads(profile.read_bytes())
        return ModelSpec.model_validate(data["model"])
    settings = home / ".jevpilot/settings.json"
    if settings.exists():
        data = json.loads(settings.read_bytes())
        return ModelSpec(
            model_id=data["model_id"], reasoning_effort=data.get("reasoning_effort"),
            fast=data.get("fast", False),
        )
    raise RuntimeError("No selected subscription profile; select a Codex model before evaluation")


def git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], env={**os.environ, "GIT_MASTER": "1"},
    )


def prepare_campaign(args: argparse.Namespace, real_home: Path) -> Campaign:
    expected = args.holdout_sha256 or HOLDOUT.with_suffix(".sha256").read_text().strip()
    check_holdout(expected, args.holdout_file)
    manifest_path = args.output / "manifest.json"
    if args.resume:
        campaign = Campaign.model_validate_json(manifest_path.read_bytes())
        check_holdout(campaign.holdout_sha256, args.holdout_file)
        if expected != campaign.holdout_sha256:
            raise HoldoutMismatch("holdout seal differs from campaign manifest")
        comparisons = {
            "split": args.split, "repeat": args.repeat,
            "fixture_base_url": args.fixture_base_url,
            "arms": args.arms.split(",") if args.arms else None,
        }
        for key, value in comparisons.items():
            if value is not None and value != getattr(campaign, key):
                raise ValueError(f"resume configuration mismatch: {key}")
        if args.scenarios and args.scenarios.split(",") != [s.scenario_id for s in campaign.scenarios]:
            raise ValueError("resume configuration mismatch: scenarios")
        return campaign
    if manifest_path.exists():
        raise FileExistsError("Campaign exists; use --resume")
    split = args.split or "dev"
    repeat = args.repeat if args.repeat is not None else 5
    arms = TypeAdapter(list[Arm]).validate_python((args.arms or "A,B,C,D").split(","))
    if repeat < 1 or len(set(arms)) != len(arms) or not arms:
        raise ValueError("repeat must be positive; arms must be unique and nonempty")
    date = campaign_date()
    fixture_url = args.fixture_base_url or DEFAULT_FIXTURE_URL
    scenarios = load_scenarios(
        split, expected_hash=expected, date=date,
        fixture_base_url=fixture_url, holdout_path=args.holdout_file,
    )
    if args.scenarios:
        ids = args.scenarios.split(",")
        by_id = {s.scenario_id: s for s in scenarios}
        if len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError("unknown or repeated scenario selection")
        scenarios = [by_id[name] for name in ids]
    campaign = Campaign(
        split=split, repeat=repeat, arms=arms, scenarios=scenarios,
        scheduled=schedule(scenarios, arms, repeat), holdout_sha256=expected,
        h4_date=date, fixture_base_url=fixture_url, model=selected_model(real_home),
        head=git_bytes("rev-parse", "HEAD").decode().strip(),
        diff_sha256=hashlib.sha256(git_bytes("diff", "--binary")).hexdigest(),
        chrome=subprocess.check_output(["/usr/bin/google-chrome", "--version"]).decode().strip(),
        real_home_before=home_hashes(real_home),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(campaign.model_dump_json(indent=2) + "\n")
    return campaign


def kill_owned_group(pid: int, home: Path) -> list[int]:
    """Kill group members and detached descendants carrying this isolated HOME."""
    owned: list[tuple[int, int]] = []
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        candidate = int(entry.name)
        try:
            in_group = os.getpgid(candidate) == pid
            in_home = b"HOME=" + os.fsencode(home) + b"\0" in (entry / "environ").read_bytes()
            if in_group or in_home:
                descriptor = libc.syscall(434, candidate, 0)  # Linux x86_64 pidfd_open
                if descriptor < 0:
                    raise ProcessLookupError(candidate)
                owned.append((candidate, descriptor))
        except (ProcessLookupError, FileNotFoundError, PermissionError):
            continue
    try:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for candidate, descriptor in owned:
            try:
                os.kill(candidate, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not select.select([descriptor], [], [], 15)[0]:
                raise TimeoutError("Owned process did not exit")
    finally:
        for _, descriptor in owned:
            os.close(descriptor)
    return [candidate for candidate, _ in owned]


def redact(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z0-9])(?:sk-|AIza)[A-Za-z0-9_-]{8,}|Bearer\s+\S+",
        "[REDACTED]", text,
    )


def trial_metrics(run: PlannedRun, attempt: Path, status: WorkerResult, elapsed: float, complete: bool) -> Trial:
    from jevpilot.ledger.models import CallRecord
    from jevpilot.ledger.recorder import RunLedger

    ledger = RunLedger()
    path = attempt / "calls.jsonl"
    calls = [CallRecord.model_validate_json(line) for line in path.read_text().splitlines()] if path.exists() else []
    for call in calls:
        ledger.record_call(call.model_copy(update={"span_id": ledger.root_span_id}))
    summary = ledger.summary()
    measures: dict[str, Measure] = {}
    for name in (
        "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens",
        "billed_usd", "api_estimate_usd", "api_equivalent_usd",
    ):
        values = [getattr(part, name) for part in summary.by_kind.values()]
        measures[name] = Measure(
            known_sum=sum(float(v.known_sum) for v in values),
            known_count=sum(v.known_count for v in values),
            unknown_count=sum(v.unknown_count for v in values),
        )
    event_file = attempt / "ledger/events.jsonl"
    events = [json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
    routes = [event["data"] for event in events if event["type"] == "route"]
    recovery = [event["data"] for event in events if event["type"] == "recovery"]
    # A recovery succeeds when execution resumes or the run verifies before the
    # next failure. Opening-evidence and run-outcome events are not attempts.
    recovery_count, recovery_successes = recovery_counts(recovery)
    return Trial(
        **run.model_dump(), status=status.status, oracle=status.oracle, oracle_reason=status.oracle_reason,
        error_kind=status.error_kind,
        elapsed_s=elapsed, llm_calls=sum(c.kind == "llm" for c in calls),
        jev_calls=sum(c.kind == "jev" for c in calls),
        tokens={k: v for k, v in measures.items() if k.endswith("_tokens")},
        costs={k: v for k, v in measures.items() if k.endswith("_usd")},
        recoveries=recovery_count, recovery_successes=recovery_successes,
        llm_reasons=[r["reason"] for r in routes if str(r.get("action", "")).startswith("llm_")],
        accounting_complete=complete,
        browser_language=status.page.language if status.page else "",
        browser_languages=list(status.page.languages) if status.page else [],
        intl_locale=status.page.intl_locale if status.page else "",
        viewport=status.page.viewport if status.page else None,
    )


def recovery_counts(events: list[dict[str, JsonValue]]) -> tuple[int, int]:
    attempts, successes, pending = 0, 0, False
    for event in events:
        if "failure" in event:
            attempts += 1
            pending = True
        elif pending and (
            event.get("event") == "execution" and event.get("outcome") == "executed"
            or event.get("event") == "run_outcome" and event.get("status") == "verified"
        ):
            successes += 1
            pending = False
    return attempts, successes


def run_one(run: PlannedRun, campaign: Campaign, output: Path, real_home: Path) -> Trial:
    scenario = next(s for s in campaign.scenarios if s.scenario_id == run.scenario)
    directory = output / run.run_id
    directory.mkdir(exist_ok=True)
    attempt = directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:02}"
    attempt.mkdir()
    spec = WorkerSpec(arm=run.arm, scenario=scenario, model=campaign.model, output=attempt)
    spec_path = attempt / "spec.json"
    spec_path.write_text(spec.model_dump_json(indent=2))
    (attempt / "provenance.json").write_text(json.dumps({
        "head": git_bytes("rev-parse", "HEAD").decode().strip(),
        "diff_sha256": hashlib.sha256(git_bytes("diff", "--binary")).hexdigest(),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }, indent=2) + "\n")
    started = time.monotonic()
    timed_out = False
    with TemporaryDirectory(prefix="bu-eval-") as temporary:
        home = Path(temporary)
        codex_home = home / ".jevpilot/codex"
        codex_home.mkdir(mode=0o700, parents=True)
        (home / "tmp").mkdir()
        source = real_home / ".jevpilot/codex/auth.json"
        shutil.copyfile(source, codex_home / "auth.json")
        (codex_home / "auth.json").chmod(0o600)
        environment = {
            **os.environ, "HOME": temporary, "CODEX_HOME": str(codex_home),
            "TMPDIR": str(home / "tmp"), "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_CACHE_HOME": str(home / "cache"), "XDG_DATA_HOME": str(home / "data"),
            "XDG_STATE_HOME": str(home / "state"),
            "JEVPILOT_DISABLE_MODEL_FALLBACK": "1",
            "ANONYMIZED_TELEMETRY": "false",
        }
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(spec_path)],
            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            try:
                raw, _ = process.communicate(timeout=scenario.deadline_s)
            except subprocess.TimeoutExpired as error:
                timed_out = True
                (attempt / "deadline-output.txt").write_text(redact(
                    (error.output or b"").decode(errors="replace"),
                ))
                killed = kill_owned_group(process.pid, home)
                raw, _ = process.communicate(timeout=20)
            else:
                killed = kill_owned_group(process.pid, home)
        finally:
            if process.poll() is None:
                kill_owned_group(process.pid, home)
                process.wait(timeout=20)
        (attempt / "worker.txt").write_text(redact(raw.decode(errors="replace")))
        result_file = attempt / "result.json"
        result = WorkerResult.model_validate_json(result_file.read_bytes()) if result_file.exists() else WorkerResult(
            status="timeout" if timed_out else "failed", oracle="failed",
            oracle_reason="deadline" if timed_out else "worker exited without result",
        )
        cleanup: dict[str, JsonValue] = {
            "worker_pid": process.pid, "worker_exit": process.returncode, "deadline_s": scenario.deadline_s,
            "deadline_reached": timed_out, "killed_pids": list[JsonValue](killed),
            "temporary_home": temporary, "profile": str(home / "tmp/chrome-profile"),
        }
    cleanup["temporary_home_removed"] = not home.exists()
    (attempt / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
    trial = trial_metrics(run, attempt, result, time.monotonic() - started, not timed_out)
    target = attempt / "provider-limited.json" if result.status == "provider_limited" else directory / "trial.json"
    target.write_text(trial.model_dump_json(indent=2) + "\n")
    return trial


async def worker(spec: WorkerSpec) -> int:
    from benchmarks.browser_use_eval.arms import execute, make_ledger
    from benchmarks.browser_use_eval.oracle import Verdict, grade
    from jevpilot.llm.errors import LLMError
    from jevpilot.llm.types import ModelRef

    ledger = make_ledger(spec.output)
    try:
        outcome = await execute(
            spec.arm, spec.scenario, ModelRef(**spec.model.model_dump()), spec.output, ledger,
        )
        verdict = (
            Verdict("unverifiable", "agent_provider_failure")
            if outcome.error_kind == "agent_provider_failure"
            else grade(spec.scenario.scenario_id, outcome.answer, outcome.page)
        )
        limited = next((
            call.error_kind for call in ledger.calls
            if call.error_kind in {"rate_limited", "quota_exhausted"}
        ), None)
        result = WorkerResult(
            status="provider_limited" if limited else outcome.status,
            oracle="unverifiable" if limited or outcome.error_kind == "agent_provider_failure" else verdict.status,
            oracle_reason=limited or outcome.error_kind or verdict.reason,
            answer=outcome.answer, page=outcome.page,
            extraction_verified=outcome.extraction_verified, error_kind=outcome.error_kind,
        )
    except LLMError as error:
        result = WorkerResult(
            status="provider_limited" if error.kind in {"rate_limited", "quota_exhausted"} else "failed",
            oracle="unverifiable" if error.kind in {"rate_limited", "quota_exhausted"} else "failed",
            oracle_reason=error.kind,
            error_kind=error.kind,
        )
    except Exception as error:  # noqa: BROAD_EXCEPT_OK - durable worker boundary, failure remains failure
        import traceback
        print(json.dumps({
            "error_type": type(error).__name__,
            "frames": [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
                       for frame in traceback.extract_tb(error.__traceback__)],
        }), flush=True)
        result = WorkerResult(status="failed", oracle="failed", oracle_reason=type(error).__name__)
    (spec.output / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    print(json.dumps({"status": result.status, "oracle": result.oracle}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--split", choices=("dev", "holdout"))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--arms")
    parser.add_argument("--scenarios")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixture-base-url")
    parser.add_argument("--holdout-file", type=Path, default=HOLDOUT)
    parser.add_argument("--holdout-sha256")
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        spec = WorkerSpec.model_validate_json(args.worker.read_bytes())
        return anyio.run(worker, spec)
    if args.output is None:
        parser.error("OUTPUT_DIR is required")
    args.output = args.output.resolve()
    real_home = Path.home()
    campaign = prepare_campaign(args, real_home)
    if git_bytes("rev-parse", "HEAD").decode().strip() != campaign.head:
        raise ValueError("checkout HEAD changed since campaign creation")
    before = home_hashes(real_home)
    try:
        for run in campaign.scheduled:
            check_holdout(campaign.holdout_sha256, args.holdout_file)
            if (args.output / run.run_id / "trial.json").exists():
                continue
            trial = run_one(run, campaign, args.output, real_home)
            print(trial.model_dump_json(), flush=True)
            if trial.status == "provider_limited":
                break
        trials = [Trial.model_validate_json(path.read_bytes()) for path in args.output.glob("*/trial.json")]
        write_report(args.output, trials, campaign.scheduled)
    finally:
        after = home_hashes(real_home)
        (args.output / f"home-receipt-{time.time_ns()}.json").write_text(json.dumps({
            "before": before, "after": after, "unchanged": before == after,
        }, indent=2) + "\n")
        if before != after:
            raise RuntimeError("Real HOME authentication/profile/settings bytes changed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HoldoutMismatch, ValueError, FileExistsError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from None
