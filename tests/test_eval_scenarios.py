from datetime import UTC, datetime
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from threading import Thread
from typing import Literal

import anyio
import pytest

from benchmarks.browser_use_eval import run_eval
from benchmarks.browser_use_eval.oracle import normalize
from benchmarks.browser_use_eval.run_eval import ModelSpec, kill_owned_group, recovery_counts, redact, schedule
from benchmarks.browser_use_eval.scenarios import (
    HOLDOUT, HoldoutMismatch, campaign_date, check_holdout, holdout_sha256, load_scenarios,
)


def test_sealed_holdout_when_one_byte_copy_changes_refuses(tmp_path: Path) -> None:
    # Given
    copy = tmp_path / "holdout.json"
    original = HOLDOUT.read_bytes()
    copy.write_bytes(original + b" ")
    expected = holdout_sha256()
    # When / Then
    with pytest.raises(HoldoutMismatch):
        load_scenarios("holdout", expected_hash=expected, date="2030-01-01", holdout_path=copy)
    assert HOLDOUT.read_bytes() == original


def test_cli_when_holdout_copy_mismatches_refuses_before_profile_or_browser(tmp_path: Path) -> None:
    # Given
    copy = tmp_path / "holdout.json"
    copy.write_bytes(HOLDOUT.read_bytes() + b" ")
    output = tmp_path / "no-trials"
    # When
    result = subprocess.run(
        [sys.executable, str(Path(run_eval.__file__)), str(output),
         "--holdout-file", str(copy), "--holdout-sha256", holdout_sha256()],
        capture_output=True, text=True, timeout=15, check=False,
    )
    # Then: no trial, profile, browser or site was touched.
    assert result.returncode == 2
    assert not output.exists()


@pytest.mark.parametrize("instant,expected", [
    (datetime(2030, 1, 1, 14, 59, tzinfo=UTC), "2030-01-02"),
    (datetime(2030, 1, 1, 15, 0, tzinfo=UTC), "2030-01-03"),
])
def test_h4_date_when_injected_clock_crosses_seoul_midnight(instant: datetime, expected: str) -> None:
    # Given / When
    result = campaign_date(lambda: instant)
    # Then
    assert result == expected


def test_campaign_when_resumed_uses_saved_date_not_new_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no credentials or real git commands are involved.
    monkeypatch.setattr(run_eval, "selected_model", lambda _: ModelSpec(model_id="offline"))
    monkeypatch.setattr(run_eval, "git_bytes", lambda *args: b"head" if args[0] == "rev-parse" else b"diff")
    monkeypatch.setattr(run_eval.subprocess, "check_output", lambda _: b"Chrome test")
    monkeypatch.setattr(run_eval, "campaign_date", lambda: campaign_date(
        lambda: datetime(2030, 1, 1, tzinfo=UTC),
    ))
    args = run_eval.argparse.Namespace(
        output=tmp_path / "campaign", split="holdout", repeat=1, arms="A,B,C,D",
        scenarios="H-4", resume=False, fixture_base_url=None,
        holdout_file=HOLDOUT, holdout_sha256=holdout_sha256(),
    )
    initial = run_eval.prepare_campaign(args, tmp_path)
    monkeypatch.setattr(run_eval, "campaign_date", lambda: pytest.fail("resume must not consult clock"))
    args.resume = True
    # When
    resumed = run_eval.prepare_campaign(args, tmp_path)
    # Then
    assert resumed.h4_date == initial.h4_date == "2030-01-02"
    assert resumed.scenarios == initial.scenarios
    assert initial.h4_date in resumed.scenarios[0].goal
    manifest = json.loads((args.output / "manifest.json").read_text())
    assert manifest["holdout_sha256"] == holdout_sha256()


def test_dev_when_fixture_origin_is_configured_and_order_rotates() -> None:
    # Given
    scenarios = load_scenarios(
        "dev", expected_hash=holdout_sha256(), date="2030-01-01",
        fixture_base_url="http://127.0.0.1:8775",
    )
    # When
    runs = schedule(scenarios[:1], ["A", "B", "C", "D"], 4)
    # Then: IDs and URLs are machine-consumed contracts, not prose pins.
    assert [s.scenario_id for s in scenarios] == ["F-02", "F-03", "F-04", "F-05", "D-1", "D-2", "D-3"]
    assert scenarios[2].start_url == "http://127.0.0.1:8775/autocomplete.html"
    assert [r.arm for r in runs] == list("ABCDBCDACDABDABC")
    assert normalize(" ＭＩＴ　\nLicense ") == "mit license"
    check_holdout(HOLDOUT.with_suffix(".sha256").read_text().strip())


def test_resume_when_configuration_changes_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(run_eval, "selected_model", lambda _: ModelSpec(model_id="offline"))
    monkeypatch.setattr(run_eval, "git_bytes", lambda *args: b"head")
    monkeypatch.setattr(run_eval.subprocess, "check_output", lambda _: b"Chrome test")
    monkeypatch.setattr(run_eval, "campaign_date", lambda: "2030-01-01")
    args = run_eval.argparse.Namespace(
        output=tmp_path / "campaign", split="dev", repeat=1, arms="A,B,C,D",
        scenarios="F-04", resume=False, fixture_base_url=None,
        holdout_file=HOLDOUT, holdout_sha256=holdout_sha256(),
    )
    run_eval.prepare_campaign(args, tmp_path)
    args.resume, args.repeat = True, 2
    # When / Then
    with pytest.raises(ValueError, match="repeat"):
        run_eval.prepare_campaign(args, tmp_path)


def test_deadline_cleanup_when_child_is_waiting_kills_owned_group(tmp_path: Path) -> None:
    # Given: readiness is signalled before cleanup, with no sleeps or polling.
    command = [
        sys.executable, "-c",
        "import os,sys; print(os.getpid(), flush=True); sys.stdin.read()",
    ]
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "HOME": str(tmp_path)}, start_new_session=True,
    ) as process:
        assert process.stdout is not None
        with selectors.DefaultSelector() as readiness:
            readiness.register(process.stdout, selectors.EVENT_READ)
            assert readiness.select(timeout=5), "child did not signal readiness"
            assert int(process.stdout.readline()) == process.pid
        # When
        killed = kill_owned_group(process.pid, tmp_path)
        code = process.wait(timeout=5)
        # Then
        assert killed == [process.pid]
        assert code == -9


def test_recovery_when_opening_and_outcome_events_are_not_attempts() -> None:
    # Given / When
    result = recovery_counts([
        {"event": "opening_evidence"},
        {"failure": "transient", "action": "reobserve"},
        {"event": "execution", "outcome": "executed"},
        {"failure": "transient", "action": "reobserve"},
        {"failure": "repeated", "action": "replan"},
        {"event": "run_outcome", "status": "verified"},
    ])
    # Then
    assert result == (3, 2)


def test_metrics_when_restoring_calls_rebinds_original_span(tmp_path: Path) -> None:
    # Given: a persisted call belongs to a now-closed worker ledger.
    from benchmarks.browser_use_eval.run_eval import WorkerResult, trial_metrics
    from benchmarks.browser_use_eval.summarize import PlannedRun
    from jevpilot.ledger.models import CallRecord

    record = CallRecord(
        call_id="original-call", span_id="original-worker-span", kind="llm",
        provider="chatgpt_subscription", auth_mode="subscription", purpose="brief",
        request_model="unknown", outcome="failed", error_kind="provider_unavailable",
        latency_s=1,
    )
    (tmp_path / "calls.jsonl").write_text(record.model_dump_json() + "\n")
    run = PlannedRun(arm="A", scenario="F-04", repeat=0, split="dev")
    result = WorkerResult(status="failed", oracle="failed", oracle_reason="provider_unavailable")
    # When
    trial = trial_metrics(run, tmp_path, result, 2, True)
    # Then
    assert trial.llm_calls == 1 and trial.jev_calls == 0
    assert trial.tokens["input_tokens"].unknown_count == 1
    assert trial.costs["api_equivalent_usd"].unknown_count == 1
    assert trial.costs["billed_usd"].known_count == 0


def test_redaction_when_evidence_path_contains_task_preserves_path() -> None:
    # Given
    path = "/evidence/task-19/trial.json"
    # When
    result = redact(path + " sk-synthetic12345678 Bearer synthetic-token")
    # Then
    assert result == path + " [REDACTED] [REDACTED]"


def test_subscription_when_worker_refreshes_only_writable_private_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: distinguish JevPilot's valid profile from the unrelated old profile.
    real_home = tmp_path / "real-home"
    source = real_home / ".jevpilot/codex/auth.json"
    source.parent.mkdir(parents=True)
    source.write_text("current-private-auth")
    old = real_home / ".codex/auth.json"
    old.parent.mkdir()
    old.write_text("expired-auth")
    monkeypatch.setattr(run_eval, "git_bytes", lambda *args: b"offline")
    args = run_eval.argparse.Namespace(
        output=tmp_path / "campaign", split="dev", repeat=1, arms="A", scenarios="F-04",
        resume=False, fixture_base_url=None, holdout_file=HOLDOUT, holdout_sha256=holdout_sha256(),
    )
    monkeypatch.setattr(run_eval, "selected_model", lambda _: ModelSpec(model_id="offline"))
    monkeypatch.setattr(run_eval.subprocess, "check_output", lambda _: b"Chrome test")
    campaign = run_eval.prepare_campaign(args, real_home)
    original_popen = subprocess.Popen
    observed_homes: list[Path] = []

    def spawn(command: list[str], **kwargs):
        observed_homes.append(Path(kwargs["env"]["HOME"]))
        # Substitute only model/browser work; retain actual process and cleanup.
        script = """
import json, os, stat, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
home = Path.home()
auth = home / '.jevpilot/codex/auth.json'
assert str(auth.parent) == os.environ['CODEX_HOME']
assert auth.read_text() == 'current-private-auth'
assert stat.S_IMODE(auth.stat().st_mode) == 0o600
assert not (home / '.codex/auth.json').exists()
auth.write_text('refreshed-copy')
(Path(spec['output']) / 'result.json').write_text(json.dumps({
    'status': 'completed', 'oracle': 'passed', 'oracle_reason': 'offline'
}))
"""
        return original_popen([sys.executable, "-c", script, command[-1]], **kwargs)

    monkeypatch.setattr(run_eval.subprocess, "Popen", spawn)
    # When
    trial = run_eval.run_one(campaign.scheduled[0], campaign, args.output, real_home)
    # Then
    assert trial.oracle == "passed"
    assert source.read_text() == "current-private-auth" and old.read_text() == "expired-auth"
    assert len(observed_homes) == 1 and not observed_homes[0].exists()
    cleanup = json.loads(next(args.output.glob("*/attempt-*/cleanup.json")).read_text())
    assert cleanup["temporary_home_removed"] and cleanup["worker_exit"] == 0


@pytest.mark.browser
@pytest.mark.parametrize("arm", ["A", "B", "C", "D"])
async def test_browser_locale_when_real_arm_launches_is_korean(
    arm: Literal["A", "B", "C", "D"], locale_home: Path, locale_fixture_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: only decisions are offline; the arm's Chrome and capture are real.
    from benchmarks.browser_use_eval import arms
    from benchmarks.browser_use_eval.scenarios import Scenario
    from jevpilot.browser.port import BrowserAction, BrowserConfig
    from jevpilot.llm.types import ModelRef

    tmp_path = locale_home
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    for name, value in {
        "HOME": str(tmp_path), "TMPDIR": str(temporary),
        "XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_STATE_HOME": str(tmp_path / "state"),
        "JEVPILOT_JEV_API_KEY": "offline-never-sent",
    }.items():
        monkeypatch.setenv(name, value)

    class ObserveOnlyAgent:
        def __init__(self, **kwargs):
            pass

        async def run(self, **kwargs):
            return self

        def is_done(self) -> bool:
            return False

    async def observe_task(goal, *, browser, start_url, **kwargs):
        await browser.start(BrowserConfig(headless=True))
        try:
            await browser.execute(BrowserAction(kind="navigate", url=start_url), await browser.observe())
        finally:
            await browser.close()
        return SimpleNamespace(status="observed", extraction=None)

    monkeypatch.setattr(arms, "Agent", ObserveOnlyAgent)
    monkeypatch.setattr(arms, "run_task", observe_task)
    ledger = arms.make_ledger(tmp_path)
    # When
    with anyio.fail_after(30):
        result = await arms.execute(
            arm, Scenario(scenario_id="F-04", goal="Observe settings", split="dev",
                          start_url=locale_fixture_url),
            ModelRef("chatgpt_subscription", "subscription", "offline"),
            tmp_path, ledger,
        )
    # Then
    assert result.page.language == result.page.intl_locale == "ko-KR"
    assert result.page.languages == ("ko-KR", "ko")
    assert result.page.viewport == ((1120, 780) if arm == "D" else (1280, 800))
    # D uses its actual native interpreter/worker and stops at the passing first
    # checkpoint, before any Jev or subscription request.
    assert ledger.calls == []


@pytest.fixture
def locale_fixture_url() -> Iterator[str]:
    handler = partial(
        SimpleHTTPRequestHandler,
        directory=str(Path(__file__).parents[1] / "benchmarks/tasks/fixtures"),
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/items/aurora.html"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.fixture
def locale_home() -> Iterator[Path]:
    # Chrome's Unix sockets require a short, unique profile root.
    with TemporaryDirectory(prefix="bu-locale-test-") as directory:
        yield Path(directory)
    assert not Path(directory).exists()
    print(json.dumps({"locale_test_home": directory, "removed": True}))
