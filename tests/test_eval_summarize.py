from pathlib import Path

import pytest

from benchmarks.browser_use_eval.summarize import (
    Measure, PlannedRun, Trial, rate, summarize, write_report,
)
from benchmarks.browser_use_eval.oracle import PageEvidence, grade


def test_blocked_when_oracle_observes_denial_excludes_both_denominators() -> None:
    # Given: all five verifier cases are crossed with all four real arm identities.
    pages = [
        PageEvidence("", "Access denied"), PageEvidence("", "403 Forbidden"),
        PageEvidence("", "403 Forbidden"), PageEvidence("", "접근이 차단되었습니다."),
        PageEvidence("", "429 Too Many Requests"),
    ]
    planned = [PlannedRun(arm=arm, scenario="D-1", repeat=repeat, split="dev")
               for repeat in range(len(pages)) for arm in ("A", "B", "C", "D")]
    trials = [Trial(
        **run.model_dump(), status="completed", elapsed_s=1,
        oracle=grade("D-1", "3.11", pages[run.repeat]).status,
    ) for run in planned]
    # When
    result = summarize(trials, planned)["dev"]
    # Then
    assert all(row.blocked == row.scheduled == 5 for row in result.values())
    assert all(row.evaluable.denominator == row.common.denominator == 0 for row in result.values())
    assert all(row.evaluable.successes == row.common.successes == 0 for row in result.values())


def test_denominators_when_blocked_unverifiable_and_missing_are_distinct(tmp_path: Path) -> None:
    # Given: only repeat zero is jointly evaluable across all four arms.
    planned = [PlannedRun(arm=arm, scenario="F-04", repeat=repeat, split="dev")
               for repeat in range(3) for arm in ("A", "B", "C", "D")]
    trials = [
        Trial.model_validate({
            **p.model_dump(), "status": "completed", "oracle": oracle,
            "elapsed_s": 10 + p.repeat, "llm_calls": 2, "jev_calls": 3,
            "recoveries": 2, "recovery_successes": 1,
            "costs": {"billed_usd": Measure(unknown_count=2),
                      "api_equivalent_usd": Measure(known_sum=0.25, known_count=2)},
        })
        for p in planned
        if not (p.repeat == 2 and p.arm == "D")
        for oracle in [
            "blocked" if p.repeat == 1 and p.arm == "A" else
            "unverifiable" if p.repeat == 1 and p.arm == "B" else
            "failed" if p.arm == "C" else "passed"
        ]
    ]
    # When
    result = summarize(trials, planned)["dev"]
    report = write_report(tmp_path, trials, planned)
    # Then
    assert result["A"].evaluable.successes == result["A"].evaluable.denominator == 2
    assert result["B"].unverifiable == result["A"].blocked == 1
    assert result["C"].evaluable.successes == 0
    assert result["C"].evaluable.denominator == 3
    assert result["D"].scheduled == 3 and result["D"].missing == 1
    assert all(row.common.denominator == 1 for row in result.values())
    assert result["C"].common.successes == 0 and result["A"].common.successes == 1
    assert result["A"].costs["billed_usd"].known_count == 0
    assert result["A"].costs["billed_usd"].unknown_count == 6
    assert result["A"].costs["api_equivalent_usd"].known_sum == 0.75
    assert result["A"].median_s == 11 and result["A"].iqr_s == 1
    assert result["A"].recovery.value == 0.5
    assert report.name == "REPORT.md" and report.stat().st_size > 0


def test_splits_when_same_scenario_repeat_is_reused_never_mix() -> None:
    # Given
    planned = [PlannedRun(arm=arm, scenario="shared", repeat=0, split=split)
               for split in ("dev", "holdout") for arm in ("A", "B", "C", "D")]
    trials = [Trial(**p.model_dump(), status="failed", oracle="failed", elapsed_s=1) for p in planned]
    trials[0] = trials[0].model_copy(update={"oracle": "passed"})
    # When
    result = summarize(trials, planned)
    # Then
    assert result["dev"]["A"].common.successes == 1
    assert result["holdout"]["A"].common.successes == 0
    assert result["holdout"]["A"].common.denominator == 1


def test_wilson_when_empty_or_small_sample_is_not_false_precision() -> None:
    # Given / When
    empty, perfect = rate(0, 0), rate(4, 4)
    # Then
    assert empty.value is None and empty.wilson_low is None
    assert perfect.wilson_low == pytest.approx(0.5101091635)
    assert perfect.wilson_high == pytest.approx(1)


def test_duplicate_trial_when_resuming_is_rejected() -> None:
    # Given
    run = PlannedRun(arm="A", scenario="F-04", repeat=0, split="dev")
    trial = Trial(**run.model_dump(), status="failed", oracle="failed", elapsed_s=1)
    # When / Then
    with pytest.raises(ValueError, match="duplicate"):
        summarize([trial, trial], [run])


def test_integration_failure_when_recorded_is_separate_from_measured_failures(
    tmp_path: Path,
) -> None:
    # Given: one ordinary failed run and one integration failure for arm A.
    planned = [PlannedRun(arm=arm, scenario="F-02", repeat=repeat, split="dev")
               for repeat in range(2) for arm in ("A", "B", "C", "D")]
    trials = [Trial(
        **run.model_dump(), status="completed", elapsed_s=1, oracle="passed",
    ) for run in planned]
    trials[0] = trials[0].model_copy(update={
        "status": "integration_failed", "oracle": "unverifiable",
        "error_kind": "agent_provider_failure",
    })
    trials[4] = trials[4].model_copy(update={"status": "incomplete", "oracle": "failed"})
    # When
    rows = summarize(trials, planned)["dev"]
    report = write_report(tmp_path, trials, planned)
    # Then: integration is visible but never a measured failure or missing run.
    assert rows["A"].scheduled == rows["A"].recorded == 2
    assert rows["A"].integration_failed == 1
    assert rows["A"].blocked == rows["A"].unverifiable == rows["A"].missing == 0
    assert rows["A"].evaluable.denominator == 1
    assert rows["A"].evaluable.successes == 0
    assert rows["B"].evaluable.denominator == 2
    assert all(row.common.denominator == 1 for row in rows.values())
    assert report.stat().st_size > 0
