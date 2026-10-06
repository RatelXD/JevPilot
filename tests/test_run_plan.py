from collections import Counter

import pytest

from benchmarks.tasks.run_plan import build_trial_plan
from benchmarks.tasks.run_settings import SettingsError, parse_options


def test_smoke_gate_runs_three_tasks_in_both_modes_before_fifty_trials() -> None:
    # Given
    options = parse_options(
        [
            "--mode",
            "compare",
            "--task",
            "all",
            "--repeat",
            "5",
            "--live",
            "--smoke-gate",
        ]
    )

    # When
    plan = build_trial_plan(options)

    # Then
    assert len(plan) == 56
    assert {trial.phase for trial in plan[:6]} == {"smoke"}
    assert {(trial.task_id, trial.mode) for trial in plan[:6]} == {
        ("T-01", "llm_only"),
        ("T-01", "jev_hybrid"),
        ("T-02", "llm_only"),
        ("T-02", "jev_hybrid"),
        ("T-03", "llm_only"),
        ("T-03", "jev_hybrid"),
    }
    comparison = plan[6:]
    assert {trial.phase for trial in comparison} == {"comparison"}
    assert Counter((trial.task_id, trial.mode) for trial in comparison) == Counter(
        {
            (task_id, mode): 5
            for task_id in ("T-01", "T-02", "T-03", "T-04", "T-05")
            for mode in ("llm_only", "jev_hybrid")
        }
    )


def test_smoke_gate_rejects_single_mode_runner() -> None:
    # Given / When / Then
    with pytest.raises(SettingsError, match="requires --mode compare"):
        _ = parse_options(
            [
                "--mode",
                "llm_only",
                "--task",
                "all",
                "--repeat",
                "5",
                "--live",
                "--smoke-gate",
            ]
        )
