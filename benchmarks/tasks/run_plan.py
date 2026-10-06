"""Construct the deterministic smoke and measurement trial order."""

from __future__ import annotations

from typing import Final

from benchmarks.tasks.run_settings import Mode, Options
from benchmarks.tasks.run_trial import ProductMode, TrialPhase, TrialSpec

_ARMS: Final[dict[Mode, tuple[ProductMode, ...]]] = {
    "llm_only": ("llm_only",),
    "jev_hybrid": ("jev_hybrid",),
    "compare": ("llm_only", "jev_hybrid"),
}
_MAIN_PHASE: Final[dict[Mode, TrialPhase]] = {
    "llm_only": "primary",
    "jev_hybrid": "primary",
    "compare": "comparison",
}
_COMPARISON_TASKS: Final = ("T-01", "T-02", "T-03", "T-04", "T-05")
_SMOKE_TASKS: Final = ("T-01", "T-02", "T-03")


def build_trial_plan(options: Options) -> tuple[TrialSpec, ...]:
    """Build a fixed first smoke gate, then the requested main measurement."""
    plan: list[TrialSpec] = []
    if options.smoke_gate:
        for task_index, task_id in enumerate(_SMOKE_TASKS):
            modes = _ARMS["compare"]
            ordered_modes = (
                modes if task_index % 2 == 0 else tuple(reversed(modes))
            )
            plan.extend(
                TrialSpec(
                    phase="smoke",
                    repeat_index=1,
                    task_id=task_id,
                    mode=mode,
                )
                for mode in ordered_modes
            )

    task_ids = (
        _COMPARISON_TASKS if options.task_id == "all" else (options.task_id,)
    )
    modes = _ARMS[options.mode]
    phase = _MAIN_PHASE[options.mode]
    for repeat_index in range(options.repeat):
        for task_index, task_id in enumerate(task_ids):
            ordered_modes = (
                modes
                if (repeat_index + task_index) % 2 == 0
                else tuple(reversed(modes))
            )
            plan.extend(
                TrialSpec(
                    phase=phase,
                    repeat_index=repeat_index + 1,
                    task_id=task_id,
                    mode=mode,
                )
                for mode in ordered_modes
            )
    return tuple(plan)
