"""Split-separated evaluation tables, preserving excluded and missing runs."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median, quantiles
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Measure(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    known_sum: float = 0
    known_count: int = 0
    unknown_count: int = 0


class Trial(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    arm: Literal["A", "B", "C", "D"]
    scenario: str
    repeat: int
    split: Literal["dev", "holdout"]
    status: str
    oracle: Literal["passed", "failed", "blocked", "unverifiable"]
    oracle_reason: str = ""
    error_kind: str | None = None
    elapsed_s: float
    llm_calls: int = 0
    jev_calls: int = 0
    tokens: dict[str, Measure] = Field(default_factory=dict)
    costs: dict[str, Measure] = Field(default_factory=dict)
    recoveries: int = 0
    recovery_successes: int = 0
    llm_reasons: list[str] = Field(default_factory=list)
    accounting_complete: bool = True
    browser_language: str = ""
    browser_languages: list[str] = Field(default_factory=list)
    intl_locale: str = ""
    viewport: tuple[int, int] | None = None


class PlannedRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    arm: Literal["A", "B", "C", "D"]
    scenario: str
    repeat: int
    split: Literal["dev", "holdout"]

    @property
    def run_id(self) -> str:
        return f"{self.scenario}-r{self.repeat:02}-{self.arm}"


class Rate(BaseModel):
    successes: int
    denominator: int
    value: float | None
    wilson_low: float | None
    wilson_high: float | None


class ArmSummary(BaseModel):
    scheduled: int
    recorded: int
    blocked: int
    unverifiable: int
    integration_failed: int
    missing: int
    evaluable: Rate
    common: Rate
    median_s: float | None
    iqr_s: float | None
    llm_calls: int
    jev_calls: int
    tokens: dict[str, Measure]
    costs: dict[str, Measure]
    recovery: Rate
    incomplete_accounting: int


def rate(successes: int, denominator: int) -> Rate:
    if not denominator:
        return Rate(successes=0, denominator=0, value=None, wilson_low=None, wilson_high=None)
    z = 1.959963984540054
    p = successes / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return Rate(
        successes=successes, denominator=denominator, value=p,
        wilson_low=max(0, center - half), wilson_high=min(1, center + half),
    )


def sum_measures(values: list[dict[str, Measure]]) -> dict[str, Measure]:
    keys = {key for value in values for key in value}
    return {
        key: Measure(
            known_sum=sum(value[key].known_sum for value in values if key in value),
            known_count=sum(value[key].known_count for value in values if key in value),
            unknown_count=sum(value[key].unknown_count for value in values if key in value),
        ) for key in sorted(keys)
    }


def summarize(trials: list[Trial], planned: list[PlannedRun]) -> dict[str, dict[str, ArmSummary]]:
    """Completion excludes blocked/unverifiable; common means all four arms."""
    keys = [(t.split, t.scenario, t.repeat, t.arm) for t in trials]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate trial identity")
    expected = {(p.split, p.scenario, p.repeat, p.arm) for p in planned}
    if not set(keys) <= expected:
        raise ValueError("unplanned trial")
    evaluable = [
        t for t in trials if t.oracle not in {"blocked", "unverifiable"}
        and t.error_kind != "agent_provider_failure"
    ]
    groups: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for trial in evaluable:
        groups[(trial.split, trial.scenario, trial.repeat)].add(trial.arm)
    common = {key for key, arms in groups.items() if arms == {"A", "B", "C", "D"}}
    result: dict[str, dict[str, ArmSummary]] = {}
    for split in sorted({p.split for p in planned}):
        result[split] = {}
        for arm in sorted({p.arm for p in planned if p.split == split}):
            rows = [t for t in trials if t.split == split and t.arm == arm]
            eligible = [t for t in evaluable if t.split == split and t.arm == arm]
            shared = [t for t in eligible if (t.split, t.scenario, t.repeat) in common]
            scheduled = sum(p.split == split and p.arm == arm for p in planned)
            times = [t.elapsed_s for t in eligible]
            quartiles = quantiles(times, n=4, method="inclusive") if len(times) > 1 else [0, 0, 0]
            result[split][arm] = ArmSummary(
                scheduled=scheduled, recorded=len(rows), missing=scheduled - len(rows),
                blocked=sum(t.oracle == "blocked" for t in rows),
                unverifiable=sum(
                    t.oracle == "unverifiable" and t.error_kind != "agent_provider_failure"
                    for t in rows
                ),
                integration_failed=sum(t.error_kind == "agent_provider_failure" for t in rows),
                evaluable=rate(sum(t.oracle == "passed" for t in eligible), len(eligible)),
                common=rate(sum(t.oracle == "passed" for t in shared), len(shared)),
                median_s=median(times) if times else None,
                iqr_s=quartiles[2] - quartiles[0] if times else None,
                llm_calls=sum(t.llm_calls for t in rows), jev_calls=sum(t.jev_calls for t in rows),
                tokens=sum_measures([t.tokens for t in rows]), costs=sum_measures([t.costs for t in rows]),
                recovery=rate(sum(t.recovery_successes for t in rows), sum(t.recoveries for t in rows)),
                incomplete_accounting=sum(not t.accounting_complete for t in rows),
            )
    return result


def format_rate(value: Rate) -> str:
    if value.value is None:
        return f"{value.successes}/{value.denominator} (미확인)"
    return (
        f"{value.successes}/{value.denominator} = {value.value:.6f} "
        f"[{value.wilson_low:.6f}, {value.wilson_high:.6f}]"
    )


def write_report(output: Path, trials: list[Trial], planned: list[PlannedRun]) -> Path:
    summaries = summarize(trials, planned)
    lines = [
        "# Browser-use evaluation", "",
        "Rates use evaluable runs and the common (scenario, repeat) set where all four arms "
        "are evaluable. Missing, blocked, unverifiable and integration-failed runs remain in scheduled counts. "
        "Integration failures are excluded from measured completion rates. "
        "Intervals are Wilson 95%. Times use evaluable runs (including failures); IQR uses inclusive quartiles. "
        "A recovery succeeds when execution resumes or the run verifies before the next failure.",
        "", "A and D have no risky-action guard. D retains the native 1120x780 viewport and "
        "Browser Harness layer; A/B/C use 1280x800 browser-use. Subscription billed cost is "
        "미확인, not zero. Partial sums are not total billed costs. No superiority is implied.", "",
    ]
    labels = {"billed_usd": "청구", "api_estimate_usd": "추정", "api_equivalent_usd": "API 환산"}
    lines.extend([
        "## Observed browser conditions", "",
        "All arms use Korean language/ICU settings. D's comparator source is unchanged; "
        "the harness configures its native CDP session before navigation. "
        "D's native viewport and Browser Harness layer remain reference-arm limitations.",
        "", "| Arm | navigator.language | navigator.languages | Intl locale | Viewport |",
        "|---|---|---|---|---|",
    ])
    for arm in ("A", "B", "C", "D"):
        conditions = sorted({
            (trial.browser_language, tuple(trial.browser_languages), trial.intl_locale, trial.viewport)
            for trial in trials if trial.arm == arm and trial.browser_language
        })
        for language, languages, intl, viewport in conditions:
            lines.append(f"| {arm} | {language} | {', '.join(languages)} | {intl} | {viewport} |")
        if not conditions:
            lines.append(f"| {arm} | 미확인 | 미확인 | 미확인 | 미확인 |")
    lines.append("")
    for split, arms in summaries.items():
        lines.extend([
            f"## {split}", "",
            "| Arm | Scheduled | Recorded | Blocked | Unverifiable | Integration failures | Missing | Evaluable completion (95%) | Common completion (95%) | Median s | IQR s | LLM | Jev | Recovery |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
        ])
        for arm, row in arms.items():
            lines.append(
                f"| {arm} | {row.scheduled} | {row.recorded} | {row.blocked} | {row.unverifiable} | "
                f"{row.integration_failed} | {row.missing} | {format_rate(row.evaluable)} | {format_rate(row.common)} | "
                f"{row.median_s} | {row.iqr_s} | {row.llm_calls} | {row.jev_calls} | {format_rate(row.recovery)} |"
            )
        for arm, row in arms.items():
            lines.extend(["", f"### {arm} accounting", ""])
            for key, value in row.costs.items():
                amount = f"${value.known_sum:.9f} ({labels[key]})" if value.known_count else "미확인"
                lines.append(f"- {key}: {amount}; known {value.known_count}; 미확인 {value.unknown_count}.")
            for key, value in row.tokens.items():
                lines.append(f"- {key}: known sum {value.known_sum:g}; known {value.known_count}; 미확인 {value.unknown_count}.")
            lines.append(f"- Incomplete call accounting: {row.incomplete_accounting}.")
        lines.extend(["", "### Scenarios", "", "| Scenario | Repeat | Arm | Status | Oracle | Seconds | LLM | Jev |", "|---|---:|---|---|---|---:|---:|---:|"])
        for trial in sorted(
            (t for t in trials if t.split == split), key=lambda t: (t.scenario, t.repeat, t.arm),
        ):
            lines.append(
                f"| {trial.scenario} | {trial.repeat} | {trial.arm} | {trial.status} | "
                f"{trial.oracle} | {trial.elapsed_s:.6f} | {trial.llm_calls} | {trial.jev_calls} |"
            )
        reasons: dict[str, int] = defaultdict(int)
        for trial in trials:
            if trial.split == split:
                for reason in trial.llm_reasons:
                    reasons[f"{trial.arm}: {reason}"] += 1
        lines.extend(["", "### LLM intervention reasons", "", *[f"- {key}: {count}" for key, count in sorted(reasons.items())]])
    path = output / "REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    planned = [PlannedRun.model_validate(p) for p in manifest["scheduled"]]
    trials = [Trial.model_validate_json(p.read_bytes()) for p in args.output.glob("*/trial.json")]
    print(write_report(args.output, trials, planned))


if __name__ == "__main__":
    main()
