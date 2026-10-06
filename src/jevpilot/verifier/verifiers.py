"""Trusted task evaluation is separate from model-authored progress hints."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import override
from urllib.parse import urlsplit

from playwright.async_api import Page

from jevpilot.observer.models import Observation
from jevpilot.planner.models import Subgoal
from jevpilot.task import TaskSpec
from jevpilot.verifier.models import Check, TaskEvaluation, VerificationResult

Initializer = Callable[[Page, TaskSpec], Awaitable[None]]
Evaluator = Callable[[Page], Awaitable[TaskEvaluation]]


@dataclass(frozen=True, slots=True)
class TaskBinding:
    """Private code-owned callbacks; never serialize these into a model request."""

    initializer_id: str
    evaluator_id: str
    initializer: Initializer
    evaluator: Evaluator


class IndependentVerifier:
    async def verify(
        self,
        page: Page,
        *,
        observation: Observation,
        subgoal: Subgoal,
        binding: TaskBinding,
    ) -> VerificationResult:
        task = await binding.evaluator(page)
        if task.evaluator_id != binding.evaluator_id:
            raise EvaluationError("evaluator_id_mismatch")
        if task.status == "passed" and (
            not task.checks or any(check.status != "passed" for check in task.checks)
        ):
            raise EvaluationError("success_without_passing_checks")
        checks: tuple[Check, ...] = ()
        hint = subgoal.progress_hint
        if hint is not None:
            matched = False
            controls = tuple(c for c in observation.controls if c.name == hint.name)
            match hint.kind:
                case "url_path":
                    matched = urlsplit(observation.url).path == hint.expected
                case "visible":
                    matched = bool(controls) == hint.expected
                case "value":
                    matched = len(controls) == 1 and controls[0].value == hint.expected
                case "checked":
                    matched = len(controls) == 1 and controls[0].checked == hint.expected
            checks = (
                Check(
                    check_id=subgoal.subgoal_id,
                    scope="subgoal",
                    status="passed" if matched else "not_yet",
                    evidence=("observed_progress_hint",),
                ),
            )
        return VerificationResult(
            snapshot_id=observation.snapshot_id,
            subgoal_id=subgoal.subgoal_id,
            subgoal_status=checks[0].status if checks else "unknown",
            subgoal_checks=checks,
            task_evaluation=task,
            effect_checks=(),
            evidence=("trusted_evaluator",),
        )


class EvaluationError(RuntimeError):
    kind: str

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(kind)

    @override
    def __str__(self) -> str:
        return f"Independent evaluation failed: {self.kind}"
