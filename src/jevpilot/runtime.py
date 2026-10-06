"""Bounded first live slice: observe, plan once, choose, execute, verify."""

import time
from datetime import UTC, datetime
from typing import Protocol, final
from urllib.parse import urlsplit
from uuid import uuid4

import anyio
from playwright.async_api import Browser, Error as BrowserError, Page, Route
from pydantic import ValidationError

from jevpilot.decision.models import DecisionInput, DecisionOutput, DecisionPolicy
from jevpilot.executor.models import ActionRequest
from jevpilot.executor.playwright_executor import PlaywrightExecutor
from jevpilot.model_client import ProviderError
from jevpilot.observer.service import Observer
from jevpilot.planner.models import (
    BudgetsRemaining, HistoryEntry, Plan, PlannerInput, PlanningFailure,
)
from jevpilot.privacy import PrivacyPolicy
from jevpilot.task import ActionType, TaskSpec
from jevpilot.telemetry.models import CallContext, ExecutionTrace, TerminalResult
from jevpilot.telemetry.recorder import BudgetExceeded, RunRecorder
from jevpilot.text import TextConstraints, TextInput, TextValueProvider
from jevpilot.verifier.verifiers import EvaluationError, IndependentVerifier, TaskBinding


class Planner(Protocol):
    async def plan(self, payload: PlannerInput, *, context: CallContext) -> Plan: ...


class Selector(Protocol):
    async def decide(
        self, payload: DecisionInput, *, context: CallContext
    ) -> DecisionOutput: ...


@final
class HybridBrowserAgent:
    planner: Planner
    selector: Selector
    text: TextValueProvider
    privacy: PrivacyPolicy
    operation_threshold: float
    target_threshold: float

    def __init__(
        self, planner: Planner, decision_provider: Selector, *,
        text: TextValueProvider,
        privacy: PrivacyPolicy,
        operation_threshold: float,
        target_threshold: float,
    ) -> None:
        self.planner = planner
        self.selector = decision_provider
        self.text = text
        self.privacy = privacy
        self.operation_threshold = operation_threshold
        self.target_threshold = target_threshold

    async def run(
        self, browser: Browser, task: TaskSpec, *,
        binding: TaskBinding, recorder: RunRecorder,
    ) -> ExecutionTrace:
        context = await browser.new_context(locale=task.locale)
        terminal = TerminalResult(
            status="failed", reason="setup_incomplete", ended_at=datetime.now(UTC)
        )
        observer: Observer | None = None
        try:
            with anyio.fail_after(max(0, recorder.deadline - time.monotonic())):
                if (binding.initializer_id, binding.evaluator_id) != (
                    task.initializer_id, task.evaluator_id
                ):
                    raise EvaluationError("binding_mismatch")
                page = await context.new_page()

                async def restrict_origin(route: Route) -> None:
                    url = urlsplit(route.request.url)
                    origin = f"{url.scheme}://{url.netloc}"
                    if origin in task.allowed_origins:
                        await route.continue_()
                    else:
                        await route.abort("blockedbyclient")

                _ = await context.route("**/*", restrict_origin)
                await binding.initializer(page, task)
                observer = Observer(page, task=task, privacy=self.privacy)
                terminal = await self._loop(page, task, binding, recorder, observer)
        except (TimeoutError, BudgetExceeded) as exc:
            reason = exc.resource if isinstance(exc, BudgetExceeded) else "wall_time"
            terminal = TerminalResult(
                status="timeout" if reason == "wall_time" else "blocked",
                reason=reason, ended_at=datetime.now(UTC),
            )
        except (ProviderError, BrowserError, EvaluationError, ValidationError) as exc:
            terminal = TerminalResult(
                status="failed", reason=type(exc).__name__, ended_at=datetime.now(UTC)
            )
        finally:
            recorder.transition(terminal.status.upper(), terminal.reason)
            recorder.finish_task()
            with anyio.CancelScope(shield=True):
                if observer is not None:
                    await observer.close()
                await context.close()
        return recorder.trace(
            task_id=task.task_id, mode=task.mode,
            browser_version=browser.version,
            fixture_version=task.fixture_version, terminal=terminal,
        )

    async def _loop(
        self, page: Page, task: TaskSpec, binding: TaskBinding,
        recorder: RunRecorder, observer: Observer,
    ) -> TerminalResult:
        executor = PlaywrightExecutor(
            page, observer, task=task, record_execution=recorder.executions.append
        )
        verifier = IndependentVerifier()
        plan: Plan | None = None
        previous_plan_id: str | None = None
        failure: PlanningFailure | None = None
        index = 0
        history: list[HistoryEntry] = []
        consumed: set[str] = set()
        last_fingerprint: str | None = None
        stagnant = 0
        while True:
            recorder.check_deadline()
            recorder.transition("OBSERVING", "next_observation")
            observation, candidates = await observer.observe(
                deadline_monotonic=recorder.deadline
            )
            if plan is None:
                recorder.start_task()
                recorder.transition(
                    "REPLANNING" if failure else "PLANNING",
                    failure.kind if failure else "initial_plan",
                )
                budget = task.budgets
                plan = await self.planner.plan(
                    PlannerInput(
                        goal=task.goal, observation=observation, history=tuple(history[-6:]),
                        previous_plan_id=previous_plan_id, failure=failure,
                        budgets_remaining=BudgetsRemaining(
                            steps=max(0, budget.max_steps - recorder.steps),
                            mutations=max(0, budget.max_mutations - recorder.mutations),
                            replans=max(0, budget.max_replans - recorder.replans),
                            model_attempts=max(0, budget.max_model_attempts - recorder.attempts),
                            stale_observations=max(0, budget.max_stale_observations - recorder.stale),
                            waits=max(0, budget.max_waits - recorder.waits),
                            wait_ms=budget.max_wait_ms,
                            wall_time_ms=max(0, int((recorder.deadline - time.monotonic()) * 1000)),
                        ),
                    ),
                    context=recorder.context("planner", "fallback" if failure else "initial"),
                )
                index = 0
                observation, candidates = await observer.observe(
                    deadline_monotonic=recorder.deadline
                )
            if recorder.steps >= task.budgets.max_steps:
                raise BudgetExceeded("steps")
            subgoal = plan.subgoals[index]
            recorder.transition("DECIDING", "select_current_candidate")
            decision = await self.selector.decide(
                DecisionInput(
                    goal=task.goal, current_subgoal=subgoal, observation=observation,
                    candidates=candidates, recent_history=tuple(history[-6:]),
                    policy=DecisionPolicy(
                        allowed_operations=task.allowed_operations,
                        max_wait_ms=task.budgets.max_wait_ms,
                    ),
                ),
                context=recorder.context("selector", "normal"),
            )
            if decision.decision_id in consumed:
                raise EvaluationError("decision_already_consumed")
            consumed.add(decision.decision_id)
            recorder.steps += 1
            reason: str | None = None
            if decision.snapshot_id != observation.snapshot_id:
                reason = "snapshot_mismatch"
            elif task.mode == "jev_hybrid" and (
                decision.operation_confidence is None
                or decision.operation_confidence < self.operation_threshold
                or (decision.operation in {
                    ActionType.CLICK,
                    ActionType.TYPE_TEXT,
                    ActionType.SELECT,
                    ActionType.WAIT,
                } and (
                    decision.target_confidence is None
                    or decision.target_confidence < self.target_threshold
                ))
            ):
                reason = "low_confidence"
            elif decision.operation is ActionType.BLOCKED:
                return TerminalResult(status="blocked", reason="model_blocked", ended_at=datetime.now(UTC))
            elif decision.operation in {
                ActionType.CLICK,
                ActionType.TYPE_TEXT,
                ActionType.SELECT,
            }:
                candidate = next(
                    (c for c in candidates if c.candidate_id == decision.candidate_id), None
                )
                if candidate is None or candidate.operation is not decision.operation:
                    reason = "invalid_candidate"
                else:
                    if recorder.mutations >= task.budgets.max_mutations:
                        raise BudgetExceeded("mutations")
                    text_value: str | None = None
                    if decision.operation is ActionType.TYPE_TEXT:
                        recorder.transition("GENERATING_TEXT", "selected_text_field")
                        generated = await self.text.generate(
                            TextInput(
                                goal=task.goal,
                                current_subgoal=subgoal,
                                observation=observation,
                                target=candidate,
                                history=tuple(history[-6:]),
                                constraints=TextConstraints(
                                    max_length=candidate.max_length or 2000,
                                    allow_empty=False,
                                ),
                            ),
                            context=recorder.context("text", "normal"),
                        )
                        text_value = generated.text
                    recorder.transition("EXECUTING", f"validated_{decision.operation.value}")
                    receipt = await executor.execute(
                        ActionRequest(
                            action_id=uuid4().hex, decision_id=decision.decision_id,
                            snapshot_id=observation.snapshot_id,
                            operation=decision.operation,
                            candidate_id=candidate.candidate_id,
                            target_ref=candidate.target_ref,
                            option_ref=candidate.option_ref,
                            text=text_value,
                        ),
                        deadline_monotonic=recorder.deadline,
                    )
                    recorder.mutations += int(receipt.dispatched)
                    history.append(HistoryEntry(
                        snapshot_id=observation.snapshot_id, subgoal_id=subgoal.subgoal_id,
                        operation=decision.operation, outcome=receipt.status,
                        action_id=receipt.action_id, evidence=receipt.evidence,
                    ))
                    if receipt.status == "failed_not_executed":
                        recorder.stale += 1
                        if recorder.stale > task.budgets.max_stale_observations:
                            raise BudgetExceeded("stale_observations")
                        continue
                    if receipt.status == "uncertain":
                        recorder.transition("VERIFYING", "read_only_uncertain_check")
                        try:
                            after, _ = await observer.observe(deadline_monotonic=recorder.deadline)
                            verification = await verifier.verify(
                                page, observation=after, subgoal=subgoal, binding=binding
                            )
                        except (BrowserError, EvaluationError, ValidationError):
                            return TerminalResult(status="uncertain_execution", reason="effect_check_unavailable", ended_at=datetime.now(UTC))
                        recorder.verifications.append(verification)
                        if verification.task_evaluation.status == "passed":
                            return TerminalResult(status="succeeded", reason="verified_after_uncertain", ended_at=datetime.now(UTC))
                        return TerminalResult(status="uncertain_execution", reason="effect_unconfirmed", ended_at=datetime.now(UTC))
            elif decision.operation in {
                ActionType.SCROLL_UP,
                ActionType.SCROLL_DOWN,
            }:
                if not any(
                    candidate.operation is decision.operation
                    for candidate in candidates
                ):
                    reason = "invalid_scroll_candidate"
                else:
                    recorder.transition("EXECUTING", f"validated_{decision.operation.value}")
                    receipt = await executor.execute(
                        ActionRequest(
                            action_id=uuid4().hex,
                            decision_id=decision.decision_id,
                            snapshot_id=observation.snapshot_id,
                            operation=decision.operation,
                        ),
                        deadline_monotonic=recorder.deadline,
                    )
                    history.append(
                        HistoryEntry(
                            snapshot_id=observation.snapshot_id,
                            subgoal_id=subgoal.subgoal_id,
                            operation=decision.operation,
                            outcome=receipt.status,
                            action_id=receipt.action_id,
                            evidence=receipt.evidence,
                        )
                    )
                    if receipt.status == "failed_not_executed":
                        recorder.stale += 1
                        if recorder.stale > task.budgets.max_stale_observations:
                            raise BudgetExceeded("stale_observations")
                        continue
                    if receipt.status == "uncertain":
                        return TerminalResult(
                            status="uncertain_execution",
                            reason="scroll_effect_unconfirmed",
                            ended_at=datetime.now(UTC),
                        )
            elif decision.operation is ActionType.WAIT:
                wait_candidate = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id == decision.candidate_id
                        and candidate.operation is ActionType.WAIT
                    ),
                    None,
                )
                if wait_candidate is None or wait_candidate.wait_condition is None:
                    reason = "invalid_wait_candidate"
                else:
                    if recorder.waits >= task.budgets.max_waits:
                        raise BudgetExceeded("waits")
                    recorder.waits += 1
                    wait_started = time.monotonic()
                    recorder.transition("EXECUTING", "validated_bounded_wait")
                    receipt = await executor.execute(
                        ActionRequest(
                            action_id=uuid4().hex,
                            decision_id=decision.decision_id,
                            snapshot_id=observation.snapshot_id,
                            operation=ActionType.WAIT,
                            candidate_id=wait_candidate.candidate_id,
                            wait_condition=wait_candidate.wait_condition,
                        ),
                        deadline_monotonic=recorder.deadline,
                    )
                    recorder.wait_ms += (time.monotonic() - wait_started) * 1000
                    history.append(
                        HistoryEntry(
                            snapshot_id=observation.snapshot_id,
                            subgoal_id=subgoal.subgoal_id,
                            operation=ActionType.WAIT,
                            outcome=receipt.status,
                            action_id=receipt.action_id,
                            evidence=receipt.evidence,
                        )
                    )
                    if receipt.status == "failed_not_executed":
                        recorder.stale += 1
                        if recorder.stale > task.budgets.max_stale_observations:
                            raise BudgetExceeded("stale_observations")
                        continue
            elif decision.operation is not ActionType.DONE:
                return TerminalResult(status="blocked", reason="unsupported_first_slice_operation", ended_at=datetime.now(UTC))
            if reason is None:
                recorder.transition("VERIFYING", "independent_task_check")
                after, _ = await observer.observe(deadline_monotonic=recorder.deadline)
                verification = await verifier.verify(
                    page, observation=after, subgoal=subgoal, binding=binding
                )
                recorder.verifications.append(verification)
                if verification.task_evaluation.status == "passed":
                    return TerminalResult(status="succeeded", reason="independently_verified", ended_at=datetime.now(UTC))
                if decision.operation is ActionType.DONE:
                    reason = "false_done"
                elif verification.subgoal_status == "passed" and index + 1 < len(plan.subgoals):
                    index += 1
                stagnant = stagnant + 1 if after.fingerprint == last_fingerprint else 0
                last_fingerprint = after.fingerprint
                if stagnant >= 2:
                    reason = "no_progress"
            if reason is not None:
                if recorder.replans >= task.budgets.max_replans:
                    raise BudgetExceeded("replans")
                recorder.replans += 1
                previous_plan_id = plan.plan_id
                failure = PlanningFailure(kind=reason, evidence=(reason,))
                plan = None
