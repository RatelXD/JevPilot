"""Reusable scripted providers for runtime browser tests."""

from datetime import UTC, datetime
from typing import final

from jevpilot.decision.models import DecisionInput, DecisionOutput
from jevpilot.planner.models import Plan, PlannerInput, ProgressHint, Subgoal
from jevpilot.task import ActionType, Budget
from jevpilot.telemetry.models import CallContext, CallEvent
from jevpilot.text import TextInput, TextOutput


def budgets() -> Budget:
    """Small deterministic budgets for local browser tests."""
    return Budget(
        max_steps=8,
        max_mutations=3,
        max_replans=2,
        max_model_attempts=8,
        max_attempts_per_call=2,
        max_stale_observations=2,
        max_waits=2,
        max_wait_ms=500,
        wall_time_ms=15000,
    )


@final
class ScriptedPlanner:
    calls: int
    failures: list[str | None]

    def __init__(self) -> None:
        self.calls = 0
        self.failures = []

    async def plan(self, payload: PlannerInput, *, context: CallContext) -> Plan:
        self.calls += 1
        self.failures.append(payload.failure.kind if payload.failure else None)
        context.record_call(
            CallEvent(
                run_id=context.run_id,
                call_id=f"plan-{self.calls}",
                logical_call_id=context.logical_call_id,
                source="test_double",
                role="planner",
                purpose=context.purpose,
                provider="scripted",
                request_model="offline",
                attempt_index=1,
                sent=False,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                outcome="succeeded",
            )
        )
        return Plan(
            plan_id=f"plan-{self.calls}",
            parent_plan_id=payload.previous_plan_id,
            subgoals=(
                Subgoal(
                    subgoal_id=f"subgoal-{self.calls}",
                    description="Open the Aurora detail page",
                    progress_hint=ProgressHint(
                        kind="url_path", expected="/items/aurora.html"
                    ),
                ),
            ),
        )


@final
class ScriptedSelector:
    calls: int
    false_first_done: bool
    confidence: float | None
    actions: tuple[ActionType, ...]
    target_names: tuple[str, ...]
    target_values: tuple[str | None, ...]
    autocomplete: bool

    def __init__(
        self,
        *,
        false_first_done: bool = False,
        confidence: float | None = None,
        actions: tuple[ActionType, ...] = (ActionType.CLICK,),
        target_names: tuple[str, ...] = ("Aurora",),
        target_values: tuple[str | None, ...] = (None,),
        autocomplete: bool = False,
    ) -> None:
        self.calls = 0
        self.false_first_done = false_first_done
        self.confidence = confidence
        self.actions = actions
        self.target_names = target_names
        self.target_values = target_values
        self.autocomplete = autocomplete

    async def decide(
        self, payload: DecisionInput, *, context: CallContext
    ) -> DecisionOutput:
        self.calls += 1
        if self.false_first_done and self.calls == 1:
            operation = ActionType.DONE
            candidate_id = None
        elif self.autocomplete:
            typed = any(
                entry.operation is ActionType.TYPE_TEXT
                and entry.outcome == "executed"
                for entry in payload.recent_history
            )
            suggestion = next(
                (
                    item
                    for item in payload.candidates
                    if item.operation is ActionType.CLICK and item.name == "Aurora"
                ),
                None,
            )
            text_field = next(
                (
                    item
                    for item in payload.candidates
                    if item.operation is ActionType.TYPE_TEXT
                    and item.name == "Search catalog"
                ),
                None,
            )
            wait = next(
                (
                    item
                    for item in payload.candidates
                    if item.operation is ActionType.WAIT
                ),
                None,
            )
            if not typed and text_field is not None:
                operation = ActionType.TYPE_TEXT
                candidate_id = text_field.candidate_id
            elif suggestion is not None:
                operation = ActionType.CLICK
                candidate_id = suggestion.candidate_id
            elif wait is not None:
                operation = ActionType.WAIT
                candidate_id = wait.candidate_id
            else:
                operation = ActionType.BLOCKED
                candidate_id = None
        else:
            position = min(
                self.calls - 1 - int(self.false_first_done),
                len(self.actions) - 1,
            )
            operation = self.actions[position]
            target_name = self.target_names[min(position, len(self.target_names) - 1)]
            target_value = self.target_values[min(
                position, len(self.target_values) - 1
            )]
            selected = next(
                (
                    item
                    for item in payload.candidates
                    if item.operation is operation
                    and item.name == target_name
                    and (target_value is None or item.value == target_value)
                ),
                None,
            )
            candidate_id = selected.candidate_id if selected is not None else None
        context.record_call(
            CallEvent(
                run_id=context.run_id,
                call_id=f"select-{self.calls}",
                logical_call_id=context.logical_call_id,
                source="test_double",
                role="selector",
                purpose=context.purpose,
                provider="scripted",
                request_model="offline",
                attempt_index=1,
                sent=False,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                outcome="succeeded",
            )
        )
        return DecisionOutput(
            decision_id=f"decision-{self.calls}",
            snapshot_id=payload.observation.snapshot_id,
            operation=operation,
            candidate_id=candidate_id,
            operation_confidence=self.confidence,
            target_confidence=(
                self.confidence
                if operation
                in {
                    ActionType.CLICK,
                    ActionType.TYPE_TEXT,
                    ActionType.SELECT,
                    ActionType.WAIT,
                }
                else None
            ),
            provider="offline_test_double",
            model=None,
            call_id=f"select-{self.calls}",
        )


@final
class ScriptedText:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self, payload: TextInput, *, context: CallContext
    ) -> TextOutput:
        self.calls += 1
        assert payload.target.operation is ActionType.TYPE_TEXT
        assert payload.constraints.max_length > 0
        context.record_call(
            CallEvent(
                run_id=context.run_id,
                call_id=f"text-{self.calls}",
                logical_call_id=context.logical_call_id,
                source="test_double",
                role="text",
                purpose=context.purpose,
                provider="scripted",
                request_model="offline",
                attempt_index=1,
                sent=False,
                started_at=datetime.now(UTC),
                ended_at=datetime.now(UTC),
                outcome="succeeded",
            )
        )
        values = {
            "Search catalog": "Aurora",
            "Full name": "Riley",
            "City": "Reykjavik",
        }
        return TextOutput(text=values[payload.target.name or ""])
