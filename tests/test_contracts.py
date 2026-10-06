from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from jevpilot.decision.models import DecisionOutput
from jevpilot.executor.models import ActionExecution
from jevpilot.observer.models import ActionCandidate, WaitCondition
from jevpilot.planner.models import Plan, Subgoal
from jevpilot.task import ActionType, Budget, TaskSpec
from jevpilot.telemetry.models import CallEvent
from jevpilot.verifier.models import Check


def valid_budget() -> Budget:
    return Budget(
        max_steps=12,
        max_mutations=6,
        max_replans=2,
        max_model_attempts=8,
        max_attempts_per_call=2,
        max_stale_observations=3,
        max_waits=2,
        max_wait_ms=1_000,
        wall_time_ms=30_000,
        cost_limit_usd=None,
    )


def test_task_spec_rejects_empty_goal() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = TaskSpec(
            task_id="T-01",
            goal=" ",
            mode="llm_only",
            initial_url="https://example.test/start",
            allowed_origins=("https://example.test",),
            allowed_operations=(ActionType.CLICK,),
            initializer_id="init-1",
            evaluator_id="eval-1",
            budgets=valid_budget(),
            locale="en-US",
            fixture_version="1",
        )


def test_task_spec_rejects_forbidden_mode() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = TaskSpec.model_validate(
            {
                "task_id": "T-01",
                "goal": "Open the item",
                "mode": "mock",
                "initial_url": "https://example.test/start",
                "allowed_origins": ("https://example.test",),
                "allowed_operations": (ActionType.CLICK,),
                "initializer_id": "init-1",
                "evaluator_id": "eval-1",
                "budgets": valid_budget(),
                "locale": "en-US",
                "fixture_version": "1",
            }
        )


@pytest.mark.parametrize(
    ("initial_url", "allowed_origins"),
    [
        ("not-a-url", ("https://example.test",)),
        ("https://example.test/start", ("https://example.test/path",)),
        ("https://example.test/start", ("https://other.test",)),
    ],
)
def test_task_spec_rejects_malformed_or_disallowed_origin(
    initial_url: str,
    allowed_origins: tuple[str, ...],
) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = TaskSpec(
            task_id="T-01",
            goal="Open the item",
            mode="jev_hybrid",
            initial_url=initial_url,
            allowed_origins=allowed_origins,
            allowed_operations=(ActionType.CLICK,),
            initializer_id="init-1",
            evaluator_id="eval-1",
            budgets=valid_budget(),
            locale="en-US",
            fixture_version="1",
        )


def test_budget_rejects_negative_values() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = Budget(
            max_steps=12,
            max_mutations=-1,
            max_replans=2,
            max_model_attempts=8,
            max_attempts_per_call=2,
            max_stale_observations=3,
            max_waits=2,
            max_wait_ms=1_000,
            wall_time_ms=30_000,
            cost_limit_usd=None,
        )


def test_targeted_action_requires_target_ref() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = ActionCandidate(
            candidate_id="c1",
            snapshot_id="s1",
            operation=ActionType.CLICK,
        )


def test_select_requires_option_ref_and_rejects_option_on_click() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = ActionCandidate(
            candidate_id="c1",
            snapshot_id="s1",
            operation=ActionType.SELECT,
            target_ref="t1",
        )
    with pytest.raises(ValidationError):
        _ = ActionCandidate(
            candidate_id="c2",
            snapshot_id="s1",
            operation=ActionType.CLICK,
            target_ref="t1",
            option_ref="o1",
        )


def test_wait_requires_bounded_condition() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = ActionCandidate(
            candidate_id="c1",
            snapshot_id="s1",
            operation=ActionType.WAIT,
        )
    candidate = ActionCandidate(
        candidate_id="c1",
        snapshot_id="s1",
        operation=ActionType.WAIT,
        wait_condition=WaitCondition(kind="state_change", timeout_ms=500),
    )
    assert candidate.wait_condition is not None
    assert candidate.wait_condition.timeout_ms == 500


def test_decision_rejects_nan_confidence() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = DecisionOutput(
            decision_id="d1",
            snapshot_id="s1",
            operation=ActionType.CLICK,
            candidate_id="c1",
            operation_confidence=float("nan"),
            provider="jev",
            model="jev-v1",
            call_id="call-1",
        )


def test_records_are_frozen_and_sequences_are_tuples() -> None:
    # Given
    plan = Plan(
        plan_id="p1",
        subgoals=(Subgoal(subgoal_id="sg1", description="Open the item"),),
    )

    # When / Then
    with pytest.raises(ValidationError):
        plan.plan_id = "p2"
    assert isinstance(plan.subgoals, tuple)


def test_public_contracts_exclude_selectors_and_evaluator_answers() -> None:
    # Given / When
    task_fields = TaskSpec.model_fields
    plan_fields = Plan.model_fields
    candidate_fields = ActionCandidate.model_fields

    # Then
    assert "expected_text" not in task_fields
    assert "expected_text" not in plan_fields
    assert "verification_selector" not in plan_fields
    assert "selector" not in candidate_fields


def test_action_execution_preserves_dispatch_certainty() -> None:
    # Given
    now = datetime.now(UTC)

    # When / Then
    with pytest.raises(ValidationError):
        _ = ActionExecution(
            action_id="a1",
            decision_id="d1",
            snapshot_id="s1",
            operation=ActionType.CLICK,
            candidate_id="c1",
            target_ref="t1",
            status="uncertain",
            dispatched=False,
            started_at=now,
            ended_at=now,
            evidence=(),
        )


def test_action_effect_check_requires_action_id() -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = Check(
            check_id="effect-1",
            scope="action_effect",
            status="unknown",
            evidence=(),
        )


def test_call_event_distinguishes_network_and_test_double() -> None:
    # Given
    now = datetime.now(UTC)

    # When / Then
    with pytest.raises(ValidationError):
        _ = CallEvent(
            run_id="run-1",
            call_id="call-1",
            logical_call_id="logical-1",
            source="test_double",
            role="selector",
            purpose="normal",
            provider="fake",
            request_model="fake-v1",
            attempt_index=1,
            sent=True,
            started_at=now,
            ended_at=now,
            outcome="succeeded",
        )
