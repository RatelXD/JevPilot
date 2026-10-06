from jevpilot.observer import ActionCandidate, ActionType


def test_action_candidate_uses_snapshot_bound_public_identity() -> None:
    candidate = ActionCandidate(
        candidate_id="candidate-1",
        snapshot_id="snapshot-1",
        operation=ActionType.CLICK,
        target_ref="target-1",
        role="button",
        name="Run",
        enabled=True,
        readonly=False,
    )

    assert candidate.operation is ActionType.CLICK
    assert candidate.model_dump(mode="json")["operation"] == "click"
    assert "selector" not in candidate.model_dump()
