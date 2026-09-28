from jevpilot.observer import ActionCandidate, ActionType


def test_action_candidate_representation() -> None:
    candidate = ActionCandidate(
        id="c1",
        action_type=ActionType.CLICK,
        selector="#run",
        text="Run",
        enabled=True,
    )

    assert candidate.action_type == ActionType.CLICK
    assert candidate.model_dump()["selector"] == "#run"
