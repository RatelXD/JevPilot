from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.browser.port import BrowserAction, ElementRef, Observation
from jevpilot.engine.brief import AnswerFields, TaskBrief, VisibleText
from jevpilot.engine.jev import StepDecision, StepSignals
from jevpilot.engine.routing import Route, RoutingHistory, RoutingPolicy, RoutingSignals, route
from jevpilot.engine.verify import VerifyResult, verify
from jevpilot.ledger.recorder import RunLedger
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.profile import Profile
from jevpilot.llm.types import ModelRef

TARGET = ElementRef(
    snapshot_id="s", index=7, backend_node_id=107, tag="input", text="Search",
    attributes={"name": "query"}, in_viewport=True,
)
PAGE = Observation(
    snapshot_id="s", url="https://example.test/results", title="", tabs=[], elements=[TARGET],
    page_text="Year 1397", fingerprint="f", observed_at_monotonic=1,
)
BRIEF = TaskBrief(
    goal_restated="Find the year", start_url=PAGE.url, input_values={"query": "supplied"},
    required_outputs=[], success_conditions=[VisibleText(text="Year")], missing_inputs=[],
    mutating_intent=False, multi_step=False, comparison=None,
)
SCORES = StepSignals(0.85, 0.20, 0.85, 0.20, None, None, 150, False)
DECISION = StepDecision(
    "type_text", BrowserAction(kind="type_text", target=TARGET), SCORES,
    ("s", PAGE.url, 107), (), 0,
)
SIGNALS = RoutingSignals(PAGE, DECISION, input_key="query")
UNAVAILABLE = [
    replace(SIGNALS, observation=None),
    replace(SIGNALS, observation=PAGE.model_copy(update={
        "elements": [], "page_text": " ", "url": "unavailable", "fingerprint": "empty",
    })),
    replace(SIGNALS, observation_fresh=False,
            observation=PAGE.model_copy(update={"fingerprint": "stale"})),
    replace(SIGNALS, decision=replace(DECISION, bound_to=("old", PAGE.url, 107)),
            observation=PAGE.model_copy(update={"fingerprint": "unbound"})),
]
ANSWERS = BRIEF.model_copy(update={"success_conditions": [AnswerFields(names=["year"])]})
DONE = replace(SIGNALS, decision=replace(DECISION, operation="done", action=None))


@pytest.mark.parametrize("scores,action", [
    (SCORES, "jev_execute"),
    (replace(SCORES, op_confidence=0.849), "llm_choose"),
    (replace(SCORES, op_margin=0.199), "llm_choose"),
    (replace(SCORES, target_confidence=0.849), "llm_choose"),
    (replace(SCORES, target_margin=0.199), "llm_choose"),
    (replace(SCORES, target_confidence=None), "llm_choose"),
    (replace(SCORES, target_margin=None), "llm_choose"),
    (replace(SCORES, op_confidence=float("nan")), "llm_choose"),
    (replace(SCORES, candidate_count=151), "llm_choose"),
    (replace(SCORES, truncated=True), "llm_choose"),
    (replace(SCORES, option_confidence=0.85, option_margin=0.20), "jev_execute"),
    (replace(SCORES, option_confidence=0.849, option_margin=0.20), "llm_choose"),
    (replace(SCORES, option_confidence=0.85, option_margin=0.199), "llm_choose"),
    (replace(SCORES, option_confidence=0.85), "llm_choose"),
])
def test_confidence_table_when_each_signal_crosses_its_boundary(
    scores: StepSignals, action: str,
) -> None:
    # Given
    signals = replace(SIGNALS, decision=replace(DECISION, signals=scores))
    # When
    result = route(signals, BRIEF, RoutingHistory(), ledger=RunLedger())
    # Then
    assert result.action == action
    print(f"signal_table {scores}: {result.level}/{result.action} ({result.reason})")


@pytest.mark.parametrize("signals,brief,history,expected", [
    (SIGNALS, BRIEF, RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (replace(SIGNALS, input_key="other"), BRIEF, RoutingHistory(),
     Route("mid", "llm_text", "input_value_missing")),
    (SIGNALS, BRIEF.model_copy(update={"input_values": {"query": ""}}), RoutingHistory(),
     Route("easy", "jev_execute", "easy_signals")),
    (DONE, ANSWERS, RoutingHistory(), Route("mid", "llm_extract", "answer_extraction_needed")),
    (replace(SIGNALS, verification=VerifyResult("FAIL", "answer_fields_missing")), ANSWERS,
     RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (replace(SIGNALS, verification=verify(PAGE, ANSWERS)), ANSWERS,
     RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (replace(SIGNALS, verification=VerifyResult("FAIL", "visible_text_mismatch")), ANSWERS,
     RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (replace(DONE, verification=VerifyResult("PASS", "conditions_met")), ANSWERS,
     RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (SIGNALS, BRIEF, RoutingHistory(fingerprints=("f", "f")),
     Route("easy", "jev_execute", "easy_signals")),
    (SIGNALS, BRIEF, RoutingHistory(fingerprints=("f", "f", "f")),
     Route("high", "llm_replan", "stalled_fingerprint")),
    (SIGNALS, BRIEF, RoutingHistory(fingerprints=("f", "f", "different", "f")),
     Route("easy", "jev_execute", "easy_signals")),
    (SIGNALS, BRIEF, RoutingHistory(consecutive_failures=1),
     Route("easy", "jev_execute", "easy_signals")),
    (SIGNALS, BRIEF, RoutingHistory(consecutive_failures=2),
     Route("high", "llm_replan", "repeated_failure")),
    (SIGNALS, BRIEF.model_copy(update={"multi_step": True}), RoutingHistory(),
     Route("high", "llm_replan", "subgoal_missing")),
    (replace(SIGNALS, subgoal="Search"), BRIEF.model_copy(update={"multi_step": True}),
     RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
    (replace(SIGNALS, observation=None), BRIEF, RoutingHistory(),
     Route("mid", "reobserve", "empty_observation")),
    (replace(SIGNALS, observation=PAGE.model_copy(update={"elements": [], "page_text": "  "})),
     BRIEF, RoutingHistory(), Route("mid", "reobserve", "empty_observation")),
    (replace(SIGNALS, observation_fresh=False), BRIEF, RoutingHistory(consecutive_failures=2),
     Route("mid", "reobserve", "stale_observation")),
    (replace(SIGNALS, decision=replace(DECISION, bound_to=("old", PAGE.url, 107))), BRIEF,
     RoutingHistory(), Route("mid", "reobserve", "stale_observation")),
    (replace(SIGNALS, decision=replace(DECISION, bound_to=("s", "other", 107))), BRIEF,
     RoutingHistory(), Route("mid", "reobserve", "stale_observation")),
    (replace(SIGNALS, observation=None), BRIEF.model_copy(update={"missing_inputs": ["date"]}),
     RoutingHistory(), Route("high", "ask_user", "missing_inputs")),
    (SIGNALS, BRIEF, RoutingHistory(steps=40), Route("high", "ask_user", "step_limit")),
    (SIGNALS, BRIEF, RoutingHistory(elapsed_s=300), Route("high", "ask_user", "wall_limit")),
    (DONE, ANSWERS, RoutingHistory(llm_calls=8), Route("high", "ask_user", "llm_call_limit")),
    (DONE, ANSWERS, RoutingHistory(llm_calls=7),
     Route("mid", "llm_extract", "answer_extraction_needed")),
    (SIGNALS, BRIEF, RoutingHistory(llm_calls=8),
     Route("easy", "jev_execute", "easy_signals")),
    (SIGNALS, BRIEF, RoutingHistory(llm_calls=9), Route("high", "ask_user", "llm_call_limit")),
    (replace(SIGNALS, decision=replace(
        DECISION, operation="scroll_down", action=BrowserAction(kind="scroll_down"),
        signals=replace(SCORES, target_confidence=None, target_margin=None),
    )), BRIEF, RoutingHistory(), Route("easy", "jev_execute", "easy_signals")),
])
def test_rule_table_when_routing_conditions_and_precedence_change(
    signals: RoutingSignals, brief: TaskBrief, history: RoutingHistory, expected: Route,
) -> None:
    # Given / When
    result = route(signals, brief, replace(history, routes=[]), ledger=RunLedger())
    # Then
    assert result == expected
    print(f"rule_table {expected.reason}: {result.level}/{result.action}")


@pytest.mark.parametrize("first", ["llm_text", "llm_choose", "llm_replan", "llm_extract"])
@pytest.mark.parametrize("gap", [None, *UNAVAILABLE])
def test_second_assistance_when_state_unchanged_asks_user(
    first: str, gap: RoutingSignals | None,
) -> None:
    # Given
    signals = replace(SIGNALS, input_key=None)
    brief = BRIEF
    if first == "llm_choose":
        signals = replace(SIGNALS, decision=replace(DECISION, signals=replace(SCORES, op_margin=0)))
    if first == "llm_replan":
        brief = BRIEF.model_copy(update={"multi_step": True})
    if first == "llm_extract":
        signals, brief = DONE, ANSWERS
    history, ledger = RoutingHistory(), RunLedger()
    initial = route(signals, brief, history, ledger=ledger)
    if gap is not None:
        assert route(gap, brief, history, ledger=ledger).action == "reobserve"
    fresh_page = PAGE.model_copy(update={"snapshot_id": "new"})
    retry = replace(signals, observation=fresh_page, decision=replace(
        signals.decision, bound_to=("new", PAGE.url, 107),
    ))
    # When
    result = route(retry, brief, history, ledger=ledger)
    # Then
    assert initial.action == first
    assert result.action == "ask_user"
    assert result.reason == (
        "unchanged_after_replan" if first == "llm_replan" else "unchanged_after_assistance"
    )
    print(f"same_state {first} gap={gap is not None} -> {result.action} ({result.reason}); requests=1")


@pytest.mark.parametrize("change", ["fingerprint", "url"])
@pytest.mark.parametrize("gap", [None, *UNAVAILABLE])
def test_assistance_when_state_changes_is_available_again(
    change: str, gap: RoutingSignals | None,
) -> None:
    # Given
    history, ledger = RoutingHistory(), RunLedger()
    _ = route(DONE, ANSWERS, history, ledger=ledger)
    if gap is not None:
        assert route(gap, ANSWERS, history, ledger=ledger).action == "reobserve"
    page = PAGE.model_copy(update={change: "changed"})
    signals = replace(DONE, observation=page, decision=replace(
        DONE.decision, bound_to=(page.snapshot_id, page.url, None),
    ))
    # When
    result = route(signals, ANSWERS, history, ledger=ledger)
    # Then
    assert result.action == "llm_extract"


@pytest.mark.parametrize("first", UNAVAILABLE)
@pytest.mark.parametrize("second", UNAVAILABLE)
def test_reobserve_when_observations_remain_unavailable_is_bounded_to_one_attempt(
    first: RoutingSignals, second: RoutingSignals,
) -> None:
    # Given
    history, ledger = RoutingHistory(), RunLedger()
    assert route(first, BRIEF, history, ledger=ledger).action == "reobserve"
    # When
    result = route(second, BRIEF, history, ledger=ledger)
    # Then
    assert result == Route("high", "ask_user", "observation_unavailable")
    print(f"unavailable_observations -> {result.action} ({result.reason}); reobservations=1")


@pytest.mark.parametrize("gap", [None, *UNAVAILABLE])
def test_successful_assistance_when_input_is_now_available_returns_to_jev(
    gap: RoutingSignals | None,
) -> None:
    # Given
    history, ledger = RoutingHistory(), RunLedger()
    _ = route(replace(SIGNALS, input_key=None), BRIEF, history, ledger=ledger)
    if gap is not None:
        assert route(gap, BRIEF, history, ledger=ledger).action == "reobserve"
    # When
    result = route(SIGNALS, BRIEF, history, ledger=ledger)
    # Then
    assert result.action == "jev_execute"


def test_different_assistance_when_state_is_unchanged_shares_one_allowance() -> None:
    # Given
    history, ledger = RoutingHistory(), RunLedger()
    _ = route(replace(SIGNALS, input_key=None), BRIEF, history, ledger=ledger)
    # When
    result = route(
        SIGNALS, BRIEF.model_copy(update={"multi_step": True}), history, ledger=ledger,
    )
    # Then
    assert result == Route("high", "ask_user", "unchanged_after_assistance")


@pytest.mark.parametrize("overrides,signals,history,action", [
    ({"easy_op_conf": 0.9}, SIGNALS, RoutingHistory(), "llm_choose"),
    ({"easy_target_conf": 0.9}, SIGNALS, RoutingHistory(), "llm_choose"),
    ({"easy_margin": 0.3}, SIGNALS, RoutingHistory(), "llm_choose"),
    ({"easy_max_candidates": 149}, SIGNALS, RoutingHistory(), "llm_choose"),
    ({"stall_fingerprint_repeats": 1}, SIGNALS, RoutingHistory(fingerprints=("f", "f")), "llm_replan"),
    ({"failure_repeats": 1}, SIGNALS, RoutingHistory(consecutive_failures=1), "llm_replan"),
    ({"max_llm_calls": 0}, replace(SIGNALS, input_key=None), RoutingHistory(), "ask_user"),
    ({"max_steps": 1}, SIGNALS, RoutingHistory(steps=1), "ask_user"),
    ({"max_wall_s": 1.0}, SIGNALS, RoutingHistory(elapsed_s=1), "ask_user"),
])
def test_profile_overrides_when_configured_change_decisions(
    overrides: dict[str, float | int], signals: RoutingSignals,
    history: RoutingHistory, action: str,
) -> None:
    # Given
    profile = Profile("api", ModelRef("openai", "api", "test"), policy_overrides=dict(overrides))
    policy = RoutingPolicy.from_profile(profile)
    # When
    result = route(signals, BRIEF, replace(history, routes=[]), ledger=RunLedger(), policy=policy)
    # Then
    assert result.action == action


@pytest.mark.parametrize("overrides", [
    {"easy_op_conf": float("nan")}, {"easy_margin": 1.1}, {"failure_repeats": 0},
    {"max_steps": True}, {"max_wall_s": float("inf")}, {"unknown_setting": 1},
])
def test_policy_when_overrides_are_invalid_rejects_them(overrides: dict[str, JsonValue]) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = RoutingPolicy.model_validate(overrides)


def test_route_event_when_persisted_contains_signals_and_policy_without_inputs(tmp_path: Path) -> None:
    # Given
    sink = LedgerSink("routing", root=tmp_path)
    ledger = RunLedger(sink=sink)
    profile = Profile("api", ModelRef("openai", "api", "test"), policy_overrides={
        "policy_version": "custom-2", "easy_op_conf": 0.9,
    })
    # When
    result = route(SIGNALS, BRIEF, RoutingHistory(), ledger=ledger, policy=RoutingPolicy.from_profile(profile))
    # Then
    lines = (sink.directory / "events.jsonl").read_text().splitlines()
    event = TypeAdapter(dict[str, JsonValue]).validate_json(lines[-1])
    assert event["type"] == "route"
    data = event["data"]
    assert isinstance(data, dict)
    assert data["policy_version"] == "custom-2" and data["reason"] == result.reason
    assert data["thresholds"] == RoutingPolicy.from_profile(profile).model_dump(exclude={"policy_version"})
    assert isinstance(data["signals"], dict) and data["signals"]["op_confidence"] == 0.85
    assert isinstance(data["history"], dict) and data["history"]["llm_calls"] == 0
    assert "supplied" not in lines[-1]


@pytest.mark.parametrize("operation,binding,navigation,confidence,expected", [
    ("click", False, True, 0.85, "jev_execute"),
    ("click", False, True, 0.849, "llm_choose"),
    ("click", False, False, 1.0, "llm_choose"),
    ("type_text", True, False, 0.85, "jev_execute"),
    ("type_text", True, False, 0.849, "llm_choose"),
    ("type_text", False, True, 1.0, "llm_choose"),
    ("scroll_down", False, True, 1.0, "llm_choose"),
])
def test_structured_routing_when_operation_and_target_contract_match(
    operation: str, binding: bool, navigation: bool, confidence: float, expected: str,
) -> None:
    # Given: the loop marks only complete structured tasks with compatible targets.
    decision = replace(DECISION, operation=operation, signals=replace(
        SCORES, op_confidence=0.5, op_margin=0.01, target_confidence=confidence,
    ))
    signals = replace(
        SIGNALS, decision=decision, input_binding_unique=binding,
        structured_navigation=navigation,
    )
    # When
    result = route(signals, BRIEF, RoutingHistory(), ledger=RunLedger())
    # Then
    assert result.action == expected
