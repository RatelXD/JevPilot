from collections.abc import Sequence

import anyio
import pytest

from jevpilot.browser.port import (
    BrowserAction, BrowserConfig, BrowserPort, ElementRef, ExecutionResult, Observation,
)
from jevpilot.engine.brief import TaskBrief, UrlContains
from jevpilot.engine.recovery import (
    BLOCK_PHRASES, Failure, FailureKind, FailureSignals, Recovery, RecoveryAction, Safety,
    StepRecord, action_key, classify, may_execute, plan_recovery, record_step, settle,
)
from jevpilot.engine.verify import VerifyResult


def el(index: int, tag: str = "button", in_viewport: bool = True, **attributes: str) -> ElementRef:
    return ElementRef(
        snapshot_id="s", index=index, backend_node_id=300 + index, tag=tag, text=f"e{index}",
        attributes=attributes, in_viewport=in_viewport,
    )


def obs(
    fingerprint: str = "f1", text: str = "Results page", title: str = "Results",
    elements: list[ElementRef] | None = None,
) -> Observation:
    return Observation(
        snapshot_id=fingerprint, url="https://example.test/p", title=title, tabs=[],
        elements=[el(0), el(1, "input", type="text")] if elements is None else elements,
        page_text=text, fingerprint=fingerprint, observed_at_monotonic=0.0,
    )


def brief(missing: list[str] | None = None) -> TaskBrief:
    return TaskBrief(
        goal_restated="g", start_url="https://example.test", input_values={},
        required_outputs=[], success_conditions=[UrlContains(text="done")],
        missing_inputs=missing or [], mutating_intent=False, multi_step=False, comparison=None,
    )


CLICK = BrowserAction(kind="click", target=el(0))
OTHER_CLICK = BrowserAction(kind="click", target=el(5))
TYPE_NO_VALUE = BrowserAction(kind="type_text", target=el(1, "input", type="text"))
EXECUTED = ExecutionResult(outcome="executed")
UNCERTAIN = ExecutionResult(outcome="uncertain", error_kind="execution_failed")
STALE = ExecutionResult(outcome="failed_not_executed", error_kind="stale_target")
FAIL = VerifyResult("FAIL", "url_condition_mismatch")
UNKNOWN = VerifyResult("UNKNOWN", "evidence_missing")


def step(
    fingerprint: str = "f1", action: BrowserAction = CLICK, result: ExecutionResult = EXECUTED,
    after: str | None = "f2",
) -> StepRecord:
    return StepRecord(
        fingerprint=fingerprint, action=action_key(action), outcome=result.outcome,
        error_kind=result.error_kind, fingerprint_after=after,
    )


CASES: list[tuple[str, ExecutionResult, FailureSignals, list[StepRecord], FailureKind, str]] = [
    ("stale_target", STALE, FailureSignals(obs(), CLICK), [step(result=STALE, after="f1")],
     "transient_page", "stale_target"),
    ("empty_observation", EXECUTED, FailureSignals(obs(text="", elements=[]), CLICK), [step()],
     "transient_page", "empty_observation"),
    ("no_observation", EXECUTED, FailureSignals(None, CLICK), [step(after=None)],
     "transient_page", "empty_observation"),
    ("page_loading", EXECUTED, FailureSignals(obs(), CLICK, page_loading=True), [step()],
     "transient_page", "page_loading"),
    ("stale_observation", EXECUTED, FailureSignals(obs(), CLICK, observation_stale=True), [step()],
     "transient_page", "stale_observation"),
    ("type_text_without_value", ExecutionResult(outcome="failed_not_executed", error_kind="invalid_value"),
     FailureSignals(obs(), TYPE_NO_VALUE), [step(action=TYPE_NO_VALUE, after="f1")],
     "needs_value", "type_text_value_missing"),
    ("no_progress_twice", EXECUTED, FailureSignals(obs("f1"), CLICK),
     [step("f0", OTHER_CLICK, after="f0"), step("f1", CLICK, after="f1")],
     "plan_error", "no_progress:2"),
    ("same_failure_twice", ExecutionResult(outcome="failed_not_executed", error_kind="invalid_option"),
     FailureSignals(obs(), CLICK),
     [step(result=ExecutionResult(outcome="failed_not_executed", error_kind="invalid_option"), after="f1"),
      step("f9", OTHER_CLICK, ExecutionResult(outcome="failed_not_executed", error_kind="invalid_option"), "f9")],
     "plan_error", "same_failure_repeated:invalid_option"),
    ("missing_inputs", EXECUTED, FailureSignals(obs(), CLICK, brief(["date"])), [step()],
     "needs_user", "missing_inputs:date"),
    ("auth_screen", EXECUTED,
     FailureSignals(obs(elements=[el(0), el(1, "input", type="password")]), CLICK), [step()],
     "needs_user", "auth_screen"),
    ("safety_handoff", EXECUTED, FailureSignals(obs(), CLICK, safety="handoff"), [step()],
     "needs_user", "safety_handoff"),
    ("http_403_title", EXECUTED, FailureSignals(obs(title="403 Forbidden", text="nginx"), CLICK), [step()],
     "access_blocked", "http_status_title:403 Forbidden"),
    ("http_429_text", EXECUTED, FailureSignals(obs(text="Error 429 Too Many Requests"), CLICK), [step()],
     "access_blocked", "http_status_text:429 too many requests"),
    ("captcha_phrase", EXECUTED, FailureSignals(obs(text="Please complete the CAPTCHA"), CLICK), [step()],
     "access_blocked", "block_phrase:captcha"),
    ("uncertain_click", UNCERTAIN, FailureSignals(obs(), CLICK), [step(result=UNCERTAIN)],
     "uncertain_mutation", "uncertain:execution_failed"),
]


@pytest.mark.parametrize(
    "name,result,signals,history,kind,evidence", CASES, ids=[case[0] for case in CASES],
)
def test_classify_when_failure_signals_differ_returns_table_kind(
    name: str, result: ExecutionResult, signals: FailureSignals, history: list[StepRecord],
    kind: FailureKind, evidence: str,
) -> None:
    # When
    failure = classify(result, FAIL, signals, history)
    # Then
    assert failure == Failure(kind, evidence)
    assert failure is not None
    print(f"classify {name}: kind={failure.kind} evidence={failure.evidence} "
          f"recovery={plan_recovery(failure).action}")


@pytest.mark.parametrize("phrase", BLOCK_PHRASES)
def test_block_phrase_when_on_screen_in_any_case_is_access_blocked(phrase: str) -> None:
    failure = classify(EXECUTED, UNKNOWN, FailureSignals(obs(text=f"x {phrase.upper()} y")), [step()])
    assert failure is not None and failure.kind == "access_blocked"


def test_progress_when_executed_and_state_changed_is_not_a_failure() -> None:
    assert classify(EXECUTED, UNKNOWN, FailureSignals(obs("f2"), CLICK), [step(after="f2")]) is None


def test_verify_pass_when_page_is_otherwise_odd_is_not_a_failure() -> None:
    signals = FailureSignals(obs(text="", elements=[]), CLICK)
    assert classify(STALE, VerifyResult("PASS", "conditions_met"), signals, [step()]) is None


def test_auth_screen_when_password_field_is_off_viewport_is_ignored() -> None:
    signals = FailureSignals(obs(elements=[el(0), el(1, "input", False, type="password")]), CLICK)
    assert classify(EXECUTED, UNKNOWN, signals, [step()]) is None


@pytest.mark.parametrize("safety,expected", [
    ("confirm", "needs_user"), ("allow", None),
])
def test_c17_safety_when_confirmation_pending_needs_user(
    safety: Safety, expected: FailureKind | None,
) -> None:
    # Given
    signals = FailureSignals(obs(), CLICK, safety=safety)
    # When
    failure = classify(None, UNKNOWN, signals, [])
    # Then
    assert (failure.kind if failure is not None else None) == expected
    if failure is not None:
        assert plan_recovery(failure).action == "ask_user"
    print(f"C17 safety={safety}: failure={failure}")


@pytest.mark.parametrize("text,expected", [
    ("HTTP 403", "access_blocked"),
    ("HTTP 429", "access_blocked"),
    ("HTTP 200", None),
    ("HTTP 4030", None),
    ("Items 403 and 429 are available", None),
])
def test_c17_status_when_visible_body_contains_http_error_stops(
    text: str, expected: FailureKind | None,
) -> None:
    # Given
    signals = FailureSignals(obs(title="Error", text=text, elements=[]))
    # When
    failure = classify(None, UNKNOWN, signals, [])
    # Then
    assert (failure.kind if failure is not None else None) == expected
    if failure is not None:
        recovery = plan_recovery(failure)
        assert recovery.action == "stop_blocked"
        assert recovery.terminal_status == "blocked_by_site"
    print(f"C17 body={text!r}: failure={failure}")


@pytest.mark.parametrize("different", [None, "fingerprint", "operation", "target"])
def test_c17_loop_when_values_differ_counts_only_same_state_operation_target(
    different: str | None,
) -> None:
    # Given: every execution changes the page, so the stall guard cannot mask the loop guard.
    before = obs("same")
    actions = [
        BrowserAction(kind="type_text", target=el(1, "input"), value=value)
        for value in ("one", "two", "six")
    ]
    if different == "operation":
        actions[1] = BrowserAction(kind="click", target=el(1, "input"))
    if different == "target":
        actions[1] = BrowserAction(kind="type_text", target=el(2, "input"), value="two")
    after = [obs(f"changed-{i}") for i in range(3)]
    history = [
        record_step(
            obs("other") if different == "fingerprint" and i == 1 else before,
            action, EXECUTED, next_observation,
        )
        for i, (action, next_observation) in enumerate(zip(actions, after, strict=True))
    ]
    # When
    failure = classify(EXECUTED, UNKNOWN, FailureSignals(after[-1], actions[-1]), history)
    # Then
    if different is None:
        assert failure is not None and failure.kind == "plan_error"
        assert plan_recovery(failure).action == "llm_replan"
    else:
        assert failure is None
    print(f"C17 loop different={different}: failure={failure}")


def test_loop_guard_when_same_state_action_target_runs_three_times_is_plan_error() -> None:
    # Given: same (fingerprint, action, target) three times, each step still changing the page
    history = [step("f1", CLICK, after=f"x{i}") for i in range(3)]
    # When
    failure = classify(EXECUTED, UNKNOWN, FailureSignals(obs("x2"), CLICK), history)
    # Then
    assert failure == Failure("plan_error", "same_state_action_repeated:3")
    assert failure is not None
    assert plan_recovery(failure).action == "llm_replan"
    print(f"loop guard: kind={failure.kind} evidence={failure.evidence}")


def test_loop_guard_when_repeated_only_twice_is_not_triggered() -> None:
    history = [step("f1", CLICK, after=f"x{i}") for i in range(2)]
    assert classify(EXECUTED, UNKNOWN, FailureSignals(obs("x1"), CLICK), history) is None


def test_transient_when_repeated_twice_escalates_to_plan_error_instead_of_looping() -> None:
    history = [step(result=STALE, after="f1"), step(result=STALE, after="f1")]
    failure = classify(STALE, FAIL, FailureSignals(obs(), CLICK), history)
    assert failure is not None and failure.kind == "plan_error"


def test_access_block_when_also_uncertain_takes_precedence_and_has_no_retry() -> None:
    signals = FailureSignals(obs(text="Access Denied"), CLICK)
    failure = classify(UNCERTAIN, FAIL, signals, [step(result=UNCERTAIN)])
    assert failure is not None
    assert plan_recovery(failure) == Recovery(
        "stop_blocked", Failure("access_blocked", "block_phrase:access denied"),
        terminal_status="blocked_by_site",
    )


MAPPING: list[tuple[Failure, RecoveryAction]] = [
    (Failure("transient_page", "stale_target"), "reobserve"),
    (Failure("needs_value", "type_text_value_missing"), "llm_text"),
    (Failure("plan_error", "no_progress:2"), "llm_replan"),
    (Failure("needs_user", "missing_inputs:date"), "ask_user"),
    (Failure("needs_user", "auth_screen"), "handoff"),
    (Failure("needs_user", "safety_handoff"), "handoff"),
    (Failure("access_blocked", "block_phrase:captcha"), "stop_blocked"),
    (Failure("uncertain_mutation", "uncertain:execution_failed"), "reobserve_verify"),
]


@pytest.mark.parametrize("failure,action", MAPPING, ids=[f"{f.kind}-{a}" for f, a in MAPPING])
def test_plan_recovery_when_failure_kind_given_maps_per_plan(
    failure: Failure, action: RecoveryAction,
) -> None:
    recovery = plan_recovery(failure)
    assert recovery.action == action
    assert recovery.terminal_status == ("blocked_by_site" if action == "stop_blocked" else None)


class FakeBrowser(BrowserPort):
    def __init__(self, observations: Sequence[Observation], results: Sequence[ExecutionResult] = ()) -> None:
        self.observations = list(observations)
        self.results = list(results)
        self.observe_calls = 0
        self.executed: list[BrowserAction] = []

    async def start(self, config: BrowserConfig) -> None:
        return None

    async def observe(self) -> Observation:
        self.observe_calls += 1
        return self.observations.pop(0)

    async def execute(self, action: BrowserAction, observation: Observation) -> ExecutionResult:
        self.executed.append(action)
        return self.results.pop(0)

    async def close(self) -> None:
        return None


async def test_uncertain_click_when_jev_proposes_it_again_is_never_re_executed() -> None:
    # Given: the first click ends uncertain; Jev then proposes the same click twice, then another
    browser = FakeBrowser(
        observations=[obs("f1"), obs("f1b"), obs("f1b"), obs("f3")],
        results=[UNCERTAIN, EXECUTED],
    )
    history: list[StepRecord] = []
    recoveries: list[RecoveryAction] = []
    skipped: list[BrowserAction] = []
    before = await browser.observe()
    # When
    for proposal in [CLICK, CLICK, CLICK, OTHER_CLICK]:
        if not may_execute(proposal, history):
            skipped.append(proposal)
            continue
        result = await browser.execute(proposal, before)
        after = await browser.observe()
        history.append(record_step(before, proposal, result, after))
        failure = classify(result, UNKNOWN, FailureSignals(after, proposal), history)
        if failure is not None:
            recoveries.append(plan_recovery(failure).action)
        before = after
    # Then
    assert browser.executed == [CLICK, OTHER_CLICK]
    assert skipped == [CLICK, CLICK]
    assert recoveries == ["reobserve_verify"]
    assert may_execute(BrowserAction(kind="click", target=el(0)), history) is False
    print(f"uncertain click: executed={[action_key(a) for a in browser.executed]} "
          f"skipped_reproposals={len(skipped)} recoveries={recoveries}")


def test_uncertain_action_when_another_failure_wins_classification_stays_forbidden() -> None:
    history = [step(result=UNCERTAIN)]
    failure = classify(UNCERTAIN, FAIL, FailureSignals(obs(), CLICK, brief(["date"])), history)
    assert failure is not None and failure.kind == "needs_user"
    assert not may_execute(CLICK, history)
    assert may_execute(OTHER_CLICK, history)
    assert may_execute(None, history)


async def test_settle_when_page_changes_returns_after_first_fresh_observation() -> None:
    browser = FakeBrowser([obs("f2"), obs("f3")])
    result = await settle(browser, obs("f1"))
    assert result is not None and result.fingerprint == "f2"
    assert browser.observe_calls == 1


async def test_settle_when_page_unchanged_reobserves_at_most_twice() -> None:
    browser = FakeBrowser([obs("f1"), obs("f1"), obs("f1")])
    result = await settle(browser, obs("f1"))
    assert result is not None and result.fingerprint == "f1"
    assert browser.observe_calls == 2


class HangingBrowser(FakeBrowser):
    async def observe(self) -> Observation:
        self.observe_calls += 1
        await anyio.Event().wait()
        raise AssertionError("unreachable")


async def test_settle_when_observation_never_arrives_is_bounded_by_timeout() -> None:
    browser = HangingBrowser([])
    with anyio.fail_after(5):
        result = await settle(browser, obs("f1"), timeout_s=0.01)
    assert result is None
    assert browser.observe_calls == 1
