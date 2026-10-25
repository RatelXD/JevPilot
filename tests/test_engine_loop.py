"""Offline loop tests keep the selector, routing, recovery and parser real."""

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import time
from typing import Literal, TypedDict, override

import anyio
import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.browser.port import BrowserAction, BrowserConfig, ElementRef, ExecutionResult, Observation, Option
from jevpilot.engine.assist import ActionChoice, PlanPatch, TextValue
from jevpilot.engine.brief import AnswerFields, Comparison, Extraction, FieldValue, OpenedTarget, StructuredTask, TaskBrief, VisibleText
from jevpilot.engine.control import InteractiveControl, RunControl, ScriptedControl
from jevpilot.engine.jev import JevSelector
from jevpilot.engine.loop import RunOutcome, run_task
from jevpilot.engine.routing import RoutingPolicy
from jevpilot.ledger.models import RunSummary
from jevpilot.ledger.recorder import CallScope as LedgerScope, RunLedger
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.types import CallScope, Capabilities, LLMProvider, LLMRequest, LLMResult, LLMUsage, ModelRef


def page(text: str = "Working", *, tag: str = "button", label: str = "Next") -> Observation:
    return Observation(
        snapshot_id="s", url="https://example.test", title="Example", tabs=[],
        elements=[ElementRef(
            snapshot_id="s", index=1, backend_node_id=10, tag=tag, text=label,
            attributes={"id": "query"} if tag == "input" else {},
            in_viewport=True,
        )],
        page_text=text, fingerprint=text, observed_at_monotonic=0,
    )


TASK = StructuredTask(start_url="https://example.test", success_conditions=[VisibleText(text="Complete")])
BRIEF = TaskBrief(
    goal_restated="Finish task", start_url=TASK.start_url, input_values={},
    required_outputs=[], success_conditions=TASK.success_conditions, missing_inputs=[],
    mutating_intent=False, multi_step=False, comparison=None,
)


class Browser:
    """Each dispatched action consumes one transition; observing never advances it."""

    def __init__(self, pages: list[Observation], results: list[ExecutionResult] | None = None) -> None:
        self.pages: list[Observation] = list(pages)
        self.results: list[ExecutionResult] = list(results or [])
        self.actions: list[BrowserAction] = []
        self.observations: int = 0
        self.closed: bool = False
        self.started: anyio.Event = anyio.Event()
        self.dispatch: anyio.Event = anyio.Event()
        self.hang: bool = False

    async def start(self, config: BrowserConfig) -> None:
        assert config.start_url is None
        self.started.set()

    async def observe(self) -> Observation:
        self.observations += 1
        original = self.pages[0]
        sid = f"s{self.observations}"
        return original.model_copy(update={
            "snapshot_id": sid,
            "elements": [e.model_copy(update={"snapshot_id": sid}) for e in original.elements],
        })

    async def execute(self, action: BrowserAction, observation: Observation) -> ExecutionResult:
        if action.kind == "navigate":
            return ExecutionResult(outcome="executed")
        assert action.target is None or action.target.snapshot_id == observation.snapshot_id
        self.actions.append(action)
        self.dispatch.set()
        if self.hang:
            await anyio.Event().wait()
        if len(self.pages) > 1:
            _ = self.pages.pop(0)
        return self.results.pop(0) if self.results else ExecutionResult(outcome="executed")

    async def close(self) -> None:
        self.closed = True


class Provider:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.ref: ModelRef = ModelRef("openai", "api", "gpt-5.3-codex")
        self.replies: list[str] = list(replies or [])
        self.purposes: list[str] = []
        self.payloads: list[JsonValue] = []
        self.closed: bool = False
        self.interrupted: bool = False
        self.started: anyio.Event = anyio.Event()
        self.hang: bool = False

    def capabilities(self) -> Capabilities:
        return Capabilities("native", False, (), False)

    async def generate[T](self, request: LLMRequest[T], context: CallScope) -> LLMResult[T]:
        self.purposes.append(request.purpose)
        self.payloads.append(request.user_payload)

        async def send(_prompt: StructuredPrompt) -> RawCompletion:
            assert isinstance(context, LedgerScope)
            context.reserve_attempt()
            self.started.set()
            try:
                if self.hang:
                    await anyio.Event().wait()
            except anyio.get_cancelled_exc_class():
                self.interrupted = True
                raise
            return RawCompletion(
                self.replies.pop(0), self.ref.model_id, LLMUsage(100, 10, 0, None),
            )

        return await generate_structured(request, context, ref=self.ref, mode="native", send=send)

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class Pick:
    operation: str = "click"
    confidence: float = 1
    target_confidence: float | None = None


def wire(picks: list[Pick], seen: list[dict[str, JsonValue]], *, status: int = 200) -> httpx2.MockTransport:
    """Mock only the network: Jev request domains and response validation stay real."""
    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        seen.append(payload)
        if status != 200:
            return httpx2.Response(status, headers={"Retry-After": "0"}, json={})
        pick = picks.pop(0)
        questions = TypeAdapter(dict[str, dict[str, JsonValue]]).validate_python(payload["questions"])
        answers: dict[str, JsonValue] = {}
        for name, question in questions.items():
            criteria = question["criteria"]
            assert isinstance(criteria, dict)
            selected = pick.operation if name == "operation" else next(iter(criteria))
            answers[name] = {
                "type": "choice", "choice": selected, "confidence": (
                    pick.target_confidence if name != "operation" and pick.target_confidence is not None
                    else pick.confidence
                ),
                "probabilities": {key: float(key == selected) for key in criteria},
            }
        return httpx2.Response(200, json={
            "model": "jev-1.14.0", "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        })
    return httpx2.MockTransport(handler)


type RunEvidence = tuple[RunOutcome, RunLedger, list[dict[str, JsonValue]]]


class Event(TypedDict):
    type: str
    data: dict[str, JsonValue]


async def run(
    tmp_path: Path, browser: Browser, provider: LLMProvider, picks: list[Pick], *,
    goal: str | StructuredTask = TASK, control: RunControl | None = None,
    policy: RoutingPolicy | None = None, mode: Literal["adaptive", "jev_only"] = "adaptive",
    status: int = 200, clock: Callable[[], float] | None = None,
) -> RunEvidence:
    sink = LedgerSink("loop", root=tmp_path)
    ledger = RunLedger("loop", sink=sink, clock=clock or time.monotonic)
    seen: list[dict[str, JsonValue]] = []
    async with httpx2.AsyncClient(transport=wire(picks, seen, status=status)) as http:
        result = await run_task(
            goal, browser=browser, llm=provider,
            jev=JevSelector(api_key="synthetic", http_client=http),
            control=control or ScriptedControl(), ledger=ledger,
            policy=policy or RoutingPolicy(), mode=mode,
        )
    # Every exit must close spans and persist an outcome plus matching receipt.
    assert all(span.end_s is not None for span in ledger.spans.values())
    assert RunSummary.model_validate_json((sink.directory / "receipt.json").read_text()) == result.receipt
    terminal = TypeAdapter(Event).validate_json((sink.directory / "events.jsonl").read_text().splitlines()[-1])
    assert terminal["type"] == "span"  # root closes after the terminal event
    events = [TypeAdapter(Event).validate_json(line)["data"] for line in (sink.directory / "events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "run_outcome" and e["status"] == result.status for e in events)
    assert browser.closed
    return result, ledger, seen


@pytest.mark.parametrize("natural", [False, True])
async def test_easy_completion_when_inputs_complete_uses_only_brief_if_natural(
    tmp_path: Path, natural: bool,
) -> None:
    # Given
    browser, provider = Browser([page(), page("Complete")]), Provider([BRIEF.model_dump_json()])
    # When
    result, ledger, seen = await run(
        tmp_path, browser, provider, [Pick()], goal="Finish task" if natural else TASK,
    )
    # Then
    assert result.status == "verified" and len(seen) == 1
    assert provider.purposes == (["brief"] if natural else [])
    assert ledger.summary().by_kind["llm"].calls == int(natural)
    assert provider.closed and len(browser.actions) == 1


async def test_missing_value_when_text_generated_returns_to_fresh_jev(tmp_path: Path) -> None:
    # Given
    browser = Browser([page(tag="input"), page("Complete")])
    provider = Provider([TextValue(value="Aurora", rationale="given").model_dump_json()])
    # When
    result, _, seen = await run(tmp_path, browser, provider, [Pick("type_text"), Pick("type_text")])
    # Then
    assert result.status == "verified" and len(seen) == 2
    assert provider.purposes == ["text"]
    assert browser.actions[0].value == "Aurora"
    assert browser.actions[0].target is not None
    assert browser.actions[0].target.snapshot_id == "s3"


async def test_multistep_brief_when_input_key_differs_does_not_spend_replan_before_text(
    tmp_path: Path,
) -> None:
    # Given: natural briefs supply semantic names, not DOM IDs.
    browser = Browser([page(tag="input"), page("Complete")])
    provider = Provider([
        BRIEF.model_copy(update={
            "multi_step": True, "input_values": {"search_term": "Aurora"},
        }).model_dump_json(),
        TextValue(value="Aurora", rationale="given").model_dump_json(),
    ])
    # When
    result, _, seen = await run(
        tmp_path, browser, provider, [Pick("type_text")] * 2, goal="Search and open",
    )
    # Then
    assert result.status == "verified" and len(seen) == 2
    assert provider.purposes == ["brief", "text"]
    assert browser.actions[0].value == "Aurora"


async def test_stall_when_replanned_next_selection_is_jev(tmp_path: Path) -> None:
    # Given
    browser = Browser([page(), page(), page(), page("Complete")])
    provider = Provider([PlanPatch(subgoals=["Try next"], next_hint="Continue").model_dump_json()])
    # When
    result, _, seen = await run(tmp_path, browser, provider, [Pick()] * 4)
    # Then
    assert result.status == "verified" and provider.purposes == ["replan"]
    assert len(seen) == 4 and len(browser.actions) == 3
    state = seen[-1]["state"]
    assert isinstance(state, dict) and state["subgoal"] == "Try next\nContinue"


async def test_stale_target_when_not_executed_reobserves_then_selects(tmp_path: Path) -> None:
    # Given
    browser = Browser(
        [page(), page(), page("Complete")],
        [ExecutionResult(outcome="failed_not_executed", error_kind="stale_target")],
    )
    provider = Provider()
    # When
    result, _, seen = await run(tmp_path, browser, provider, [Pick(), Pick()])
    # Then
    assert result.status == "verified" and len(seen) == 2
    assert browser.actions[0].target is not None and browser.actions[1].target is not None
    assert browser.actions[0].target.snapshot_id != browser.actions[1].target.snapshot_id
    assert provider.purposes == [] and browser.observations == 5


async def test_missing_inputs_when_answered_are_supplied_to_jev(tmp_path: Path) -> None:
    # Given
    provider = Provider([BRIEF.model_copy(update={"missing_inputs": ["query"]}).model_dump_json()])
    browser = Browser([page(tag="input"), page("Complete")])
    control = InteractiveControl()
    results: list[RunEvidence] = []

    async def engine() -> None:
        results.append(await run(
            tmp_path, browser, provider, [Pick("type_text")] * 2, goal="Finish task", control=control,
        ))

    # When
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            assert (await control.prompts.receive()).kind == "ask"
            control.answer("Aurora")
    # Then
    assert results[0][0].status == "verified"
    assert browser.actions[0].value == "Aurora" and provider.purposes == ["brief"]
    assert len(results[0][2]) == 2  # Select before asking, then reselect the fresh field.


@pytest.mark.parametrize("label,status,reason", [
    ("Reserve", "failed", "confirmation_denied"),
    ("Sign in", "handoff_ended", "needs_user"),
    ("Pay now", "handoff_ended", "needs_user"),
])
async def test_pending_inputs_when_guarded_action_ready_reach_guard_first(
    tmp_path: Path, label: str, status: str, reason: str,
) -> None:
    # Given: future form details are not prerequisites for reaching a guard.
    provider = Provider([BRIEF.model_copy(update={
        "missing_inputs": ["details needed later"],
    }).model_dump_json()])
    browser = Browser([page(label=label)])
    # When
    result, _, _ = await run(tmp_path, browser, provider, [Pick()], goal="Continue")
    # Then
    assert (result.status, result.reason) == (status, reason)
    assert browser.actions == []


@pytest.mark.parametrize("attributes", [
    {"autocomplete": "cc-number"}, {"aria-label": "Card Number"},
])
async def test_sensitive_input_when_value_missing_hands_off_before_asking(
    tmp_path: Path, attributes: dict[str, str],
) -> None:
    # Given: the live route requested payment information before its guard ran.
    initial = page(tag="input")
    initial = initial.model_copy(update={"elements": [
        initial.elements[0].model_copy(update={"attributes": attributes}),
    ]})
    provider = Provider([BRIEF.model_copy(update={
        "missing_inputs": ["payment information"],
    }).model_dump_json()])
    browser = Browser([initial])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick("type_text", 0.69)],
        goal="Continue payment",
    )
    # Then
    assert result.status == "handoff_ended" and result.reason == "needs_user"
    assert provider.purposes == ["brief"] and browser.actions == []


async def test_input_length_progress_when_page_text_unchanged_allows_next_assistance(
    tmp_path: Path,
) -> None:
    # Given: private input text is retained only as a length by the BrowserPort.
    empty = page(tag="input")
    filled = empty.model_copy(update={"elements": [
        empty.elements[0].model_copy(update={"attributes": {"id": "query", "value": "6"}}),
    ]})
    browser = Browser([empty, filled, page("Complete")])
    provider = Provider([
        BRIEF.model_dump_json(),
        TextValue(value="secret", rationale="supplied").model_dump_json(),
        ActionChoice(operation="type_text", target_index=1, option_value=None).model_dump_json(),
    ])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider,
        [Pick("type_text"), Pick("type_text"), Pick("type_text", 0.5), Pick("type_text", 0.5)],
        goal="Complete the fields",
    )
    # Then
    assert result.status == "verified"
    assert provider.purposes == ["brief", "text", "choose"]
    assert len(browser.actions) == 2
    choice_payload = provider.payloads[-1]
    assert isinstance(choice_payload, dict)
    assert choice_payload["recent_actions"] == ["type_text target=10 outcome=executed"]


async def test_confirmation_when_denied_never_dispatches(tmp_path: Path) -> None:
    # Given / When
    browser = Browser([page(label="Submit")])
    result, _, _ = await run(tmp_path, browser, Provider(), [Pick()])
    # Then
    assert (result.status, result.reason) == ("failed", "confirmation_denied")
    assert browser.actions == []


async def test_payment_when_resumed_reobserves_without_recording_handoff(tmp_path: Path) -> None:
    # Given
    browser, provider, control = Browser([page(label="Pay now")]), Provider(), InteractiveControl()
    results: list[RunEvidence] = []
    count = 0

    async def engine() -> None:
        results.append(await run(tmp_path, browser, provider, [Pick()], control=control))

    # When
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            assert (await control.prompts.receive()).kind == "handoff"
            assert control.recording_paused and browser.actions == []
            count = browser.observations
            browser.pages[:] = [page("Complete")]
            control.resume()
    # Then
    assert results[0][0].status == "verified"
    assert browser.observations == count + 1 and browser.actions == []


@pytest.mark.parametrize("external", [False, True])
async def test_cancel_when_llm_inflight_interrupts_and_closes_all_spans(
    tmp_path: Path, external: bool,
) -> None:
    # Given
    browser, provider, control = Browser([page()]), Provider(), InteractiveControl()
    provider.hang = True
    results: list[RunEvidence] = []
    cancel = anyio.CancelScope()

    async def engine() -> None:
        with cancel:
            results.append(await run(tmp_path, browser, provider, [], goal="Finish", control=control))

    # When
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            await provider.started.wait()
            if external:
                cancel.cancel()
            else:
                control.request_stop()
    # Then
    assert results[0][0].status == "cancelled"
    assert provider.interrupted and provider.closed
    assert results[0][1].calls[0].outcome == "cancelled"


async def test_cancel_when_mutation_dispatched_records_uncertain(tmp_path: Path) -> None:
    # Given
    browser, control = Browser([page()]), InteractiveControl()
    browser.hang = True
    results: list[RunEvidence] = []

    async def engine() -> None:
        results.append(await run(tmp_path, browser, Provider(), [Pick()], control=control))

    # When
    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(engine)
            await browser.dispatch.wait()
            control.request_stop()
    # Then
    assert results[0][0].status == "cancelled" and len(browser.actions) == 1
    events = [TypeAdapter(Event).validate_json(line)["data"] for line in (tmp_path / "loop/events.jsonl").read_text().splitlines()]
    assert any(e.get("outcome") == "uncertain" for e in events)


async def test_jev_only_when_low_confidence_and_stalled_never_chooses_or_replans(tmp_path: Path) -> None:
    # Given
    browser, provider = Browser([page(), page(), page(), page("Complete")]), Provider()
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick(confidence=0.5)] * 3, mode="jev_only",
    )
    # Then
    assert result.status == "verified" and provider.purposes == []


async def test_jev_when_429_persists_ends_unavailable(tmp_path: Path) -> None:
    # Given / When
    result, ledger, seen = await run(tmp_path, Browser([page()]), Provider(), [], status=429)
    # Then
    assert (result.status, result.reason) == ("failed", "jev_unavailable")
    assert len(seen) == len(ledger.calls) == 4


@pytest.mark.parametrize("limit", ["steps", "llm", "wall"])
async def test_budget_when_exhausted_persists_incomplete(tmp_path: Path, limit: str) -> None:
    # Given
    browser = Browser([page(), page()])
    provider = Provider()
    policy = RoutingPolicy(max_steps=1)
    clock = None
    picks = [Pick()]
    if limit == "llm":
        policy = RoutingPolicy(max_llm_calls=0)
        picks = [Pick(confidence=0.5)]
    if limit == "wall":
        ticks = iter([0.0, 0.0, *range(1, 100)])
        clock = lambda: float(next(ticks))
        policy = RoutingPolicy(max_wall_s=1)
        picks = []
    # When
    result, _, _ = await run(tmp_path, browser, provider, picks, policy=policy, clock=clock)
    # Then
    assert result.status == "incomplete" and result.reason.endswith("_limit")
    assert provider.purposes == []


@pytest.mark.parametrize("end", ["blocked", "handoff", "uncertain"])
async def test_terminal_when_blocked_handed_off_or_uncertain_never_repeats(
    tmp_path: Path, end: str,
) -> None:
    # Given
    browser = Browser([page("HTTP 403") if end == "blocked" else page(label="Pay" if end == "handoff" else "Next")])
    if end == "uncertain":
        browser.results = [ExecutionResult(outcome="uncertain")]
    # When
    result, _, _ = await run(tmp_path, browser, Provider(), [] if end == "blocked" else [Pick()])
    # Then
    assert result.status == {"blocked": "blocked_by_site", "handoff": "handoff_ended", "uncertain": "incomplete"}[end]
    assert len(browser.actions) == int(end == "uncertain")


@pytest.mark.parametrize("grounded", [False, True])
async def test_extraction_when_quotes_checked_in_code_cannot_invent_success(
    tmp_path: Path, grounded: bool,
) -> None:
    # Given
    task = StructuredTask(start_url=TASK.start_url, success_conditions=[AnswerFields(names=["year"])])
    provider = Provider([Extraction(
        fields={"year": FieldValue(value="1397", quote="Year 1397" if grounded else "Fake 1397")},
        no_result=False, no_result_quote=None, compared_items=[],
    ).model_dump_json()] * 2)
    # When
    result, _, _ = await run(tmp_path, Browser([page("Year 1397")]), provider, [Pick("done")], goal=task)
    # Then
    assert result.status == ("verified" if grounded else "failed")
    if not grounded:
        assert result.reason == "llm_invalid_output" and result.receipt.by_kind["llm"].calls == 2
    assert provider.purposes == ["extract"]


async def test_choose_when_supported_rebinds_after_new_jev_selection(tmp_path: Path) -> None:
    # Given
    browser = Browser([page(), page("Complete")])
    provider = Provider([ActionChoice(operation="click", target_index=1, option_value=None).model_dump_json()])
    # When
    result, _, seen = await run(tmp_path, browser, provider, [Pick(confidence=0.5)] * 2)
    # Then
    assert result.status == "verified" and len(seen) == 2
    assert browser.actions[0].target is not None
    assert provider.purposes == ["choose"] and browser.actions[0].target.snapshot_id == "s3"


async def test_structured_supplied_field_when_unique_needs_no_llm(tmp_path: Path) -> None:
    # Given: the DOM binding determines the sole supplied field, not a score override.
    browser = Browser([page(tag="input"), page("Complete")])
    provider = Provider([
        ActionChoice(operation="type_text", target_index=1, option_value=None).model_dump_json(),
    ])
    task = TASK.model_copy(update={"input_values": {"query": "supplied"}})
    # When
    result, ledger, _ = await run(
        tmp_path, browser, provider, [Pick("type_text", 0.5, 1)] * 2, goal=task,
    )
    # Then
    assert result.status == "verified"
    assert browser.actions[0].value == "supplied"
    assert provider.purposes == [] and ledger.summary().by_kind["llm"].calls == 0


@pytest.mark.parametrize("natural,duplicate", [(True, False), (False, True)])
async def test_supplied_field_when_binding_not_structurally_unique_keeps_assistance(
    tmp_path: Path, natural: bool, duplicate: bool,
) -> None:
    # Given: neither natural interpretation nor duplicate DOM identifiers proves uniqueness.
    initial = page(tag="input")
    if duplicate:
        initial = initial.model_copy(update={"elements": [
            *initial.elements,
            initial.elements[0].model_copy(update={"index": 2, "backend_node_id": 20}),
        ]})
    brief = BRIEF.model_copy(update={"input_values": {"query": "supplied"}})
    provider = Provider([
        *([brief.model_dump_json()] if natural else []),
        ActionChoice(operation="type_text", target_index=1, option_value=None).model_dump_json(),
    ])
    browser = Browser([initial, page("Complete")])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick("type_text", 0.5)] * 2,
        goal="Finish task" if natural else TASK.model_copy(update={"input_values": brief.input_values}),
    )
    # Then
    assert result.status == "verified" and browser.actions[0].value == "supplied"
    assert provider.purposes == (["brief", "choose"] if natural else ["choose"])


@pytest.mark.parametrize("next_pick", ["click", "type_text"])
async def test_chosen_type_when_supplied_value_exists_keeps_matching_value(
    tmp_path: Path, next_pick: str,
) -> None:
    # Given: the fresh Jev action need not target the field chosen by assistance.
    initial = page(tag="input")
    button = page().elements[0].model_copy(update={"index": 2, "backend_node_id": 20})
    initial = initial.model_copy(update={"elements": [*initial.elements, button]})
    browser = Browser([initial, page("Complete")])
    provider = Provider([
        ActionChoice(operation="type_text", target_index=1, option_value=None).model_dump_json(),
    ])
    task = TASK.model_copy(update={"input_values": {"query": "supplied"}})
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick("click", 0.5), Pick(next_pick, 0.5)], goal=task,
    )
    # Then
    assert result.status == "verified"
    assert provider.purposes == ["choose"]
    assert browser.actions[0].kind == "type_text" and browser.actions[0].value == "supplied"


async def test_answer_when_confident_navigation_pending_executes_before_extraction(tmp_path: Path) -> None:
    # Given: only the destination contains the requested evidence.
    browser = Browser([page("Read the details"), page("Year 1397")])
    provider = Provider([Extraction(
        fields={"year": FieldValue(value="1397", quote="Year 1397")},
        no_result=False, no_result_quote=None, compared_items=[],
    ).model_dump_json()])
    task = TASK.model_copy(update={"success_conditions": [AnswerFields(names=["year"])]})
    # When
    result, _, _ = await run(tmp_path, browser, provider, [Pick(), Pick("done")], goal=task)
    # Then
    assert result.status == "verified" and result.steps == 1
    assert [action.kind for action in browser.actions] == ["click"]
    assert provider.purposes == ["extract"]


async def test_receipt_when_clock_overlaps_startup_attributes_time_and_costs(tmp_path: Path) -> None:
    # Given
    now = [0.0]
    browser = Browser([page(), page("Complete")])

    class TimedProvider(Provider):
        @override
        async def generate[T](self, request: LLMRequest[T], context: CallScope) -> LLMResult[T]:
            await browser.started.wait()
            now[0] = 2
            result = await super().generate(request, context)
            now[0] = 4
            return result

    class TimedBrowser(Browser):
        @override
        async def start(self, config: BrowserConfig) -> None:
            assert config.start_url is None
            browser.started.set()
            await provider.started.wait()
            now[0] = 5

    provider = TimedProvider([BRIEF.model_dump_json()])
    timed = TimedBrowser(browser.pages)
    # When
    result, ledger, _ = await run(
        tmp_path, timed, provider, [Pick()], goal="Finish", clock=lambda: now[0],
    )
    # Then
    assert result.status == "verified"
    assert result.receipt.overlap_s == 4
    assert sum(result.receipt.duration_s.values()) == result.receipt.elapsed_s == 5
    for kind in ("jev", "llm"):
        expected = sum((c.api_estimate_usd or Decimal(0) for c in ledger.calls if c.kind == kind), Decimal(0))
        assert result.receipt.by_kind[kind].api_estimate_usd.known_sum == expected


async def test_startup_when_browser_fails_returns_failed_and_receipt(tmp_path: Path) -> None:
    # Given
    class UnavailableBrowser(Browser):
        @override
        async def start(self, config: BrowserConfig) -> None:
            assert config.start_url is None
            raise RuntimeError("synthetic-private-error")

    browser = UnavailableBrowser([page()])
    # When
    result, _, _ = await run(tmp_path, browser, Provider(), [])
    # Then
    assert (result.status, result.reason) == ("failed", "browser_unavailable")
    assert "synthetic-private-error" not in (tmp_path / "loop/events.jsonl").read_text()


async def test_brief_when_zero_budget_returns_precise_limit_reason(tmp_path: Path) -> None:
    # Given / When
    result, _, _ = await run(
        tmp_path, Browser([page()]), Provider(), [], goal="Finish",
        policy=RoutingPolicy(max_llm_calls=0),
    )
    # Then
    assert (result.status, result.reason) == ("incomplete", "llm_call_limit")


async def test_select_when_supplied_option_executes_and_records_public_value(tmp_path: Path) -> None:
    # Given
    initial = page(tag="select")
    initial = initial.model_copy(update={"elements": [
        initial.elements[0].model_copy(update={"options": [Option(value="north", label="North")]}),
    ]})
    browser = Browser([initial, page("Complete")])
    # When
    result, _, _ = await run(
        tmp_path, browser, Provider(), [Pick("select_option")],
        goal=TASK.model_copy(update={"input_values": {"region": "North"}}),
    )
    # Then
    assert result.status == "verified" and browser.actions[0].value == "north"
    events = [
        TypeAdapter(Event).validate_json(line)["data"]
        for line in (tmp_path / "loop/events.jsonl").read_text().splitlines()
    ]
    execution = next(e for e in events if e.get("event") == "execution")
    assert execution["operation"] == "select_option"
    assert execution["value"] == "north"


@pytest.mark.parametrize("max_steps", [1, 2])
async def test_open_goal_when_suggestion_visible_requires_actual_open(
    tmp_path: Path, max_steps: int,
) -> None:
    # Given: the same name appears after typing, but the requested item is not open.
    initial = page("Search", tag="input")
    suggestion = page("Aurora", label="Aurora")
    destination = page("Aurora details").model_copy(update={"url": "https://example.test/item"})
    browser = Browser([initial, suggestion, destination])
    brief = BRIEF.model_copy(update={
        "success_conditions": [OpenedTarget(text="Aurora")],
        "input_values": {"query": "Aurora"},
    })
    provider = Provider([brief.model_dump_json()])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick("type_text"), Pick()],
        goal="Search and open the result", policy=RoutingPolicy(max_steps=max_steps),
    )
    # Then
    assert result.status == ("incomplete" if max_steps == 1 else "verified")
    assert len(browser.actions) == max_steps


async def test_initial_listing_when_target_already_visible_is_not_completion(tmp_path: Path) -> None:
    # Given
    initial = page("Aurora", label="Aurora").model_copy(update={"elements": [
        page(label="Aurora").elements[0].model_copy(update={"attributes": {"href": "/item?region=north"}}),
    ]})
    final = page("Aurora details").model_copy(update={"url": "https://example.test/item?region=north"})
    brief = BRIEF.model_copy(update={"success_conditions": [OpenedTarget(text="Aurora")]})
    browser = Browser([initial, final])
    # When
    result, _, seen = await run(
        tmp_path, browser, Provider([brief.model_dump_json()]), [Pick()], goal="Open an item",
    )
    # Then
    assert result.status == "verified" and len(seen) == len(browser.actions) == 1
    assert result.final_url == final.url


@pytest.mark.parametrize("outcome", ["executed", "uncertain", "failed_not_executed"])
@pytest.mark.parametrize("settled", [True, False])
async def test_delayed_open_waits_for_dispatched_navigation_without_second_click(
    tmp_path: Path, outcome: str, settled: bool,
) -> None:
    # Given: an observation can still describe the source after dispatch.
    initial = page("Widget", label="Widget")
    destination = page("Widget details").model_copy(update={
        "url": "https://example.test/detail", "document_id": "loader-click",
    })

    class DelayedBrowser(Browser):
        waited: bool = False

        @override
        async def execute(self, action: BrowserAction, observation: Observation) -> ExecutionResult:
            result = await super().execute(action, observation)
            if action.kind == "click":
                return ExecutionResult.model_validate({"outcome": outcome}).model_copy(
                    update={"navigation_id": "loader-click"},
                )
            return result

        async def wait_for_navigation(self, navigation_id: str) -> None:
            assert navigation_id == "loader-click"
            self.waited = True
            if settled:
                self.pages[:] = [destination]

    browser = DelayedBrowser([initial])
    task = StructuredTask(
        start_url=initial.url, success_conditions=[OpenedTarget(text="Widget")],
    )
    # When
    result, _, _ = await run(
        tmp_path, browser, Provider(), [Pick(), Pick("done")],
        goal=task, policy=RoutingPolicy(max_steps=1),
    )
    # Then
    assert result.status == ("verified" if outcome == "executed" and settled else "incomplete")
    if outcome == "executed" and not settled:
        assert result.reason == "navigation_unsettled"
    assert browser.waited == (outcome == "executed")
    assert len(browser.actions) == 1


@pytest.mark.parametrize("natural,tag,label,attributes,target_score,expected", [
    (False, "a", "Details", {"href": "/detail"}, 0.85, []),
    (False, "button", "Next", {"type": "button"}, 0.85, []),
    (True, "a", "Details", {"href": "/detail"}, 0.85, ["brief", "choose"]),
    (False, "a", "Details", {"href": "/detail"}, 0.849, ["choose"]),
    (False, "input", "Search", {"id": "query"}, 1.0, ["choose"]),
])
async def test_structured_target_when_compatible_uses_target_confidence(
    tmp_path: Path, natural: bool, tag: str, label: str,
    attributes: dict[str, str], target_score: float, expected: list[str],
) -> None:
    # Given: low operation confidence alone must not force structured navigation assistance.
    initial = page(tag=tag, label=label)
    initial = initial.model_copy(update={"elements": [
        initial.elements[0].model_copy(update={"attributes": attributes}),
    ]})
    # A click on an input is not a candidate; use type_text without a supplied value.
    operation = "type_text" if tag == "input" else "click"
    provider = Provider([
        *([BRIEF.model_dump_json()] if natural else []),
        *([TextValue(value="supplied", rationale="needed").model_dump_json()]
          if tag == "input" else [
              ActionChoice(operation="click", target_index=1, option_value=None).model_dump_json(),
          ]),
    ])
    browser = Browser([initial, page("Complete")])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick(operation, 0.5, target_score)] * 2,
        goal="Finish task" if natural else TASK,
    )
    # Then
    assert result.status == "verified"
    assert provider.purposes == (["text"] if tag == "input" else expected)


@pytest.mark.parametrize("observed", [True, False])
async def test_comparison_when_prior_list_observed_grounds_only_seen_items(
    tmp_path: Path, observed: bool,
) -> None:
    # Given: only the current detail contains the selected answer.
    listing = page("A $10 B $20" if observed else "A $10")
    detail = page("B $20").model_copy(update={"url": "https://example.test/detail"})
    brief = BRIEF.model_copy(update={
        "success_conditions": [AnswerFields(names=["price"])],
        "comparison": Comparison(field="price", kind="min"),
    })
    # Winner stays current while one compared value exists only in a prior list.
    detail = detail.model_copy(update={"page_text": "A $10", "fingerprint": "detail"})
    provider = Provider([brief.model_dump_json(), Extraction(
        fields={"price": FieldValue(value="$10", quote="A $10")},
        no_result=False, no_result_quote=None,
        compared_items=[FieldValue(value="$10", quote="A $10"), FieldValue(value="$20", quote="B $20")],
    ).model_dump_json()])
    browser = Browser([listing, detail])
    # When
    result, _, _ = await run(
        tmp_path, browser, provider, [Pick(), Pick("done")], goal="Find the lowest price",
    )
    # Then
    assert result.status == ("verified" if observed else "unverified")
    events = [
        TypeAdapter(Event).validate_json(line)["data"]
        for line in (tmp_path / "loop/events.jsonl").read_text().splitlines()
    ]
    if observed:
        grounding = next(e for e in events if e.get("event") == "comparison_sources")
        assert listing.url in str(grounding["sources"])


@pytest.mark.parametrize("listing_text,accepted", [
    ("A Journey Through ... $20", True),
    ("a journey through… $20", True),
    ("A Journey Through Mountains $20", True),
    ("An Unrelated Journey ... $20", False),
    ("A Journey Through ... $30", False),
    ("A Journey ... $20", False),
])
async def test_comparison_when_name_truncated_requires_observed_prefix_and_value(
    tmp_path: Path, listing_text: str, accepted: bool,
) -> None:
    # Given: the losing item's full name is absent from the current detail.
    listing = page(listing_text)
    detail = page("Winner $50").model_copy(update={"url": "https://example.test/detail"})
    brief = BRIEF.model_copy(update={
        "success_conditions": [AnswerFields(names=["price"])],
        "comparison": Comparison(field="price", kind="max"),
    })
    extraction = Extraction(
        fields={"price": FieldValue(value="$50", quote="Winner $50")},
        no_result=False, no_result_quote=None,
        compared_items=[
            FieldValue(value="$20", quote="A Journey Through Mountains $20"),
            FieldValue(value="$50", quote="Winner $50"),
        ],
    )
    provider = Provider([brief.model_dump_json(), extraction.model_dump_json()])
    # When
    result, _, _ = await run(
        tmp_path, Browser([listing, detail]), provider, [Pick(), Pick("done")],
        goal="Find the highest price",
    )
    # Then
    assert result.status == ("verified" if accepted else "unverified")
    events = [
        TypeAdapter(Event).validate_json(line)["data"]
        for line in (tmp_path / "loop/events.jsonl").read_text().splitlines()
    ]
    raw = next(event for event in events if event.get("event") == "extraction_response")
    assert Extraction.model_validate_json(str(raw["raw_output"])) == extraction
    if accepted:
        sources = next(event for event in events if event.get("event") == "comparison_sources")
        assert listing.url in str(sources["sources"])
