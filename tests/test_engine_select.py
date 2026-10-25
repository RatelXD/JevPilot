from collections.abc import Mapping, Sequence
import json

import httpx2
import pytest

from jevpilot.browser.port import ElementRef, Observation, Option
from jevpilot.engine.candidates import build_candidates
from jevpilot.engine.jev import (
    GoalContext, JevChoice, JevSelector, JevUnavailable, OptionChoice, SelectorState,
    select_step,
)
from jevpilot.ledger import RunLedger
from jevpilot.ledger.recorder import CallScope
from jevpilot.telemetry.models import CallUsage

OPTIONS = [
    Option(value="", label="Select a region"),
    Option(value="north", label="North"),
    Option(value="south", label="South"),
]


def el(index: int, tag: str, **attributes: str) -> ElementRef:
    return ElementRef(
        snapshot_id="snap", index=index, backend_node_id=500 + index, tag=tag, role=None,
        text=f"e{index}", attributes=attributes, in_viewport=True,
        options=list(OPTIONS) if tag == "select" else [],
    )


OBSERVATION = Observation(
    snapshot_id="snap", url="https://example.com/f", title="t", tabs=[],
    elements=[el(0, "select"), el(1, "button"), el(2, "input", type="text"),
              el(3, "input", type="password")],
    page_text="", fingerprint="f", observed_at_monotonic=0.0,
)
CONTEXT = GoalContext(goal="g")


def scope() -> CallScope:
    ledger = RunLedger()
    return ledger.scope(deadline=ledger.clock() + 60)


class FakeSelector(JevSelector):
    def __init__(self, choice: JevChoice, option: OptionChoice | None = None) -> None:
        super().__init__(api_key="k")
        self.choice, self.option = choice, option
        self.choose_calls = 0
        self.option_calls = 0
        self.seen: list[SelectorState] = []

    async def choose(
        self, state: SelectorState, *, scope: CallScope,
        domains: Mapping[str, Sequence[ElementRef]] | None = None,
    ) -> JevChoice:
        self.choose_calls += 1
        self.seen.append(state)
        return self.choice

    async def choose_option(
        self, state: SelectorState, target: ElementRef, *, scope: CallScope,
    ) -> OptionChoice:
        self.option_calls += 1
        assert self.option is not None
        return self.option


def choice(operation: str, target_index: int | None = None) -> JevChoice:
    return JevChoice(
        operation=operation, op_confidence=0.9, op_margin=0.5, target_index=target_index,
        target_confidence=0.8 if target_index is not None else None,
        target_margin=0.4 if target_index is not None else None,
        candidate_count=3, truncated=False, usage=CallUsage(input_tokens=10), latency_s=0.1,
    )


def option(value: str) -> OptionChoice:
    return OptionChoice(value=value, confidence=0.7, margin=0.3, usage=CallUsage(input_tokens=5),
                        latency_s=0.2)


async def test_exact_input_value_selects_option_without_extra_jev_call() -> None:
    selector = FakeSelector(choice("select_option", 0))
    step = await select_step(OBSERVATION, CONTEXT, {"region": "  NORTH "}, selector=selector, scope=scope())
    assert step.action is not None
    assert (step.action.kind, step.action.value) == ("select_option", "north")
    assert selector.option_calls == 0 and len(step.usages) == 1
    assert step.signals.option_confidence is None
    assert step.bound_to == ("snap", "https://example.com/f", 500)


async def test_fullwidth_and_option_value_match_normalize() -> None:
    selector = FakeSelector(choice("select_option", 0))
    step = await select_step(OBSERVATION, CONTEXT, {"r": "ｓｏｕｔｈ"}, selector=selector, scope=scope())
    assert step.action is not None and step.action.value == "south"


async def test_no_match_calls_choose_option_and_records_signals() -> None:
    selector = FakeSelector(choice("select_option", 0), option("south"))
    step = await select_step(OBSERVATION, CONTEXT, {"region": "West"}, selector=selector, scope=scope())
    assert selector.option_calls == 1
    assert step.action is not None and step.action.value == "south"
    assert (step.signals.option_confidence, step.signals.option_margin) == (0.7, 0.3)
    assert len(step.usages) == 2 and step.latency_s == pytest.approx(0.3)


async def test_ambiguous_match_falls_back_to_choose_option() -> None:
    selector = FakeSelector(choice("select_option", 0), option("north"))
    step = await select_step(
        OBSERVATION, CONTEXT, {"a": "North", "b": "South"}, selector=selector, scope=scope(),
    )
    assert selector.option_calls == 1
    assert step.action is not None and step.action.value == "north"


async def test_option_outside_target_options_is_rejected() -> None:
    selector = FakeSelector(choice("select_option", 0), option("east"))
    with pytest.raises(JevUnavailable) as raised:
        _ = await select_step(OBSERVATION, CONTEXT, {}, selector=selector, scope=scope())
    assert raised.value.kind == "invalid_output"


async def test_type_text_has_no_value_and_selector_never_sees_password_field() -> None:
    selector = FakeSelector(choice("type_text", 2))
    step = await select_step(OBSERVATION, CONTEXT, {"q": "hello"}, selector=selector, scope=scope())
    assert step.action is not None
    assert (step.action.kind, step.action.value) == ("type_text", None)
    assert [e.index for e in selector.seen[0].observation.elements] == [0, 1, 2]
    assert step.signals.target_margin == 0.4 and step.signals.candidate_count == 3


async def test_target_outside_candidates_is_rejected() -> None:
    selector = FakeSelector(choice("type_text", 3))
    with pytest.raises(JevUnavailable):
        _ = await select_step(OBSERVATION, CONTEXT, {}, selector=selector, scope=scope())


def real_el(
    index: int, tag: str, role: str | None = None, *,
    has_onclick: bool = False, contenteditable: bool = False,
) -> ElementRef:
    attributes = {"type": "password"} if tag == "password" else {"type": "text"} if tag == "input" else {}
    return ElementRef(
        snapshot_id="snap", index=index, backend_node_id=500 + index,
        tag="input" if tag == "password" else tag, role=role, text=f"e{index}",
        attributes=attributes, in_viewport=True,
        options=list(OPTIONS) if tag == "select" else [],
        has_onclick=has_onclick, contenteditable=contenteditable,
    )


REAL_OBSERVATION = Observation(
    snapshot_id="snap", url="https://example.com/f", title="t", tabs=[],
    elements=[
        real_el(0, "select", "combobox"), real_el(1, "button"), real_el(2, "input"),
        real_el(3, "password"), real_el(4, "div", has_onclick=True),
        real_el(5, "div", contenteditable=True), real_el(6, "textarea"),
    ],
    page_text="", fingerprint="f", observed_at_monotonic=0.0,
)


def choice_answer(selected: str, criteria: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "choice", "choice": selected, "confidence": 1.0,
        "probabilities": {key: float(key == selected) for key in criteria},
    }


async def test_real_selector_target_domains_equal_candidates_and_round_trip() -> None:
    candidates = build_candidates(REAL_OBSERVATION)
    expected = {
        op: {str(e.index) for e in candidates.for_operation(op)}
        for op in ("click", "type_text", "select_option")
    }
    assert expected == {"click": {"1", "4"}, "type_text": {"2", "5", "6"}, "select_option": {"0"}}
    picked: dict[str, str] = {}
    sent: list[dict[str, dict[str, object]]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        questions = json.loads(request.content)["questions"]
        sent.append(questions)
        op = picked["op"]
        answers = {
            "operation": choice_answer(op, questions["operation"]["criteria"]),
            f"{op}_target": choice_answer(picked["target"], questions[f"{op}_target"]["criteria"]),
        }
        return httpx2.Response(200, json={
            "model": "jev-1.14.0", "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        })

    round_trips = 0
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        selector = JevSelector(api_key="synthetic-key", http_client=http)
        for op, targets in expected.items():
            for target in sorted(targets):
                picked.update(op=op, target=target)
                step = await select_step(
                    REAL_OBSERVATION, CONTEXT, {"region": "North"}, selector=selector, scope=scope(),
                )
                questions = sent[-1]
                assert {
                    o: set(questions[f"{o}_target"]["criteria"]) for o in expected  # pyright: ignore[reportArgumentType]
                } == expected
                assert step.action is not None and step.action.target is not None
                assert (step.action.kind, str(step.action.target.index)) == (op, target)
                assert step.bound_to == ("snap", "https://example.com/f", 500 + int(target))
                round_trips += 1
    assert round_trips == 6 and len(sent) == 6
    print("advertised target domains:", {o: sorted(t) for o, t in expected.items()})


async def test_done_and_non_target_operations_bind_without_node() -> None:
    done = await select_step(OBSERVATION, CONTEXT, {}, selector=FakeSelector(choice("done")), scope=scope())
    assert done.action is None and done.bound_to == ("snap", "https://example.com/f", None)
    scroll = await select_step(
        OBSERVATION, CONTEXT, {}, selector=FakeSelector(choice("scroll_down")), scope=scope(),
    )
    assert scroll.action is not None and scroll.action.kind == "scroll_down"
