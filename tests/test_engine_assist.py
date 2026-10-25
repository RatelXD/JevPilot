from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from typing import Literal, assert_never

import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.browser.port import BrowserAction, ElementRef, Observation, Option
from jevpilot.engine.assist import (
    ActionChoice, PlanPatch, TextValue, llm_choose, llm_extract, llm_replan, llm_text,
)
from jevpilot.engine.brief import AnswerFields, Comparison, Extraction, FieldValue, TaskBrief
from jevpilot.engine.jev import GoalContext
from jevpilot.engine.verify import verify
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import Profile
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.types import (
    AuthMode, CallScope, Capabilities, LLMCallRecord, LLMRequest, LLMResult, LLMUsage, ModelRef,
)

AssistPurpose = Literal["text", "choose", "replan", "extract"]
REF = ModelRef("openai", "api", "chosen-model", "high", True)
ELEMENTS = [
    ElementRef(snapshot_id="s", index=index, backend_node_id=100 + index, tag=tag,
               text=tag, attributes=attrs, in_viewport=True, options=options)
    for index, tag, attrs, options in [
        (3, "button", {}, []), (7, "input", {"type": "text"}, []),
        (11, "select", {}, [Option(value="", label="Choose"), Option(value="north", label="North")]),
        (15, "input", {"type": "password"}, []),
    ]
]
PAGE = Observation(
    snapshot_id="s", url="https://example.test", title="", tabs=[], elements=ELEMENTS,
    page_text="A $10 B $20", fingerprint="f", observed_at_monotonic=0,
)
BRIEF = TaskBrief(
    goal_restated="Find the maximum price", start_url=PAGE.url, input_values={"region": "North"},
    required_outputs=[], success_conditions=[AnswerFields(names=["price"])], missing_inputs=[],
    mutating_intent=False, multi_step=False, comparison=Comparison(field="price", kind="max"),
)
ANSWER = Extraction(
    fields={"price": FieldValue(value="$20", quote="B $20")}, no_result=False, no_result_quote=None,
    compared_items=[FieldValue(value="$10", quote="A $10"), FieldValue(value="$20", quote="B $20")],
)
RESPONSES = {
    "text": TextValue(value="generated", rationale="field").model_dump_json(),
    "choose": ActionChoice(operation="select_option", target_index=11, option_value="north").model_dump_json(),
    "replan": PlanPatch(subgoals=["first", "second"], next_hint="hint").model_dump_json(),
    "extract": ANSWER.model_dump_json(),
}


@dataclass(frozen=True, slots=True)
class Scope:
    run_id: str = "assist-test"
    parent_span_id: str | None = None
    deadline: float = float("inf")
    records: list[LLMCallRecord] = field(default_factory=list)

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class Provider:
    """Offline responses through the real parser, repair path, and call recorder."""

    def __init__(self, replies: list[str | LLMError], ref: ModelRef = REF) -> None:
        self.replies: Iterator[str | LLMError] = iter(replies)
        self.ref: ModelRef = ref
        self.prompts: list[StructuredPrompt] = []
        self.refs: list[ModelRef] = []

    def capabilities(self) -> Capabilities:
        return Capabilities("prompted", False, ("low", "high"), True)

    async def generate[T](self, request: LLMRequest[T], context: CallScope) -> LLMResult[T]:
        self.refs.append(self.ref)

        async def send(prompt: StructuredPrompt) -> RawCompletion:
            self.prompts.append(prompt)
            reply = next(self.replies)
            match reply:
                case str():
                    return RawCompletion(reply, self.ref.model_id, LLMUsage(10, 2, 0, None))
                case LLMError():
                    raise reply
                case _:
                    assert_never(reply)

        return await generate_structured(request, context, ref=self.ref, mode="prompted", send=send)

    async def aclose(self) -> None:
        return None


async def invoke(
    purpose: AssistPurpose, provider: Provider, scope: Scope,
) -> TextValue | BrowserAction | PlanPatch | Extraction:
    match purpose:
        case "text":
            return await llm_text(PAGE, BRIEF, ELEMENTS[1], provider=provider, context=scope)
        case "choose":
            return await llm_choose(PAGE, BRIEF, provider=provider, context=scope)
        case "replan":
            return await llm_replan(PAGE, GoalContext("original"), provider=provider, context=scope)
        case "extract":
            return await llm_extract(PAGE, BRIEF, provider=provider, context=scope)
        case _:
            assert_never(purpose)


@pytest.mark.parametrize("purpose", ["text", "choose", "replan", "extract"])
@pytest.mark.parametrize("effort", [None, "low", "high"])
@pytest.mark.parametrize("auth_mode", ["api", "subscription"])
async def test_assistance_when_called_preserves_profile_model_and_effort(
    purpose: AssistPurpose, effort: str | None, auth_mode: AuthMode,
) -> None:
    # Given
    profile = Profile(auth_mode, ModelRef(
        "openai" if auth_mode == "api" else "chatgpt_subscription",
        auth_mode, "user-selected-model", effort, True,
    ))
    provider, scope = Provider([RESPONSES[purpose]], profile.model), Scope()
    # When
    result = await invoke(purpose, provider, scope)
    # Then
    assert result is not None
    assert provider.refs == [profile.model] and provider.ref is profile.model
    assert [(r.purpose, r.model_id, r.auth_mode, r.outcome) for r in scope.records] == [
        (purpose, "user-selected-model", auth_mode, "ok"),
    ]
    assert scope.records[0].usage == LLMUsage(10, 2, 0, None)


async def test_text_when_generated_is_one_value_for_the_requested_field() -> None:
    # Given
    provider = Provider([RESPONSES["text"]])
    # When
    result = await llm_text(PAGE, BRIEF, ELEMENTS[1], provider=provider, context=Scope())
    # Then
    assert result == TextValue(value="generated", rationale="field")
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(provider.prompts[0].user_text)
    assert payload["target"] == ELEMENTS[1].model_dump(mode="json")
    assert payload["input_values"] == BRIEF.input_values


@pytest.mark.parametrize("target", [
    ELEMENTS[1].model_copy(update={"snapshot_id": "old"}), ELEMENTS[3], ELEMENTS[0],
])
async def test_text_when_target_is_stale_password_or_wrong_kind_sends_nothing(target: ElementRef) -> None:
    # Given
    provider, scope = Provider([]), Scope()
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = await llm_text(PAGE, BRIEF, target, provider=provider, context=scope)
    assert raised.value.kind == "configuration"
    assert provider.prompts == [] and scope.records == []


@pytest.mark.parametrize("operation,index,value", [
    ("select_option", 11, None), ("select_option", 11, "south"), ("select_option", 3, "north"),
    ("click", 0, None), ("click", 7, None), ("type_text", 15, None), ("click", None, None),
    ("click", 3, "north"), ("scroll_down", 3, None), ("wait", None, "north"),
])
async def test_choice_when_outside_operation_domain_is_invalid(
    operation: str, index: int | None, value: str | None,
) -> None:
    # Given
    provider = Provider([json.dumps({"operation": operation, "target_index": index, "option_value": value})])
    scope = Scope()
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = await llm_choose(PAGE, BRIEF, provider=provider, context=scope)
    assert raised.value.kind == "invalid_output" and not raised.value.retryable
    assert len(provider.prompts) == len(scope.records) == 1


@pytest.mark.parametrize("choice,expected", [
    (ActionChoice(operation="select_option", target_index=11, option_value="north"),
     BrowserAction(kind="select_option", target=ELEMENTS[2], value="north")),
    (ActionChoice(operation="select_option", target_index=11, option_value=""),
     BrowserAction(kind="select_option", target=ELEMENTS[2], value="")),
    (ActionChoice(operation="click", target_index=3, option_value=None),
     BrowserAction(kind="click", target=ELEMENTS[0])),
    (ActionChoice(operation="type_text", target_index=7, option_value=None),
     BrowserAction(kind="type_text", target=ELEMENTS[1])),
    (ActionChoice(operation="go_back", target_index=None, option_value=None), BrowserAction(kind="go_back")),
])
async def test_choice_when_valid_binds_original_target_and_option(
    choice: ActionChoice, expected: BrowserAction,
) -> None:
    # Given
    provider = Provider([choice.model_dump_json()])
    # When
    result = await llm_choose(PAGE, BRIEF, provider=provider, context=Scope())
    # Then
    assert result == expected
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(provider.prompts[0].user_text)
    candidates = payload["candidates"]
    assert isinstance(candidates, dict)
    assert candidates == {
        "click": [ELEMENTS[0].model_dump(mode="json")],
        "type_text": [ELEMENTS[1].model_dump(mode="json")],
        "select_option": [ELEMENTS[2].model_dump(mode="json")],
    }


async def test_choice_when_jev_domain_is_truncated_cannot_restore_an_excluded_target() -> None:
    # Given
    page = PAGE.model_copy(update={"elements": [
        ELEMENTS[0].model_copy(update={"index": i, "backend_node_id": i, "in_viewport": i == 200})
        for i in range(201)
    ]})
    provider = Provider(['{"operation":"click","target_index":199,"option_value":null}'])
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = await llm_choose(page, BRIEF, provider=provider, context=Scope())
    assert raised.value.kind == "invalid_output"
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(provider.prompts[0].user_text)
    candidates = payload["candidates"]
    assert isinstance(candidates, dict) and isinstance(candidates["click"], list)
    assert [e["index"] for e in candidates["click"] if isinstance(e, dict)] == [*range(199), 200]


async def test_replan_when_applied_updates_jev_context_without_changing_original_goal() -> None:
    # Given
    original = GoalContext("original", "brief", "old", ("clicked",))
    provider = Provider([RESPONSES["replan"]])
    # When
    patch = await llm_replan(PAGE, original, provider=provider, context=Scope())
    updated = patch.apply(original)
    # Then
    assert updated == GoalContext("original", "brief", "first\nsecond\nhint", ("clicked",))
    assert original.subgoal == "old"


@pytest.mark.parametrize("answer,status", [
    (ANSWER, "PASS"),
    (ANSWER.model_copy(update={"fields": {"price": FieldValue(value="$10", quote="A $10")}}), "FAIL"),
    (ANSWER.model_copy(update={"fields": {"price": FieldValue(value="$99", quote="C $99")}}), "FAIL"),
])
async def test_extraction_when_comparing_preserves_all_quotes_for_code_verification(
    answer: Extraction, status: str,
) -> None:
    # Given
    provider = Provider([answer.model_dump_json()])
    # When
    result = await llm_extract(PAGE, BRIEF, provider=provider, context=Scope())
    # Then
    assert result == answer and verify(PAGE, BRIEF, result).status == status
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(provider.prompts[0].user_text)
    assert payload["comparison"] == {"field": "price", "kind": "max"}
    assert payload["include_all_compared_items"] is True


@pytest.mark.parametrize("purpose", ["text", "choose", "replan", "extract"])
@pytest.mark.parametrize("repaired", [False, True])
async def test_structured_boundary_when_output_is_malformed_repairs_once(
    purpose: AssistPurpose, repaired: bool,
) -> None:
    # Given
    provider = Provider(["PRIVATE-OUTPUT", RESPONSES[purpose] if repaired else "PRIVATE-OUTPUT"])
    scope = Scope()
    # When / Then
    if repaired:
        _ = await invoke(purpose, provider, scope)
    else:
        with pytest.raises(LLMError) as raised:
            _ = await invoke(purpose, provider, scope)
        assert raised.value.kind == "invalid_output"
        assert "PRIVATE-OUTPUT" not in str(raised.value)
    assert [r.outcome for r in scope.records] == ["invalid_output", "ok" if repaired else "invalid_output"]
    assert [r.usage for r in scope.records] == [LLMUsage(10, 2, 0, None)] * 2
    assert len(provider.prompts) == 2
    assert "PRIVATE-OUTPUT" not in provider.prompts[1].user_text


async def test_provider_failure_when_unavailable_propagates_without_fallback() -> None:
    # Given
    error = LLMError("provider_unavailable", retryable=False, message="unavailable")
    provider, scope = Provider([error]), Scope()
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = await invoke("replan", provider, scope)
    assert raised.value is error
    assert provider.refs == [REF] and len(provider.prompts) == 1
    assert [r.outcome for r in scope.records] == ["provider_unavailable"]


@pytest.mark.parametrize("purpose,bad", [
    ("text", '{"value":"v","rationale":"r","extra":"PRIVATE"}'),
    ("choose", '{"operation":"click","target_index":true,"option_value":null}'),
    ("choose", '{"operation":"click","target_index":-1,"option_value":null}'),
    ("replan", '{"subgoals":[],"next_hint":"PRIVATE"}'),
    ("extract", '{"fields":{},"no_result":"yes"}'),
])
async def test_schema_when_response_shape_is_invalid_cannot_bypass_validation(
    purpose: AssistPurpose, bad: str,
) -> None:
    # Given
    provider, scope = Provider([bad, bad]), Scope()
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = await invoke(purpose, provider, scope)
    assert raised.value.kind == "invalid_output"
    assert [r.outcome for r in scope.records] == ["invalid_output", "invalid_output"]
