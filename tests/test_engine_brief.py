from dataclasses import dataclass, field
from typing import assert_never

import anyio
import pytest
from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.engine.brief import (
    AnswerFields, Comparison, Condition, Extraction, FieldValue, OutputField, StructuredTask,
    TaskBrief, UrlContains, VisibleText, build_brief,
)
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import (
    CallScope, Capabilities, LLMCallRecord, LLMRequest, LLMResult, ModelRef,
)


@dataclass(frozen=True, slots=True)
class Scope:
    run_id: str = "brief-test"
    parent_span_id: str | None = None
    deadline: float = float("inf")
    records: list[LLMCallRecord] = field(default_factory=list)

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


def brief(start_url: str | None = "https://example.test") -> TaskBrief:
    return TaskBrief(
        goal_restated="Find the year", start_url=start_url, input_values={},
        required_outputs=[OutputField(name="year", description="Birth year")],
        success_conditions=[AnswerFields(names=["year"])], missing_inputs=[],
        mutating_intent=False, multi_step=True, comparison=None,
    )


class Provider:
    """Offline wire response using the production structured-output boundary."""

    def __init__(self, answer: TaskBrief) -> None:
        self.answer: TaskBrief = answer
        self.ref: ModelRef = ModelRef(provider="openai", auth_mode="api", model_id="test")
        self.prompts: list[StructuredPrompt] = []
        self.started: anyio.Event = anyio.Event()
        self.release: anyio.Event | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities("native", False, (), False)

    async def generate[T](self, request: LLMRequest[T], context: CallScope) -> LLMResult[T]:
        async def send(prompt: StructuredPrompt) -> RawCompletion:
            self.prompts.append(prompt)
            self.started.set()
            if self.release is not None:
                await self.release.wait()
            return RawCompletion(self.answer.model_dump_json(), "test", None)

        return await generate_structured(
            request, context, ref=self.ref, mode="native", send=send,
        )

    async def aclose(self) -> None:
        pass


async def test_structured_task_when_complete_never_calls_llm() -> None:
    # Given
    provider, scope = Provider(brief()), Scope()
    task = StructuredTask(
        start_url="https://example.test",
        success_conditions=[UrlContains(text="/done"), VisibleText(text="Complete")],
        input_values={"query": "supplied"},
        required_outputs=[OutputField(name="year", description="Birth year")],
    )
    # When
    result = await build_brief("Original goal", provider, scope, structured_task=task)
    # Then
    assert result.start_url == task.start_url
    assert result.success_conditions == task.success_conditions
    assert result.input_values == task.input_values
    assert result.required_outputs == task.required_outputs
    assert result.missing_inputs == []
    assert provider.prompts == []
    assert scope.records == []


async def test_natural_language_when_multiline_and_long_is_one_brief_call() -> None:
    # Given
    goal = "first line\n" + "detail\n" * 5000 + "last line"
    provider, scope = Provider(brief()), Scope()
    # When
    result = await build_brief(goal, provider, scope)
    # Then
    assert result == brief()
    assert len(provider.prompts) == len(scope.records) == 1
    payload = TypeAdapter(dict[str, str | None]).validate_json(provider.prompts[0].user_text)
    assert payload == {"goal": goal, "start_url": None}
    assert scope.records[0].purpose == "brief"


async def test_brief_when_browser_observes_concurrently_does_not_block_first_observation() -> None:
    # Given
    provider = Provider(brief())
    provider.release = anyio.Event()
    completed: list[TaskBrief] = []

    async def interpret() -> None:
        completed.append(await build_brief("goal", provider, Scope()))

    async def first_observation() -> None:
        await provider.started.wait()
        assert not completed
        assert provider.release is not None
        provider.release.set()

    # When
    with anyio.fail_after(2):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(interpret)
            tasks.start_soon(first_observation)
    # Then
    assert completed == [brief()]
    assert len(provider.prompts) == 1


@pytest.mark.parametrize("supplied", [None, "https://explicit.test"])
async def test_missing_start_url_when_request_also_lacks_url_needs_user(
    supplied: str | None,
) -> None:
    # Given
    provider = Provider(brief(None))
    # When
    result = await build_brief("goal", provider, Scope(), start_url=supplied)
    # Then
    assert result.start_url == supplied
    assert result.missing_inputs == (["start_url"] if supplied is None else [])


async def test_request_url_when_model_suggests_another_url_takes_precedence() -> None:
    # Given
    provider = Provider(brief())
    # When
    result = await build_brief("goal", provider, Scope(), start_url="https://explicit.test")
    # Then
    assert result.start_url == "https://explicit.test"


async def test_missing_inputs_when_model_lists_them_are_preserved_without_duplicates() -> None:
    # Given
    answer = TaskBrief.model_validate({
        **brief(None).model_dump(), "missing_inputs": ["date", "start_url"],
    })
    # When
    result = await build_brief("goal", Provider(answer), Scope())
    # Then
    assert result.missing_inputs == ["date", "start_url"]


async def test_supplied_url_when_model_reports_missing_url_clears_that_question() -> None:
    # Given
    answer = TaskBrief.model_validate({
        **brief(None).model_dump(), "missing_inputs": ["date", "start_url"],
    })
    # When
    result = await build_brief(
        "goal", Provider(answer), Scope(), start_url="https://explicit.test",
    )
    # Then
    assert result.missing_inputs == ["date"]


@pytest.mark.parametrize("url,conditions", [
    ("", [UrlContains(text="/done")]), ("https://example.test", []),
])
def test_structured_task_when_incomplete_is_rejected(
    url: str, conditions: list[Condition],
) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = StructuredTask(start_url=url, success_conditions=conditions)


def test_schema_when_roundtripped_preserves_condition_tags_and_extraction_contract() -> None:
    # Given
    task = TaskBrief.model_validate({
        **brief().model_dump(),
        "comparison": Comparison(field="price", kind="min").model_dump(),
        "success_conditions": [
            UrlContains(text="/done").model_dump(), VisibleText(text="Complete").model_dump(),
            AnswerFields(names=["price"]).model_dump(),
        ],
    })
    extraction = Extraction(
        fields={"price": FieldValue(value="$10", quote="Item $10")},
        no_result=False, no_result_quote=None,
        compared_items=[FieldValue(value="$10", quote="Item $10")],
    )
    # When
    parsed = TaskBrief.model_validate_json(task.model_dump_json())
    extracted = Extraction.model_validate_json(extraction.model_dump_json())
    # Then
    assert parsed == task
    assert extracted == extraction
    assert isinstance(parsed.success_conditions[0], UrlContains)
    assert isinstance(parsed.success_conditions[1], VisibleText)
    assert isinstance(parsed.success_conditions[2], AnswerFields)
    assert TypeAdapter(TaskBrief).json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("value", ["", " \n\t", "\u3000"])
def test_condition_when_text_is_blank_cannot_vacuously_pass(value: str) -> None:
    # Given / When / Then
    with pytest.raises(ValidationError):
        _ = VisibleText(text=value)


def test_native_schemas_when_objects_and_unions_are_nested_use_supported_shapes() -> None:
    # Given: the real Codex server rejects objects without properties and oneOf.
    schemas: list[JsonValue] = [
        TypeAdapter[JsonValue](JsonValue).validate_python(TaskBrief.model_json_schema()),
        TypeAdapter[JsonValue](JsonValue).validate_python(Extraction.model_json_schema()),
    ]

    def check(schema: JsonValue) -> None:
        match schema:
            case dict() as node:
                assert "oneOf" not in node
                if node.get("type") == "object":
                    assert "properties" in node
                for child in node.values():
                    check(child)
                return
            case list() as nodes:
                for child in nodes:
                    check(child)
                return
            case str() | int() | float() | None:
                return
        assert_never(schema)

    # When / Then
    for schema in schemas:
        check(schema)


async def test_natural_brief_when_final_conditions_empty_is_rejected() -> None:
    # Given: a model must provide an observable final-state condition.
    provider = Provider(brief().model_copy(update={"success_conditions": []}))
    # When / Then
    with pytest.raises(LLMError) as raised:
        await build_brief("Finish all requested steps", provider, Scope())
    assert raised.value.kind == "invalid_output"
    assert len(provider.prompts) == 2
