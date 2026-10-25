"""Task interpretation and evidence schemas shared with the engine's caller."""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from jevpilot.llm.types import CallScope, LLMProvider, LLMRequest

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Model(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


class OutputField(_Model):
    name: NonBlank
    description: str


class UrlContains(_Model):
    kind: Literal["url_contains"] = "url_contains"
    text: NonBlank


class VisibleText(_Model):
    kind: Literal["visible_text"] = "visible_text"
    text: NonBlank


class AnswerFields(_Model):
    kind: Literal["answer_fields"] = "answer_fields"
    names: list[NonBlank] = Field(min_length=1)


class OpenedTarget(_Model):
    """A requested item must be opened, not merely listed or suggested."""

    kind: Literal["opened_target"] = "opened_target"
    text: NonBlank = Field(description=(
        "Literal name of the item to open, copied from the task, in its original "
        "language. This is a substring matched against the clicked element and "
        "destination content, NOT a description of the action or UI container. "
        "Exclude words describing suggestions, results, buttons, opening, or searching "
        "unless those words are part of the item's proper name."
    ))


# Literal tags still disambiguate validation; a plain union emits anyOf, which
# native structured-output providers accept (discriminated unions emit oneOf).
Condition = UrlContains | VisibleText | AnswerFields | OpenedTarget


class Comparison(_Model):
    field: NonBlank
    kind: Literal["max", "min"]


class TaskBrief(_Model):
    goal_restated: str
    start_url: NonBlank | None
    input_values: dict[str, str] = Field(json_schema_extra={"properties": {}})
    required_outputs: list[OutputField]
    success_conditions: list[Condition]
    missing_inputs: list[str]
    mutating_intent: bool
    multi_step: bool
    comparison: Comparison | None


class StructuredTask(_Model):
    """Caller asserts that all necessary inputs are supplied; no LLM is needed."""

    start_url: NonBlank
    success_conditions: list[Condition] = Field(min_length=1)
    input_values: dict[str, str] = Field(default_factory=dict)
    required_outputs: list[OutputField] = Field(default_factory=list)


class FieldValue(_Model):
    value: str
    quote: str


class Extraction(_Model):
    fields: dict[str, FieldValue] = Field(json_schema_extra={"properties": {}})
    no_result: bool
    no_result_quote: str | None
    compared_items: list[FieldValue]


async def build_brief(
    goal: str,
    provider: LLMProvider,
    context: CallScope,
    *,
    structured_task: StructuredTask | None = None,
    start_url: str | None = None,
) -> TaskBrief:
    """Resolve a brief without owning browser startup or user interaction.

    The loop schedules this coroutine alongside browser startup/first observation
    and can use the original goal before awaiting it. Input values, extraction,
    and verification must await this result. ``missing_inputs`` is the caller's
    ask-user contract (``needs_user`` for a noninteractive caller).
    """
    if structured_task is not None:
        return TaskBrief(
            goal_restated=goal,
            start_url=structured_task.start_url,
            input_values=structured_task.input_values,
            required_outputs=structured_task.required_outputs,
            success_conditions=structured_task.success_conditions,
            missing_inputs=[],
            mutating_intent=False,
            multi_step=False,
            comparison=None,
        )
    result = await provider.generate(
        LLMRequest(
            purpose="brief",
            system=(
                "Interpret the complete browser task, including every line. "
                "Restate the goal, collect supplied inputs, required answer fields "
                "and observable completion conditions. Never invent missing inputs "
                "or destination URLs. Conditions describe ONLY the requested FINAL "
                "state, not intermediate controls or steps. When the task asks to "
                "open an item, result, or suggestion, use opened_target with its "
                "literal item name alone, not a paraphrase such as a named search "
                "result or autocomplete suggestion: seeing its name in a list is "
                "not completion. Preserve the item's spelling and language. Do not require "
                "search fields, filter controls or option labels to remain visible "
                "after opening the destination. Use visible_text for a requested "
                "visible outcome and answer_fields for information requests; these "
                "tasks do not necessarily require navigation. Never invent "
                "or claim completion. Use start_url only when supplied or unambiguous; "
                "otherwise null. List missing inputs. Mark mutating intent and "
                "multi-step tasks. For numeric comparisons identify the answer field "
                "and max/min kind. Do not browse, execute code, or use tools."
            ),
            user_payload={"goal": goal, "start_url": start_url},
            output_type=TaskBrief,
        ),
        context,
    )
    brief = result.value
    resolved_url = start_url or brief.start_url
    missing = [
        name for name in brief.missing_inputs
        if name != "start_url" or resolved_url is None
    ]
    if resolved_url is None and "start_url" not in missing:
        missing.append("start_url")
    return TaskBrief(
        goal_restated=brief.goal_restated,
        start_url=resolved_url,
        input_values=brief.input_values,
        required_outputs=brief.required_outputs,
        success_conditions=brief.success_conditions,
        missing_inputs=missing,
        mutating_intent=brief.mutating_intent,
        multi_step=brief.multi_step,
        comparison=brief.comparison,
    )
