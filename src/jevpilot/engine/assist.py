"""Bounded LLM assistance using the caller's selected provider and effort."""

from dataclasses import replace
from typing import ClassVar, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevpilot.browser.port import BrowserAction, ElementRef, Observation
from jevpilot.engine.brief import Extraction, NonBlank, TaskBrief
from jevpilot.engine.candidates import ActionCandidates, build_candidates
from jevpilot.engine.jev import GoalContext, select_elements
from jevpilot.engine.verify import _grounded, _normalize, _number
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import CallScope, LLMProvider, LLMRequest


class _Model(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)


class TextValue(_Model):
    value: str = Field(min_length=1)
    rationale: str


class ActionChoice(_Model):
    operation: Literal[
        "click", "type_text", "select_option", "scroll_down", "scroll_up", "go_back", "wait",
    ]
    target_index: int | None = Field(ge=0)
    option_value: str | None

    @model_validator(mode="after")
    def validate_option(self) -> Self:
        """Reject option-only data at the repairable structured-output boundary."""
        if self.operation != "select_option" and self.option_value is not None:
            raise ValueError("option_value must be null except for select_option")
        return self

    def to_action(self, candidates: ActionCandidates) -> BrowserAction:
        """Bind to the supplied candidate domain, never an index invented by a model."""
        target = next(
            (e for e in candidates.for_operation(self.operation) if e.index == self.target_index),
            None,
        )
        match self.operation:
            case "select_option":
                if target is None or self.option_value not in {o.value for o in target.options}:
                    raise LLMError("invalid_output", retryable=False, message="choice option outside target options")
                return BrowserAction(kind=self.operation, target=target, value=self.option_value)
            case "click" | "type_text":
                if target is None:
                    raise LLMError("invalid_output", retryable=False, message="choice target outside candidates")
                if self.option_value is not None:
                    raise LLMError("invalid_output", retryable=False, message="unexpected choice option")
                return BrowserAction(kind=self.operation, target=target)
            case "scroll_down" | "scroll_up" | "go_back" | "wait":
                if self.target_index is not None or self.option_value is not None:
                    raise LLMError("invalid_output", retryable=False, message="unexpected choice target or option")
                return BrowserAction(kind=self.operation)
            case _:
                assert_never(self.operation)


class PlanPatch(_Model):
    subgoals: list[NonBlank] = Field(min_length=1)
    next_hint: str

    def apply(self, goal_context: GoalContext) -> GoalContext:
        """Supply the plan and next hint to Jev without replacing the user's goal."""
        return replace(goal_context, subgoal="\n".join([*self.subgoals, self.next_hint]).strip())


def _domains(observation: Observation) -> ActionCandidates:
    """Use the same rule-built, viewport-prioritized domain as select_step."""
    candidates = build_candidates(observation)
    elements, _ = select_elements(candidates.all_elements())
    kept = {element.index for element in elements}
    return ActionCandidates(
        click=tuple(e for e in candidates.click if e.index in kept),
        type_text=tuple(e for e in candidates.type_text if e.index in kept),
        select_option=tuple(e for e in candidates.select_option if e.index in kept),
        safety_flags=candidates.safety_flags,
    )


def supplied_input_key(target: ElementRef, observation: Observation, brief: TaskBrief) -> str | None:
    """Resolve a field by its observed identifiers, never by its position."""
    return next((
        name for name in (
            target.attributes.get("name"), target.attributes.get("id"),
            target.attributes.get("aria-label"), target.attributes.get("placeholder"),
            f"{observation.url}#{target.backend_node_id}",
        ) if name in brief.input_values
    ), None)


async def llm_text(
    observation: Observation, brief: TaskBrief, target: ElementRef, *,
    provider: LLMProvider, context: CallScope,
) -> TextValue:
    """Generate exactly one field value; do not choose or execute browser actions."""
    if target.snapshot_id != observation.snapshot_id or target not in _domains(observation).type_text:
        raise LLMError("configuration", retryable=False, message="text target outside current candidates")
    result = await provider.generate(LLMRequest(
        purpose="text",
        system=(
            "Produce one input value for the specified field and a rationale. "
            "Use the supplied task inputs and page context. Do not invent missing "
            "personal information, select other fields, execute code, or use tools."
        ),
        user_payload={
            "goal": brief.goal_restated,
            "input_values": {name: value for name, value in brief.input_values.items()},
            "page_text": observation.page_text, "target": target.model_dump(mode="json"),
        },
        output_type=TextValue,
    ), context)
    return result.value


async def llm_choose(
    observation: Observation, brief: TaskBrief, *,
    provider: LLMProvider, context: CallScope,
    recent_actions: tuple[str, ...] = (),
) -> BrowserAction:
    """Return a snapshot-bound action, including the observed select option value."""
    candidates = _domains(observation)
    result = await provider.generate(LLMRequest(
        purpose="choose",
        system=(
            "Choose one operation from the schema. For targeted operations choose "
            "only an index in that operation's candidates. For select_option use "
            "an exact option value from that target. Other operations require null "
            "option_value; operations without a target require null target_index. "
            "Use the recent executed actions to track progress. Input value "
            "attributes and retained_input_lengths are character counts, not "
            "field contents; an already filled field may need no further typing. "
            "Do not execute actions, code, or tools."
        ),
        user_payload={
            "goal": brief.goal_restated,
            "input_values": {name: value for name, value in brief.input_values.items()},
            "page_text": observation.page_text,
            "recent_actions": list(recent_actions),
            "retained_input_lengths": {
                str(element.index): element.attributes.get("value")
                for element in candidates.type_text
            },
            "candidates": {
                op: [e.model_dump(mode="json") for e in candidates.for_operation(op)]
                for op in ("click", "type_text", "select_option")
            },
        },
        output_type=ActionChoice,
    ), context)
    action = result.value.to_action(candidates)
    if action.kind == "type_text" and action.target is not None:
        key = supplied_input_key(action.target, observation, brief)
        if key is not None:
            action = action.model_copy(update={"value": brief.input_values[key]})
    return action


async def llm_replan(
    observation: Observation, goal_context: GoalContext, *,
    provider: LLMProvider, context: CallScope,
) -> PlanPatch:
    """Return subgoals for PlanPatch.apply; never change provider/model/effort."""
    result = await provider.generate(LLMRequest(
        purpose="replan",
        system=(
            "Revise the browser plan into ordered subgoals and a next-step hint "
            "using the current page and recent actions. Preserve the user's goal. "
            "Return data only, not executable code, tools, or a completion claim."
        ),
        user_payload={
            "goal": goal_context.goal, "brief": goal_context.brief,
            "subgoal": goal_context.subgoal, "recent_actions": list(goal_context.recent_actions),
            "url": observation.url, "page_text": observation.page_text,
        },
        output_type=PlanPatch,
    ), context)
    return result.value


async def llm_extract(
    observation: Observation, brief: TaskBrief, *,
    provider: LLMProvider, context: CallScope,
    observed_pages: tuple[Observation, ...] = (),
) -> Extraction:
    """Extract claims with evidence; the code-owned verifier alone decides success."""
    class _ObservedExtraction(Extraction):
        """Repair malformed claims without weakening the final verifier."""

        @model_validator(mode="after")
        def validate_observed_fields(self) -> Self:
            if not self.no_result and not all(
                _grounded(field, _normalize(observation.page_text)) for field in self.fields.values()
            ):
                raise ValueError("fields must contain values and verbatim quotes from the current page; do not reorder or omit words")
            if brief.comparison is not None and any(
                _number(item.value) is None for item in self.compared_items
            ):
                raise ValueError("compared_items.value must contain only the numeric comparison field, not an item name")
            return self

    result = await provider.generate(LLMRequest(
        purpose="extract",
        system=(
            "Extract the required answer fields from the supplied page. Each value "
            "needs an exact supporting page quote containing that value. A no-result "
            "claim needs the page's explicit no-result quote. When comparison is "
            "supplied, include every compared item with its value and quote in "
            "compared_items, not just the winner. Do not execute code or use tools."
            " Each compared_items.value is only the numeric comparison field, "
            "with its displayed currency if any; put the item name in quote, not value."
            " Compared items may quote earlier observed pages, each independently. "
            "Required answer fields must still quote the current page."
        ),
        user_payload={
            "page_text": observation.page_text,
            "required_outputs": [f.model_dump(mode="json") for f in brief.required_outputs],
            "success_conditions": [c.model_dump(mode="json") for c in brief.success_conditions],
            "comparison": brief.comparison.model_dump(mode="json") if brief.comparison else None,
            "include_all_compared_items": brief.comparison is not None,
            "observed_pages": [
                {"url": page.url, "page_text": page.page_text} for page in observed_pages
            ] if brief.comparison is not None else [],
        },
        output_type=_ObservedExtraction,
    ), context)
    return result.value
