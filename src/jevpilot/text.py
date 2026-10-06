"""Generate only free-form values for observed editable fields."""

from typing import Annotated, ClassVar, Protocol, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from jevpilot.model_client import ProviderError, StructuredModelClient
from jevpilot.observer.models import ActionCandidate, Observation
from jevpilot.planner.models import HistoryEntry, Subgoal
from jevpilot.privacy import PrivacyPolicy, redact_json
from jevpilot.task import ActionType, ContractViolation
from jevpilot.telemetry.models import CallContext

_JSON: TypeAdapter[JsonValue] = TypeAdapter[JsonValue](JsonValue)


class TextConstraints(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    max_length: Annotated[int, Field(gt=0)]
    allow_empty: bool = False


class TextInput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    goal: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    current_subgoal: Subgoal
    observation: Observation
    target: ActionCandidate
    history: tuple[HistoryEntry, ...]
    constraints: TextConstraints

    @model_validator(mode="after")
    def validate_target(self) -> "TextInput":
        if self.target.operation is not ActionType.TYPE_TEXT:
            raise ContractViolation(
                field="target.operation",
                reason="text generation requires an observed TYPE_TEXT target",
            )
        if self.target.snapshot_id != self.observation.snapshot_id:
            raise ContractViolation(
                field="target.snapshot_id",
                reason="text target must belong to current observation",
            )
        return self


class TextOutput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    text: str


class TextValueProvider(Protocol):
    async def generate(
        self, payload: TextInput, *, context: CallContext
    ) -> TextOutput: ...


@final
class LLMTextValueProvider:
    """Use a structured model client to generate input values."""

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        model: str,
        privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._privacy = privacy_policy or PrivacyPolicy()

    async def generate(
        self, payload: TextInput, *, context: CallContext
    ) -> TextOutput:
        target = payload.target
        max_length = min(
            payload.constraints.max_length,
            target.max_length if target.max_length is not None else payload.constraints.max_length,
        )
        schema: JsonValue = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 0 if payload.constraints.allow_empty else 1,
                    "maxLength": max_length,
                }
            },
            "required": ["text"],
            "additionalProperties": False,
        }
        projection = _JSON.validate_python(
            {
                "goal": payload.goal,
                "subgoal": payload.current_subgoal.model_dump(mode="json"),
                "snapshot_id": payload.observation.snapshot_id,
                "page": {
                    "title": payload.observation.title,
                    "text": payload.observation.text,
                },
                "target": {
                    "role": target.role,
                    "name": target.name,
                    "current_value": target.value,
                    "max_length": max_length,
                },
                "recent_history": [
                    entry.model_dump(mode="json") for entry in payload.history[-6:]
                ],
            }
        )
        result = await self._client.generate_json(
            instructions=(
                "Return one value for the observed editable field. "
                "Use the user goal and current subgoal. "
                "Return only an object with a text string. "
                "Do not return selectors, commands, or explanations."
            ),
            payload=redact_json(projection, policy=self._privacy),
            schema=schema,
            request_model=self.model,
            context=context,
        )
        if not isinstance(result.data, dict) or set(result.data) != {"text"}:
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription text generator returned an invalid result",
            )
        try:
            output = TextOutput.model_validate(result.data)
        except ValueError:
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription text generator returned an invalid result",
            ) from None
        if (
            len(output.text) > max_length
            or (not payload.constraints.allow_empty and not output.text.strip())
        ):
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription text value violates field constraints",
            )
        return output
