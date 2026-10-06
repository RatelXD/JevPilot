from datetime import datetime
from typing import Annotated, ClassVar, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from jevpilot.task import ActionType, ContractViolation

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class OptionObservation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    option_ref: NonEmptyStr = Field(
        validation_alias=AliasChoices("option_ref", "optionRef")
    )
    name: NonEmptyStr
    value: str
    selected: bool
    disabled: bool


class ControlObservation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    target_ref: NonEmptyStr
    role: NonEmptyStr
    name: NonEmptyStr
    value: str | None = None
    enabled: bool
    readonly: bool
    checked: bool | None = None
    selected: bool | None = None
    expanded: bool | None = None
    href: str | None = None
    max_length: NonNegativeInt | None = None
    options: tuple[OptionObservation, ...] = ()


class OmittedCounts(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    hidden: NonNegativeInt
    disabled: NonNegativeInt
    readonly: NonNegativeInt
    unnamed: NonNegativeInt
    over_limit: NonNegativeInt


class Observation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: NonEmptyStr
    document_id: NonEmptyStr
    navigation_id: NonEmptyStr
    url: NonEmptyStr
    title: str
    text: str
    controls: tuple[ControlObservation, ...]
    scroll_y: NonNegativeInt = 0
    scroll_height: PositiveInt = 1
    viewport_height: PositiveInt = 1
    fingerprint: NonEmptyStr
    observed_at: datetime
    omitted_counts: OmittedCounts


class WaitCondition(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal[
        "url_change",
        "control_appears",
        "option_appears",
        "state_change",
    ]
    target_ref: NonEmptyStr | None = None
    timeout_ms: PositiveInt


class ActionCandidate(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    candidate_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    operation: ActionType
    target_ref: NonEmptyStr | None = None
    option_ref: NonEmptyStr | None = None
    role: str | None = None
    name: str | None = None
    value: str | None = None
    enabled: bool | None = None
    readonly: bool | None = None
    max_length: NonNegativeInt | None = None
    wait_condition: WaitCondition | None = None

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "ActionCandidate":
        targeted = {
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
        }
        if self.operation in targeted:
            if self.target_ref is None:
                raise ContractViolation(
                    field="target_ref",
                    reason="targeted operations require it",
                )
        elif self.target_ref is not None:
            raise ContractViolation(
                field="target_ref",
                reason="control operations must not include it",
            )

        if self.operation is ActionType.SELECT:
            if self.option_ref is None:
                raise ContractViolation(
                    field="option_ref",
                    reason="SELECT requires it",
                )
        elif self.option_ref is not None:
            raise ContractViolation(
                field="option_ref",
                reason="only SELECT may include it",
            )

        if self.operation is ActionType.WAIT:
            if self.wait_condition is None:
                raise ContractViolation(
                    field="wait_condition",
                    reason="WAIT requires a bounded condition",
                )
        elif self.wait_condition is not None:
            raise ContractViolation(
                field="wait_condition",
                reason="only WAIT may include it",
            )
        return self
