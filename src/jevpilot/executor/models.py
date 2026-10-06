from datetime import datetime
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)

from jevpilot.observer.models import WaitCondition
from jevpilot.task import ActionType, ContractViolation

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ActionRequest(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    action_id: NonEmptyStr
    decision_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    operation: ActionType
    candidate_id: NonEmptyStr | None = None
    target_ref: NonEmptyStr | None = None
    option_ref: NonEmptyStr | None = None
    text: str | None = None
    wait_condition: WaitCondition | None = None

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "ActionRequest":
        targeted = {
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
        }
        candidate_bound = targeted | {ActionType.WAIT}

        if self.operation in candidate_bound:
            if self.candidate_id is None:
                raise ContractViolation(
                    field="candidate_id",
                    reason="selected operation requires it",
                )
        elif self.candidate_id is not None:
            raise ContractViolation(
                field="candidate_id",
                reason="control operation must not include it",
            )

        if self.operation in targeted:
            if self.target_ref is None:
                raise ContractViolation(
                    field="target_ref",
                    reason="targeted operation requires it",
                )
        elif self.target_ref is not None:
            raise ContractViolation(
                field="target_ref",
                reason="control operation must not include it",
            )

        if self.operation is ActionType.TYPE_TEXT:
            if self.text is None:
                raise ContractViolation(
                    field="text",
                    reason="TYPE_TEXT requires it",
                )
        elif self.text is not None:
            raise ContractViolation(
                field="text",
                reason="only TYPE_TEXT may include it",
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
                    reason="WAIT requires it",
                )
        elif self.wait_condition is not None:
            raise ContractViolation(
                field="wait_condition",
                reason="only WAIT may include it",
            )
        return self


class ActionExecution(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    action_id: NonEmptyStr
    decision_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    operation: ActionType
    candidate_id: NonEmptyStr | None = None
    target_ref: NonEmptyStr | None = None
    option_ref: NonEmptyStr | None = None
    status: Literal["executed", "failed_not_executed", "uncertain"]
    dispatched: bool
    started_at: datetime
    ended_at: datetime
    evidence: tuple[str, ...]
    error_kind: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_receipt(self) -> "ActionExecution":
        if self.ended_at < self.started_at:
            raise ContractViolation(
                field="ended_at",
                reason="must not precede started_at",
            )
        if self.status == "failed_not_executed" and self.dispatched:
            raise ContractViolation(
                field="dispatched",
                reason="failed_not_executed cannot be dispatched",
            )
        if self.status in {"executed", "uncertain"} and not self.dispatched:
            raise ContractViolation(
                field="dispatched",
                reason="executed or uncertain actions must be dispatched",
            )
        return self
