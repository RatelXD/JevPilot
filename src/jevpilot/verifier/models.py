from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    model_validator,
)

from jevpilot.task import ContractViolation

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
VerificationStatus = Literal["passed", "not_yet", "failed", "unknown"]


class Check(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    check_id: NonEmptyStr
    scope: Literal["subgoal", "task", "action_effect"]
    status: VerificationStatus
    evidence: tuple[str, ...]
    action_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> "Check":
        if self.scope == "action_effect":
            if self.action_id is None:
                raise ContractViolation(
                    field="action_id",
                    reason="action_effect checks require it",
                )
        elif self.action_id is not None:
            raise ContractViolation(
                field="action_id",
                reason="only action_effect checks may include it",
            )
        return self


class TaskEvaluation(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    evaluator_id: NonEmptyStr
    status: VerificationStatus
    checks: tuple[Check, ...]

    @model_validator(mode="after")
    def validate_checks(self) -> "TaskEvaluation":
        if any(check.scope != "task" for check in self.checks):
            raise ContractViolation(
                field="checks",
                reason="task evaluation may contain only task checks",
            )
        return self


class VerificationResult(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: NonEmptyStr
    subgoal_id: NonEmptyStr
    subgoal_status: VerificationStatus
    subgoal_checks: tuple[Check, ...]
    task_evaluation: TaskEvaluation
    effect_checks: tuple[Check, ...]
    evidence: tuple[str, ...]
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_check_scopes(self) -> "VerificationResult":
        if any(check.scope != "subgoal" for check in self.subgoal_checks):
            raise ContractViolation(
                field="subgoal_checks",
                reason="may contain only subgoal checks",
            )
        if any(check.scope != "action_effect" for check in self.effect_checks):
            raise ContractViolation(
                field="effect_checks",
                reason="may contain only action effect checks",
            )
        return self
