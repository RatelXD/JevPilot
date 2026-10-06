from math import isfinite
from typing import Annotated, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from jevpilot.observer.models import ActionCandidate, Observation
from jevpilot.planner.models import HistoryEntry, Subgoal
from jevpilot.task import ActionType, ContractViolation

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(gt=0)]


class DecisionPolicy(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    allowed_operations: tuple[ActionType, ...]
    max_wait_ms: PositiveInt


class DecisionInput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    goal: NonEmptyStr
    current_subgoal: Subgoal
    observation: Observation
    candidates: tuple[ActionCandidate, ...]
    policy: DecisionPolicy
    recent_history: tuple[HistoryEntry, ...]

    @model_validator(mode="after")
    def validate_snapshot_scope(self) -> "DecisionInput":
        if any(
            candidate.snapshot_id != self.observation.snapshot_id
            for candidate in self.candidates
        ):
            raise ContractViolation(
                field="candidates",
                reason="all candidates must belong to the observation snapshot",
            )
        if any(
            candidate.operation not in self.policy.allowed_operations
            for candidate in self.candidates
        ):
            raise ContractViolation(
                field="candidates",
                reason="operation is not allowed by policy",
            )
        return self


class DecisionOutput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    decision_id: NonEmptyStr
    snapshot_id: NonEmptyStr
    operation: ActionType
    candidate_id: NonEmptyStr | None = None
    operation_confidence: Probability | None = None
    target_confidence: Probability | None = None
    operation_probabilities: dict[ActionType, Probability] | None = None
    target_probabilities: dict[str, Probability] | None = None
    provider: NonEmptyStr
    model: NonEmptyStr | None
    call_id: NonEmptyStr

    @field_validator("operation_probabilities", "target_probabilities")
    @classmethod
    def validate_probability_map(
        cls,
        values: dict[ActionType, float] | dict[str, float] | None,
    ) -> dict[ActionType, float] | dict[str, float] | None:
        if values is not None and any(not isfinite(value) for value in values.values()):
            raise ContractViolation(
                field="probabilities",
                reason="must be finite",
            )
        return values

    @model_validator(mode="after")
    def validate_operation_shape(self) -> "DecisionOutput":
        candidate_bound = {
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
            ActionType.WAIT,
        }
        target_head_operations = {
            ActionType.CLICK,
            ActionType.TYPE_TEXT,
            ActionType.SELECT,
            ActionType.WAIT,
        }
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

        if self.operation not in target_head_operations:
            if self.target_confidence is not None or self.target_probabilities is not None:
                raise ContractViolation(
                    field="target_confidence",
                    reason="target head data is invalid for this operation",
                )
        return self
