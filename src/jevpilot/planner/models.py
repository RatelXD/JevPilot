from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from jevpilot.observer.models import Observation
from jevpilot.task import ActionType, ContractViolation

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ProgressHint(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["url_path", "visible", "value", "checked"]
    name: str | None = None
    expected: str | bool


class Subgoal(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    subgoal_id: NonEmptyStr
    description: NonEmptyStr
    progress_hint: ProgressHint | None = None


class Plan(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    plan_id: NonEmptyStr
    parent_plan_id: NonEmptyStr | None = None
    subgoals: tuple[Subgoal, ...]

    @field_validator("subgoals")
    @classmethod
    def validate_subgoals(cls, values: tuple[Subgoal, ...]) -> tuple[Subgoal, ...]:
        if not values:
            raise ContractViolation(field="subgoals", reason="must not be empty")
        ids = tuple(subgoal.subgoal_id for subgoal in values)
        if len(ids) != len(set(ids)):
            raise ContractViolation(field="subgoals", reason="IDs must be unique")
        return values


class HistoryEntry(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: NonEmptyStr
    subgoal_id: NonEmptyStr
    operation: ActionType
    outcome: Literal[
        "executed",
        "failed_not_executed",
        "uncertain",
        "verified",
        "rejected",
    ]
    action_id: NonEmptyStr | None = None
    evidence: tuple[str, ...] = ()


class PlanningFailure(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    kind: NonEmptyStr
    evidence: tuple[str, ...]


class BudgetsRemaining(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    steps: NonNegativeInt
    mutations: NonNegativeInt
    replans: NonNegativeInt
    model_attempts: NonNegativeInt
    stale_observations: NonNegativeInt
    waits: NonNegativeInt
    wait_ms: NonNegativeInt
    wall_time_ms: NonNegativeInt
    cost_usd: NonNegativeFloat | None = None


class PlannerInput(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    goal: NonEmptyStr
    observation: Observation
    history: tuple[HistoryEntry, ...]
    previous_plan_id: NonEmptyStr | None = None
    failure: PlanningFailure | None = None
    budgets_remaining: BudgetsRemaining
