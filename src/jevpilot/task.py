from enum import StrEnum
from typing import Annotated, ClassVar, Literal, override
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ContractViolation(ValueError):
    field: str
    reason: str

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return f"{self.field}: {self.reason}"


class ActionType(StrEnum):
    CLICK = "click"
    TYPE_TEXT = "type_text"
    SELECT = "select"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    WAIT = "wait"
    DONE = "done"
    BLOCKED = "blocked"


class Budget(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    max_steps: PositiveInt
    max_mutations: NonNegativeInt
    max_replans: NonNegativeInt
    max_model_attempts: PositiveInt
    max_attempts_per_call: PositiveInt
    max_stale_observations: NonNegativeInt
    max_waits: NonNegativeInt
    max_wait_ms: PositiveInt
    wall_time_ms: PositiveInt
    cost_limit_usd: NonNegativeFloat | None = None


class TaskSpec(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    task_id: NonEmptyStr
    goal: NonEmptyStr
    mode: Literal["llm_only", "jev_hybrid"]
    initial_url: NonEmptyStr
    allowed_origins: tuple[NonEmptyStr, ...]
    allowed_operations: tuple[ActionType, ...]
    initializer_id: NonEmptyStr
    evaluator_id: NonEmptyStr
    budgets: Budget
    locale: NonEmptyStr
    fixture_version: NonEmptyStr

    @field_validator("initial_url")
    @classmethod
    def validate_initial_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ContractViolation(
                field="initial_url",
                reason="must be an absolute HTTP(S) URL",
            )
        return value

    @field_validator("allowed_origins")
    @classmethod
    def validate_allowed_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ContractViolation(
                field="allowed_origins",
                reason="must not be empty",
            )
        if len(values) != len(set(values)):
            raise ContractViolation(
                field="allowed_origins",
                reason="must be unique",
            )
        for value in values:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ContractViolation(
                    field="allowed_origins",
                    reason="entries must be exact HTTP(S) origins",
                )
        return values

    @field_validator("allowed_operations")
    @classmethod
    def validate_allowed_operations(
        cls, values: tuple[ActionType, ...]
    ) -> tuple[ActionType, ...]:
        if not values:
            raise ContractViolation(
                field="allowed_operations",
                reason="must not be empty",
            )
        if len(values) != len(set(values)):
            raise ContractViolation(
                field="allowed_operations",
                reason="must be unique",
            )
        return values

    @model_validator(mode="after")
    def validate_initial_origin(self) -> "TaskSpec":
        parsed = urlsplit(self.initial_url)
        host = parsed.hostname
        if host is None:
            return self
        port_value = parsed.port
        if (parsed.scheme == "http" and port_value == 80) or (
            parsed.scheme == "https" and port_value == 443
        ):
            port_value = None
        port = f":{port_value}" if port_value is not None else ""
        origin = f"{parsed.scheme}://{host}{port}"
        if origin not in self.allowed_origins:
            raise ContractViolation(
                field="initial_url",
                reason="origin must be allowed",
            )
        return self
