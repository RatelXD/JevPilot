"""Browser-independent observations, actions, and lifecycle contract."""

from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

ATTRIBUTE_NAMES: Final = frozenset(
    {
        "aria-label", "placeholder", "name", "id", "type", "href", "value",
        "autocomplete", "role", "title", "alt",
    }
)


class Option(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str
    label: str

    @field_validator("label")
    @classmethod
    def truncate_label(cls, value: str) -> str:
        return value[:100]


class ElementRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    index: int
    backend_node_id: int
    tag: str
    role: str | None = None
    text: str
    attributes: dict[str, str]
    in_viewport: bool
    form_key: str | None = None
    form_has_password: bool = False
    form_has_search: bool = False
    form_has_card_field: bool = False
    form_action_host: str | None = None
    options: list[Option] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def truncate_text(cls, value: str) -> str:
        return value[:300]

    @field_validator("attributes")
    @classmethod
    def truncate_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        return {key: text[:200] for key, text in value.items() if key in ATTRIBUTE_NAMES}

    @field_validator("options")
    @classmethod
    def truncate_options(cls, value: list[Option]) -> list[Option]:
        return value[:50]


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    url: str
    title: str
    tabs: list[str]
    elements: list[ElementRef]
    page_text: str
    fingerprint: str
    observed_at_monotonic: float

    @field_validator("page_text")
    @classmethod
    def truncate_page_text(cls, value: str) -> str:
        return value[:20000]


ActionKind = Literal[
    "click", "type_text", "select_option", "scroll_down", "scroll_up",
    "go_back", "wait", "navigate",
]
ExecutionOutcome = Literal["executed", "failed_not_executed", "uncertain"]


class BrowserAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ActionKind
    target: ElementRef | None = None
    value: str | None = None
    url: str | None = None


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: ExecutionOutcome
    error_kind: str | None = None
    detail: str | None = None


class BrowserConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    executable_path: str = "/usr/bin/google-chrome"
    user_data_dir: Path | None = None
    headless: bool = False
    viewport: tuple[int, int] = (1280, 800)
    locale: str = "ko-KR"
    start_url: str | None = None


@runtime_checkable
class BrowserPort(Protocol):
    async def start(self, config: BrowserConfig) -> None: ...

    async def observe(self) -> Observation: ...

    async def execute(
        self, action: BrowserAction, observation: Observation,
    ) -> ExecutionResult: ...

    async def close(self) -> None: ...
