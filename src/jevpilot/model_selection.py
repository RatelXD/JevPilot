"""Persist the selected model reference without storing credentials."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    ValidationError,
    field_validator,
)
from typing_extensions import TypeAlias

from jevpilot.codex_auth import ensure_jevpilot_home

ModelId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ReasoningEffort: TypeAlias = Literal["low", "medium", "high", "xhigh", "max"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class ModelSelectionError(ValueError):
    """A saved model selection is invalid or cannot be stored safely."""


class ModelSelection(BaseModel):
    """A subscription model and its Codex generation preferences."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["chatgpt-subscription"]
    model_id: ModelId
    reasoning_effort: ReasoningEffort = "medium"
    fast: bool = False
    supports_fast: bool = False
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = (
        "low",
        "medium",
        "high",
    )

    @field_validator("model_id")
    @classmethod
    def reject_model_id_whitespace(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("model id must not contain whitespace")
        return value

    @property
    def model_ref(self) -> str:
        """Return the provider/model display value."""
        return f"{self.provider}/{self.model_id}"


def selection_path() -> Path:
    """Return the private model-selection settings file."""
    return ensure_jevpilot_home() / "settings.json"


def load_model_selection() -> ModelSelection | None:
    """Load the user's saved model reference, if one exists."""
    path = selection_path()
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ModelSelectionError("saved model settings must be a regular file")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise ModelSelectionError("saved model settings are owned by another user")
    try:
        return ModelSelection.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError):
        raise ModelSelectionError("saved model settings are invalid; select a model again") from None


def save_model_selection(selection: ModelSelection) -> None:
    """Atomically save the selected provider/model without credentials."""
    directory = ensure_jevpilot_home()
    path = directory / "settings.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".settings-",
        suffix=".tmp",
        dir=directory,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(selection.model_dump_json().encode("utf-8"))
            _ = stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_model_ref(value: str) -> ModelSelection:
    """Parse a provider/model choice at the interactive CLI boundary."""
    provider, separator, model_id = value.partition("/")
    if not separator:
        raise ModelSelectionError("use provider/model, for example chatgpt-subscription/model-id")
    if provider != "chatgpt-subscription":
        raise ModelSelectionError("only chatgpt-subscription models are available here")
    try:
        return ModelSelection(provider="chatgpt-subscription", model_id=model_id)
    except ValidationError:
        raise ModelSelectionError("model reference is malformed") from None


def select_available_model(
    value: str, available_model_ids: frozenset[str]
) -> ModelSelection:
    """Accept a well-formed model reference only if the catalog contains it."""
    selection = parse_model_ref(value)
    if selection.model_id not in available_model_ids:
        raise ModelSelectionError("model is not available in the signed-in Codex account")
    return selection


def next_reasoning_effort(
    value: ReasoningEffort,
    supported: tuple[ReasoningEffort, ...],
) -> ReasoningEffort:
    """Cycle the available reasoning efforts in the prompt composer."""
    levels: tuple[ReasoningEffort, ...] = tuple(
        level for level in REASONING_EFFORTS if level in supported
    )
    if not levels:
        return "medium"
    try:
        index = levels.index(value)
    except ValueError:
        index = -1
    return levels[(index + 1) % len(levels)]
