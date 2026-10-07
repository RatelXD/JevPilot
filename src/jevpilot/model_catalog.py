"""Read the signed-in ChatGPT account's model catalog through Codex app-server."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

import anyio
from anyio import EndOfStream
from anyio.abc import ByteReceiveStream, ByteSendStream
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from jevpilot.codex_auth import codex_command_prefix, codex_environment
from jevpilot.model_client import ProviderError
from jevpilot.model_selection import REASONING_EFFORTS, ReasoningEffort


_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_APP_NAME = "jevpilot"
_APP_VERSION = "0.1.0"
_REQUEST_TIMEOUT_SECONDS = 30
_REQUESTED_MODEL_IDS = (
    "gpt-6-astra",
    "gpt-5.3-codex-spark",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-sol",
)
_NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@dataclass(frozen=True, slots=True)
class ChatGPTModel:
    """A visible model entry returned by the signed-in Codex account."""

    model_id: str
    display_name: str
    is_default: bool
    supports_fast: bool = False
    reasoning_efforts: tuple[ReasoningEffort, ...] = ("low", "medium", "high")

    @property
    def model_ref(self) -> str:
        """Return the public provider/model display reference."""
        return f"chatgpt-subscription/{self.model_id}"


class _ReasoningEffortOption(BaseModel):
    """One effort and its display description from model/list."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    reasoning_effort: str = Field(alias="reasoningEffort")
    description: str


class _ModelServiceTier(BaseModel):
    """One service tier advertised by model/list."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str
    description: str


class _ModelEntry(BaseModel):
    """Subset of the official app-server model/list response used by JevPilot."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    id: _NonEmpty
    model: _NonEmpty
    display_name: _NonEmpty = Field(alias="displayName")
    hidden: bool
    is_default: bool = Field(alias="isDefault")
    service_tiers: tuple[_ModelServiceTier, ...] = Field(
        default=(),
        validation_alias=AliasChoices("serviceTiers", "service_tiers"),
    )
    reasoning_efforts: tuple[_ReasoningEffortOption, ...] = Field(
        validation_alias=AliasChoices(
            "supportedReasoningEfforts",
            "supported_reasoning_efforts",
        ),
    )


class _ModelPage(BaseModel):
    """One paginated model/list response."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    data: tuple[_ModelEntry, ...]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


def _reasoning_efforts(
    values: tuple[_ReasoningEffortOption, ...],
) -> tuple[ReasoningEffort, ...]:
    advertised = tuple(value.reasoning_effort for value in values)
    return tuple(level for level in REASONING_EFFORTS if level in advertised)


class _JsonLineReader:
    """Read newline-delimited JSON-RPC messages from app-server stdout."""

    def __init__(self, stream: ByteReceiveStream) -> None:
        self.stream: ByteReceiveStream = stream
        self.buffer: bytearray = bytearray()

    async def receive(self) -> JsonValue:
        """Return the next parsed JSON-RPC message."""
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                return _JSON.validate_json(line)
            try:
                chunk = await self.stream.receive()
            except EndOfStream:
                raise ProviderError(
                    kind="invalid_response",
                    call_id="codex-model-catalog",
                    retryable=False,
                    message="Codex app-server closed before returning its model catalog",
                ) from None
            if not chunk:
                raise ProviderError(
                    kind="invalid_response",
                    call_id="codex-model-catalog",
                    retryable=False,
                    message="Codex app-server closed before returning its model catalog",
                )
            self.buffer.extend(chunk)


async def _send(stream: ByteSendStream, message: JsonValue) -> None:
    """Send one newline-delimited app-server request."""
    await stream.send(json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n")


async def _request(
    reader: _JsonLineReader,
    writer: ByteSendStream,
    *,
    request_id: int,
    method: str,
    params: dict[str, JsonValue],
) -> JsonValue:
    """Send a JSON-RPC request and return its matching result."""
    await _send(
        writer,
        {"id": request_id, "method": method, "params": params},
    )
    while True:
        message = await reader.receive()
        if not isinstance(message, dict):
            raise ProviderError(
                kind="invalid_response",
                call_id="codex-model-catalog",
                retryable=False,
                message="Codex app-server returned an invalid model catalog message",
            )
        if message.get("id") != request_id:
            continue
        if "error" in message:
            raise ProviderError(
                kind="provider_unavailable",
                call_id="codex-model-catalog",
                retryable=False,
                message="Codex could not read models for the signed-in ChatGPT account",
            )
        return message.get("result")


async def list_chatgpt_models(
    *,
    command: tuple[str, ...] = ("codex",),
    profile: Path | None = None,
) -> tuple[ChatGPTModel, ...]:
    """List visible models from the account used by the official Codex profile."""
    try:
        process = await anyio.open_process(
            (
                *codex_command_prefix(command),
                "app-server",
                "--listen",
                "stdio://",
            ),
            env=codex_environment(profile=profile),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if process.stdin is None or process.stdout is None:
            process.kill()
            _ = await process.wait()
            raise ProviderError(
                kind="configuration",
                call_id="codex-model-catalog",
                retryable=False,
                message="Codex app-server did not provide its JSON-RPC streams",
            )

        models: dict[str, ChatGPTModel] = {}
        cursors: set[str] = set()
        reader = _JsonLineReader(process.stdout)
        async with process:
            with anyio.fail_after(_REQUEST_TIMEOUT_SECONDS):
                initialize = await _request(
                    reader,
                    process.stdin,
                    request_id=1,
                    method="initialize",
                    params={
                        "clientInfo": {
                            "name": _APP_NAME,
                            "title": "JevPilot",
                            "version": _APP_VERSION,
                        }
                    },
                )
                if not isinstance(initialize, dict):
                    raise ProviderError(
                        kind="invalid_response",
                        call_id="codex-model-catalog",
                        retryable=False,
                        message="Codex app-server initialization response was invalid",
                    )
                await _send(process.stdin, {"method": "initialized"})

                cursor: str | None = None
                request_id = 2
                while True:
                    params: dict[str, JsonValue] = {"includeHidden": False, "limit": 200}
                    if cursor is not None:
                        params["cursor"] = cursor
                    raw_page = await _request(
                        reader,
                        process.stdin,
                        request_id=request_id,
                        method="model/list",
                        params=params,
                    )
                    page = _ModelPage.model_validate(raw_page)
                    for entry in page.data:
                        if not entry.hidden:
                            supported_efforts = _reasoning_efforts(entry.reasoning_efforts)
                            models[entry.id] = ChatGPTModel(
                                model_id=entry.id,
                                display_name=entry.display_name,
                                is_default=entry.is_default,
                                supports_fast=any(
                                    tier.id == "priority" for tier in entry.service_tiers
                                ),
                                reasoning_efforts=supported_efforts,
                            )
                    cursor = page.next_cursor
                    request_id += 1
                    if cursor is None:
                        break
                    if cursor in cursors:
                        raise ProviderError(
                            kind="invalid_response",
                            call_id="codex-model-catalog",
                            retryable=False,
                            message="Codex app-server repeated a model catalog cursor",
                        )
                    cursors.add(cursor)
                await process.stdin.aclose()
                _ = await process.wait()
        ordered_ids = tuple(
            model_id for model_id in _REQUESTED_MODEL_IDS if model_id in models
        )
        requested = tuple(models[model_id] for model_id in ordered_ids)
        additional = tuple(
            model for model_id, model in models.items()
            if model_id not in _REQUESTED_MODEL_IDS
        )
        return (*requested, *additional)
    except (
        OSError,
        TimeoutError,
        ValidationError,
        ValueError,
        anyio.BrokenResourceError,
        anyio.ClosedResourceError,
    ):
        raise ProviderError(
            kind="provider_unavailable",
            call_id="codex-model-catalog",
            retryable=False,
            message="Codex could not read models for the signed-in ChatGPT account",
        ) from None
