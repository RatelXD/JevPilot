"""Bridge a JevPilot ``LLMProvider`` to browser-use's ``BaseChatModel`` protocol.

Comparison arm A only: this adapter lets browser-use's own ``Agent`` loop run on
a JevPilot provider. It must never be used inside the hybrid loop.

Vision: when the provider reports ``capabilities().images == False`` the bridge
drops every image part before calling the provider. Construct the browser-use
agent with ``use_vision=JevPilotChatModel.supports_vision`` (``False`` in that
case) so browser-use does not spend effort producing screenshots nobody reads.

Usage: the provider records the original usage in the JevPilot ledger through
the ``CallScope``. The browser-use ``ChatInvokeUsage`` is only filled when the
provider reported both input and output tokens; otherwise ``usage=None``.
"""

from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Callable
from typing import Any, cast

from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import (
    AssistantMessage,
    BaseMessage,
    ContentPartImageParam,
    ContentPartRefusalParam,
    ContentPartTextParam,
    SystemMessage,
    UserMessage,
)
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel, JsonValue
from pydantic_core import core_schema

from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import (
    CallScope,
    LLMCallRecord,
    LLMProvider,
    LLMRequest,
    LLMUsage,
)

DEFAULT_TIMEOUT_S = 120.0
_DATA_URL_PREFIX = "data:"


class _DefaultScope:
    def __init__(self, timeout_s: float) -> None:
        self._deadline = time.monotonic() + timeout_s
        self.records: list[LLMCallRecord] = []

    @property
    def run_id(self) -> str:
        return "browser-use-bridge"

    @property
    def parent_span_id(self) -> str | None:
        return None

    @property
    def deadline(self) -> float:
        return self._deadline

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


def _decode_data_url(url: str) -> bytes | None:
    if not url.startswith(_DATA_URL_PREFIX) or "," not in url:
        return None
    header, _, data = url.partition(",")
    if not header.endswith(";base64"):
        return None
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None


def _text_and_images(message: BaseMessage) -> tuple[str, list[bytes]]:
    content = message.content
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[bytes] = []
    for part in cast(
        "list[ContentPartTextParam | ContentPartImageParam | ContentPartRefusalParam]",
        content,
    ):
        if isinstance(part, ContentPartTextParam):
            texts.append(part.text)
        elif isinstance(part, ContentPartImageParam):
            data = _decode_data_url(part.image_url.url)
            if data is not None:
                images.append(data)
        else:
            texts.append(f"[Refusal] {part.refusal}")
    return "\n".join(texts), images


def _convert_usage(usage: LLMUsage | None) -> ChatInvokeUsage | None:
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    return ChatInvokeUsage(
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=usage.cached_input_tokens,
        prompt_cache_creation_tokens=None,
        prompt_image_tokens=None,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
    )


class JevPilotChatModel:
    _verified_api_keys: bool = True

    def __init__(
        self,
        provider: LLMProvider,
        *,
        scope_factory: Callable[[], CallScope] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._llm = provider
        self._scope_factory = scope_factory or (lambda: _DefaultScope(timeout_s))
        self.model: str = provider.ref.model_id

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: type, handler: Any) -> Any:
        _ = (source_type, handler)
        return core_schema.any_schema()

    @property
    def provider(self) -> str:
        return self._llm.ref.provider

    @property
    def name(self) -> str:
        return f"{self._llm.ref.provider}/{self._llm.ref.model_id}"

    @property
    def model_name(self) -> str:
        return self.model

    @property
    def supports_vision(self) -> bool:
        return self._llm.capabilities().images

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        output_format: type[Any] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[Any]:
        _ = kwargs
        keep_images = self.supports_vision
        system_parts: list[str] = []
        transcript: list[str] = []
        images: list[bytes] = []
        for message in messages:
            text, message_images = _text_and_images(message)
            if keep_images:
                images.extend(message_images)
            if isinstance(message, SystemMessage):
                system_parts.append(text)
            elif isinstance(message, UserMessage):
                transcript.append(f"[user]\n{text}")
            elif isinstance(message, AssistantMessage):
                transcript.append(f"[assistant]\n{text}")
        payload = cast(JsonValue, "\n\n".join(transcript))
        output_type: type[Any] = str if output_format is None else output_format
        request = LLMRequest(
            purpose="agent_step",
            system="\n\n".join(system_parts),
            user_payload=payload,
            output_type=output_type,
            images=images,
        )
        try:
            result = await self._llm.generate(request, self._scope_factory())
        except LLMError as exc:
            message = f"{exc.message} (kind={exc.kind})"
            if exc.kind in ("rate_limited", "quota_exhausted"):
                raise ModelRateLimitError(message, model=self.model) from exc
            raise ModelProviderError(message, model=self.model) from exc
        value = result.value
        if output_format is not None and not isinstance(value, BaseModel):
            raise ModelProviderError(
                "provider returned a value that is not the requested output_format",
                model=self.model,
            )
        return ChatInvokeCompletion(
            completion=value, usage=_convert_usage(result.usage)
        )

