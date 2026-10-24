"""API-key LLM providers (OpenAI, Anthropic) built on browser-use chat models.

D7: only ``openai`` and ``anthropic`` exist. ``JEVPILOT_OPENAI_BASE_URL`` is a
test/QA-only override that points the ``openai`` provider at a local fake
server. There is no fallback to any other provider.

Notes on browser-use 0.13.11 (the pinned version):
- ``ChatOpenAI``/``ChatAnthropic`` retry inside the SDK (``max_retries`` 5/10).
  We set ``max_retries=0`` so one transmission equals one ledger call record.
- ``ChatInvokeUsage`` carries no billed cost for either provider, so
  ``billed_usd`` is always unknown here; ``api_estimate_usd`` comes from the
  static table in ``jevpilot.llm.pricing`` (unknown models stay ``None``).
- The browser-use online cost service is never used.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Literal, cast, get_args, override

import httpx
from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.exceptions import (
    ModelOutputTruncatedError,
    ModelProviderError,
    ModelRateLimitError,
)
from browser_use.llm.messages import (
    BaseMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.llm.views import ChatInvokeUsage
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, ValidationError

from jevpilot.ledger.models import CallRecord, CallUsage
from jevpilot.ledger.recorder import CallScope as LedgerCallScope
from jevpilot.llm.credentials import get_api_key
from jevpilot.llm.errors import LLMError
from jevpilot.llm.pricing import estimate_cost_usd
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.types import (
    CallScope,
    Capabilities,
    LLMCallRecord,
    LLMRequest,
    LLMResult,
    LLMUsage,
    ModelRef,
)

ApiProviderId = Literal["openai", "anthropic"]

OPENAI_BASE_URL_ENV: Final = "JEVPILOT_OPENAI_BASE_URL"
_OPENAI_DEFAULT_BASE: Final = "https://api.openai.com/v1"
_ANTHROPIC_DEFAULT_BASE: Final = "https://api.anthropic.com"
_ANTHROPIC_VERSION: Final = "2023-06-01"
_LIST_TIMEOUT_S: Final = 15.0

# OpenAI: the SDK's ``reasoning_effort`` literal (minus None). Anthropic: the
# ``thinking={"type": "enabled", "budget_tokens": N}`` argument of ChatAnthropic.
OPENAI_REASONING_EFFORTS: Final[tuple[str, ...]] = tuple(
    str(value) for value in get_args(get_args(ReasoningEffort)[0])
)
ANTHROPIC_THINKING_BUDGETS: Final[dict[str, int]] = {
    "low": 2048,
    "medium": 8192,
    "high": 24576,
}
_ANTHROPIC_OUTPUT_TOKENS: Final = 8192


def openai_base_url(explicit: str | None = None) -> str | None:
    """Explicit base URL, else the QA-only environment override, else SDK default."""
    if explicit:
        return explicit
    env = os.environ.get(OPENAI_BASE_URL_ENV, "").strip()
    return env or None


def _to_ledger_usage(usage: LLMUsage) -> CallUsage:
    return CallUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


@dataclass(frozen=True, slots=True)
class _ApiScope:
    """Translate structured attempts only at the concrete ledger boundary."""

    target: CallScope | LedgerCallScope

    @property
    def run_id(self) -> str:
        return self.target.run_id

    @property
    def parent_span_id(self) -> str | None:
        return self.target.parent_span_id

    @property
    def deadline(self) -> float:
        return self.target.deadline

    def record_call(self, record: LLMCallRecord) -> None:
        match self.target:
            case LedgerCallScope() as scope:
                scope.record_call(CallRecord(
                    call_id=record.call_id, span_id=scope.parent_span_id,
                    kind="llm", provider=record.provider, auth_mode=record.auth_mode,
                    purpose=record.purpose, request_model=record.model_id,
                    resolved_model=record.resolved_model,
                    outcome="ok" if record.outcome == "ok" else (
                        "cancelled" if record.outcome == "cancelled" else "failed"
                    ),
                    error_kind=None if record.outcome == "ok" else record.outcome,
                    usage=_to_ledger_usage(record.usage) if record.usage else CallUsage(),
                    latency_s=record.ended_monotonic - record.started_monotonic,
                ))
            case target:
                target.record_call(record)


def estimate_api_cost_usd(ref: ModelRef, usage: LLMUsage | None) -> Decimal | None:
    """Public-price estimate for an API call; ``None`` when price or usage is unknown."""
    if usage is None:
        return None
    return estimate_cost_usd(ref.provider, ref.model_id, _to_ledger_usage(usage))


def convert_usage(usage: ChatInvokeUsage | None) -> LLMUsage | None:
    """browser-use usage → JevPilot usage. ``prompt_tokens`` already includes cached."""
    if usage is None:
        return None
    return LLMUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_input_tokens=usage.prompt_cached_tokens,
        reasoning_tokens=None,
    )


def _media_type(data: bytes) -> Literal["image/png", "image/jpeg", "image/gif", "image/webp"]:
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def to_messages(prompt: StructuredPrompt) -> list[BaseMessage]:
    user: UserMessage
    if prompt.images:
        parts: list[ContentPartTextParam | ContentPartImageParam] = [
            ContentPartTextParam(text=prompt.user_text)
        ]
        for image in prompt.images:
            media = _media_type(image)
            encoded = base64.b64encode(image).decode("ascii")
            parts.append(
                ContentPartImageParam(
                    image_url=ImageURL(url=f"data:{media};base64,{encoded}", media_type=media)
                )
            )
        user = UserMessage(content=parts)
    else:
        user = UserMessage(content=prompt.user_text)
    return [SystemMessage(content=prompt.system), user]


def _cause_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _status_error(status: int, *, quota: bool = False) -> LLMError:
    if status in (401, 403):
        return LLMError("auth_invalid", retryable=False, message="provider rejected the API key")
    if status == 429:
        if quota:
            return LLMError("quota_exhausted", retryable=False, message="provider quota is exhausted")
        return LLMError("rate_limited", retryable=True, message="provider rate limit reached")
    if status in (400, 404, 409, 413, 422):
        return LLMError("configuration", retryable=False, message=f"provider rejected the request (HTTP {status})")
    if status in (408, 504):
        return LLMError("timeout", retryable=True, message=f"provider timed out (HTTP {status})")
    return LLMError("provider_unavailable", retryable=True, message=f"provider unavailable (HTTP {status})")


def normalize_error(exc: Exception) -> LLMError:
    """Map a browser-use/SDK failure to ``LLMError`` with fixed, secret-free text."""
    if isinstance(exc, LLMError):
        return exc
    chain = _cause_chain(exc)
    if any(isinstance(item, (httpx.TimeoutException, asyncio.TimeoutError)) or type(item).__name__ == "APITimeoutError" for item in chain):
        return LLMError("timeout", retryable=True, message="provider request timed out")
    if any(isinstance(item, ValidationError) for item in chain):
        return LLMError("invalid_output", retryable=False, message="provider output failed schema validation")
    if isinstance(exc, ModelOutputTruncatedError):
        return LLMError("invalid_output", retryable=False, message="provider output was truncated at the token limit")
    if isinstance(exc, ModelRateLimitError):
        return _status_error(429, quota="insufficient_quota" in exc.message)
    if any(type(item).__name__ == "APIConnectionError" for item in chain):
        return LLMError("provider_unavailable", retryable=True, message="could not connect to the provider")
    if isinstance(exc, ModelProviderError):
        if exc.status_code == 502 and any(isinstance(item, ValueError) for item in chain[1:]):
            return LLMError("configuration", retryable=False, message="provider rejected the model configuration")
        return _status_error(exc.status_code)
    return LLMError("provider_unavailable", retryable=True, message=f"provider call failed ({type(exc).__name__})")


class ApiKeyProvider:
    def __init__(self, ref: ModelRef, api_key: str, *, base_url: str | None = None) -> None:
        if ref.provider not in get_args(ApiProviderId) or ref.auth_mode != "api":
            raise LLMError("configuration", retryable=False, message="not an API-key provider")
        if not api_key.strip():
            raise LLMError("auth_missing", retryable=False, message="API key is not set")
        self._ref: ModelRef = ref
        self._api_key: str = api_key.strip()
        self._base_url: str | None = (
            openai_base_url(base_url or ref.base_url)
            if ref.provider == "openai"
            else (base_url or ref.base_url)
        )
        efforts = self.capabilities().reasoning_efforts
        if ref.reasoning_effort is not None and ref.reasoning_effort not in efforts:
            raise LLMError("configuration", retryable=False, message="unsupported reasoning effort")
        self._http: httpx.AsyncClient = httpx.AsyncClient()

    @override
    def __repr__(self) -> str:
        return f"ApiKeyProvider(provider={self._ref.provider!r}, model={self._ref.model_id!r})"

    @property
    def ref(self) -> ModelRef:
        return self._ref

    def capabilities(self) -> Capabilities:
        efforts = (
            OPENAI_REASONING_EFFORTS
            if self._ref.provider == "openai"
            else tuple(ANTHROPIC_THINKING_BUDGETS)
        )
        return Capabilities(
            structured_output="native", images=True, reasoning_efforts=efforts, supports_fast=False
        )

    def estimate_usd(self, usage: LLMUsage | None) -> Decimal | None:
        return estimate_api_cost_usd(self._ref, usage)

    def _chat(self, timeout_s: float) -> ChatOpenAI | ChatAnthropic:
        ref = self._ref
        if ref.provider == "openai":
            effort = ref.reasoning_effort
            return ChatOpenAI(
                model=ref.model_id,
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout_s,
                max_retries=0,
                http_client=self._http,
                # Only the user's chosen effort is sent; no browser-use default.
                temperature=None,
                frequency_penalty=None,
                reasoning_effort=cast(ReasoningEffort, effort or "low"),
                reasoning_models=[ref.model_id] if effort else [],
            )
        budget = ANTHROPIC_THINKING_BUDGETS.get(ref.reasoning_effort or "")
        return ChatAnthropic(
            model=ref.model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=timeout_s,
            max_retries=0,
            http_client=self._http,
            thinking=None if budget is None else {"type": "enabled", "budget_tokens": budget},
            max_tokens=_ANTHROPIC_OUTPUT_TOKENS + (budget or 0),
        )

    async def generate[T](
        self, request: LLMRequest[T], context: CallScope | LedgerCallScope,
    ) -> LLMResult[T]:
        output_type = request.output_type
        structured = isinstance(output_type, type) and issubclass(output_type, BaseModel)
        if output_type is not str and not structured:
            raise LLMError(
                "configuration", retryable=False,
                message="API providers need a pydantic model or str output type",
            )

        async def send(prompt: StructuredPrompt) -> RawCompletion:
            remaining = context.deadline - time.monotonic()
            if remaining <= 0:
                raise LLMError("timeout", retryable=False, message="model call deadline passed")
            chat = self._chat(remaining)
            messages = to_messages(prompt)
            match context:
                case LedgerCallScope() as scope:
                    scope.reserve_attempt()
            try:
                if structured:
                    completion = await chat.ainvoke(
                        messages, output_format=cast(type[BaseModel], output_type)
                    )
                    text = cast(BaseModel, completion.completion).model_dump_json()
                else:
                    completion = await chat.ainvoke(messages)
                    text = cast(str, completion.completion)
            except Exception as exc:
                raise normalize_error(exc) from None
            return RawCompletion(text=text, resolved_model=None, usage=convert_usage(completion.usage))

        return await generate_structured(
            request, _ApiScope(context), ref=self._ref, mode="native", send=send
        )

    async def aclose(self) -> None:
        await self._http.aclose()


def create_api_provider(ref: ModelRef) -> ApiKeyProvider:
    """Build a provider with the key from env/credential store; never another provider."""
    if ref.provider not in get_args(ApiProviderId):
        raise LLMError("configuration", retryable=False, message="not an API-key provider")
    key = get_api_key(cast(ApiProviderId, ref.provider))
    if not key:
        raise LLMError("auth_missing", retryable=False, message="API key is not set")
    return ApiKeyProvider(ref, key)


def _models_request(provider: ApiProviderId, key: str, base_url: str | None) -> tuple[str, dict[str, str], dict[str, str]]:
    if provider == "openai":
        base = (openai_base_url(base_url) or _OPENAI_DEFAULT_BASE).rstrip("/")
        return f"{base}/models", {"Authorization": f"Bearer {key}"}, {}
    base = (base_url or _ANTHROPIC_DEFAULT_BASE).rstrip("/")
    headers = {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION}
    return f"{base}/v1/models", headers, {"limit": "1000"}


def _is_quota(response: httpx.Response) -> bool:
    try:
        body = cast(object, response.json())
    except ValueError:
        return False
    if not isinstance(body, dict):
        return False
    error = cast(dict[str, Any], body).get("error")
    return isinstance(error, dict) and cast(dict[str, Any], error).get("code") == "insufficient_quota"


async def list_models(
    provider: ApiProviderId,
    key: str,
    base_url: str | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[str]:
    if provider not in get_args(ApiProviderId):
        raise LLMError("configuration", retryable=False, message="not an API-key provider")
    if not key.strip():
        raise LLMError("auth_missing", retryable=False, message="API key is not set")
    url, headers, params = _models_request(provider, key.strip(), base_url)
    try:
        async with httpx.AsyncClient(transport=transport, timeout=_LIST_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers, params=params)
    except httpx.TimeoutException:
        raise LLMError("timeout", retryable=True, message="model list request timed out") from None
    except httpx.HTTPError:
        raise LLMError("provider_unavailable", retryable=True, message="could not reach the provider") from None
    if not 200 <= response.status_code < 300:
        raise _status_error(response.status_code, quota=_is_quota(response))
    if response.status_code == 204:
        return []
    try:
        data = cast(dict[str, Any], response.json())["data"]
        return [str(cast(dict[str, Any], item)["id"]) for item in cast(list[object], data)]
    except (ValueError, KeyError, TypeError):
        raise LLMError("provider_unavailable", retryable=True, message="unexpected model list response") from None


async def validate_key(
    provider: ApiProviderId,
    key: str,
    base_url: str | None = None,
    model_id: str | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Pass iff the authenticated model list returns 2xx (and lists ``model_id`` if given).

    Raises ``LLMError``; callers must not store the key when this raises.
    """
    models = await list_models(provider, key, base_url, transport=transport)
    if model_id is not None and model_id not in models:
        raise LLMError("configuration", retryable=False, message="model is not available for this key")


__all__ = [
    "ANTHROPIC_THINKING_BUDGETS",
    "OPENAI_BASE_URL_ENV",
    "OPENAI_REASONING_EFFORTS",
    "ApiKeyProvider",
    "ApiProviderId",
    "convert_usage",
    "create_api_provider",
    "estimate_api_cost_usd",
    "list_models",
    "normalize_error",
    "to_messages",
    "validate_key",
]
