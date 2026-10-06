"""Bounded HTTP transport and telemetry for model providers."""
# noqa: SIZE_OK — transport attempt state and CallEvent accounting are indivisible here.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
import math
import socket
import time
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    NoReturn,
    Protocol,
    TypeAlias,
    TypeVar,
    final,
    override,
)
from uuid import uuid4

import anyio
import httpx2
from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.privacy import PrivacyPolicy, redact_json
if TYPE_CHECKING:
    from jevpilot.telemetry.models import CallContext, CallUsage

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_LIMITS = httpx2.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=30.0,
)
_SOCKET_OPTIONS = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
_SAFE_CONFIGURATION_MESSAGE = "model provider configuration is incomplete"
_SAFE_INVALID_RESPONSE_MESSAGE = "model provider returned an invalid response"
_SAFE_TIMEOUT_MESSAGE = "model provider request exceeded its deadline"
_SAFE_UNAVAILABLE_MESSAGE = "model provider is unavailable"

T = TypeVar("T")
CreditEstimator: TypeAlias = Callable[[str, "CallUsage | None"], Decimal | None]
CostEstimator: TypeAlias = Callable[[str, "CallUsage | None"], Decimal | None]


class ProviderError(RuntimeError):
    """Safe provider failure surfaced to runtime and traces."""

    kind: str
    call_id: str
    retryable: bool
    message: str
    status_code: int | None
    retry_after_seconds: float | None

    def __init__(
        self, *, kind: str, call_id: str, retryable: bool,
        message: str, status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.kind = kind
        self.call_id = call_id
        self.retryable = retryable
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)

    @override
    def __str__(self) -> str:
        status = (
            f", status_code={self.status_code}"
            if self.status_code is not None
            else ""
        )
        return f"{self.message} (kind={self.kind}, call_id={self.call_id}{status})"


@dataclass(frozen=True, slots=True)
class ModelClientConfig:
    provider: str
    endpoint: str
    api_key: str
    retry_statuses: frozenset[int]
    credit_estimator: CreditEstimator | None = None
    credit_price_source: str | None = None
    credit_price_checked_on: str | None = None
    cost_estimator: CostEstimator | None = None
    cost_price_source: str | None = None
    cost_price_checked_on: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseMetadata:
    call_id: str
    response_model: str | None
    usage: CallUsage | None
    status_code: int


ResponseParser = Callable[[JsonValue, ResponseMetadata], T]


def retry_after_delay(value: str | None) -> float | None:
    """Parse a bounded Retry-After delay from either supported header form."""
    if value is None:
        return None
    try:
        seconds = float(value)
        return seconds if math.isfinite(seconds) and seconds >= 0 else None
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


@dataclass(frozen=True, slots=True)
class StructuredModelResult:
    """Safe decoded result from an external structured model client."""

    data: JsonValue
    call_id: str
    response_model: str | None


class StructuredModelClient(Protocol):
    """Client seam implemented by a structured external model provider.

    Implementations own CallEvent emission, including usage and resolved model.
    """

    async def generate_json(
        self,
        *,
        instructions: str,
        payload: JsonValue,
        schema: JsonValue,
        request_model: str,
        context: CallContext,
    ) -> StructuredModelResult: ...


def create_model_http_client() -> httpx2.AsyncClient:
    """Create the shared no-redirect HTTP/2 transport for provider requests."""
    transport = httpx2.AsyncHTTPTransport(
        http2=True,
        retries=0,
        limits=_LIMITS,
        socket_options=_SOCKET_OPTIONS,
    )
    return httpx2.AsyncClient(
        transport=transport,
        follow_redirects=False,
        timeout=None,
    )


@final
class ModelClient(Generic[T]):
    """Send JSON requests with explicit attempts and one event per transmission."""

    def __init__(
        self,
        config: ModelClientConfig,
        *,
        http_client: httpx2.AsyncClient | None = None,
        privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        self.config = config
        self.privacy_policy = privacy_policy or PrivacyPolicy()
        self._owns_client = http_client is None
        if http_client is not None:
            self._http = http_client
            return
        self._http = create_model_http_client()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self) -> "ModelClient[T]":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def request_json(
        self,
        *,
        request_model: str,
        payload: JsonValue,
        context: CallContext,
        parser: ResponseParser[T],
    ) -> T:
        """POST sanitized JSON, parse it, and record every actual attempt."""
        preflight = self._preflight_error(request_model=request_model)
        if preflight is not None:
            self._record_preflight(
                error=preflight,
                request_model=request_model or "[missing]",
                context=context,
            )
            raise preflight
        if context.max_attempts < 1:
            self.raise_preflight_error(
                request_model=request_model,
                context=context,
                message="model call max_attempts must be positive",
            )

        sanitized = redact_json(payload, policy=self.privacy_policy)
        last_error: ProviderError | None = None
        for attempt_index in range(1, context.max_attempts + 1):
            remaining = context.deadline - time.monotonic()
            if remaining <= 0:
                error = ProviderError(
                    kind="transport_timeout",
                    call_id=uuid4().hex,
                    retryable=False,
                    message=_SAFE_TIMEOUT_MESSAGE,
                )
                self._record_preflight(
                    error=error,
                    request_model=request_model,
                    context=context,
                    attempt_index=attempt_index,
                )
                raise error
            result = await self._attempt(
                request_model=request_model,
                payload=sanitized,
                context=context,
                parser=parser,
                attempt_index=attempt_index,
                timeout=remaining,
            )
            if not isinstance(result, ProviderError):
                return result
            last_error = result
            if not result.retryable or attempt_index >= context.max_attempts:
                raise result
            delay = result.retry_after_seconds
            if delay is None:
                delay = 0.25 * float(1 << (attempt_index - 1))
            if delay >= remaining:
                raise result
            if delay > 0:
                await anyio.sleep(delay)
        if last_error is None:
            self.raise_preflight_error(
                request_model=request_model,
                context=context,
                message="model request did not execute an attempt",
            )
        raise last_error

    def raise_preflight_error(
        self,
        *,
        request_model: str,
        context: CallContext,
        message: str,
        kind: str = "configuration",
    ) -> NoReturn:
        """Record a safe no-send failure, then raise it."""
        error = ProviderError(
            kind=kind,
            call_id=uuid4().hex,
            retryable=False,
            message=message,
        )
        self._record_preflight(
            error=error,
            request_model=request_model or "[missing]",
            context=context,
        )
        raise error

    def _preflight_error(self, *, request_model: str) -> ProviderError | None:
        if self.config.provider and self.config.endpoint and self.config.api_key and request_model:
            return None
        return ProviderError(
            kind="configuration",
            call_id=uuid4().hex,
            retryable=False,
            message=_SAFE_CONFIGURATION_MESSAGE,
        )

    async def _attempt(
        self,
        *,
        request_model: str,
        payload: JsonValue,
        context: CallContext,
        parser: ResponseParser[T],
        attempt_index: int,
        timeout: float,
    ) -> T | ProviderError:
        call_id = uuid4().hex
        started_at = datetime.now(UTC)
        context.reserve_attempt()
        try:
            response = await self._http.post(
                self.config.endpoint,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "JevPilot/0.1.0",
                },
                json=payload,
                timeout=timeout,
            )
        except httpx2.TimeoutException:
            return self._record_transport_error(
                call_id=call_id,
                request_model=request_model,
                context=context,
                attempt_index=attempt_index,
                started_at=started_at,
                kind="transport_timeout",
                message=_SAFE_TIMEOUT_MESSAGE,
            )
        except httpx2.TransportError:
            return self._record_transport_error(
                call_id=call_id,
                request_model=request_model,
                context=context,
                attempt_index=attempt_index,
                started_at=started_at,
                kind="provider_unavailable",
                message=_SAFE_UNAVAILABLE_MESSAGE,
            )

        body = self._parse_body(response)
        usage = self._usage(body)
        response_model = self._response_model(body)
        if not response.is_success:
            error = self._http_error(
                call_id=call_id,
                status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )
            self._record_event(
                context=context,
                call_id=call_id,
                request_model=request_model,
                response_model=response_model,
                attempt_index=attempt_index,
                sent=True,
                started_at=started_at,
                outcome="failed",
                usage=usage,
                status_code=response.status_code,
                error=error,
            )
            return error
        if body is None:
            error = ProviderError(
                kind="invalid_response",
                call_id=call_id,
                retryable=False,
                message=_SAFE_INVALID_RESPONSE_MESSAGE,
                status_code=response.status_code,
            )
            self._record_event(
                context=context,
                call_id=call_id,
                request_model=request_model,
                response_model=None,
                attempt_index=attempt_index,
                sent=True,
                started_at=started_at,
                outcome="failed",
                usage=None,
                status_code=response.status_code,
                error=error,
            )
            return error
        metadata = ResponseMetadata(
            call_id=call_id,
            response_model=response_model,
            usage=usage,
            status_code=response.status_code,
        )
        try:
            parsed = parser(body, metadata)
        except ProviderError as error:
            self._record_event(
                context=context,
                call_id=call_id,
                request_model=request_model,
                response_model=response_model,
                attempt_index=attempt_index,
                sent=True,
                started_at=started_at,
                outcome="failed",
                usage=usage,
                status_code=response.status_code,
                error=error,
            )
            return error
        except ValidationError:
            error = ProviderError(
                kind="invalid_response",
                call_id=call_id,
                retryable=False,
                message=_SAFE_INVALID_RESPONSE_MESSAGE,
                status_code=response.status_code,
            )
            self._record_event(
                context=context,
                call_id=call_id,
                request_model=request_model,
                response_model=response_model,
                attempt_index=attempt_index,
                sent=True,
                started_at=started_at,
                outcome="failed",
                usage=usage,
                status_code=response.status_code,
                error=error,
            )
            return error
        self._record_event(
            context=context,
            call_id=call_id,
            request_model=request_model,
            response_model=response_model,
            attempt_index=attempt_index,
            sent=True,
            started_at=started_at,
            outcome="succeeded",
            usage=usage,
            status_code=response.status_code,
            error=None,
        )
        return parsed

    def _parse_body(self, response: httpx2.Response) -> JsonValue | None:
        try:
            return _JSON.validate_json(response.content)
        except ValidationError:
            return None

    def _record_transport_error(
        self,
        *,
        call_id: str,
        request_model: str,
        context: CallContext,
        attempt_index: int,
        started_at: datetime,
        kind: str,
        message: str,
    ) -> ProviderError:
        error = ProviderError(
            kind=kind,
            call_id=call_id,
            retryable=True,
            message=message,
        )
        self._record_event(
            context=context,
            call_id=call_id,
            request_model=request_model,
            response_model=None,
            attempt_index=attempt_index,
            sent=True,
            started_at=started_at,
            outcome="failed",
            usage=None,
            status_code=None,
            error=error,
        )
        return error

    def _record_preflight(
        self,
        *,
        error: ProviderError,
        request_model: str,
        context: CallContext,
        attempt_index: int = 1,
    ) -> None:
        now = datetime.now(UTC)
        self._record_event(
            context=context,
            call_id=error.call_id,
            request_model=request_model,
            response_model=None,
            attempt_index=attempt_index,
            sent=False,
            started_at=now,
            outcome="failed",
            usage=None,
            status_code=None,
            error=error,
        )

    def _record_event(
        self,
        *,
        context: CallContext,
        call_id: str,
        request_model: str,
        response_model: str | None,
        attempt_index: int,
        sent: bool,
        started_at: datetime,
        outcome: Literal["succeeded", "failed"],
        usage: CallUsage | None,
        status_code: int | None,
        error: ProviderError | None,
    ) -> None:
        from jevpilot.telemetry.models import CallEvent

        estimated_cost_usd: float | None = None
        if self.config.cost_estimator is not None and usage is not None:
            estimated = self.config.cost_estimator(
                response_model or request_model, usage
            )
            if estimated is not None:
                estimated_cost_usd = float(estimated)
        estimated_cost_credits: Decimal | None = None
        if self.config.credit_estimator is not None and usage is not None:
            estimated = self.config.credit_estimator(request_model, usage)
            if estimated is not None:
                estimated_cost_credits = estimated
        context.record_call(
            CallEvent(
                run_id=context.run_id,
                call_id=call_id,
                logical_call_id=context.logical_call_id,
                source="network",
                role=context.role,
                purpose=context.purpose,
                provider=self.config.provider,
                request_model=request_model,
                response_model=response_model,
                attempt_index=attempt_index,
                sent=sent,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                outcome=outcome,
                usage=usage,
                billed_cost_usd=None,
                estimated_cost_usd=estimated_cost_usd,
                price_source=self.config.cost_price_source,
                price_checked_on=self.config.cost_price_checked_on,
                estimated_cost_credits=estimated_cost_credits,
                credit_price_source=self.config.credit_price_source,
                credit_price_checked_on=self.config.credit_price_checked_on,
                status_code=status_code,
                error_kind=error.kind if error is not None else None,
                error_message=error.message if error is not None else None,
            )
        )

    def _http_error(
        self, *, call_id: str, status_code: int, retry_after: str | None
    ) -> ProviderError:
        if status_code in {401, 403}:
            return ProviderError(
                kind="authentication",
                call_id=call_id,
                retryable=False,
                message="model provider authentication failed",
                status_code=status_code,
            )
        if status_code == 429:
            kind = "rate_limited"
        elif status_code == 402:
            kind = "insufficient_credit"
        elif status_code in self.config.retry_statuses:
            kind = "provider_unavailable"
        else:
            kind = "configuration"
        return ProviderError(
            kind=kind,
            call_id=call_id,
            retryable=status_code in self.config.retry_statuses,
            message=(
                _SAFE_UNAVAILABLE_MESSAGE
                if kind == "provider_unavailable"
                else "model provider credits are exhausted"
                if kind == "insufficient_credit"
                else "model provider rejected the request"
            ),
            status_code=status_code,
            retry_after_seconds=retry_after_delay(retry_after),
        )

    def _usage(self, body: JsonValue | None) -> CallUsage | None:
        from jevpilot.telemetry.models import CallUsage

        if not isinstance(body, dict):
            return None
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return None
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        cache_details = usage.get(
            "input_tokens_details", usage.get("prompt_tokens_details")
        )
        cache_tokens: int | None = None
        if isinstance(cache_details, dict):
            candidate = cache_details.get("cached_tokens")
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                cache_tokens = candidate
        return CallUsage(
            input_tokens=(
                input_tokens
                if isinstance(input_tokens, int)
                and not isinstance(input_tokens, bool)
                and input_tokens >= 0
                else None
            ),
            output_tokens=(
                output_tokens
                if isinstance(output_tokens, int)
                and not isinstance(output_tokens, bool)
                and output_tokens >= 0
                else None
            ),
            cache_read_tokens=cache_tokens,
        )

    def _response_model(self, body: JsonValue | None) -> str | None:
        if not isinstance(body, dict):
            return None
        model = body.get("model")
        return model if isinstance(model, str) and model else None

