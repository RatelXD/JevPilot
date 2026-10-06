import time

import httpx2
import pytest
from pydantic import JsonValue

from jevpilot.model_client import (
    ModelClient,
    ModelClientConfig,
    ProviderError,
    ResponseMetadata,
    retry_after_delay,
)
from jevpilot.telemetry.models import CallContext, CallEvent


def make_context() -> tuple[CallContext, list[CallEvent], list[None]]:
    events: list[CallEvent] = []
    attempts: list[None] = []
    return (
        CallContext(
            run_id="run-1",
            logical_call_id="logical-1",
            role="selector",
            purpose="normal",
            deadline=time.monotonic() + 10,
            max_attempts=2,
            reserve_attempt=lambda: attempts.append(None),
            record_call=events.append,
        ),
        events,
        attempts,
    )


@pytest.mark.asyncio
async def test_configuration_failure_records_no_send() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("transport must not be called")

    context, events, attempts = make_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = ModelClient[JsonValue](
            ModelClientConfig(
                provider="jev",
                endpoint="https://api.typesafe.ai/v1/systemone",
                api_key="",
                retry_statuses=frozenset({429, 529}),
            ),
            http_client=http_client,
        )

        with pytest.raises(ProviderError) as raised:
            _ = await client.request_json(
                request_model="jev-1.13.0",
                payload={"state": "safe"},
                context=context,
                parser=lambda body, metadata: body,
            )

    assert raised.value.kind == "configuration"
    assert attempts == []
    assert len(events) == 1
    assert events[0].source == "network"
    assert events[0].sent is False


@pytest.mark.asyncio
async def test_retry_records_each_actual_transmission() -> None:
    responses = [
        httpx2.Response(429, json={"error": {"type": "rate_limit"}}),
        httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {},
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        ),
    ]

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return responses.pop(0)

    context, events, attempts = make_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = ModelClient[JsonValue](
            ModelClientConfig(
                provider="jev",
                endpoint="https://api.typesafe.ai/v1/systemone",
                api_key="test-key",
                retry_statuses=frozenset({429, 529}),
            ),
            http_client=http_client,
        )
        result = await client.request_json(
            request_model="jev-1.13.0",
            payload={"state": "safe"},
            context=context,
            parser=lambda body, metadata: body,
        )

    assert isinstance(result, dict)
    assert len(attempts) == 2
    assert [event.sent for event in events] == [True, True]
    assert [event.outcome for event in events] == ["failed", "succeeded"]
    assert events[0].error_kind == "rate_limited"
    assert events[1].usage is not None
    assert events[1].usage.input_tokens == 7


@pytest.mark.parametrize(
    ("header", "expected"),
    [("0", 0.0), ("-1", None), ("NaN", None), ("not-a-date", None)],
)
def test_retry_after_header_accepts_only_bounded_numeric_delays(
    header: str,
    expected: float | None,
) -> None:
    # Given / When
    delay = retry_after_delay(header)

    # Then
    assert delay == expected


@pytest.mark.asyncio
async def test_retry_after_beyond_request_deadline_does_not_retry() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            429,
            json={"error": {"type": "rate_limit"}},
            headers={"Retry-After": "20"},
        )

    context, events, attempts = make_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = ModelClient[JsonValue](
            ModelClientConfig(
                provider="jev",
                endpoint="https://api.typesafe.ai/v1/systemone",
                api_key="test-key",
                retry_statuses=frozenset({429}),
            ),
            http_client=http_client,
        )

        with pytest.raises(ProviderError) as raised:
            _ = await client.request_json(
                request_model="jev-1.13.0",
                payload={"state": "safe"},
                context=context,
                parser=lambda body, metadata: body,
            )

    assert raised.value.retry_after_seconds == 20.0
    assert attempts == [None]
    assert len(events) == 1


@pytest.mark.asyncio
async def test_parse_failure_preserves_response_usage() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {},
                "usage": {"input_tokens": 11, "output_tokens": 5},
            },
        )

    def reject(body: JsonValue, metadata: ResponseMetadata) -> JsonValue:
        _ = body
        raise ProviderError(
            kind="invalid_response",
            call_id=metadata.call_id,
            retryable=False,
            message="safe invalid response",
            status_code=metadata.status_code,
        )

    context, events, attempts = make_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = ModelClient[JsonValue](
            ModelClientConfig(
                provider="jev",
                endpoint="https://api.typesafe.ai/v1/systemone",
                api_key="test-key",
                retry_statuses=frozenset({429, 529}),
            ),
            http_client=http_client,
        )

        with pytest.raises(ProviderError):
            _ = await client.request_json(
                request_model="jev-1.13.0",
                payload={"state": "safe"},
                context=context,
                parser=reject,
            )

    assert len(attempts) == 1
    assert len(events) == 1
    assert events[0].outcome == "failed"
    assert events[0].usage is not None
    assert events[0].usage.input_tokens == 11
    assert events[0].usage.output_tokens == 5
    assert events[0].estimated_cost_usd is None
