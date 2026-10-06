import json
import time

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.gateway_client import GatewayModelClient
from jevpilot.model_client import ProviderError
from jevpilot.telemetry.models import CallContext, CallEvent

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def context(events: list[CallEvent]) -> CallContext:
    return CallContext(
        run_id="run-1",
        logical_call_id="logical-1",
        role="planner",
        purpose="initial",
        deadline=time.monotonic() + 10,
        max_attempts=2,
        reserve_attempt=lambda: None,
        record_call=events.append,
    )


def schema() -> JsonValue:
    return {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["c1"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }


def completion(model: str, content: str) -> JsonValue:
    return _JSON.validate_python(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
    )


@pytest.mark.asyncio
async def test_gateway_retries_rate_limit_using_zero_retry_after() -> None:
    responses = [
        httpx2.Response(
            429,
            json={"detail": {"code": 6005}},
            headers={"Retry-After": "0"},
        ),
        httpx2.Response(
            200,
            json=completion("gpt-6-luna", json.dumps({"choice": "c1"})),
        ),
    ]
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return responses.pop(0)

    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(api_key="gateway-test-key", http_client=http_client)

        # When
        result = await client.generate_json(
            instructions="Select the observed value.",
            payload={"goal": "synthetic goal"},
            schema=schema(),
            request_model="gpt-6-luna",
            context=context(events),
        )

    # Then
    assert result.data == {"choice": "c1"}
    assert len(requests) == 2
    assert [event.outcome for event in events] == ["failed", "succeeded"]
    assert [event.attempt_index for event in events] == [1, 2]


@pytest.mark.asyncio
async def test_gateway_credit_exhaustion_does_not_retry() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            402,
            json={"detail": {"message": "Credit balance exhausted"}},
        )

    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(api_key="gateway-test-key", http_client=http_client)

        # When / Then
        with pytest.raises(ProviderError) as raised:
            _ = await client.generate_json(
                instructions="Select the observed value.",
                payload={"goal": "synthetic goal"},
                schema=schema(),
                request_model="gpt-6-luna",
                context=context(events),
            )

    assert raised.value.kind == "insufficient_credit"
    assert len(requests) == 1
    assert len(events) == 1
    assert events[0].status_code == 402


@pytest.mark.asyncio
async def test_new_gateway_model_keeps_usage_when_credit_price_is_unknown() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json=completion("new-account-model", json.dumps({"choice": "c1"})),
        )

    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(api_key="gateway-test-key", http_client=http_client)

        # When
        result = await client.generate_json(
            instructions="Select the observed value.",
            payload={"goal": "synthetic goal"},
            schema=schema(),
            request_model="new-account-model",
            context=context(events),
        )

    # Then
    assert result.data == {"choice": "c1"}
    assert events[0].usage is not None
    assert events[0].estimated_cost_credits is None
    assert events[0].credit_price_source is not None


@pytest.mark.asyncio
async def test_gateway_rejects_schema_invalid_output_without_retry() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json=completion("gpt-6-luna", json.dumps({"choice": "unknown"})),
        )

    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(api_key="gateway-test-key", http_client=http_client)

        # When / Then
        with pytest.raises(ProviderError, match="invalid_response"):
            _ = await client.generate_json(
                instructions="Select the observed value.",
                payload={"goal": "synthetic goal"},
                schema=schema(),
                request_model="gpt-6-luna",
                context=context(events),
            )

    assert len(events) == 1
    assert events[0].outcome == "failed"
    assert events[0].usage is not None
    assert events[0].usage.output_tokens == 3


@pytest.mark.asyncio
async def test_gateway_missing_key_is_a_no_send_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("missing key must not send")

    monkeypatch.delenv("JEVPILOT_LLM_API_KEY", raising=False)
    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient.from_env(http_client=http_client)

        # When / Then
        with pytest.raises(ProviderError, match="configuration"):
            _ = await client.generate_json(
                instructions="Select the observed value.",
                payload={"goal": "synthetic goal"},
                schema=schema(),
                request_model="gpt-6-luna",
                context=context(events),
            )

    assert len(events) == 1
    assert events[0].sent is False
    assert events[0].outcome == "failed"
