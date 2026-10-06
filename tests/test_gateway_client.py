import json
import time
from decimal import Decimal

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.gateway_client import GatewayModelClient
from jevpilot.privacy import PrivacyPolicy
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
@pytest.mark.parametrize(
    ("model", "expected_credits"),
    [("gpt-6-luna", Decimal("0.0026")), ("claude-sonnet-5", Decimal("0.052"))],
)
async def test_gateway_returns_schema_validated_output_for_selected_models(
    model: str,
    expected_credits: Decimal,
) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=completion(model, json.dumps({"choice": "c1"})),
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
            request_model=model,
            context=context(events),
        )

    # Then
    assert result.data == {"choice": "c1"}
    assert result.response_model == model
    assert len(requests) == 1
    assert requests[0].url == (
        "https://factchat-cloud.mindlogic.ai/v1/gateway/chat/completions/"
    )
    assert requests[0].headers["Authorization"] == "Bearer gateway-test-key"
    body = _JSON.validate_json(requests[0].content)
    assert isinstance(body, dict)
    assert body["model"] == model
    response_format = body["response_format"]
    assert isinstance(response_format, dict)
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["strict"] is True
    assert json_schema["schema"] == schema()
    assert len(events) == 1
    assert events[0].provider == "chosun_api_gateway"
    assert events[0].outcome == "succeeded"
    assert events[0].usage is not None
    assert events[0].usage.input_tokens == 11
    assert events[0].usage.output_tokens == 3
    assert events[0].usage.cache_read_tokens == 2
    assert events[0].estimated_cost_credits == expected_credits
    assert events[0].credit_price_checked_on == "2026-09-30"


@pytest.mark.asyncio
async def test_gateway_redacts_payload_before_sending() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json=completion("gpt-6-luna", json.dumps({"choice": "c1"})),
        )

    events: list[CallEvent] = []
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(
            api_key="gateway-test-key",
            http_client=http_client,
            privacy_policy=PrivacyPolicy(secret_values=("synthetic-secret",)),
        )

        # When
        _ = await client.generate_json(
            instructions="Select the observed value.",
            payload={"value": "synthetic-secret"},
            schema=schema(),
            request_model="gpt-6-luna",
            context=context(events),
        )

    # Then
    body = _JSON.validate_json(requests[0].content)
    assert isinstance(body, dict)
    messages = body["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    assert "synthetic-secret" not in str(user_message["content"])
    assert events[0].sent is True


@pytest.mark.asyncio
async def test_gateway_credit_balance_reads_only_the_account_total() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={"object": "credit_balance", "total": {"remaining": 10000}},
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        client = GatewayModelClient(api_key="gateway-test-key", http_client=http_client)

        # When
        balance = await client.credit_balance()

    # Then
    assert balance.total.remaining == Decimal("10000")
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert requests[0].url == (
        "https://factchat-cloud.mindlogic.ai/v1/gateway/credits/"
    )
    assert requests[0].headers["Authorization"] == "Bearer gateway-test-key"
