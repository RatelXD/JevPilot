# noqa: SIZE_OK — scope fixes decision provider coverage to this single owned test file.

from datetime import UTC, datetime
from decimal import Decimal
import time
from typing import final

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.decision import (
    DecisionInput,
    JevDecisionProvider,
    LLMDecisionProvider,
)
from jevpilot.decision.models import DecisionPolicy
from jevpilot.model_client import (
    ModelClient,
    ModelClientConfig,
    ProviderError,
    StructuredModelResult,
)
from jevpilot.observer.models import (
    ActionCandidate,
    Observation,
    OmittedCounts,
    WaitCondition,
)
from jevpilot.planner.models import Subgoal
from jevpilot.task import ActionType
from jevpilot.telemetry.models import CallContext, CallEvent

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def sample_input() -> DecisionInput:
    observation = Observation(
        snapshot_id="snapshot-1",
        document_id="document-1",
        navigation_id="navigation-1",
        url="https://example.test/items",
        title="Items",
        text="Open Alpha",
        controls=(),
        fingerprint="fingerprint-1",
        observed_at=datetime.now(UTC),
        omitted_counts=OmittedCounts(
            hidden=0,
            disabled=0,
            readonly=0,
            unnamed=0,
            over_limit=0,
        ),
    )
    return DecisionInput(
        goal="Open Alpha",
        current_subgoal=Subgoal(
            subgoal_id="subgoal-1",
            description="Open the Alpha item",
        ),
        observation=observation,
        candidates=(
            ActionCandidate(
                candidate_id="candidate-1",
                snapshot_id="snapshot-1",
                operation=ActionType.CLICK,
                target_ref="target-1",
                role="link",
                name="Alpha",
                enabled=True,
                readonly=False,
            ),
        ),
        policy=DecisionPolicy(
            allowed_operations=(
                ActionType.CLICK,
                ActionType.DONE,
                ActionType.BLOCKED,
            ),
            max_wait_ms=1000,
        ),
        recent_history=(),
    )


def text_input() -> DecisionInput:
    payload = sample_input()
    candidate = ActionCandidate(
        candidate_id="text-1",
        snapshot_id="snapshot-1",
        operation=ActionType.TYPE_TEXT,
        target_ref="target-text",
        role="searchbox",
        name="Search catalog",
        value="",
        enabled=True,
        readonly=False,
        max_length=24,
    )
    policy = payload.policy.model_copy(
        update={
            "allowed_operations": (
                ActionType.CLICK,
                ActionType.TYPE_TEXT,
                ActionType.DONE,
                ActionType.BLOCKED,
            )
        }
    )
    return payload.model_copy(
        update={"candidates": (*payload.candidates, candidate), "policy": policy}
    )


def call_context() -> tuple[CallContext, list[CallEvent]]:
    events: list[CallEvent] = []
    return (
        CallContext(
            run_id="run-1",
            logical_call_id="logical-1",
            role="selector",
            purpose="normal",
            deadline=time.monotonic() + 10,
            max_attempts=1,
            reserve_attempt=lambda: None,
            record_call=events.append,
        ),
        events,
    )


@pytest.mark.asyncio
async def test_jev_click_uses_one_operation_and_target_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEVPILOT_JEV_API_KEY", "test-key")
    monkeypatch.setenv("JEVPILOT_JEV_MODEL", "jev-latest")
    monkeypatch.setenv(
        "JEVPILOT_JEV_ENDPOINT", "https://api.typesafe.ai/v1/systemone"
    )
    requests: list[JsonValue] = []

    def handler(_request: httpx2.Request) -> httpx2.Response:
        requests.append(_JSON.validate_json(_request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "operation": {
                        "type": "choice",
                        "choice": "click",
                        "probabilities": {
                            "click": 0.8,
                            "done": 0.1,
                            "blocked": 0.1,
                        },
                        "confidence": 0.7,
                    },
                    "click_target": {
                        "type": "choice",
                        "choice": "candidate-1",
                        "probabilities": {"candidate-1": 1.0},
                        "confidence": 1.0,
                    },
                },
                "usage": {"input_tokens": 30, "output_tokens": 12},
            },
        )

    context, events = call_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        provider = JevDecisionProvider.from_env(http_client=http_client)
        result = await provider.decide(sample_input(), context=context)

    assert result.operation is ActionType.CLICK
    assert result.candidate_id == "candidate-1"
    assert result.operation_confidence == 0.7
    assert result.target_confidence == 1.0
    assert result.model == "jev-1.13.0"
    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request, dict)
    assert request["model"] == "jev-latest"
    questions = request["questions"]
    assert isinstance(questions, dict)
    assert set(questions) == {"operation", "click_target"}
    assert events[0].request_model == "jev-latest"
    assert events[0].response_model == "jev-1.13.0"
    assert Decimal(str(events[0].estimated_cost_usd)) == Decimal("0.00000126")
    assert events[0].price_source is not None
    assert events[0].price_checked_on == "2026-10-06"


@pytest.mark.asyncio
async def test_jev_ignores_unselected_malformed_click_head() -> None:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "operation": {
                        "type": "choice",
                        "choice": "done",
                        "probabilities": {
                            "click": 0.1,
                            "done": 0.8,
                            "blocked": 0.1,
                        },
                        "confidence": 0.7,
                    },
                    "click_target": "unused malformed head",
                },
                "usage": {"input_tokens": 20, "output_tokens": 8},
            },
        )

    context, _ = call_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        provider = JevDecisionProvider(
            ModelClient(
                ModelClientConfig(
                    provider="jev",
                    endpoint="https://api.typesafe.ai/v1/systemone",
                    api_key="test-key",
                    retry_statuses=frozenset({429, 529}),
                ),
                http_client=http_client,
            ),
            model="jev-requested",
        )
        result = await provider.decide(sample_input(), context=context)

    assert result.operation is ActionType.DONE
    assert result.candidate_id is None
    assert result.target_confidence is None


@pytest.mark.asyncio
async def test_jev_rejects_nonmaximal_selected_target_and_preserves_usage() -> None:
    payload = sample_input()
    second_candidate = payload.candidates[0].model_copy(
        update={
            "candidate_id": "candidate-2",
            "target_ref": "target-2",
            "name": "Beta",
        }
    )
    payload = payload.model_copy(
        update={"candidates": (*payload.candidates, second_candidate)}
    )

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "operation": {
                        "type": "choice",
                        "choice": "click",
                        "probabilities": {
                            "click": 0.8,
                            "done": 0.1,
                            "blocked": 0.1,
                        },
                        "confidence": 0.7,
                    },
                    "click_target": {
                        "type": "choice",
                        "choice": "candidate-1",
                        "probabilities": {
                            "candidate-1": 0.4,
                            "candidate-2": 0.6,
                        },
                        "confidence": 0.2,
                    },
                },
                "usage": {"input_tokens": 44, "output_tokens": 10},
            },
        )

    context, events = call_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        provider = JevDecisionProvider(
            ModelClient(
                ModelClientConfig(
                    provider="jev",
                    endpoint="https://api.typesafe.ai/v1/systemone",
                    api_key="test-key",
                    retry_statuses=frozenset({429, 529}),
                ),
                http_client=http_client,
            ),
            model="jev-requested",
        )

        with pytest.raises(ProviderError) as raised:
            _ = await provider.decide(payload, context=context)

    assert raised.value.kind == "invalid_response"
    assert events[0].usage is not None
    assert events[0].usage.input_tokens == 44
    assert events[0].outcome == "failed"


@final
class FakeStructuredClient:
    def __init__(self, result: StructuredModelResult) -> None:
        self.result = result
        self.schema: JsonValue | None = None

    async def generate_json(
        self,
        *,
        instructions: str,
        payload: JsonValue,
        schema: JsonValue,
        request_model: str,
        context: CallContext,
    ) -> StructuredModelResult:
        _ = (instructions, payload, request_model, context)
        self.schema = schema
        return self.result


@pytest.mark.asyncio
async def test_subscription_selector_keeps_confidence_unknown() -> None:
    client = FakeStructuredClient(
        StructuredModelResult(
            data={"operation": "click", "candidate_id": "candidate-1"},
            call_id="call-subscription",
            response_model="codex-response-model",
        )
    )
    provider = LLMDecisionProvider(client, model="codex-requested")
    context, _ = call_context()

    result = await provider.decide(sample_input(), context=context)

    assert result.operation is ActionType.CLICK
    assert result.operation_confidence is None
    assert result.target_confidence is None
    assert result.model == "codex-response-model"
    assert client.schema is not None


@pytest.mark.asyncio
async def test_jev_batches_click_and_text_target_heads_for_one_snapshot() -> None:
    requests: list[JsonValue] = []

    def handler(_request: httpx2.Request) -> httpx2.Response:
        requests.append(_JSON.validate_json(_request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "operation": {
                        "type": "choice",
                        "choice": "type_text",
                        "probabilities": {
                            "click": 0.1,
                            "type_text": 0.8,
                            "done": 0.05,
                            "blocked": 0.05,
                        },
                        "confidence": 0.8,
                    },
                    "click_target": "unused malformed head",
                    "type_text_target": {
                        "type": "choice",
                        "choice": "text-1",
                        "probabilities": {"text-1": 1.0},
                        "confidence": 0.9,
                    },
                },
                "usage": {"input_tokens": 50, "output_tokens": 14},
            },
        )

    context, _ = call_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        provider = JevDecisionProvider(
            ModelClient(
                ModelClientConfig(
                    provider="jev",
                    endpoint="https://api.typesafe.ai/v1/systemone",
                    api_key="test-key",
                    retry_statuses=frozenset({429, 529}),
                ),
                http_client=http_client,
            ),
            model="jev-requested",
        )
        result = await provider.decide(text_input(), context=context)

    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request, dict)
    state = request.get("state")
    assert isinstance(state, dict)
    assert state.get("snapshot_id") == "snapshot-1"
    questions = request["questions"]
    assert isinstance(questions, dict)
    assert set(questions) == {"operation", "click_target", "type_text_target"}
    assert result.operation is ActionType.TYPE_TEXT
    assert result.candidate_id == "text-1"
    assert result.operation_confidence == 0.8
    assert result.target_confidence == 0.9


@pytest.mark.asyncio
async def test_jev_wait_selects_only_the_observed_bounded_condition() -> None:
    # Given
    payload = sample_input()
    wait_candidate = ActionCandidate(
        candidate_id="wait-1",
        snapshot_id="snapshot-1",
        operation=ActionType.WAIT,
        wait_condition=WaitCondition(kind="state_change", timeout_ms=500),
    )
    wait_policy = payload.policy.model_copy(
        update={
            "allowed_operations": (
                ActionType.CLICK,
                ActionType.WAIT,
                ActionType.DONE,
                ActionType.BLOCKED,
            )
        }
    )
    payload = payload.model_copy(
        update={
            "candidates": (*payload.candidates, wait_candidate),
            "policy": wait_policy,
        }
    )
    requests: list[JsonValue] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(_JSON.validate_json(request.content))
        return httpx2.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "operation": {
                        "type": "choice",
                        "choice": "wait",
                        "probabilities": {
                            "click": 0.1,
                            "wait": 0.8,
                            "done": 0.05,
                            "blocked": 0.05,
                        },
                        "confidence": 0.8,
                    },
                    "click_target": "unselected malformed head",
                    "wait_target": {
                        "type": "choice",
                        "choice": "wait-1",
                        "probabilities": {"wait-1": 1.0},
                        "confidence": 0.9,
                    },
                },
                "usage": {"input_tokens": 45, "output_tokens": 14},
            },
        )

    context, _ = call_context()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler)
    ) as http_client:
        provider = JevDecisionProvider(
            ModelClient(
                ModelClientConfig(
                    provider="jev",
                    endpoint="https://api.typesafe.ai/v1/systemone",
                    api_key="test-key",
                    retry_statuses=frozenset({429, 529}),
                ),
                http_client=http_client,
            ),
            model="jev-requested",
        )

        # When
        decision = await provider.decide(payload, context=context)

    # Then
    assert decision.operation is ActionType.WAIT
    assert decision.candidate_id == "wait-1"
    assert decision.target_confidence == 0.9
    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request, dict)
    questions = request["questions"]
    assert isinstance(questions, dict)
    assert "wait_target" in questions
