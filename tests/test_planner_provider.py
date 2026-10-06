from datetime import UTC, datetime
import time

import pytest
from pydantic import JsonValue

from jevpilot.model_client import StructuredModelResult
from jevpilot.observer.models import Observation, OmittedCounts
from jevpilot.planner import LLMPlannerProvider, PlannerInput
from jevpilot.planner.models import BudgetsRemaining
from jevpilot.telemetry.models import CallContext


class FakeStructuredClient:
    def __init__(self) -> None:
        self.payload: JsonValue | None = None
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
        _ = (instructions, request_model, context)
        self.payload = payload
        self.schema = schema
        return StructuredModelResult(
            data={
                "plan_id": "plan-1",
                "parent_plan_id": None,
                "subgoals": [
                    {
                        "subgoal_id": "subgoal-1",
                        "description": "Open the Alpha item",
                        "progress_hint": {
                            "kind": "url_path",
                            "name": None,
                            "expected": "/items/alpha",
                        },
                    }
                ],
            },
            call_id="call-1",
            response_model="codex-response-model",
        )


def planner_input(*, previous_plan_id: str | None = None) -> PlannerInput:
    return PlannerInput(
        goal="Open Alpha",
        observation=Observation(
            snapshot_id="snapshot-1",
            document_id="document-1",
            navigation_id="navigation-1",
            url="https://example.test/items?token=synthetic-secret",
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
        ),
        history=(),
        previous_plan_id=previous_plan_id,
        budgets_remaining=BudgetsRemaining(
            steps=3,
            mutations=1,
            replans=1,
            model_attempts=2,
            stale_observations=1,
            waits=0,
            wait_ms=1000,
            wall_time_ms=5000,
        ),
    )


@pytest.mark.asyncio
async def test_subscription_planner_stamps_plan_identity_and_parses_progress_hint() -> None:
    client = FakeStructuredClient()
    provider = LLMPlannerProvider(client, model="codex-requested")
    context = CallContext(
        run_id="run-1",
        logical_call_id="logical-1",
        role="planner",
        purpose="initial",
        deadline=time.monotonic() + 10,
        max_attempts=1,
        reserve_attempt=lambda: None,
        record_call=lambda event: None,
    )

    result = await provider.plan(
        planner_input(previous_plan_id="trusted-parent"),
        context=context,
    )

    assert result.plan_id != "plan-1"
    assert result.parent_plan_id == "trusted-parent"
    assert result.subgoals[0].progress_hint is not None
    assert result.subgoals[0].progress_hint.kind == "url_path"
    assert client.payload is not None
    assert isinstance(client.payload, dict)
    assert "evaluator" not in client.payload
    observation = client.payload["observation"]
    assert isinstance(observation, dict)
    assert observation["url"] == "https://example.test/items"
    assert client.schema is not None
    assert isinstance(client.schema, dict)
    properties = client.schema["properties"]
    assert isinstance(properties, dict)
    assert "plan_id" not in properties
    assert "parent_plan_id" not in properties
    subgoals = properties["subgoals"]
    assert isinstance(subgoals, dict)
    items = subgoals["items"]
    assert isinstance(items, dict)
    item_properties = items["properties"]
    assert isinstance(item_properties, dict)
    assert "progress_hint" in item_properties
