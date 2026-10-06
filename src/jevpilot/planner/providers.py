"""Subscription-authenticated planning provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, final
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.model_client import ProviderError, StructuredModelClient
from jevpilot.privacy import PrivacyPolicy, redact_json

from .models import Plan, PlannerInput, Subgoal

if TYPE_CHECKING:
    from jevpilot.telemetry.models import CallContext

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_SUBGOALS: TypeAdapter[tuple[Subgoal, ...]] = TypeAdapter(tuple[Subgoal, ...])
_INSTRUCTIONS = (
    "Create a short browser plan. Return only the requested JSON. "
    "Subgoals describe observable progress and must not contain selectors, code, "
    "hidden answers, or final-task evaluator criteria."
)
_PLAN_SCHEMA: JsonValue = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "subgoal_id": {"type": "string"},
                    "description": {"type": "string"},
                    "progress_hint": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": [
                                            "url_path",
                                            "visible",
                                            "value",
                                            "checked",
                                        ],
                                    },
                                    "name": {"type": ["string", "null"]},
                                    "expected": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "boolean"},
                                        ]
                                    },
                                },
                                "required": ["kind", "name", "expected"],
                                "additionalProperties": False,
                            },
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["subgoal_id", "description", "progress_hint"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}


class PlannerProvider(Protocol):
    async def plan(
        self,
        payload: PlannerInput,
        *,
        context: CallContext,
    ) -> Plan: ...


@final
class LLMPlannerProvider:
    """Generate first-slice plans through a structured model client."""

    def __init__(
        self,
        client: StructuredModelClient,
        *,
        model: str,
        privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self._privacy_policy = privacy_policy or PrivacyPolicy()

    async def plan(
        self,
        payload: PlannerInput,
        *,
        context: CallContext,
    ) -> Plan:
        projection: JsonValue = _JSON.validate_python(
            {
                "goal": payload.goal,
                "observation": {
                    "snapshot_id": payload.observation.snapshot_id,
                    "url": payload.observation.url,
                    "title": payload.observation.title,
                    "text": payload.observation.text,
                    "controls": [
                        control.model_dump(mode="json")
                        for control in payload.observation.controls
                    ],
                    "omitted_counts": payload.observation.omitted_counts.model_dump(
                        mode="json"
                    ),
                },
                "history": [
                    entry.model_dump(mode="json") for entry in payload.history
                ],
                "previous_plan_id": payload.previous_plan_id,
                "failure": (
                    payload.failure.model_dump(mode="json")
                    if payload.failure is not None
                    else None
                ),
                "budgets_remaining": payload.budgets_remaining.model_dump(mode="json"),
            }
        )
        result = await self._client.generate_json(
            instructions=_INSTRUCTIONS,
            payload=redact_json(projection, policy=self._privacy_policy),
            schema=_PLAN_SCHEMA,
            request_model=self.model,
            context=context,
        )
        if not isinstance(result.data, dict):
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription planner returned an invalid plan",
            )
        try:
            subgoals = _SUBGOALS.validate_python(result.data.get("subgoals"))
        except ValidationError:
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription planner returned invalid subgoals",
            ) from None
        return Plan(
            plan_id=uuid4().hex,
            parent_plan_id=payload.previous_plan_id,
            subgoals=subgoals,
        )
