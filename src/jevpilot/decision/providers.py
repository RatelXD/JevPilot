"""TypeSafe Jev decision provider."""
# noqa: SIZE_OK — scope restricts both first-slice decision adapters to this module.

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Protocol, TypeGuard, final
from uuid import uuid4

import httpx2
from pydantic import JsonValue, TypeAdapter

from jevpilot.model_client import (
    ModelClient,
    ModelClientConfig,
    ProviderError,
    ResponseMetadata,
    StructuredModelClient,
    StructuredModelResult,
)
from jevpilot.jev_pricing import (
    estimate_jev_cost_usd,
    jev_price_checked_on,
    jev_price_source,
)
from jevpilot.observer.models import ActionCandidate
from jevpilot.privacy import PrivacyPolicy, redact_json
from jevpilot.task import ActionType

from .models import DecisionInput, DecisionOutput

if TYPE_CHECKING:
    from jevpilot.telemetry.models import CallContext

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_JEV_RETRY_STATUSES = frozenset({429, 529})
_FIRST_SLICE_OPERATIONS = frozenset(
    {
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SELECT,
        ActionType.SCROLL_UP,
        ActionType.SCROLL_DOWN,
        ActionType.WAIT,
        ActionType.DONE,
        ActionType.BLOCKED,
    }
)
_TARGET_OPERATIONS = frozenset(
    {ActionType.CLICK, ActionType.TYPE_TEXT, ActionType.SELECT, ActionType.WAIT}
)
_DIRECTION_OPERATIONS = frozenset({ActionType.SCROLL_UP, ActionType.SCROLL_DOWN})
_PROBABILITY_SUM_TOLERANCE = 1e-6
_MAX_CHOICE_OPTIONS = 255


class DecisionProvider(Protocol):
    async def decide(
        self,
        payload: DecisionInput,
        *,
        context: CallContext,
    ) -> DecisionOutput: ...


@final
class LLMDecisionProvider:
    """Select first-slice actions through a structured model client."""

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

    async def decide(
        self,
        payload: DecisionInput,
        *,
        context: CallContext,
    ) -> DecisionOutput:
        operations, target_candidates = _first_slice_choices(payload)
        candidate_ids = [candidate.candidate_id for candidate in target_candidates]
        schema = _JSON.validate_python(
            {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [operation.value for operation in operations],
                    },
                    "candidate_id": {
                        "anyOf": [
                            {"type": "string", "enum": candidate_ids},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["operation", "candidate_id"],
                "additionalProperties": False,
            }
        )
        result = await self._client.generate_json(
            instructions=(
                "Choose the next browser operation. CLICK, TYPE_TEXT, SELECT, and WAIT require a "
                "candidate_id for that operation. DONE and BLOCKED require null. "
                "Return only the requested JSON."
            ),
            payload=redact_json(
                _decision_projection(payload, target_candidates),
                policy=self._privacy_policy,
            ),
            schema=schema,
            request_model=self.model,
            context=context,
        )
        return _parse_llm_decision(
            result=result,
            payload=payload,
            operations=operations,
            target_candidates=target_candidates,
        )


@final
class JevDecisionProvider:
    """Select CLICK, DONE, or BLOCKED using one SystemOne request."""

    def __init__(self, client: ModelClient[DecisionOutput], *, model: str) -> None:
        self._client = client
        self.model = model

    async def aclose(self) -> None:
        await self._client.aclose()

    @classmethod
    def from_env(
        cls,
        *,
        http_client: httpx2.AsyncClient | None = None,
        privacy_policy: PrivacyPolicy | None = None,
    ) -> "JevDecisionProvider":
        api_key = os.getenv("JEVPILOT_JEV_API_KEY", "")
        model = os.getenv("JEVPILOT_JEV_MODEL", "")
        endpoint = os.getenv("JEVPILOT_JEV_ENDPOINT", _JEV_ENDPOINT)
        if not api_key or not model or not endpoint:
            raise ProviderError(
                kind="configuration",
                call_id="configuration",
                retryable=False,
                message="Jev provider configuration is incomplete",
            )
        client = ModelClient[DecisionOutput](
            ModelClientConfig(
                provider="jev",
                endpoint=endpoint,
                api_key=api_key,
                retry_statuses=_JEV_RETRY_STATUSES,
                cost_estimator=estimate_jev_cost_usd,
                cost_price_source=jev_price_source(),
                cost_price_checked_on=jev_price_checked_on(),
            ),
            http_client=http_client,
            privacy_policy=privacy_policy,
        )
        return cls(client, model=model)

    async def decide(
        self,
        payload: DecisionInput,
        *,
        context: CallContext,
    ) -> DecisionOutput:
        try:
            operations, target_candidates = _first_slice_choices(payload)
        except ProviderError as error:
            self._client.raise_preflight_error(
                request_model=self.model,
                context=context,
                message=error.message,
            )
        if len(target_candidates) > _MAX_CHOICE_OPTIONS:
            self._client.raise_preflight_error(
                request_model=self.model,
                context=context,
                message="Jev target candidate count exceeds the Choice limit",
            )

        operation_options = {operation.value: None for operation in operations}
        questions = _JSON.validate_python(
            {
                "operation": {
                    "type": "choice",
                    "instructions": "Choose the next allowed browser operation.",
                    "criteria": operation_options,
                }
            }
        )
        if not isinstance(questions, dict):
            raise ProviderError(
                kind="configuration",
                call_id="configuration",
                retryable=False,
                message="Jev questions must be a JSON object",
            )
        for operation in operations:
            if operation not in _TARGET_OPERATIONS:
                continue
            candidates = tuple(
                candidate
                for candidate in target_candidates
                if candidate.operation is operation
            )
            questions[f"{operation.value}_target"] = {
                "type": "choice",
                "instructions": (
                    f"If {operation.value.upper()} is selected, choose the observed "
                    "candidate or bounded wait condition that best advances the subgoal."
                ),
                "criteria": {
                    candidate.candidate_id: {
                        "operation": candidate.operation.value,
                        "role": candidate.role,
                        "name": candidate.name,
                        "value": candidate.value,
                        "wait_condition": (
                            candidate.wait_condition.model_dump(mode="json")
                            if candidate.wait_condition is not None
                            else None
                        ),
                    }
                    for candidate in candidates
                },
            }

        request: JsonValue = {
            "state": _decision_projection(payload, target_candidates),
            "model": self.model,
            "questions": questions,
        }
        return await self._client.request_json(
            request_model=self.model,
            payload=request,
            context=context,
            parser=lambda body, metadata: _parse_jev_decision(
                body=body,
                metadata=metadata,
                payload=payload,
                operations=operations,
                target_candidates=target_candidates,
            ),
        )


def _first_slice_choices(
    payload: DecisionInput,
) -> tuple[tuple[ActionType, ...], tuple[ActionCandidate, ...]]:
    if set(payload.policy.allowed_operations) - _FIRST_SLICE_OPERATIONS:
        raise ProviderError(
            kind="configuration",
            call_id="configuration",
            retryable=False,
            message="decision includes an unsupported operation",
        )
    target_candidates = tuple(
        candidate
        for candidate in payload.candidates
        if candidate.operation in _TARGET_OPERATIONS
    )
    operations = tuple(
        operation
        for operation in payload.policy.allowed_operations
        if (
            operation not in _TARGET_OPERATIONS | _DIRECTION_OPERATIONS
            or any(
                candidate.operation is operation
                for candidate in payload.candidates
            )
        )
    )
    if not operations:
        raise ProviderError(
            kind="configuration",
            call_id="configuration",
            retryable=False,
            message="decision request has no selectable operations",
        )
    return operations, target_candidates


def _decision_projection(
    payload: DecisionInput,
    target_candidates: tuple[ActionCandidate, ...],
) -> JsonValue:
    return _JSON.validate_python(
        {
            "snapshot_id": payload.observation.snapshot_id,
            "goal": payload.goal,
            "subgoal": payload.current_subgoal.model_dump(mode="json"),
            "page": {
                "url": payload.observation.url,
                "title": payload.observation.title,
                "text": payload.observation.text,
                "scroll": {
                    "y": payload.observation.scroll_y,
                    "height": payload.observation.scroll_height,
                    "viewport_height": payload.observation.viewport_height,
                },
            },
            "elements": [
                {
                    "id": candidate.candidate_id,
                    "operation": candidate.operation.value,
                    "role": candidate.role,
                    "name": candidate.name,
                    "value": candidate.value,
                    "enabled": candidate.enabled,
                }
                for candidate in target_candidates
            ],
            "recent_history": [
                entry.model_dump(mode="json") for entry in payload.recent_history
            ],
        }
    )


def _parse_llm_decision(
    *,
    result: StructuredModelResult,
    payload: DecisionInput,
    operations: tuple[ActionType, ...],
    target_candidates: tuple[ActionCandidate, ...],
) -> DecisionOutput:
    if not isinstance(result.data, dict) or set(result.data) != {
        "operation",
        "candidate_id",
    }:
        raise ProviderError(
            kind="invalid_response",
            call_id=result.call_id,
            retryable=False,
            message="subscription model returned an invalid decision",
        )
    raw_operation = result.data.get("operation")
    raw_candidate = result.data.get("candidate_id")
    allowed = frozenset(operation.value for operation in operations)
    if not isinstance(raw_operation, str) or raw_operation not in allowed:
        raise ProviderError(
            kind="invalid_response",
            call_id=result.call_id,
            retryable=False,
            message="subscription model selected an invalid operation",
        )
    operation = ActionType(raw_operation)
    candidate_id = raw_candidate if isinstance(raw_candidate, str) else None
    if operation in _TARGET_OPERATIONS:
        selected = next(
            (
                candidate
                for candidate in target_candidates
                if candidate.candidate_id == candidate_id
                and candidate.operation is operation
            ),
            None,
        )
        if selected is None or selected.snapshot_id != payload.observation.snapshot_id:
            raise ProviderError(
                kind="invalid_response",
                call_id=result.call_id,
                retryable=False,
                message="subscription model selected an invalid action candidate",
            )
    elif raw_candidate is not None:
        raise ProviderError(
            kind="invalid_response",
            call_id=result.call_id,
            retryable=False,
            message="control operation must not include a candidate",
        )
    return DecisionOutput(
        decision_id=uuid4().hex,
        snapshot_id=payload.observation.snapshot_id,
        operation=operation,
        candidate_id=candidate_id,
        operation_confidence=None,
        target_confidence=None,
        operation_probabilities=None,
        target_probabilities=None,
        provider="openai_subscription",
        model=result.response_model,
        call_id=result.call_id,
    )


def _parse_jev_decision(
    *,
    body: JsonValue,
    metadata: ResponseMetadata,
    payload: DecisionInput,
    operations: tuple[ActionType, ...],
    target_candidates: tuple[ActionCandidate, ...],
) -> DecisionOutput:
    if not isinstance(body, dict) or metadata.response_model is None:
        raise _invalid_response(metadata)
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise _invalid_response(metadata)
    operation_choice, operation_probabilities, operation_confidence = _parse_choice(
        answer=answers.get("operation"),
        expected=frozenset(operation.value for operation in operations),
        metadata=metadata,
    )
    operation = ActionType(operation_choice)
    candidate_id: str | None = None
    target_probabilities: dict[str, float] | None = None
    target_confidence: float | None = None
    if operation in _TARGET_OPERATIONS:
        selected_candidates = tuple(
            candidate
            for candidate in target_candidates
            if candidate.operation is operation
        )
        expected_candidates = frozenset(
            candidate.candidate_id for candidate in selected_candidates
        )
        candidate_id, target_probabilities, target_confidence = _parse_choice(
            answer=answers.get(f"{operation.value}_target"),
            expected=expected_candidates,
            metadata=metadata,
        )
        selected = next(
            (
                candidate
                for candidate in selected_candidates
                if candidate.candidate_id == candidate_id
            ),
            None,
        )
        if selected is None or selected.snapshot_id != payload.observation.snapshot_id:
            raise _invalid_response(metadata)
    return DecisionOutput(
        decision_id=uuid4().hex,
        snapshot_id=payload.observation.snapshot_id,
        operation=operation,
        candidate_id=candidate_id,
        operation_confidence=operation_confidence,
        target_confidence=target_confidence,
        operation_probabilities={
            ActionType(key): value for key, value in operation_probabilities.items()
        },
        target_probabilities=target_probabilities,
        provider="jev",
        model=metadata.response_model,
        call_id=metadata.call_id,
    )


def _parse_choice(
    *,
    answer: JsonValue | None,
    expected: frozenset[str],
    metadata: ResponseMetadata,
) -> tuple[str, dict[str, float], float]:
    if not isinstance(answer, dict) or set(answer) != {
        "type",
        "choice",
        "probabilities",
        "confidence",
    }:
        raise _invalid_response(metadata)
    choice = answer.get("choice")
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if (
        answer.get("type") != "choice"
        or not isinstance(choice, str)
        or choice not in expected
        or not isinstance(probabilities, dict)
        or frozenset(probabilities) != expected
        or not _finite_probability(confidence)
    ):
        raise _invalid_response(metadata)
    parsed: dict[str, float] = {}
    for key, value in probabilities.items():
        if not _finite_probability(value):
            raise _invalid_response(metadata)
        parsed[key] = float(value)
    if not math.isclose(
        sum(parsed.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_SUM_TOLERANCE,
    ):
        raise _invalid_response(metadata)
    selected = parsed[choice]
    if selected < max(parsed.values()) and not math.isclose(
        selected,
        max(parsed.values()),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise _invalid_response(metadata)
    return choice, parsed, float(confidence)


def _finite_probability(
    value: JsonValue | None,
) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _invalid_response(metadata: ResponseMetadata) -> ProviderError:
    return ProviderError(
        kind="invalid_response",
        call_id=metadata.call_id,
        retryable=False,
        message="Jev returned an invalid decision envelope",
        status_code=metadata.status_code,
    )
