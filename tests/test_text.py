from datetime import UTC, datetime
import time
from typing import final

import pytest
from pydantic import JsonValue

from jevpilot.model_client import ProviderError, StructuredModelResult
from jevpilot.observer.models import (
    ActionCandidate,
    Observation,
    OmittedCounts,
)
from jevpilot.planner.models import Subgoal
from jevpilot.task import ActionType
from jevpilot.telemetry.models import CallContext
from jevpilot.text import (
    LLMTextValueProvider,
    TextConstraints,
    TextInput,
)


@final
class ScriptedTextClient:
    data: JsonValue

    def __init__(self, data: JsonValue) -> None:
        self.data = data
        self.output_schema: JsonValue | None = None

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
        self.output_schema = schema
        return StructuredModelResult(
            data=self.data, call_id="text-call", response_model=None
        )


def text_input(*, maximum: int = 16, allow_empty: bool = False) -> TextInput:
    return TextInput(
        goal="Search the catalog for Aurora",
        current_subgoal=Subgoal(
            subgoal_id="search",
            description="Enter the item name",
        ),
        observation=Observation(
            snapshot_id="snapshot-1",
            document_id="document-1",
            navigation_id="navigation-1",
            url="https://example.test/search",
            title="Catalog",
            text="Search the catalog",
            controls=(),
            fingerprint="fingerprint-1",
            observed_at=datetime.now(UTC),
            omitted_counts=OmittedCounts(
                hidden=0, disabled=0, readonly=0, unnamed=0, over_limit=0
            ),
        ),
        target=ActionCandidate(
            candidate_id="candidate-1",
            snapshot_id="snapshot-1",
            operation=ActionType.TYPE_TEXT,
            target_ref="target-1",
            role="searchbox",
            name="Search catalog",
            value="",
            enabled=True,
            readonly=False,
            max_length=maximum,
        ),
        history=(),
        constraints=TextConstraints(max_length=32, allow_empty=allow_empty),
    )


def context() -> CallContext:
    return CallContext(
        run_id="run-1",
        logical_call_id="text-1",
        role="text",
        purpose="normal",
        deadline=time.monotonic() + 5,
        max_attempts=1,
        reserve_attempt=lambda: None,
        record_call=lambda event: None,
    )


@pytest.mark.asyncio
async def test_text_provider_generates_only_bounded_field_value() -> None:
    # Given
    client = ScriptedTextClient({"text": "Aurora"})
    provider = LLMTextValueProvider(client, model="codex-model")

    # When
    result = await provider.generate(text_input(maximum=8), context=context())

    # Then
    assert result.text == "Aurora"
    assert client.output_schema == {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1, "maxLength": 8}
        },
        "required": ["text"],
        "additionalProperties": False,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "a" * 9])
async def test_text_provider_rejects_empty_or_oversized_value(value: str) -> None:
    # Given
    provider = LLMTextValueProvider(
        ScriptedTextClient({"text": value}), model="codex-model"
    )

    # When / Then
    with pytest.raises(ProviderError, match="field constraints"):
        _ = await provider.generate(text_input(maximum=8), context=context())


@pytest.mark.asyncio
async def test_text_provider_rejects_extra_output_fields() -> None:
    # Given
    provider = LLMTextValueProvider(
        ScriptedTextClient({"text": "Aurora", "selector": "#query"}),
        model="codex-model",
    )

    # When / Then
    with pytest.raises(ProviderError, match="invalid result"):
        _ = await provider.generate(text_input(), context=context())


def test_text_input_rejects_candidate_from_another_snapshot() -> None:
    # Given
    payload = text_input().model_dump()
    payload["target"]["snapshot_id"] = "stale-snapshot"

    # When / Then
    with pytest.raises(ValueError, match="current observation"):
        _ = TextInput.model_validate(payload)
