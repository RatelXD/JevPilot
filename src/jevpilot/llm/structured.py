from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import (
    CallOutcome,
    CallScope,
    LLMCallRecord,
    LLMRequest,
    LLMResult,
    LLMUsage,
    ModelRef,
    StructuredOutputMode,
)

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n?\s*```\s*$", re.DOTALL)
_MAX_REPORTED_ERRORS = 8
_MAX_ERROR_CHARS = 200
_DEADLINE_MESSAGE = "model call deadline passed before the request was sent"
_INVALID_OUTPUT_MESSAGE = (
    "model output failed schema validation after one repair attempt"
)


class StructuredOutputError(ValueError):
    errors: tuple[str, ...]

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass(frozen=True, slots=True)
class StructuredPrompt:
    """One transmission a provider adapter must send.

    ``json_schema`` is set for structured output so ``native`` providers can
    pass it to their API; ``prompted`` providers already have it in ``system``.
    """

    call_id: str
    attempt: int
    system: str
    user_text: str
    images: list[bytes]
    json_schema: JsonValue | None
    repair_errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class RawCompletion:
    text: str
    resolved_model: str | None
    usage: LLMUsage | None


SendAttempt = Callable[[StructuredPrompt], Awaitable[RawCompletion]]


def output_schema(output_type: type[object]) -> JsonValue:
    return cast(JsonValue, TypeAdapter(output_type).json_schema())


def parse_structured[T](text: str, output_type: type[T]) -> T:
    if output_type is str:
        return cast(T, text)
    fenced = _FENCE.match(text)
    body = fenced.group(1) if fenced else text
    try:
        return TypeAdapter(output_type).validate_json(body)
    except ValidationError as exc:
        raise StructuredOutputError(_summarize(exc)) from None


def _summarize(exc: ValidationError) -> tuple[str, ...]:
    # include_input=False keeps raw model output out of the repair prompt and logs.
    details = exc.errors(include_input=False, include_url=False, include_context=False)
    lines: list[str] = []
    for detail in details[:_MAX_REPORTED_ERRORS]:
        location = ".".join(str(part) for part in detail["loc"]) or "$"
        lines.append(f"{location}: {detail['msg']}"[:_MAX_ERROR_CHARS])
    if len(details) > _MAX_REPORTED_ERRORS:
        lines.append(f"... {len(details) - _MAX_REPORTED_ERRORS} more errors")
    return tuple(lines)


def prompted_system(system: str, schema: JsonValue) -> str:
    rendered = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    return (
        f"{system}\n\n"
        "Respond with exactly one JSON value that validates against the JSON "
        "Schema below. Do not add prose or code fences.\n"
        f"JSON Schema:\n{rendered}"
    )


def repair_instruction(errors: tuple[str, ...]) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        "Your previous response did not validate against the JSON Schema.\n"
        f"Validation errors:\n{listed}\n"
        "Respond again with only the corrected JSON value."
    )


def render_user_payload(payload: JsonValue) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


async def generate_structured[T](
    request: LLMRequest[T],
    scope: CallScope,
    *,
    ref: ModelRef,
    mode: StructuredOutputMode,
    send: SendAttempt,
) -> LLMResult[T]:
    """Send ``request`` through ``send`` with at most one validation repair.

    Every transmission is reported to ``scope.record_call``; the returned
    usage sums all attempts and stays ``None`` if any attempt lacked usage.
    No other provider is ever tried.
    """
    schema = None if request.output_type is str else output_schema(request.output_type)
    system = (
        prompted_system(request.system, schema)
        if schema is not None and mode == "prompted"
        else request.system
    )
    base_user_text = render_user_payload(request.user_payload)
    max_attempts = 1 if schema is None else 2
    call_ids: list[str] = []
    total_usage: LLMUsage | None = None
    repair_errors: tuple[str, ...] = ()

    for attempt in range(1, max_attempts + 1):
        if time.monotonic() >= scope.deadline:
            raise LLMError("timeout", retryable=False, message=_DEADLINE_MESSAGE)
        call_id = uuid.uuid4().hex
        call_ids.append(call_id)
        user_text = (
            f"{base_user_text}\n\n{repair_instruction(repair_errors)}"
            if repair_errors
            else base_user_text
        )
        prompt = StructuredPrompt(
            call_id=call_id,
            attempt=attempt,
            system=system,
            user_text=user_text,
            images=list(request.images),
            json_schema=schema,
            repair_errors=repair_errors,
        )
        started = time.monotonic()

        def record(
            outcome: CallOutcome,
            usage: LLMUsage | None,
            resolved_model: str | None,
            *,
            call_id: str = call_id,
            attempt: int = attempt,
            started: float = started,
        ) -> None:
            scope.record_call(
                LLMCallRecord(
                    call_id=call_id,
                    purpose=request.purpose,
                    provider=ref.provider,
                    auth_mode=ref.auth_mode,
                    model_id=ref.model_id,
                    attempt=attempt,
                    resolved_model=resolved_model,
                    usage=usage,
                    started_monotonic=started,
                    ended_monotonic=time.monotonic(),
                    outcome=outcome,
                )
            )

        try:
            raw = await send(prompt)
        except LLMError as exc:
            record(exc.kind, None, None)
            raise
        except asyncio.CancelledError:
            record("cancelled", None, None)
            raise

        if attempt == 1:
            total_usage = raw.usage
        elif total_usage is not None and raw.usage is not None:
            total_usage = total_usage.plus(raw.usage)
        else:
            total_usage = None

        try:
            value = parse_structured(raw.text, request.output_type)
        except StructuredOutputError as exc:
            record("invalid_output", raw.usage, raw.resolved_model)
            repair_errors = exc.errors
            continue
        record("ok", raw.usage, raw.resolved_model)
        return LLMResult(
            value=value,
            resolved_model=raw.resolved_model,
            usage=total_usage,
            attempts=attempt,
            call_ids=call_ids,
        )

    raise LLMError("invalid_output", retryable=False, message=_INVALID_OUTPUT_MESSAGE)
