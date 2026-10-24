"""Jev choice client: picks the next browser operation and its target element.

Jev only answers closed-set ``choice`` questions; it never generates free text.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
import math
import os
import sys
import time
from typing import Final

import httpx2
from pydantic import JsonValue

from jevpilot.browser.port import ActionKind, ElementRef, Observation
from jevpilot.ledger.legacy_bridge import legacy_call_context
from jevpilot.ledger.models import CallRecord
from jevpilot.ledger.recorder import CallScope, RunLedger
from jevpilot.llm.credentials import get_api_key
from jevpilot.llm.errors import LLMErrorKind
from jevpilot.llm.pricing import PRICES
from jevpilot.model_client import ModelClient, ModelClientConfig, ProviderError, ResponseMetadata
from jevpilot.telemetry.models import CallContext, CallEvent, CallUsage

JEV_ENDPOINT: Final = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL: Final = "jev-latest"
MAX_ELEMENTS: Final = 200
PAGE_TEXT_LIMIT: Final = 4000
RECENT_ACTION_LIMIT: Final = 5
MAX_ATTEMPTS: Final = 4
_RETRY_STATUSES: Final = frozenset({429, 529})
_PROBABILITY_SUM_TOLERANCE: Final = 1e-6
_TEXT_TARGET_ROLES: Final = frozenset({"textbox", "searchbox", "combobox"})
_NON_TEXT_INPUT_TYPES: Final = frozenset({
    "button", "submit", "reset", "checkbox", "radio", "image", "file", "hidden", "range", "color",
})
_OPERATIONS: Final[tuple[ActionKind, ...]] = (
    "click", "type_text", "select_option", "scroll_down", "scroll_up", "go_back", "wait",
)
_TARGET_OPERATIONS: Final = frozenset({"click", "type_text", "select_option"})


class JevUnavailable(Exception):
    """Jev could not give a usable answer; ``kind`` is a safe, key-free reason."""

    kind: LLMErrorKind

    def __init__(self, kind: LLMErrorKind) -> None:
        self.kind = kind
        super().__init__(f"Jev unavailable (kind={kind})")


@dataclass(frozen=True, slots=True)
class SelectorState:
    goal: str
    brief: str
    subgoal: str
    observation: Observation
    recent_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class JevChoice:
    operation: str
    op_confidence: float
    op_margin: float
    target_index: int | None
    target_confidence: float | None
    target_margin: float | None
    candidate_count: int
    truncated: bool
    usage: CallUsage | None
    latency_s: float


@dataclass(frozen=True, slots=True)
class OptionChoice:
    value: str
    confidence: float
    margin: float
    usage: CallUsage | None
    latency_s: float


@dataclass(frozen=True, slots=True)
class _Choice:
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def margin(self) -> float:
        ranked = sorted(self.probabilities.values(), reverse=True)
        return ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)


@dataclass(frozen=True, slots=True)
class _Answers:
    answers: dict[str, JsonValue]
    usage: CallUsage | None


type _Interpret[T] = Callable[[_Answers, ResponseMetadata], T]


def _invalid(metadata: ResponseMetadata) -> ProviderError:
    return ProviderError(
        kind="invalid_response", call_id=metadata.call_id, retryable=False,
        message="model provider returned an invalid response",
        status_code=metadata.status_code,
    )


def _probability(value: JsonValue | None) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and 0.0 <= value <= 1.0
    )


def _parse_choice(
    answer: JsonValue | None, expected: frozenset[str], metadata: ResponseMetadata,
) -> _Choice:
    """Same acceptance rules as v1: allowed value, probabilities sum to 1, argmax."""
    if not isinstance(answer, dict) or set(answer) != {
        "type", "choice", "probabilities", "confidence",
    }:
        raise _invalid(metadata)
    choice = answer["choice"]
    probabilities = answer["probabilities"]
    confidence = answer["confidence"]
    if (
        answer["type"] != "choice" or not isinstance(choice, str) or choice not in expected
        or not isinstance(probabilities, dict) or frozenset(probabilities) != expected
        or not _probability(confidence)
    ):
        raise _invalid(metadata)
    parsed: dict[str, float] = {}
    for key, value in probabilities.items():
        if not _probability(value):
            raise _invalid(metadata)
        parsed[key] = float(value)  # pyright: ignore[reportArgumentType]
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_SUM_TOLERANCE):
        raise _invalid(metadata)
    top = max(parsed.values())
    if parsed[choice] < top and not math.isclose(parsed[choice], top, rel_tol=0.0, abs_tol=1e-12):
        raise _invalid(metadata)
    return _Choice(choice, parsed, float(confidence))  # pyright: ignore[reportArgumentType]


def _parse_answers(body: JsonValue, metadata: ResponseMetadata) -> _Answers:
    if not isinstance(body, dict) or metadata.response_model is None:
        raise _invalid(metadata)
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise _invalid(metadata)
    return _Answers(answers, metadata.usage)


def _text_input(element: ElementRef) -> bool:
    if element.tag == "textarea" or element.role in _TEXT_TARGET_ROLES:
        return True
    return element.tag == "input" and element.attributes.get("type", "text") not in _NON_TEXT_INPUT_TYPES


def _selectable(element: ElementRef) -> bool:
    return element.tag == "select" or bool(element.options)


def _candidates(elements: list[ElementRef], operation: str) -> list[ElementRef]:
    if operation == "type_text":
        return [e for e in elements if _text_input(e)]
    if operation == "select_option":
        return [e for e in elements if _selectable(e)]
    return list(elements)


def select_elements(elements: list[ElementRef]) -> tuple[list[ElementRef], bool]:
    """Keep viewport elements first, then the rest, all in document order, up to 200."""
    if len(elements) <= MAX_ELEMENTS:
        return list(elements), False
    ordered = [e for e in elements if e.in_viewport] + [e for e in elements if not e.in_viewport]
    kept = {e.index for e in ordered[:MAX_ELEMENTS]}
    return [e for e in elements if e.index in kept], True


def _element_line(element: ElementRef) -> str:
    text = " ".join(element.text.split())
    attrs = " ".join(f"{k}={v}" for k, v in element.attributes.items())
    return f"[{element.index}] {element.tag} {element.role or '-'} {text} {attrs}".rstrip()


def _state_json(state: SelectorState, elements: list[ElementRef]) -> JsonValue:
    observation = state.observation
    return {
        "goal": state.goal,
        "brief": state.brief,
        "subgoal": state.subgoal,
        "url": observation.url,
        "title": observation.title,
        "page_text": observation.page_text[:PAGE_TEXT_LIMIT],
        "elements": [_element_line(e) for e in elements],
        "recent_actions": list(state.recent_actions[-RECENT_ACTION_LIMIT:]),
    }


def _choice_question(instructions: str, criteria: dict[str, JsonValue]) -> JsonValue:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _unavailable(error: ProviderError) -> JevUnavailable:
    if error.kind == "authentication":
        return JevUnavailable("auth_invalid")
    if error.kind == "insufficient_credit":
        return JevUnavailable("quota_exhausted")
    if error.kind == "transport_timeout":
        return JevUnavailable("timeout")
    if error.kind == "invalid_response":
        return JevUnavailable("invalid_output")
    if error.kind == "rate_limited" or error.status_code in _RETRY_STATUSES:
        return JevUnavailable("rate_limited")
    if error.kind == "provider_unavailable" or error.status_code is not None:
        return JevUnavailable("provider_unavailable")
    return JevUnavailable("configuration")


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


_PRICE_ANCHOR: Final = ("jev", "jev-1.14.0")


def _estimate_cost(model: str, usage: CallUsage | None) -> Decimal | None:
    """Input tokens only (output is free); unknown model or usage stays unknown."""
    price = PRICES.get(("jev", model))
    if price is None or usage is None or usage.input_tokens is None:
        return None
    return Decimal(usage.input_tokens) * price.input_per_m / Decimal(1_000_000)


def _client_config(api_key: str, endpoint: str) -> ModelClientConfig:
    anchor = PRICES[_PRICE_ANCHOR]
    return ModelClientConfig(
        provider="jev", endpoint=endpoint, api_key=api_key,
        retry_statuses=_RETRY_STATUSES,
        cost_estimator=_estimate_cost,
        cost_price_source=(
            f"{anchor.source_url}: ${anchor.input_per_m} per million input tokens; output free"
        ),
        cost_price_checked_on=anchor.checked_on,
    )


def format_cost_line(record: CallRecord) -> str:
    """One real-time line per transport attempt."""
    cost = f"${record.api_estimate_usd:.6f}" if record.api_estimate_usd is not None else "unknown"
    tokens = record.usage.input_tokens if record.usage.input_tokens is not None else "?"
    return (
        f"jev {record.purpose} {record.outcome} in_tokens={tokens} "
        f"est_cost={cost} latency={record.latency_s:.2f}s"
    )


@dataclass(frozen=True, slots=True)
class _AttemptScope(CallScope):
    """Numbers each transport attempt's record 1, 2, ... within one logical call."""

    seen: list[int] = field(default_factory=list)

    def record_call(self, record: CallRecord) -> None:
        self.seen.append(1)
        self.ledger.record_call(record.model_copy(update={"attempts": len(self.seen)}))


class JevSelector:
    """Operation/target/option selection through Jev ``choice`` questions."""

    def __init__(
        self, *, api_key: str | None = None, endpoint: str | None = None,
        model: str | None = None, http_client: httpx2.AsyncClient | None = None,
        on_call: Callable[[CallRecord], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self.endpoint = endpoint or _setting("JEVPILOT_JEV_ENDPOINT", JEV_ENDPOINT)
        self.model = model or _setting("JEVPILOT_JEV_MODEL", JEV_MODEL)
        self._http_client = http_client
        self.on_call = on_call

    def _key(self) -> str:
        key = self._api_key or get_api_key("jev")
        if not key:
            raise JevUnavailable("auth_missing")
        return key

    def _context(self, scope: CallScope) -> CallContext:
        numbered = _AttemptScope(
            scope.ledger, scope.parent_span_id, scope.deadline, scope.step_index,
            scope.reserve_attempt,
        )
        context = legacy_call_context(numbered, role="selector", max_attempts=MAX_ATTEMPTS)
        if self.on_call is None:
            return context
        record, on_call = context.record_call, self.on_call

        def record_and_report(event: CallEvent) -> None:
            record(event)
            on_call(scope.ledger.calls[-1])

        return replace(context, record_call=record_and_report)

    async def _ask[T](
        self, scope: CallScope, state: JsonValue, questions: dict[str, JsonValue],
        interpret: _Interpret[T],
    ) -> tuple[T, CallUsage | None, float]:
        usage: list[CallUsage | None] = []

        def parse(body: JsonValue, metadata: ResponseMetadata) -> T:
            answers = _parse_answers(body, metadata)
            usage.append(answers.usage)
            return interpret(answers, metadata)

        client = ModelClient[T](
            _client_config(self._key(), self.endpoint), http_client=self._http_client,
        )
        started = time.monotonic()
        try:
            parsed = await client.request_json(
                request_model=self.model,
                payload={"state": state, "model": self.model, "questions": questions},
                context=self._context(scope), parser=parse,
            )
        except ProviderError as error:
            raise _unavailable(error) from None
        finally:
            await client.aclose()
        return parsed, usage[-1], time.monotonic() - started

    async def choose(self, state: SelectorState, *, scope: CallScope) -> JevChoice:
        elements, truncated = select_elements(state.observation.elements)
        by_operation = {op: _candidates(elements, op) for op in _OPERATIONS}
        operations = [
            op for op in _OPERATIONS if op not in _TARGET_OPERATIONS or by_operation[op]
        ] + ["done"]
        questions: dict[str, JsonValue] = {
            "operation": _choice_question(
                "Choose the next allowed browser operation, or done if the goal is met.",
                {op: None for op in operations},
            ),
        }
        for op in operations:
            if op in _TARGET_OPERATIONS:
                questions[f"{op}_target"] = _choice_question(
                    f"If {op} is selected, choose the element that best advances the subgoal.",
                    {str(e.index): _element_line(e) for e in by_operation[op]},
                )

        def interpret(
            result: _Answers, metadata: ResponseMetadata,
        ) -> tuple[_Choice, _Choice | None]:
            operation = _parse_choice(
                result.answers.get("operation"), frozenset(operations), metadata,
            )
            if operation.choice not in _TARGET_OPERATIONS:
                return operation, None
            expected = frozenset(str(e.index) for e in by_operation[operation.choice])
            return operation, _parse_choice(
                result.answers.get(f"{operation.choice}_target"), expected, metadata,
            )

        (operation, target), usage, latency = await self._ask(
            scope, _state_json(state, elements), questions, interpret,
        )
        return JevChoice(
            operation=operation.choice, op_confidence=operation.confidence,
            op_margin=operation.margin,
            target_index=int(target.choice) if target else None,
            target_confidence=target.confidence if target else None,
            target_margin=target.margin if target else None,
            candidate_count=len(elements), truncated=truncated,
            usage=usage, latency_s=latency,
        )

    async def choose_option(
        self, state: SelectorState, target: ElementRef, *, scope: CallScope,
    ) -> OptionChoice:
        if not target.options:
            raise JevUnavailable("configuration")
        elements, _ = select_elements(state.observation.elements)
        questions: dict[str, JsonValue] = {
            "option": _choice_question(
                "Choose the option of the selected element that best advances the subgoal.",
                {str(i): option.label for i, option in enumerate(target.options)},
            ),
        }
        option, usage, latency = await self._ask(
            scope, _state_json(state, elements), questions,
            lambda result, metadata: _parse_choice(
                result.answers.get("option"),
                frozenset(str(i) for i in range(len(target.options))), metadata,
            ),
        )
        return OptionChoice(
            value=target.options[int(option.choice)].value, confidence=option.confidence,
            margin=option.margin, usage=usage, latency_s=latency,
        )


async def validate_key(
    key: str, *, endpoint: str | None = None, model: str | None = None,
    http_client: httpx2.AsyncClient | None = None,
) -> None:
    """One minimal two-option choice request: 2xx passes, otherwise JevUnavailable."""
    target = endpoint or _setting("JEVPILOT_JEV_ENDPOINT", JEV_ENDPOINT)
    request_model = model or _setting("JEVPILOT_JEV_MODEL", JEV_MODEL)
    context = CallContext(
        run_id="key-check", logical_call_id="key-check", role="selector", purpose="normal",
        deadline=time.monotonic() + 30.0, max_attempts=1,
        reserve_attempt=lambda: None, record_call=lambda _event: None,
    )
    client = ModelClient[None](_client_config(key, target), http_client=http_client)
    try:
        _ = await client.request_json(
            request_model=request_model,
            payload={
                "state": {"goal": "key check"}, "model": request_model,
                "questions": {"check": _choice_question("Choose a.", {"a": None, "b": None})},
            },
            context=context, parser=lambda _body, _metadata: None,
        )
    except ProviderError as error:
        # The JSON transport rejects empty/non-JSON bodies after accepting HTTP 2xx.
        # Key validation depends only on that authenticated status, not the body.
        if (
            error.kind == "invalid_response" and error.status_code is not None
            and 200 <= error.status_code < 300
        ):
            return
        raise _unavailable(error) from None
    finally:
        await client.aclose()


def _smoke_state() -> SelectorState:
    def element(index: int, tag: str, role: str | None, text: str, **attrs: str) -> ElementRef:
        return ElementRef(
            snapshot_id="smoke", index=index, backend_node_id=index, tag=tag, role=role,
            text=text, attributes=attrs, in_viewport=True,
        )

    observation = Observation(
        snapshot_id="smoke", url="https://example.com/", title="Example Shop",
        tabs=["Example Shop"], page_text="Example Shop. Search for products.",
        fingerprint="smoke", observed_at_monotonic=0.0,
        elements=[
            element(0, "a", "link", "Home", href="/"),
            element(1, "input", "searchbox", "", placeholder="Search products", type="search"),
            element(2, "button", "button", "Search"),
        ],
    )
    return SelectorState(
        goal="Search for blue shoes", brief="Find blue shoes on the shop site.",
        subgoal="Type 'blue shoes' into the search box", observation=observation,
    )


async def _smoke() -> int:
    ledger = RunLedger()
    scope = ledger.scope(deadline=ledger.clock() + 60)
    selector = JevSelector(on_call=lambda record: print(format_cost_line(record)))
    try:
        choice = await selector.choose(_smoke_state(), scope=scope)
    except JevUnavailable as error:
        print(f"jev smoke failed: {error.kind}")
        return 2
    print(
        f"operation={choice.operation} op_confidence={choice.op_confidence:.3f} "
        f"op_margin={choice.op_margin:.3f} target_index={choice.target_index} "
        f"target_confidence={choice.target_confidence} candidates={choice.candidate_count} "
        f"truncated={choice.truncated} latency={choice.latency_s:.2f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jevpilot.engine.jev")
    _ = parser.add_argument("--smoke", action="store_true", help="one real Jev call")
    args = parser.parse_args(argv)
    if not args.smoke:
        parser.error("only --smoke is supported")
    return asyncio.run(_smoke())


if __name__ == "__main__":
    sys.exit(main())
