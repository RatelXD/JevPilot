"""Jev choice client: picks the next browser operation and its target element.

Jev only answers closed-set ``choice`` questions; it never generates free text.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
import math
import os
import sys
import time
from typing import Final
import unicodedata
from uuid import uuid4

import anyio
import httpx2
from pydantic import JsonValue, ValidationError

from jevpilot.browser.port import ActionKind, BrowserAction, ElementRef, Observation
from jevpilot.engine.candidates import build_candidates
from jevpilot.ledger.models import CallRecord, CallUsage as LedgerUsage
from jevpilot.ledger.recorder import CallScope, RunLedger
from jevpilot.llm.credentials import get_api_key
from jevpilot.llm.errors import LLMErrorKind
from jevpilot.llm.pricing import PRICES
from jevpilot.model_client import ModelClient, ModelClientConfig, ProviderError, ResponseMetadata
from jevpilot.telemetry.models import CallUsage
from jevpilot.privacy import PrivacyPolicy, redact_json

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

    async def _ask[T](
        self, scope: CallScope, state: JsonValue, questions: dict[str, JsonValue],
        interpret: _Interpret[T],
    ) -> tuple[T, CallUsage | None, float]:
        # Reuse v1's wire parsing/error classification, but never construct its
        # UTC-ordered CallEvent. New-engine accounting measures actual attempts
        # directly with the ledger's monotonic clock.
        client = ModelClient[T](
            _client_config(self._key(), self.endpoint), http_client=self._http_client,
        )
        payload = redact_json(
            {"state": state, "model": self.model, "questions": questions},
            policy=PrivacyPolicy(),
        )
        started = time.monotonic()
        try:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                remaining = scope.deadline - time.monotonic()
                if remaining <= 0:
                    raise JevUnavailable("timeout")
                scope.reserve_attempt()
                call_id = uuid4().hex
                sent_at = scope.ledger.clock()
                usage: CallUsage | None = None
                resolved_model: str | None = None
                error: ProviderError | None = None
                outcome = "failed"
                try:
                    response = await client._http.post(
                        self.endpoint,
                        headers={
                            "Authorization": f"Bearer {client.config.api_key}",
                            "Content-Type": "application/json", "User-Agent": "JevPilot/0.1.0",
                        },
                        json=payload, timeout=remaining,
                    )
                    body = client._parse_body(response)
                    usage = client._usage(body)
                    resolved_model = client._response_model(body)
                    metadata = ResponseMetadata(call_id, resolved_model, usage, response.status_code)
                    if not response.is_success:
                        raise client._http_error(
                            call_id=call_id, status_code=response.status_code,
                            retry_after=response.headers.get("Retry-After"),
                        )
                    try:
                        parsed = interpret(_parse_answers(body, metadata), metadata)
                    except ValidationError:
                        raise _invalid(metadata) from None
                    outcome = "ok"
                    return parsed, usage, time.monotonic() - started
                except httpx2.TransportError as failure:
                    error = ProviderError(
                        kind="transport_timeout" if isinstance(failure, httpx2.TimeoutException) else "provider_unavailable",
                        call_id=call_id, retryable=True, message="Jev transport failed",
                    )
                except ProviderError as failure:
                    error = failure
                except anyio.get_cancelled_exc_class():
                    outcome = "cancelled"
                    raise
                finally:
                    price = PRICES.get(("jev", resolved_model or self.model))
                    scope.record_call(CallRecord(
                        call_id=call_id, span_id=scope.parent_span_id,
                        kind="jev", provider="jev", auth_mode="api", purpose="normal",
                        request_model=self.model, resolved_model=resolved_model,
                        attempts=attempt, outcome=outcome,
                        error_kind=error.kind if error else ("cancelled" if outcome == "cancelled" else None),
                        usage=LedgerUsage(
                            input_tokens=usage.input_tokens if usage else None,
                            output_tokens=usage.output_tokens if usage else None,
                            cached_input_tokens=usage.cache_read_tokens if usage else None,
                        ),
                        api_estimate_usd=_estimate_cost(resolved_model or self.model, usage),
                        price_source=price.source_url if price else None,
                        price_checked_on=price.checked_on if price else None,
                        latency_s=scope.ledger.clock() - sent_at,
                    ))
                    if self.on_call is not None:
                        self.on_call(scope.ledger.calls[-1])
                assert error is not None
                if not error.retryable or attempt == MAX_ATTEMPTS:
                    raise _unavailable(error)
                delay = error.retry_after_seconds
                if delay is None:
                    delay = 0.25 * (1 << (attempt - 1))
                if delay >= scope.deadline - time.monotonic():
                    raise _unavailable(error)
                if delay:
                    await anyio.sleep(delay)
            raise AssertionError("attempt loop must return or raise")
        finally:
            with anyio.CancelScope(shield=True):
                await client.aclose()

    async def choose(
        self, state: SelectorState, *, scope: CallScope,
        domains: Mapping[str, Sequence[ElementRef]] | None = None,
    ) -> JevChoice:
        """``domains`` fixes each target operation's candidates (kept to the sent elements)."""
        elements, truncated = select_elements(state.observation.elements)
        kept = {e.index for e in elements}
        by_operation = {
            op: _candidates(elements, op) if domains is None
            else [e for e in domains.get(op, ()) if e.index in kept]
            for op in _OPERATIONS
        }
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


@dataclass(frozen=True, slots=True)
class GoalContext:
    goal: str
    brief: str = ""
    subgoal: str = ""
    recent_actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StepSignals:
    op_confidence: float
    op_margin: float
    target_confidence: float | None
    target_margin: float | None
    option_confidence: float | None
    option_margin: float | None
    candidate_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class StepDecision:
    """``action`` is None when Jev answered done; ``bound_to`` is (snapshot_id, url, backend_node_id)."""

    operation: str
    action: BrowserAction | None
    signals: StepSignals
    bound_to: tuple[str, str, int | None]
    usages: tuple[CallUsage | None, ...]
    latency_s: float


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _exact_option(target: ElementRef, input_values: dict[str, str]) -> str | None:
    wanted = {_normalize(v) for v in input_values.values()} - {""}
    matched = {
        o.value for o in target.options
        if _normalize(o.label) in wanted or _normalize(o.value) in wanted
    }
    return next(iter(matched)) if len(matched) == 1 else None


async def select_step(
    observation: Observation, goal_context: GoalContext, input_values: dict[str, str],
    *, selector: JevSelector, scope: CallScope,
) -> StepDecision:
    """One Jev step over rule-built candidates; type_text carries no value."""
    candidates = build_candidates(observation)
    visible = observation.model_copy(update={"elements": candidates.all_elements()})
    state = SelectorState(
        goal=goal_context.goal, brief=goal_context.brief, subgoal=goal_context.subgoal,
        observation=visible, recent_actions=goal_context.recent_actions,
    )
    choice = await selector.choose(
        state, scope=scope,
        domains={op: candidates.for_operation(op) for op in _TARGET_OPERATIONS},
    )
    usages = [choice.usage]
    latency = choice.latency_s
    target: ElementRef | None = None
    if choice.operation in _TARGET_OPERATIONS:
        target = next(
            (e for e in candidates.for_operation(choice.operation) if e.index == choice.target_index),
            None,
        )
        if target is None:
            raise JevUnavailable("invalid_output")
    value: str | None = None
    option_confidence = option_margin = None
    if choice.operation == "select_option" and target is not None:
        value = _exact_option(target, input_values)
        if value is None:
            option = await selector.choose_option(state, target, scope=scope)
            if option.value not in {o.value for o in target.options}:
                raise JevUnavailable("invalid_output")
            value, option_confidence, option_margin = option.value, option.confidence, option.margin
            usages.append(option.usage)
            latency += option.latency_s
    action = None if choice.operation == "done" else BrowserAction(
        kind=choice.operation, target=target, value=value,  # pyright: ignore[reportArgumentType]
    )
    return StepDecision(
        operation=choice.operation, action=action,
        signals=StepSignals(
            choice.op_confidence, choice.op_margin, choice.target_confidence,
            choice.target_margin, option_confidence, option_margin,
            choice.candidate_count, choice.truncated,
        ),
        bound_to=(
            observation.snapshot_id, observation.url,
            target.backend_node_id if target is not None else None,
        ),
        usages=tuple(usages), latency_s=latency,
    )


async def validate_key(
    key: str, *, endpoint: str | None = None, model: str | None = None,
    http_client: httpx2.AsyncClient | None = None,
) -> None:
    """One minimal two-option choice request: 2xx passes, otherwise JevUnavailable."""
    target = endpoint or _setting("JEVPILOT_JEV_ENDPOINT", JEV_ENDPOINT)
    request_model = model or _setting("JEVPILOT_JEV_MODEL", JEV_MODEL)
    client = ModelClient[None](_client_config(key, target), http_client=http_client)
    try:
        response = await client._http.post(
            target, headers={"Authorization": f"Bearer {key}"},
            json={
                "state": {"goal": "key check"}, "model": request_model,
                "questions": {"check": _choice_question("Choose a.", {"a": None, "b": None})},
            },
            timeout=30,
        )
        if not response.is_success:
            raise client._http_error(
                call_id=uuid4().hex, status_code=response.status_code,
                retry_after=response.headers.get("Retry-After"),
            )
    except httpx2.TransportError as error:
        raise JevUnavailable(
            "timeout" if isinstance(error, httpx2.TimeoutException) else "provider_unavailable",
        ) from None
    except ProviderError as error:
        raise _unavailable(error) from None
    finally:
        with anyio.CancelScope(shield=True):
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
