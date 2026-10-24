from collections.abc import Callable
from decimal import Decimal

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.browser.port import ElementRef, Observation, Option
from jevpilot.engine.jev import JevSelector, JevUnavailable, SelectorState, validate_key
from jevpilot.ledger import CallRecord, RunLedger

KEY = "synthetic-jev-key-12345"


def element(index: int, tag: str = "button", *, in_viewport: bool = True,
            options: list[Option] | None = None, **attributes: str) -> ElementRef:
    return ElementRef(
        snapshot_id="s", index=index, backend_node_id=index, tag=tag, role=None,
        text=f"element {index}", attributes=attributes, in_viewport=in_viewport,
        options=options or [],
    )


def state_of(elements: list[ElementRef]) -> SelectorState:
    return SelectorState(
        goal="goal", brief="brief", subgoal="subgoal", recent_actions=("a",) * 7,
        observation=Observation(
            snapshot_id="s", url="https://example.com", title="t", tabs=[],
            elements=elements, page_text="x" * 5000, fingerprint="f",
            observed_at_monotonic=0.0,
        ),
    )


NO_ELEMENT_OPS = {
    "scroll_down": 0.0, "scroll_up": 0.0, "go_back": 0.0, "wait": 0.0, "done": 1.0,
}


def answer(choice: str, probabilities: dict[str, float], confidence: float = 0.9) -> JsonValue:
    return {
        "type": "choice", "choice": choice, "probabilities": dict(probabilities),
        "confidence": confidence,
    }


def body_for(answers: dict[str, JsonValue], input_tokens: int = 1000) -> JsonValue:
    return {
        "model": "jev-1.13.0", "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


def client_for(handler: Callable[[httpx2.Request], httpx2.Response]) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def questions_of(request: httpx2.Request) -> dict[str, dict[str, JsonValue]]:
    payload = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
    return TypeAdapter(dict[str, dict[str, JsonValue]]).validate_python(payload["questions"])


def ledger_scope() -> tuple[RunLedger, "object"]:
    ledger = RunLedger()
    return ledger, ledger.scope(deadline=ledger.clock() + 60)


async def test_choose_returns_operation_target_and_records_one_call() -> None:
    # Given
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=body_for({
            "operation": answer("click", {
                "click": 0.7, "scroll_down": 0.1, "scroll_up": 0.1,
                "go_back": 0.04, "wait": 0.04, "done": 0.02,
            }),
            "click_target": answer("2", {"1": 0.2, "2": 0.8}, 0.8),
        }))

    ledger, scope = ledger_scope()
    lines: list[CallRecord] = []
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http, on_call=lines.append)
        # When
        choice = await selector.choose(state_of([element(1), element(2)]), scope=scope)  # pyright: ignore[reportArgumentType]
    # Then
    assert choice.operation == "click"
    assert choice.op_confidence == 0.9
    assert choice.op_margin == pytest.approx(0.6)
    assert choice.target_index == 2
    assert choice.target_margin == pytest.approx(0.6)
    assert (choice.candidate_count, choice.truncated) == (2, False)
    assert choice.usage is not None and choice.usage.input_tokens == 1000
    assert len(ledger.calls) == 1 and lines == ledger.calls
    assert ledger.calls[0].kind == "jev"
    assert ledger.calls[0].api_estimate_usd == Decimal("0.000042")
    request = seen[0]
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    payload = request.read().decode()
    assert "navigate" not in payload
    assert "type_text_target" not in payload and "select_option" not in payload
    assert payload.count("x" * 4000) == 1 and "x" * 4001 not in payload


@pytest.mark.parametrize("wait_probability", [0.4, 0.3], ids=["valid-sum", "invalid-sum"])
async def test_probability_sum_error_is_rejected_and_recorded_as_failed(
    wait_probability: float,
) -> None:
    # Given
    ops = NO_ELEMENT_OPS | {"done": 0.6, "wait": wait_probability}

    def handler(request: httpx2.Request) -> httpx2.Response:
        criteria = questions_of(request)["operation"]["criteria"]
        assert isinstance(criteria, dict) and set(ops) == set(criteria)
        return httpx2.Response(200, json=body_for({
            "operation": answer("done", ops),
        }))

    ledger, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http)
        # When / Then: the paired control differs only in the probability sum.
        if wait_probability == 0.4:
            choice = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
            assert choice.operation == "done"
            assert [c.outcome for c in ledger.calls] == ["ok"]
        else:
            with pytest.raises(JevUnavailable) as raised:
                _ = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
            assert raised.value.kind == "invalid_output"
            assert [(c.outcome, c.error_kind) for c in ledger.calls] == [
                ("failed", "invalid_response"),
            ]


@pytest.mark.parametrize("done_probability", [0.6, 0.4], ids=["valid-argmax", "invalid-argmax"])
async def test_choice_that_is_not_the_argmax_is_rejected(done_probability: float) -> None:
    # Given
    ops = NO_ELEMENT_OPS | {"done": done_probability, "wait": 1.0 - done_probability}

    def handler(request: httpx2.Request) -> httpx2.Response:
        criteria = questions_of(request)["operation"]["criteria"]
        assert isinstance(criteria, dict) and set(ops) == set(criteria)
        return httpx2.Response(200, json=body_for({"operation": answer("done", ops)}))

    ledger, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http)
        # When / Then: both distributions sum to one and use the same valid domain.
        if done_probability == 0.6:
            choice = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
            assert choice.operation == "done"
            assert [c.outcome for c in ledger.calls] == ["ok"]
        else:
            with pytest.raises(JevUnavailable) as raised:
                _ = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
            assert raised.value.kind == "invalid_output"
            assert [(c.outcome, c.error_kind) for c in ledger.calls] == [
                ("failed", "invalid_response"),
            ]


async def test_429_then_success_records_two_calls() -> None:
    # Given
    replies = iter([
        httpx2.Response(429, headers={"Retry-After": "0"}, json={}),
        httpx2.Response(200, json=body_for({
            "operation": answer("done", NO_ELEMENT_OPS),
        })),
    ])

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return next(replies)

    ledger, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http)
        # When
        choice = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
    # Then
    assert choice.operation == "done"
    first, second = ledger.calls
    assert (first.outcome, first.error_kind) == ("failed", "rate_limited")
    assert second.outcome == "ok"
    assert first.call_id != second.call_id
    assert [c.attempts for c in ledger.calls] == [1, 2]


async def test_more_than_200_elements_keep_viewport_first_and_flag_truncation() -> None:
    # Given
    elements = [element(i, in_viewport=i >= 150) for i in range(300)]
    seen: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.read().decode())
        return httpx2.Response(200, json=body_for({
            "operation": answer("click", {
                "click": 1.0, "scroll_down": 0.0, "scroll_up": 0.0, "go_back": 0.0,
                "wait": 0.0, "done": 0.0,
            }),
            "click_target": answer("150", {str(i): 1.0 if i == 150 else 0.0
                                           for i in [*range(50), *range(150, 300)]}),
        }))

    _, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http)
        # When
        choice = await selector.choose(state_of(elements), scope=scope)  # pyright: ignore[reportArgumentType]
    # Then
    assert (choice.candidate_count, choice.truncated) == (200, True)
    assert choice.target_index == 150
    assert '"[299] button' in seen[0] and '"[49] button' in seen[0]
    assert '"[50] button' not in seen[0] and '"[149] button' not in seen[0]


async def test_missing_key_raises_auth_missing_without_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr("jevpilot.engine.jev.get_api_key", lambda _provider: None)
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json={})

    ledger, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(http_client=http)
        # When / Then
        with pytest.raises(JevUnavailable) as raised:
            _ = await selector.choose(state_of([]), scope=scope)  # pyright: ignore[reportArgumentType]
    assert raised.value.kind == "auth_missing"
    assert not sent and not ledger.calls


async def test_choose_option_returns_a_value_of_the_target_options() -> None:
    # Given
    options = [Option(value="kr", label="Korea"), Option(value="us", label="United States")]
    target = element(5, "select", options=options)

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body_for({
            "option": answer("1", {"0": 0.25, "1": 0.75}, 0.75),
        }))

    _, scope = ledger_scope()
    async with client_for(handler) as http:
        selector = JevSelector(api_key=KEY, http_client=http)
        # When
        picked = await selector.choose_option(state_of([target]), target, scope=scope)  # pyright: ignore[reportArgumentType]
    # Then
    assert picked.value == "us"
    assert picked.margin == pytest.approx(0.5)


@pytest.mark.parametrize(("status", "content"), [
    (200, b'{"model":"jev-1.14.0","answers":{}}'),
    (200, b""),
    (201, b"not JSON"),
    (204, b""),
    (299, b"null"),
])
async def test_validate_key_passes_on_2xx(status: int, content: bytes) -> None:
    # Given
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        questions = questions_of(request)
        assert len(questions) == 1
        question = next(iter(questions.values()))
        assert question["type"] == "choice"
        criteria = question["criteria"]
        assert isinstance(criteria, dict) and len(criteria) == 2
        return httpx2.Response(status, content=content)

    async with client_for(handler) as http:
        # When / Then
        assert await validate_key(KEY, http_client=http) is None
    assert len(sent) == 1


@pytest.mark.parametrize(("status", "kind"), [
    (401, "auth_invalid"), (403, "auth_invalid"), (429, "rate_limited"), (529, "rate_limited"),
])
async def test_validate_key_maps_failures_without_leaking_the_key(status: int, kind: str) -> None:
    # Given
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(status, json={})

    async with client_for(handler) as http:
        # When / Then
        with pytest.raises(JevUnavailable) as raised:
            await validate_key(KEY, http_client=http)
    assert raised.value.kind == kind
    assert KEY not in str(raised.value) and KEY not in repr(raised.value)
    assert len(sent) == 1
