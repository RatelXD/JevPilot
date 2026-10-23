import json
from decimal import Decimal
from pathlib import Path
from typing import TypedDict

import anyio
from anyio.lowlevel import checkpoint
import pytest
from pydantic import JsonValue, TypeAdapter

from jevpilot.ledger import CallRecord, CallUsage, LedgerSink, RunLedger
from jevpilot.privacy import PrivacyPolicy
from jevpilot.task import ContractViolation


class Event(TypedDict):
    type: str
    data: JsonValue


class Clock:
    def __init__(self) -> None:
        self.now: float = 0.0

    def __call__(self) -> float:
        return self.now


def call(
    ledger: RunLedger, call_id: str = "c", *, billed_usd: Decimal | None = None,
    api_estimate_usd: Decimal | None = None,
) -> CallRecord:
    return CallRecord(
        call_id=call_id, span_id=ledger.root_span_id, kind="llm",
        provider="custom", auth_mode="api", purpose="choose",
        request_model="unknown", outcome="ok", latency_s=1,
        billed_usd=billed_usd, api_estimate_usd=api_estimate_usd,
    )


def test_summary_keeps_independent_partial_cost_and_token_totals() -> None:
    # Given
    ledger = RunLedger()
    ledger.record_call(call(ledger, "known", billed_usd=Decimal("1"), api_estimate_usd=Decimal("2")))
    ledger.record_call(call(ledger, "unknown"))
    # When
    totals = ledger.summary().by_kind["llm"]
    # Then
    assert totals.calls == 2
    assert totals.billed_usd.model_dump() == {
        "known_sum": Decimal("1"), "unknown_count": 1, "complete": False,
    }
    assert totals.api_estimate_usd.known_sum == Decimal("2")
    assert totals.api_estimate_usd.unknown_count == 1
    assert totals.api_equivalent_usd.unknown_count == 0
    assert totals.input_tokens.unknown_count == 2
    assert not totals.input_tokens.complete
    assert ledger.summary().by_kind["jev"].billed_usd.complete


def test_subscription_is_only_priced_as_exact_openai_api_equivalent() -> None:
    # Given
    ledger = RunLedger()
    record = CallRecord(
        call_id="sub", span_id=ledger.root_span_id, kind="llm",
        provider="chatgpt_subscription", auth_mode="subscription",
        purpose="choose", request_model="codex-default", resolved_model="gpt-5.3-codex",
        usage=CallUsage(input_tokens=1000, output_tokens=100, cached_input_tokens=200),
        outcome="ok", latency_s=1,
    )
    # When
    ledger.record_call(record)
    # Then
    actual = ledger.calls[0]
    assert actual.billed_usd is None
    assert actual.api_estimate_usd is None
    assert actual.api_equivalent_usd == Decimal("0.002835")
    assert actual.price_source is not None
    assert actual.price_checked_on == "2026-10-23"


def test_mixed_auth_counts_only_applicable_estimates_as_unknown() -> None:
    # Given
    ledger = RunLedger()
    ledger.record_call(call(ledger, "unknown-api"))
    ledger.record_call(CallRecord(
        call_id="subscription", span_id=ledger.root_span_id, kind="llm",
        provider="chatgpt_subscription", auth_mode="subscription", purpose="brief",
        request_model="gpt-5.3-codex", outcome="ok", latency_s=1,
        usage=CallUsage(input_tokens=1000, output_tokens=100, cached_input_tokens=200),
    ))
    # When
    totals = ledger.summary().by_kind["llm"]
    # Then
    assert totals.api_estimate_usd.unknown_count == 1
    assert not totals.api_estimate_usd.complete
    assert totals.api_equivalent_usd.unknown_count == 0
    assert totals.api_equivalent_usd.complete
    assert totals.billed_usd.unknown_count == 2


def test_unknown_resolved_model_does_not_inherit_requested_model_rate() -> None:
    # Given
    ledger = RunLedger()
    record = CallRecord(
        call_id="sub", span_id=ledger.root_span_id, kind="llm",
        provider="chatgpt_subscription", auth_mode="subscription",
        purpose="choose", request_model="gpt-5.3-codex", resolved_model="unknown",
        usage=CallUsage(input_tokens=1000, output_tokens=100, cached_input_tokens=0),
        outcome="ok", latency_s=1,
    )
    # When
    ledger.record_call(record)
    # Then
    assert ledger.calls[0].api_equivalent_usd is None


def test_overlap_belongs_to_first_started_direct_child_only() -> None:
    # Given
    clock = Clock()
    ledger = RunLedger(clock=clock)
    clock.now = 1
    first = ledger.open_span("observe")
    clock.now = 2
    second = ledger.open_span("llm")
    child = ledger.open_span("jev", parent_id=first.span_id)
    clock.now = 3
    _ = ledger.close_span(child.span_id)
    clock.now = 4
    _ = ledger.close_span(first.span_id)
    clock.now = 6
    _ = ledger.close_span(second.span_id)
    clock.now = 8
    # When
    summary = ledger.finish()
    # Then
    assert summary.duration_s == {"observe": 3, "llm": 2}
    assert summary.overlap_s == 2
    assert summary.elapsed_s == 8
    assert child.parent_id == first.span_id
    assert summary.run_started_at_utc.endswith("+00:00")


def test_nested_and_triple_overlap_counts_only_excess_duration() -> None:
    # Given
    clock = Clock()
    ledger = RunLedger(clock=clock)
    first = ledger.open_span("observe")
    clock.now = 1
    second = ledger.open_span("llm")
    third = ledger.open_span("execute")
    clock.now = 2
    _ = ledger.close_span(third.span_id)
    clock.now = 3
    _ = ledger.close_span(second.span_id)
    clock.now = 5
    _ = ledger.close_span(first.span_id)
    # When
    summary = ledger.finish()
    # Then
    assert summary.duration_s == {"observe": 5, "llm": 0, "execute": 0}
    assert summary.overlap_s == 3


def test_context_manager_uses_nested_parent_and_failure_outcome() -> None:
    # Given
    ledger = RunLedger()
    # When
    with pytest.raises(RuntimeError):
        with ledger.span("brief"):
            with ledger.span("llm"):
                raise RuntimeError("test failure")
    # Then
    parent = next(s for s in ledger.spans.values() if s.kind == "brief")
    child = next(s for s in ledger.spans.values() if s.kind == "llm")
    assert child.parent_id == parent.span_id
    assert ledger.spans[child.span_id].outcome == "failed"
    assert ledger.spans[parent.span_id].outcome == "failed"
    with ledger.span("verify") as sibling:
        assert sibling.parent_id == ledger.root_span_id


def test_normal_span_inside_outer_exception_handler_still_succeeds() -> None:
    # Given
    ledger = RunLedger()
    # When
    try:
        raise RuntimeError("outer handled error")
    except RuntimeError:
        with ledger.span("recovery"):
            pass
    # Then
    recovery = next(s for s in ledger.spans.values() if s.kind == "recovery")
    assert recovery.outcome == "ok"


@pytest.mark.asyncio
async def test_cancellation_closes_span_and_restores_parent() -> None:
    # Given
    ledger = RunLedger()
    # When
    with anyio.CancelScope() as cancellation:
        with ledger.span("wait"):
            cancellation.cancel()
            await checkpoint()
    # Then
    waiting = next(s for s in ledger.spans.values() if s.kind == "wait")
    assert ledger.spans[waiting.span_id].outcome == "cancelled"
    with ledger.span("verify") as sibling:
        assert sibling.parent_id == ledger.root_span_id


@pytest.mark.asyncio
async def test_concurrent_tasks_keep_independent_span_parents() -> None:
    # Given
    ledger = RunLedger()
    first_opened = anyio.Event()
    second_opened = anyio.Event()
    pairs: list[tuple[str, str | None]] = []

    async def first() -> None:
        with ledger.span("brief") as parent:
            first_opened.set()
            await second_opened.wait()
            with ledger.span("llm") as child:
                pairs.append((parent.span_id, child.parent_id))

    async def second() -> None:
        await first_opened.wait()
        with ledger.span("observe") as parent:
            second_opened.set()
            with ledger.span("verify") as child:
                pairs.append((parent.span_id, child.parent_id))

    # When
    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(first)
            group.start_soon(second)
    # Then
    assert len(pairs) == 2
    assert all(parent == actual for parent, actual in pairs)
    assert sum(s.parent_id == ledger.root_span_id for s in ledger.spans.values()) == 2


def test_persistence_redacts_events_and_receipt_and_only_keeps_input_lengths(tmp_path: Path) -> None:
    # Given
    sentinel = "synthetic-private-key"
    sink = LedgerSink(
        "run", root=tmp_path, privacy_policy=PrivacyPolicy(secret_values=(sentinel,)),
    )
    ledger = RunLedger("run", sink=sink)
    with ledger.span("observe", f"label {sentinel}"):
        ledger.event("route", {
            "api_key": sentinel, "input": {"field": "private-input"},
            "action": {"value": "never-write-me"}, "reason": sentinel,
        })
        ledger.event("recovery", {"text": "private-recovery"})
    ledger.record_call(call(ledger))
    # When
    _ = ledger.finish()
    # Then
    events = (sink.directory / "events.jsonl").read_text()
    receipt = (sink.directory / "receipt.json").read_text()
    assert sentinel not in events + receipt
    assert "never-write-me" not in events
    assert "private-input" not in events
    assert "private-recovery" not in events
    records = [
        TypeAdapter(Event).validate_json(line)
        for line in events.splitlines()
    ]
    assert {e["type"] for e in records} == {"span", "call", "route", "recovery"}
    route = next(e["data"] for e in records if e["type"] == "route")
    assert isinstance(route, dict)
    action = route["action"]
    assert isinstance(action, dict)
    assert action["value"] == len("never-write-me")
    assert route["input"] == len(json.dumps({"field": "private-input"}))
    assert json.loads(receipt)["by_kind"]["llm"]["billed_usd"]["complete"] is False


def test_default_sink_location_is_run_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setenv("HOME", str(tmp_path))
    # When
    sink = LedgerSink("run")
    # Then
    assert sink.directory == tmp_path / ".jevpilot" / "runs" / "run"


def test_record_call_callback_observes_each_priced_request_immediately() -> None:
    # Given
    observed: list[CallRecord] = []
    ledger = RunLedger(on_call=observed.append)
    # When
    for index in range(3):
        ledger.record_call(CallRecord(
            call_id=str(index), span_id=ledger.root_span_id,
            kind="jev", provider="jev", auth_mode="api", purpose="choose",
            request_model="jev-1.13.0", outcome="ok",
            usage=CallUsage(input_tokens=1000), latency_s=1,
        ))
        assert len(observed) == index + 1
    # Then
    assert all(c.api_estimate_usd == Decimal("0.000042") for c in observed)
    assert ledger.summary().by_kind["jev"].api_estimate_usd.known_sum == Decimal("0.000126")


def test_invalid_references_and_duplicate_records_are_rejected() -> None:
    # Given
    ledger = RunLedger()
    record = call(ledger)
    ledger.record_call(record)
    opened = ledger.open_span("llm")
    # When / Then
    with pytest.raises(ContractViolation, match="duplicate"):
        ledger.record_call(record)
    with pytest.raises(ContractViolation, match="child"):
        _ = ledger.finish()
    _ = ledger.close_span(opened.span_id)
    with pytest.raises(ContractViolation, match="already closed"):
        _ = ledger.close_span(opened.span_id)
    _ = ledger.finish()
    with pytest.raises(ContractViolation, match="parent span is closed"):
        _ = ledger.open_span("verify")


@pytest.mark.parametrize("run_id", ["../outside", ".", "..", ""])
def test_sink_rejects_run_directory_escape(run_id: str, tmp_path: Path) -> None:
    # Given / When / Then
    with pytest.raises(ContractViolation):
        _ = LedgerSink(run_id, root=tmp_path)
