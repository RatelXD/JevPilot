"""Run-owned mutable accumulator with monotonic span attribution."""

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import sys
import time
from typing import Literal
from uuid import uuid4

from pydantic import JsonValue

from jevpilot.ledger.models import (
    Aggregate, CallRecord, KindSummary, Outcome, RunSummary, Span, SpanKind,
)
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.pricing import PRICES, estimate_cost_usd
from jevpilot.task import ContractViolation


def _aggregate(values: Iterable[int | Decimal | None]) -> Aggregate:
    known = Decimal("0")
    unknown = 0
    for value in values:
        if value is None:
            unknown += 1
        else:
            known += Decimal(value)
    return Aggregate(known_sum=known, unknown_count=unknown, complete=unknown == 0)


@dataclass(frozen=True, slots=True)
class CallScope:
    ledger: "RunLedger"
    parent_span_id: str
    deadline: float
    step_index: int = 0
    reserve_attempt: Callable[[], None] = lambda: None

    @property
    def run_id(self) -> str:
        return self.ledger.run_id

    def record_call(self, record: CallRecord) -> None:
        self.ledger.record_call(record)


class RunLedger:
    """Mutable run-local span tree and call accumulator; no global totals."""

    def __init__(
        self, run_id: str | None = None, *, sink: LedgerSink | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_call: Callable[[CallRecord], None] | None = None,
    ) -> None:
        self.run_id: str = run_id or uuid4().hex
        self.run_started_at_utc: str = datetime.now(UTC).isoformat()
        self.clock: Callable[[], float] = clock
        self.sink: LedgerSink | None = sink
        self.on_call: Callable[[CallRecord], None] | None = on_call
        self.root_span_id: str = uuid4().hex
        self.spans: dict[str, Span] = {}
        self.calls: list[CallRecord] = []
        self._parent: ContextVar[str] = ContextVar("ledger_parent", default=self.root_span_id)
        root = Span(
            span_id=self.root_span_id, parent_id=None, kind="run",
            label="run", start_s=clock(),
        )
        self.spans[root.span_id] = root
        self._emit_span(root)

    def _emit_span(self, span: Span) -> None:
        if self.sink is not None:
            self.sink.event("span", span.model_dump(mode="json"))

    def open_span(
        self, kind: SpanKind, label: str = "", *, parent_id: str | None = None,
    ) -> Span:
        parent = self.spans[parent_id or self._parent.get()]
        if parent.end_s is not None:
            raise ContractViolation("parent_id", "parent span is closed")
        span = Span(
            span_id=uuid4().hex, parent_id=parent.span_id, kind=kind,
            label=label, start_s=self.clock(),
        )
        self.spans[span.span_id] = span
        self._emit_span(span)
        return span

    def close_span(self, span_id: str, outcome: Outcome = "ok") -> Span:
        span = self.spans[span_id]
        if span.end_s is not None:
            raise ContractViolation("span_id", "span already closed")
        if any(s.parent_id == span_id and s.end_s is None for s in self.spans.values()):
            raise ContractViolation("span_id", "child spans are still open")
        closed = Span.model_validate({
            **span.model_dump(), "end_s": self.clock(), "outcome": outcome,
        })
        self.spans[span_id] = closed
        self._emit_span(closed)
        return closed

    @contextmanager
    def span(
        self, kind: SpanKind, label: str = "", *, parent_id: str | None = None,
    ) -> Generator[Span]:
        opened = self.open_span(kind, label, parent_id=parent_id)
        token = self._parent.set(opened.span_id)
        completed = False
        try:
            yield opened
            completed = True
        finally:
            error = sys.exception()
            outcome: Outcome = "ok"
            if not completed and error is not None:
                outcome = "failed" if isinstance(error, Exception) else "cancelled"
            self._parent.reset(token)
            _ = self.close_span(opened.span_id, outcome)

    def scope(
        self, *, deadline: float, parent_span_id: str | None = None,
        step_index: int = 0, reserve_attempt: Callable[[], None] = lambda: None,
    ) -> CallScope:
        parent = parent_span_id or self._parent.get()
        if parent not in self.spans:
            raise ContractViolation("parent_span_id", "unknown span")
        return CallScope(self, parent, deadline, step_index, reserve_attempt)

    def record_call(self, record: CallRecord) -> None:
        if record.span_id not in self.spans:
            raise ContractViolation("span_id", "unknown span")
        if any(call.call_id == record.call_id for call in self.calls):
            raise ContractViolation("call_id", "duplicate call")
        family = "openai" if record.auth_mode == "subscription" else record.provider
        model = record.resolved_model or record.request_model
        price = PRICES.get((family, model))
        cost_field = "api_equivalent_usd" if record.auth_mode == "subscription" else "api_estimate_usd"
        if getattr(record, cost_field) is None and price is not None:
            estimate = estimate_cost_usd(family, model, record.usage)
            if estimate is not None:
                record = CallRecord.model_validate({
                    **record.model_dump(), cost_field: estimate,
                    "price_source": price.source_url, "price_checked_on": price.checked_on,
                })
        self.calls.append(record)
        if self.sink is not None:
            self.sink.event("call", record.model_dump(mode="json"))
        if self.on_call is not None:
            self.on_call(record)

    def event(self, kind: Literal["route", "recovery"], payload: JsonValue) -> None:
        if self.sink is not None:
            self.sink.event(kind, payload)

    def finish(self, outcome: Outcome = "ok") -> RunSummary:
        _ = self.close_span(self.root_span_id, outcome)
        summary = self.summary()
        if self.sink is not None:
            _ = self.sink.write_receipt(summary)
        return summary

    def summary(self) -> RunSummary:
        by_kind: dict[Literal["jev", "llm"], KindSummary] = {}
        for kind in ("jev", "llm"):
            calls = [call for call in self.calls if call.kind == kind]
            by_kind[kind] = KindSummary(
                calls=len(calls),
                input_tokens=_aggregate(c.usage.input_tokens for c in calls),
                output_tokens=_aggregate(c.usage.output_tokens for c in calls),
                cached_input_tokens=_aggregate(c.usage.cached_input_tokens for c in calls),
                reasoning_tokens=_aggregate(c.usage.reasoning_tokens for c in calls),
                billed_usd=_aggregate(c.billed_usd for c in calls),
                api_estimate_usd=_aggregate(
                    c.api_estimate_usd for c in calls if c.auth_mode == "api"
                ),
                api_equivalent_usd=_aggregate(
                    c.api_equivalent_usd for c in calls if c.auth_mode == "subscription"
                ),
            )
        root = self.spans[self.root_span_id]
        end = root.end_s if root.end_s is not None else self.clock()
        duration: dict[SpanKind, float] = {}
        overlap = 0.0
        covered_end = root.start_s
        children = sorted(
            (s for s in self.spans.values() if s.parent_id == self.root_span_id),
            key=lambda s: s.start_s,
        )
        for span in children:
            start = max(root.start_s, span.start_s)
            stop = min(end, span.end_s if span.end_s is not None else end)
            overlap += max(0.0, min(stop, covered_end) - start)
            duration[span.kind] = duration.get(span.kind, 0.0) + max(
                0.0, stop - max(start, covered_end),
            )
            covered_end = max(covered_end, stop)
        return RunSummary(
            run_id=self.run_id, run_started_at_utc=self.run_started_at_utc,
            elapsed_s=end - root.start_s, by_kind=by_kind,
            duration_s=duration, overlap_s=overlap,
        )
