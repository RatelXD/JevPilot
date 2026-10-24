from decimal import Decimal
from typing import Literal

import pytest

from jevpilot.ledger.models import Aggregate, CallRecord, CallUsage, KindSummary, RunSummary
from jevpilot.ledger.recorder import RunLedger
from jevpilot.ledger.receipt import render_call_line, render_receipt


def aggregate(value: int | str | None = None, unknown: int = 0) -> Aggregate:
    return Aggregate(
        known_sum=Decimal(value) if value is not None else Decimal("0"),
        known_count=int(value is not None), unknown_count=unknown, complete=unknown == 0,
    )


def kind_summary(
    *, calls: int = 0, input_tokens: int | str = 0, output_tokens: int | str = 0,
    estimate: int | str | None = None, unknown_estimates: int = 0,
    billed: int | str | None = None,
) -> KindSummary:
    return KindSummary(
        calls=calls, input_tokens=aggregate(input_tokens),
        output_tokens=aggregate(output_tokens), cached_input_tokens=aggregate(),
        reasoning_tokens=aggregate(), billed_usd=aggregate(billed),
        api_estimate_usd=aggregate(estimate, unknown_estimates),
        api_equivalent_usd=aggregate(),
    )


def summary(*, unknown_estimates: int = 0) -> RunSummary:
    return RunSummary(
        run_id="run", run_started_at_utc="2026-10-23T00:00:00+00:00",
        elapsed_s=23.4,
        by_kind={
            "jev": kind_summary(calls=9, input_tokens=31000, estimate="0.0013"),
            "llm": kind_summary(
                calls=2, input_tokens=5210, output_tokens=410, estimate="0.021",
                unknown_estimates=unknown_estimates,
            ),
        },
        duration_s={"observe": 3.1, "jev": 2.0, "llm": 9.8, "execute": 6.2, "wait": 2.3},
        overlap_s=1.1,
    )


def test_receipt_renders_time_and_cost_bases_with_explicit_labels() -> None:
    rendered = render_receipt(summary())

    assert "결과: 미검증" in rendered
    assert "시간: 총 23.4초" in rendered
    assert "관찰 3.1 · Jev 2.0(9회) · LLM 9.8(2회)" in rendered
    assert "병렬 겹침 1.1초 별도" in rendered
    assert "$0.021 (추정)" in rendered
    assert "청구액 미확인" in rendered
    assert "LLM: 토큰 입력 5,210 / 출력 410" in rendered
    assert "Jev: 9회 · 토큰 입력 31,000 · $0.0013 (추정)" in rendered


def test_unknown_cost_is_not_rendered_as_zero() -> None:
    rendered = render_receipt(summary(unknown_estimates=1))

    assert "미확인 1건" in rendered
    assert "$0.021 (추정) + 미확인 1건" in rendered


def test_all_unknown_cost_is_not_rendered_as_zero() -> None:
    data = summary(unknown_estimates=1)
    no_known = data.model_copy(update={
        "by_kind": {
            **data.by_kind,
            "llm": data.by_kind["llm"].model_copy(update={
                "api_estimate_usd": aggregate(unknown=1),
            }),
        },
    })

    assert "미확인" in render_receipt(no_known)
    assert "LLM: 2회 · 토큰 입력 미확인" not in render_receipt(no_known)
    assert "LLM:" in render_receipt(no_known)
    assert "$0 (추정)" not in render_receipt(no_known)


def test_receipt_uses_call_metadata_for_result_evidence_provider_and_reasons() -> None:
    call = CallRecord(
        call_id="llm-1", span_id="span", kind="llm", provider="openai",
        auth_mode="subscription", purpose="brief", request_model="gpt-5-codex",
        resolved_model="gpt-5-codex", outcome="ok", latency_s=7.9,
        usage=CallUsage(input_tokens=2100, output_tokens=180),
        api_equivalent_usd=Decimal("0.009"),
    )

    data = summary()
    data = data.model_copy(update={
        "by_kind": {
            **data.by_kind,
            "llm": data.by_kind["llm"].model_copy(update={
                "billed_usd": aggregate(unknown=2),
                "api_estimate_usd": aggregate(),
                "api_equivalent_usd": aggregate("0.009"),
            }),
        },
    })
    rendered = render_receipt(
        data, outcome="검증됨", answer_summary="완료",
        evidence_quote="화면에서 확인됨", calls=[call],
    )

    assert "결과: 검증됨 — 완료" in rendered
    assert '근거: "화면에서 확인됨"' in rendered
    assert "LLM: openai/gpt-5-codex · 구독" in rendered
    assert "$0.009 (API 환산)" in rendered
    assert "LLM 개입 이유: brief 1" in rendered


def test_mixed_api_and_subscription_costs_keep_their_own_labels_and_unknowns() -> None:
    data = summary()
    llm = data.by_kind["llm"].model_copy(update={
        "api_estimate_usd": aggregate("0.021", 1),
        "api_equivalent_usd": aggregate("0.009", 2),
    })
    rendered = render_receipt(
        data.model_copy(update={"by_kind": {**data.by_kind, "llm": llm}})
    )

    assert "$0.021 (추정) + 미확인 1건" in rendered
    assert "$0.009 (API 환산) + 미확인 2건" in rendered
    assert "LLM: openai/" not in rendered


def test_receipt_omits_evidence_line_when_quote_is_absent() -> None:
    rendered = render_receipt(summary())

    assert "결과: 미검증" in rendered
    assert "근거:" not in rendered
    assert '"미확인"' not in rendered


def test_call_line_requires_an_explicit_ordinal() -> None:
    call = CallRecord(
        call_id="jev-1", span_id="span", kind="jev", provider="jev",
        auth_mode="api", purpose="choose", request_model="jev-model",
        outcome="ok", attempts=9, usage=CallUsage(input_tokens=100), latency_s=0.1,
        api_estimate_usd=Decimal("0.0000042"),
    )

    assert render_call_line(call, ordinal=3).startswith("Jev #3 ")
    assert "Jev #9 " not in render_call_line(call, ordinal=3)


def test_mixed_auth_summary_keeps_provider_and_model_and_cost_labels() -> None:
    api_call = CallRecord(
        call_id="api", span_id="span", kind="llm", provider="openai",
        auth_mode="api", purpose="choose", request_model="gpt-api",
        resolved_model="gpt-api", outcome="ok", latency_s=1,
        usage=CallUsage(input_tokens=100, output_tokens=10),
        api_estimate_usd=Decimal("0.021"),
    )
    subscription_call = CallRecord(
        call_id="sub", span_id="span", kind="llm", provider="openai",
        auth_mode="subscription", purpose="brief", request_model="gpt-codex",
        resolved_model="gpt-codex", outcome="ok", latency_s=2,
        usage=CallUsage(input_tokens=200, output_tokens=20),
        api_equivalent_usd=Decimal("0.009"),
    )
    data = summary()
    llm = data.by_kind["llm"].model_copy(update={
        "api_estimate_usd": aggregate("0.021", 1),
        "api_equivalent_usd": aggregate("0.009", 2),
    })

    rendered = render_receipt(
        data.model_copy(update={"by_kind": {**data.by_kind, "llm": llm}}),
        calls=[api_call, subscription_call],
    )

    assert "LLM: openai/gpt-codex · 구독" in rendered
    assert "$0.021 (추정) + 미확인 1건" in rendered
    assert "$0.009 (API 환산) + 미확인 2건" in rendered
    assert "· 청구액 미확인" in rendered


def test_subscription_cost_basis_comes_from_summary_without_call_records() -> None:
    data = summary()
    llm = data.by_kind["llm"].model_copy(update={
        "api_estimate_usd": aggregate(),
        "api_equivalent_usd": aggregate("0.009"),
    })
    data = data.model_copy(update={"by_kind": {**data.by_kind, "llm": llm}})
    assert "$0.009 (API 환산)" in render_receipt(data)


def test_no_calls_do_not_emit_zero_cost_lines() -> None:
    data = summary()
    no_calls = data.model_copy(update={
        "by_kind": {
            **data.by_kind,
            "llm": kind_summary(),
            "jev": kind_summary(),
        },
    })
    rendered = render_receipt(no_calls)
    assert "LLM:" not in rendered
    assert "Jev:" not in rendered
    assert "$0 (청구)" not in rendered


def test_unknown_cost_rendering_distinguishes_all_unknown_and_mixed() -> None:
    data = summary()
    llm = data.by_kind["llm"].model_copy(update={
        "api_estimate_usd": aggregate(unknown=2),
    })
    rendered_all_unknown = render_receipt(
        data.model_copy(update={"by_kind": {**data.by_kind, "llm": llm}})
    )
    assert "· 미확인 ·" in rendered_all_unknown
    assert "미확인 2건" not in rendered_all_unknown

    llm = llm.model_copy(update={"api_estimate_usd": aggregate("0.021", 2)})
    rendered_mixed = render_receipt(
        data.model_copy(update={"by_kind": {**data.by_kind, "llm": llm}})
    )
    assert "$0.021 (추정) + 미확인 2건" in rendered_mixed


def test_call_line_prefers_billed_cost_and_uses_call_ordinal() -> None:
    call = CallRecord(
        call_id="jev-3", span_id="span", kind="jev", provider="jev",
        auth_mode="api", purpose="choose", request_model="jev-model",
        outcome="ok", attempts=3, usage=CallUsage(input_tokens=3410), latency_s=0.21,
        billed_usd=Decimal("0.00014"), api_estimate_usd=Decimal("0.0002"),
    )
    assert render_call_line(call, ordinal=5).startswith("Jev #5 ")
    assert render_call_line(call, ordinal=5).endswith("$0.00014 (청구)")


def test_subscription_call_line_uses_api_equivalent_cost() -> None:
    call = CallRecord(
        call_id="llm-sub", span_id="span", kind="llm", provider="openai",
        auth_mode="subscription", purpose="brief", request_model="gpt-5-codex",
        outcome="ok", latency_s=7.9, usage=CallUsage(input_tokens=2100, output_tokens=180),
        api_equivalent_usd=Decimal("0.002835"),
    )
    assert render_call_line(call, ordinal=1).endswith("$0.002835 (API 환산)")


def test_live_call_lines_keep_unknown_cost_and_cost_label() -> None:
    call = CallRecord(
        call_id="jev-3", span_id="span", kind="jev", provider="jev",
        auth_mode="api", purpose="choose", request_model="jev-model",
        outcome="ok", attempts=3, usage=CallUsage(input_tokens=3410), latency_s=0.21,
        api_estimate_usd=Decimal("0.00014"),
    )
    unknown = call.model_copy(update={"call_id": "jev-4", "api_estimate_usd": None})

    assert "Jev #7 0.21초 · 입력 3,410 · $0.00014 (추정)" == render_call_line(call, ordinal=7)
    assert render_call_line(unknown, ordinal=8).endswith("· 미확인")


def test_final_receipt_prefers_real_jev_billed_amount() -> None:
    ledger = RunLedger(run_id="jev-billed", clock=lambda: 0.0)
    ledger.record_call(CallRecord(
        call_id="jev-1", span_id=ledger.root_span_id, kind="jev", provider="jev",
        auth_mode="api", purpose="choose", request_model="jev-model",
        outcome="ok", usage=CallUsage(input_tokens=100), latency_s=0.1,
        billed_usd=Decimal("0.05"), api_estimate_usd=Decimal("0.021"),
    ))

    receipt = render_receipt(ledger.finish(), calls=ledger.calls)

    assert "Jev: 1회" in receipt
    assert "$0.05 (청구)" in receipt
    assert "$0.021 (추정)" not in receipt


def test_final_receipt_preserves_real_known_zero_amount() -> None:
    ledger = RunLedger(run_id="known-zero", clock=lambda: 0.0)
    ledger.record_call(CallRecord(
        call_id="llm-zero", span_id=ledger.root_span_id, kind="llm", provider="openai",
        auth_mode="api", purpose="brief", request_model="gpt-test",
        outcome="ok", usage=CallUsage(input_tokens=100, output_tokens=10), latency_s=0.1,
        billed_usd=Decimal("0"), api_estimate_usd=Decimal("0"),
    ))

    receipt = render_receipt(ledger.finish(), calls=ledger.calls)

    assert "LLM:" in receipt
    assert "$0 (청구)" in receipt
    assert "청구액 미확인" not in receipt
    assert "토큰 입력 100 / 출력 10" in receipt


@pytest.mark.parametrize(
    ("kind", "auth", "billed", "estimate", "equivalent", "unknown", "expected"),
    [
        ("llm", "api", "0", "0.021", None, False, "$0 (청구)"),
        ("jev", "api", "0", "0.021", None, False, "$0 (청구)"),
        ("llm", "api", None, "0", None, False, "$0 (추정)"),
        ("llm", "subscription", None, None, "0", False, "$0 (API 환산)"),
        ("llm", "api", None, "0", None, True, "$0 (추정) + 미확인 1건"),
        ("llm", "api", None, None, None, False, "· 미확인 · 청구액 미확인"),
        ("llm", "subscription", None, None, None, False, "· 미확인 · 청구액 미확인"),
    ],
)
def test_real_ledger_cost_presence_boundaries(
    kind: Literal["jev", "llm"], auth: Literal["api", "subscription"],
    billed: str | None, estimate: str | None, equivalent: str | None,
    unknown: bool, expected: str,
) -> None:
    # Given
    ledger = RunLedger(clock=lambda: 0.0)
    ledger.record_call(CallRecord(
        call_id="known", span_id=ledger.root_span_id, kind=kind, provider="custom",
        auth_mode=auth, purpose="brief", request_model="unpriced",
        outcome="ok", latency_s=0.1,
        billed_usd=Decimal(billed) if billed is not None else None,
        api_estimate_usd=Decimal(estimate) if estimate is not None else None,
        api_equivalent_usd=Decimal(equivalent) if equivalent is not None else None,
    ))
    if unknown:
        ledger.record_call(CallRecord(
            call_id="unknown", span_id=ledger.root_span_id, kind=kind, provider="custom",
            auth_mode=auth, purpose="brief", request_model="unpriced",
            outcome="ok", latency_s=0.1,
        ))
    # When
    receipt = render_receipt(ledger.summary())
    # Then
    assert expected in receipt
    if billed is not None:
        assert "$0.021 (추정)" not in receipt
    if billed is None and estimate is None and equivalent is None:
        assert "$0" not in receipt
    assert "API 환산" not in receipt if auth == "api" else "(추정)" not in receipt
    live_cost = "미확인" if expected.startswith("·") else expected.split(" + ")[0]
    assert render_call_line(ledger.calls[0], ordinal=1).endswith(live_cost)


def test_real_empty_ledger_does_not_claim_free_usage() -> None:
    # Given
    ledger = RunLedger(clock=lambda: 0.0)
    # When
    receipt = render_receipt(ledger.summary())
    # Then
    assert "LLM:" not in receipt
    assert "Jev:" not in receipt
    assert "$0" not in receipt
