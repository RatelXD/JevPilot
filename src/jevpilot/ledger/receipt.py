"""Human-readable Korean rendering for run and call receipts."""

from collections.abc import Iterable
from decimal import Decimal

from jevpilot.ledger.models import Aggregate, CallRecord, RunSummary


def _money(value: Decimal) -> str:
    return f"${value:.6f}".rstrip("0").rstrip(".")


def _cost(aggregate: Aggregate, label: str) -> str:
    if aggregate.known_count == 0:
        return "미확인"
    result = f"{_money(aggregate.known_sum)} ({label})"
    if aggregate.unknown_count:
        result += f" + 미확인 {aggregate.unknown_count}건"
    return result


def _tokens(aggregate: Aggregate) -> str:
    if aggregate.unknown_count and aggregate.known_sum == 0:
        return "미확인"
    result = f"{aggregate.known_sum:,}"
    if aggregate.unknown_count:
        result += f" + 미확인 {aggregate.unknown_count}건"
    return result


def _has_cost(aggregate: Aggregate) -> bool:
    return aggregate.unknown_count > 0 or aggregate.known_count > 0


def render_receipt(
    summary: RunSummary, *, outcome: str = "미검증", answer_summary: str = "",
    evidence_quote: str | None = None, calls: Iterable[CallRecord] = (),
) -> str:
    """Render stable Korean receipt fields from a ledger summary."""
    durations = summary.duration_s
    llm = summary.by_kind["llm"]
    jev = summary.by_kind["jev"]
    call_records = list(calls)
    llm_calls = [record for record in call_records if record.kind == "llm"]
    lines = [
        f"결과: {outcome} — {answer_summary}",
        (
            f"시간: 총 {summary.elapsed_s:.1f}초 | "
            f"관찰 {durations.get('observe', 0):.1f} · "
            f"Jev {durations.get('jev', 0):.1f}({jev.calls}회) · "
            f"LLM {durations.get('llm', 0):.1f}({llm.calls}회) · "
            f"실행 {durations.get('execute', 0):.1f} · "
            f"대기/복구 {durations.get('wait', 0) + durations.get('recovery', 0):.1f} "
            f"(병렬 겹침 {summary.overlap_s:.1f}초 별도)"
        ),
    ]
    if evidence_quote:
        lines.insert(1, f'근거: "{evidence_quote[:120]}"')
    if llm.calls == 0:
        cost = "미확인"
    elif llm.billed_usd.known_count:
        cost = _cost(llm.billed_usd, "청구")
    elif _has_cost(llm.api_estimate_usd) and _has_cost(llm.api_equivalent_usd):
        cost = (
            f"{_cost(llm.api_estimate_usd, '추정')} + "
            f"{_cost(llm.api_equivalent_usd, 'API 환산')}"
        )
    elif _has_cost(llm.api_estimate_usd):
        cost = _cost(llm.api_estimate_usd, "추정")
    elif _has_cost(llm.api_equivalent_usd):
        cost = _cost(llm.api_equivalent_usd, "API 환산")
    else:
        cost = _cost(llm.billed_usd, "청구") if _has_cost(llm.billed_usd) else "미확인"
    if llm_calls:
        latest = llm_calls[-1]
        identity = (
            f"{latest.provider}/{latest.resolved_model or latest.request_model} · "
            f"{'구독' if latest.auth_mode == 'subscription' else 'API'} · "
        )
    else:
        identity = ""
    if llm.calls:
        billed = (
            _cost(llm.billed_usd, "청구")
            if llm.billed_usd.known_count
            else "미확인"
        )
        lines.append(
            "LLM: "
            + identity
            + f"토큰 입력 {_tokens(llm.input_tokens)} / 출력 {_tokens(llm.output_tokens)} · "
            + f"{cost} · 청구액 {billed}"
        )
    if jev.calls:
        lines.append(
            "Jev: "
            + f"{jev.calls}회 · 토큰 입력 {_tokens(jev.input_tokens)} · "
            + (
                _cost(jev.billed_usd, "청구")
                if jev.billed_usd.known_count
                else _cost(jev.api_estimate_usd, "추정")
            )
        )
    reasons: dict[str, int] = {}
    for record in llm_calls:
        reasons[record.purpose] = reasons.get(record.purpose, 0) + 1
    if reasons:
        lines.append("LLM 개입 이유: " + ", ".join(f"{key} {count}" for key, count in reasons.items()))
    return "\n".join(lines)


def render_call_line(record: CallRecord, ordinal: int) -> str:
    """Render one real-time call line without substituting unknown costs."""
    if record.kind == "jev":
        prefix = f"Jev #{ordinal}"
        detail = f"{record.latency_s:.2f}초 · 입력 {_tokens_for_call(record.usage.input_tokens)}"
        cost = record.billed_usd if record.billed_usd is not None else record.api_estimate_usd
        label = "청구" if record.billed_usd is not None else "추정"
    else:
        prefix = f"LLM {record.purpose}"
        detail = (
            f"{record.latency_s:.1f}초 · 입력 {_tokens_for_call(record.usage.input_tokens)}"
            f" / 출력 {_tokens_for_call(record.usage.output_tokens)}"
        )
        if record.billed_usd is not None:
            cost, label = record.billed_usd, "청구"
        elif record.auth_mode == "subscription":
            cost, label = record.api_equivalent_usd, "API 환산"
        else:
            cost, label = record.api_estimate_usd, "추정"
    rendered_cost = "미확인" if cost is None else f"{_money(cost)} ({label})"
    return f"{prefix} {detail} · {rendered_cost}"


def _tokens_for_call(value: int | None) -> str:
    return "미확인" if value is None else f"{value:,}"
