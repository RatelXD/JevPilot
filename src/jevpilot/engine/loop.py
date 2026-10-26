"""Jev-first, single-action execution with code-owned verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import re
import time
from typing import Literal, Protocol, override, runtime_checkable

import anyio
from pydantic import JsonValue

from jevpilot.browser.port import BrowserAction, BrowserConfig, BrowserPort, NavigationWaiter, Observation
from jevpilot.engine.assist import llm_choose, llm_extract, llm_replan, llm_text, supplied_input_key
from jevpilot.engine.brief import AnswerFields, Extraction, FieldValue, OpenedTarget, StructuredTask, TaskBrief, build_brief
from jevpilot.engine.candidates import build_candidates
from jevpilot.engine.control import InteractiveControl, NeedsUser, RunControl, RunStopped, ScriptedControl
from jevpilot.engine.jev import GoalContext, JevSelector, JevUnavailable, select_step
from jevpilot.engine.recovery import FailureSignals, StepRecord, classify, may_execute, plan_recovery, record_step
from jevpilot.engine.routing import Route, RoutingHistory, RoutingPolicy, RoutingSignals, route
from jevpilot.engine.safety import assess
from jevpilot.engine.verify import OpenEvidence, VerifyResult, _normalize, verify
from jevpilot.ledger.models import CallRecord, CallUsage, RunSummary
from jevpilot.ledger.recorder import CallScope as LedgerScope, RunLedger
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import LLMCallRecord, LLMProvider

RunStatus = Literal[
    "verified", "unverified", "failed", "blocked_by_site", "cancelled",
    "handoff_ended", "incomplete",
]


@dataclass(frozen=True, slots=True)
class RunOutcome:
    status: RunStatus
    reason: str
    final_url: str | None
    steps: int
    extraction: Extraction | None
    receipt: RunSummary


@runtime_checkable
class _WarmProvider(Protocol):
    async def start(self) -> None: ...


class _LLMScope(LedgerScope):
    """Accept the public LLM protocol and the adapters' native ledger records."""

    @override
    def record_call(self, record: LLMCallRecord | CallRecord) -> None:
        match record:
            case CallRecord():
                self.ledger.record_call(record)
            case LLMCallRecord():
                usage = record.usage
                self.ledger.record_call(CallRecord(
                    call_id=record.call_id, span_id=self.parent_span_id, kind="llm",
                    provider=record.provider, auth_mode=record.auth_mode,
                    purpose=record.purpose, request_model=record.model_id,
                    resolved_model=record.resolved_model, attempts=record.attempt,
                    outcome="ok" if record.outcome == "ok" else (
                        "cancelled" if record.outcome == "cancelled" else "failed"
                    ),
                    error_kind=None if record.outcome == "ok" else record.outcome,
                    usage=CallUsage(
                        input_tokens=usage.input_tokens if usage else None,
                        output_tokens=usage.output_tokens if usage else None,
                        cached_input_tokens=usage.cached_input_tokens if usage else None,
                        reasoning_tokens=usage.reasoning_tokens if usage else None,
                    ),
                    latency_s=record.ended_monotonic - record.started_monotonic,
                ))


class _Limit(Exception):
    reason: str

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class _Termination:
    """Mutable exit state shared by execution, cancellation and finalization."""

    status: RunStatus = "failed"
    reason: str = "unexpected_error"
    active: BrowserAction | None = None


async def run_task(
    goal: str | StructuredTask, *, browser: BrowserPort, llm: LLMProvider,
    jev: JevSelector, control: RunControl, ledger: RunLedger, policy: RoutingPolicy,
    start_url: str | None = None, mode: Literal["adaptive", "jev_only"] = "adaptive",
) -> RunOutcome:
    """Own browser/provider lifetimes; persist a receipt on every exit.

    Assistance changes context, never executes a stale snapshot. A fresh Jev
    selection follows every assistance. A choose hint is usable once, only if
    the page and backend target identity still match after re-observation.
    """
    if ledger.sink is None:
        ledger.sink = LedgerSink(ledger.run_id)
    started = ledger.clock()
    deadline = time.monotonic() + policy.max_wall_s
    terminal = _Termination()
    observation: Observation | None = None
    extraction: Extraction | None = None
    brief: TaskBrief | None = None
    steps = 0
    reserved = 0
    history = RoutingHistory()
    records: list[StepRecord] = []
    opened: list[OpenEvidence] = []
    observed_pages: list[Observation] = []
    in_handoff = False
    original = goal if isinstance(goal, str) else goal.model_dump_json()
    goal_context: GoalContext = GoalContext(original)
    # One successful assistance earns one Jev execution, not another assistance.
    assisted_state: tuple[str, str] | None = None
    chosen: BrowserAction | None = None

    def count_calls() -> int:
        return max(reserved, sum(c.kind == "llm" for c in ledger.calls))

    def reserve() -> None:
        nonlocal reserved
        if count_calls() >= policy.max_llm_calls:
            raise _Limit("llm_call_limit")
        reserved = count_calls() + 1

    def scope() -> _LLMScope:
        current = ledger.scope(deadline=deadline, step_index=steps)
        return _LLMScope(
            ledger, current.parent_span_id, deadline, steps, reserve,
        )

    async def observe() -> Observation:
        with ledger.span("observe"):
            observed = await browser.observe()
            if not any(
                page.url == observed.url and page.page_text == observed.page_text
                for page in observed_pages
            ):
                observed_pages.append(observed.model_copy(update={"elements": []}))
                del observed_pages[:-8]
            # The adapter's page fingerprint omits private input text. Retained
            # lengths still prove a changed field state without exposing values.
            lengths = sorted(
                (field.backend_node_id, field.attributes.get("value"))
                for field in build_candidates(observed).type_text
            )
            return observed.model_copy(update={"fingerprint": hashlib.sha256(
                f"{observed.fingerprint}:{lengths!r}".encode(),
            ).hexdigest()})

    async def handoff(why: str) -> None:
        nonlocal in_handoff
        opened.clear()
        observed_pages.clear()
        # No observation or event is produced while control owns the browser.
        in_handoff = True
        await control.handoff(why)
        in_handoff = False

    def verification() -> VerifyResult:
        """Keep final-state claims current; ground comparison items per observed page."""
        if brief is None or brief.comparison is None or extraction is None or observation is None:
            return verify(observation, brief, extraction, opened=tuple(opened))
        current = verify(observation, brief.model_copy(update={
            "comparison": None,
            "success_conditions": [
                *brief.success_conditions, AnswerFields(names=[brief.comparison.field]),
            ],
        }), extraction, opened=tuple(opened))
        if current.status != "PASS" or extraction.no_result:
            return current
        sources: list[JsonValue] = []
        grounded_items: list[FieldValue] = []
        for item in extraction.compared_items:
            quote, value = _normalize(item.quote), _normalize(item.value)
            # Compared quotes may name an item whose observed list label was
            # truncated. The numeric value and name must share one observed page.
            name = quote.replace(value, "").strip(" :;-—–") if value and quote.count(value) == 1 else ""

            def grounded(page: Observation) -> bool:
                text = _normalize(page.page_text)
                if not quote or not value or value not in text:
                    return False
                if quote in text and value in quote:
                    return True
                if not name:
                    return False
                if name in text:
                    return True
                if len(name) < 10:
                    return False
                return any(
                    len(match.group()[:-3].rstrip()) >= 10
                    and name.casefold().startswith(match.group()[:-3].rstrip())
                    for match in re.finditer(
                        r"(?<!\w)" + re.escape(name[:10].casefold()) + r".*?\.{3}",
                        text.casefold(),
                    )
                )

            source = next((
                page for page in observed_pages if grounded(page)
            ), None)
            if source is None:
                return VerifyResult("FAIL", "compared_item_not_grounded")
            sources.append(source.url)
            # Only the code-checked numeric field reaches the extrema verifier.
            # Keep the original extraction, quotes and source URLs in the ledger.
            grounded_items.append(item.model_copy(update={"quote": value}))
        ledger.event("recovery", {"event": "comparison_sources", "sources": sources})
        return verify(observation.model_copy(update={
            "page_text": "\n".join(page.page_text for page in observed_pages),
        }), brief, extraction.model_copy(update={"compared_items": grounded_items}), opened=tuple(opened))

    async def interpret() -> None:
        nonlocal brief
        with ledger.span("brief"):
            if isinstance(goal, str) and policy.max_llm_calls == 0:
                raise _Limit("llm_call_limit")
            brief = await build_brief(
                original, llm, scope(),
                structured_task=goal if isinstance(goal, StructuredTask) else None,
                start_url=start_url,
            )

    async def start_browser() -> None:
        with ledger.span("observe", "browser_start"):
            await browser.start(BrowserConfig())

    async def start_provider() -> None:
        if isinstance(llm, _WarmProvider):
            with ledger.span("llm", "provider_start"):
                await llm.start()

    async def execute_loop() -> None:
        nonlocal observation, brief, steps, history
        nonlocal goal_context, extraction, assisted_state, chosen
        # Tasks inherit the run parent, so startup overlap is attributed once.
        async with anyio.create_task_group() as startup:
            startup.start_soon(start_browser)
            startup.start_soon(start_provider)
            startup.start_soon(interpret)
        assert brief is not None
        goal_context = replace(
            goal_context, brief=brief.model_dump_json(), subgoal=brief.goal_restated,
        )
        while "start_url" in brief.missing_inputs:
            key = "start_url"
            with ledger.span("wait", "missing_input"):
                answer = await control.ask(key)
            brief = brief.model_copy(update={
                "input_values": {**brief.input_values, key: answer},
                "missing_inputs": [name for name in brief.missing_inputs if name != key],
                "start_url": answer if key == "start_url" else brief.start_url,
            })
        observation = await observe()
        url = start_url or brief.start_url
        if url is not None:
            if control.stop_requested:
                raise RunStopped
            with ledger.span("execute", "navigate"):
                result = await browser.execute(BrowserAction(kind="navigate", url=url), observation)
            if result.outcome != "executed":
                terminal.status, terminal.reason = "failed", "navigation_failed"
                return
            observation = await observe()
        history = replace(history, fingerprints=(observation.fingerprint,))
        while True:
            if control.stop_requested:
                raise RunStopped
            if control.handoff_requested:
                await handoff("user_requested")
                observation = await observe()
                chosen = None
                assisted_state = None
            with ledger.span("verify"):
                verified = verification()
            if verified.status == "PASS":
                terminal.status, terminal.reason = "verified", verified.reason
                return
            if extraction is not None:
                terminal.status, terminal.reason = "unverified", verified.reason
                return
            if ledger.clock() - started >= policy.max_wall_s:
                raise _Limit("wall_limit")
            if steps >= policy.max_steps:
                raise _Limit("step_limit")
            failure = classify(None, verified, FailureSignals(observation), [])
            if failure is not None:
                recovery = plan_recovery(failure)
                if recovery.action == "stop_blocked":
                    terminal.status, terminal.reason = "blocked_by_site", failure.evidence
                    return
                if recovery.action == "handoff":
                    await handoff(failure.evidence)
                    observation = await observe()
                    continue
            goal_context = replace(
                goal_context, brief=brief.model_dump_json(),
                recent_actions=tuple(
                    f"{r.action[0]} target={r.action[1]} outcome={r.outcome}"
                    + (f" option_value={r.action[2]}" if r.action[0] == "select_option" else "")
                    for r in records[-5:]
                ),
            )
            with ledger.span("jev"):
                decision = await select_step(
                    observation, goal_context, brief.input_values, selector=jev,
                    scope=ledger.scope(deadline=deadline, step_index=steps),
                )
            action = decision.action
            key: str | None = None
            if action is not None and action.kind == "type_text" and action.target is not None:
                target = action.target
                key = supplied_input_key(target, observation, brief)
                if key is None:
                    key = f"{observation.url}#{target.backend_node_id}"
                if key in brief.input_values:
                    action = action.model_copy(update={"value": brief.input_values[key]})
            history = replace(
                history, llm_calls=count_calls(), steps=steps,
                elapsed_s=ledger.clock() - started,
            )
            with ledger.span("route"):
                selected = route(RoutingSignals(
                    observation, decision, input_key=key, subgoal=goal_context.subgoal,
                    verification=verified,
                    input_binding_unique=isinstance(goal, StructuredTask) and key in brief.input_values
                    and sum(
                        supplied_input_key(candidate, observation, brief) == key
                        for candidate in build_candidates(observation).type_text
                    ) == 1,
                    structured_navigation=isinstance(goal, StructuredTask)
                    and action is not None and action.kind == "click"
                    and action.target in build_candidates(observation).click
                    and action.target is not None
                    and action.target.attributes.get("type") != "submit"
                    and assess(action, observation).verdict == "allow",
                ), brief, history, ledger=ledger, policy=policy)
            state = (observation.url, observation.fingerprint)
            if assisted_state == state:
                # Consume a successful assistance once. Never renew its reservation.
                assisted_state = None
                if chosen is not None:
                    target = chosen.target
                    rebound = None if target is None else next((
                        e for e in observation.elements
                        if e.backend_node_id == target.backend_node_id
                        and e.tag == target.tag and e.attributes == target.attributes
                    ), None)
                    if target is None or rebound is not None:
                        action = chosen.model_copy(update={"target": rebound})
                    chosen = None
                selected = Route("easy", "jev_execute", "assistance_applied")
            else:
                chosen = None
                assisted_state = None
            if mode == "jev_only" and selected.action in {"llm_choose", "llm_replan"}:
                selected = Route("easy", "jev_execute", "jev_only")
                history.routes[-1] = (state, selected)
                ledger.event("route", {
                    "action": selected.action, "reason": selected.reason,
                    "policy_version": policy.policy_version,
                })
            match selected.action:
                case "ask_user":
                    if selected.reason.endswith("_limit"):
                        raise _Limit(selected.reason)
                    if action is not None and (safety := assess(action, observation)).verdict == "handoff":
                        await handoff(safety.reason)
                        observation = await observe()
                        continue
                    if mode == "jev_only":
                        terminal.status, terminal.reason = "failed", selected.reason
                        return
                    with ledger.span("wait", "ask_user"):
                        question = brief.missing_inputs[0] if selected.reason == "missing_inputs" else selected.reason
                        hint = await control.ask(question)
                    if selected.reason == "missing_inputs":
                        brief = brief.model_copy(update={
                            "input_values": {
                                **brief.input_values, question: hint,
                                **({key: hint} if key is not None else {}),
                            },
                            "missing_inputs": brief.missing_inputs[1:],
                        })
                    else:
                        goal_context = replace(goal_context, subgoal=hint)
                    assisted_state = state
                    observation = await observe()
                    continue
                case "reobserve":
                    observation = await observe()
                    continue
                case "llm_text" | "llm_choose" | "llm_replan" | "llm_extract":
                    if count_calls() >= policy.max_llm_calls:
                        raise _Limit("llm_call_limit")
                    with ledger.span("llm", selected.action):
                        match selected.action:
                            case "llm_text":
                                assert action is not None and action.target is not None and key is not None
                                safety = assess(action, observation)
                                if safety.verdict == "handoff":
                                    await handoff(safety.reason)
                                else:
                                    value = await llm_text(
                                        observation, brief, action.target, provider=llm, context=scope(),
                                    )
                                    brief = brief.model_copy(update={
                                        "input_values": {**brief.input_values, key: value.value},
                                    })
                            case "llm_choose":
                                chosen = await llm_choose(
                                    observation, brief, provider=llm, context=scope(),
                                    recent_actions=goal_context.recent_actions,
                                )
                                if chosen.kind == "type_text" and chosen.value is None and chosen.target is not None:
                                    safety = assess(chosen, observation)
                                    if safety.verdict == "handoff":
                                        await handoff(safety.reason)
                                        chosen = None
                                    else:
                                        value = await llm_text(
                                            observation, brief, chosen.target,
                                            provider=llm, context=scope(),
                                        )
                                        chosen = chosen.model_copy(update={"value": value.value})
                            case "llm_replan":
                                patch = await llm_replan(observation, goal_context, provider=llm, context=scope())
                                goal_context = patch.apply(goal_context)
                            case "llm_extract":
                                extraction = await llm_extract(
                                    observation, brief, provider=llm, context=scope(),
                                    observed_pages=tuple(observed_pages),
                                )
                                ledger.event("recovery", {
                                    "event": "extraction_response",
                                    "url": observation.url,
                                    "raw_output": extraction.model_dump_json(),
                                })
                    assisted_state = state
                    observation = await observe()
                    continue
                case "jev_execute":
                    pass
            if action is None:
                terminal.status, terminal.reason = "failed", "done_without_evidence"
                return
            if action.kind == "type_text" and action.value is None:
                terminal.status, terminal.reason = "incomplete", "input_value_missing"
                return
            if not may_execute(action, records):
                terminal.status, terminal.reason = "incomplete", "uncertain_mutation"
                return
            safety = assess(action, observation)
            if safety.verdict == "handoff":
                await handoff(safety.reason)
                observation = await observe()
                continue
            if safety.verdict == "confirm":
                with ledger.span("confirm"):
                    approved = await control.confirm(safety.reason)
                if not approved:
                    terminal.status = "cancelled" if control.stop_requested else "failed"
                    terminal.reason = "stopped" if control.stop_requested else "confirmation_denied"
                    return
            if control.stop_requested:
                raise RunStopped
            if control.handoff_requested:
                continue
            before = observation
            opened.clear()
            steps += 1
            terminal.active = action
            with ledger.span("execute", action.kind):
                result = await browser.execute(action, observation)
            terminal.active = None
            # Input values remain length-only. Public option values are discrete
            # observed choices, not user-entered text.
            ledger.event("recovery", {
                "event": "execution", "operation": action.kind,
                "target": action.target.backend_node_id if action.target else None,
                "value": action.value,
                "observed_options": [
                    option.value for option in action.target.options
                ] if action.kind == "select_option" and action.target else [],
                "outcome": result.outcome, "error_kind": result.error_kind,
            })
            observation = await observe()
            if result.outcome == "executed" and action.kind == "click":
                opened.append(OpenEvidence(before, action, observation, result))
                if result.navigation_id is not None and isinstance(browser, NavigationWaiter):
                    with ledger.span("wait", "click_navigation"):
                        await browser.wait_for_navigation(result.navigation_id)
                    observation = await observe()
                # Persist semantic binding, not private goal/input/target text.
                ledger.event("recovery", {
                    "event": "opening_evidence",
                    "conditions": [condition.model_dump(mode="json") for condition in brief.success_conditions],
                    "target": action.target.backend_node_id if action.target else None,
                    "matches": [
                        {
                            "target_matches": action.target is not None and (
                                _normalize(condition.text).casefold() in _normalize(action.target.text).casefold()
                            ),
                            "page_matches": _normalize(condition.text).casefold() in _normalize(observation.page_text).casefold(),
                        }
                        for condition in brief.success_conditions if isinstance(condition, OpenedTarget)
                    ],
                    "before_url": before.url, "first_after_url": opened[-1].after.url,
                    "current_url": observation.url,
                    "navigation_id": result.navigation_id, "document_id": observation.document_id,
                })
                if result.navigation_id is not None and observation.document_id != result.navigation_id:
                    terminal.status, terminal.reason = "incomplete", "navigation_unsettled"
                    return
            records.append(record_step(before, action, result, observation))
            history = replace(
                history, fingerprints=(*history.fingerprints, observation.fingerprint),
                consecutive_failures=history.consecutive_failures + 1 if result.outcome != "executed" else 0,
            )
            with ledger.span("verify"):
                verified = verify(observation, brief, opened=tuple(opened))
            if verified.status == "PASS":
                terminal.status, terminal.reason = "verified", verified.reason
                return
            with ledger.span("recovery"):
                failure = classify(result, verified, FailureSignals(observation, action, brief), records)
                if failure is not None:
                    recovery = plan_recovery(failure)
                    ledger.event("recovery", {
                        "failure": failure.kind, "reason": failure.evidence,
                        "action": recovery.action,
                    })
                    if recovery.action == "stop_blocked":
                        terminal.status, terminal.reason = "blocked_by_site", failure.evidence
                        return
                    if recovery.action == "reobserve_verify":
                        observation = await observe()
                        with ledger.span("verify"):
                            verified = verify(observation, brief, opened=tuple(opened))
                        terminal.status = "verified" if verified.status == "PASS" else "incomplete"
                        terminal.reason = verified.reason if terminal.status == "verified" else "uncertain_mutation"
                        return
                    if recovery.action == "handoff":
                        await handoff(failure.evidence)
                        observation = await observe()
                    elif recovery.action == "reobserve":
                        observation = await observe()
                    # plan_error is handled by the next Jev + routing decision.

    async def stop_watcher(cancel: anyio.CancelScope) -> None:
        if isinstance(control, InteractiveControl):
            await control.stop_event.wait()
            cancel.cancel()

    try:
        with anyio.move_on_after(policy.max_wall_s) as budget:
            try:
                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(stop_watcher, tasks.cancel_scope)
                    await execute_loop()
                    tasks.cancel_scope.cancel()
                if control.stop_requested:
                    terminal.status, terminal.reason = "cancelled", "stopped"
            except* (_Limit, LLMError) as errors:
                failures: list[BaseException] = [errors]
                while failures:
                    error = failures.pop()
                    match error:
                        case BaseExceptionGroup():
                            failures.extend(error.exceptions)
                        case _Limit():
                            terminal.status, terminal.reason = "incomplete", error.reason
                        case LLMError():
                            terminal.status = "cancelled" if error.kind == "cancelled" else "failed"
                            terminal.reason = f"llm_{error.kind}"
                        case _:
                            raise error
            except* JevUnavailable:
                terminal.status, terminal.reason = "failed", "jev_unavailable"
            except* NeedsUser:
                terminal.status = "handoff_ended" if in_handoff else "incomplete"
                terminal.reason = "needs_user"
            except* RunStopped:
                terminal.status, terminal.reason = "cancelled", "stopped"
            except* (RuntimeError, OSError):
                # BrowserPort exposes startup/transport failures as exceptions.
                # Do not persist external exception text (it may contain inputs).
                terminal.status, terminal.reason = "failed", "browser_unavailable"
        if budget.cancel_called:
            terminal.status, terminal.reason = "incomplete", "wall_limit"
    except anyio.get_cancelled_exc_class():
        terminal.status, terminal.reason = "cancelled", "cancelled"
    finally:
        with anyio.CancelScope(shield=True):
            if terminal.active is not None:
                ledger.event("recovery", {
                    "event": "execution", "operation": terminal.active.kind, "outcome": "uncertain",
                    "reason": "interrupted_after_dispatch",
                })
            try:
                with ledger.span("wait", "cleanup"):
                    try:
                        await llm.aclose()
                    finally:
                        await browser.close()
            finally:
                ledger.event("recovery", {
                    "event": "run_outcome", "status": terminal.status, "reason": terminal.reason,
                    "final_url": observation.url if observation else None, "steps": steps,
                })
                _ = ledger.finish(
                    "ok" if terminal.status == "verified" else "cancelled" if terminal.status == "cancelled" else "failed",
                )
    return RunOutcome(
        terminal.status, terminal.reason, observation.url if observation else None,
        steps, extraction, ledger.summary(),
    )


def main() -> int:
    """Explicit live smoke entry point; ordinary tests never call providers."""
    import argparse

    from jevpilot.browser.env import prepare_browser_use_environment
    prepare_browser_use_environment()
    from jevpilot.browser.browser_use_adapter import BrowserUseAdapter
    from jevpilot.llm.codex_app_server import CodexAppServerProvider
    from jevpilot.llm.types import ModelRef

    parser = argparse.ArgumentParser()

    class Arguments(argparse.Namespace):
        goal: str = ""
        start_url: str = ""
        max_steps: int = 40
        evidence_root: Path | None = None

    _ = parser.add_argument("--smoke", action="store_true", required=True)
    _ = parser.add_argument("--goal", required=True)
    _ = parser.add_argument("--start-url", required=True)
    _ = parser.add_argument("--max-steps", type=int, default=40)
    _ = parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args(namespace=Arguments())

    async def smoke() -> int:
        ledger = RunLedger()
        ledger.sink = LedgerSink(ledger.run_id, root=args.evidence_root)
        outcome = await run_task(
            args.goal, browser=BrowserUseAdapter(),
            llm=CodexAppServerProvider(ModelRef(
                "chatgpt_subscription", "subscription", "codex-default", "low",
            )),
            jev=JevSelector(), control=ScriptedControl(), ledger=ledger,
            policy=RoutingPolicy(max_steps=args.max_steps), start_url=args.start_url,
        )
        print(f"status={outcome.status} reason={outcome.reason} final_url={outcome.final_url}")
        print(f"artifacts={ledger.sink.directory}")
        print("cleanup: browser closed; provider closed and owned processes reaped")
        return 0 if outcome.status in {"verified", "incomplete"} else 1

    return anyio.run(smoke)


if __name__ == "__main__":
    raise SystemExit(main())
