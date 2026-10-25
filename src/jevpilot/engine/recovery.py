"""Failure classification and recovery routing for the Jev-first loop.

Every rule is site-independent. The loop calls ``classify`` after each executed
step that did not verify, maps the failure with ``plan_recovery``, and returns
to Jev selection afterwards. Two invariants are enforced here:

- access blocks end the run as ``blocked_by_site``; nothing tries to get around them.
- an action whose result is ``uncertain`` may already have changed the page, so
  it is never executed again; recovery only re-observes and verifies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Final, Literal, assert_never

import anyio

from jevpilot.browser.port import BrowserAction, BrowserPort, ExecutionResult, Observation
from jevpilot.engine.brief import TaskBrief
from jevpilot.engine.candidates import is_password_input
from jevpilot.engine.verify import VerifyResult

FailureKind = Literal[
    "transient_page", "needs_value", "plan_error", "needs_user", "access_blocked",
    "uncertain_mutation",
]
RecoveryAction = Literal[
    "reobserve", "llm_text", "llm_replan", "ask_user", "handoff", "stop_blocked",
    "reobserve_verify",
]
Safety = Literal["allow", "confirm", "handoff"]

# Generic phrases of automation-block screens (case-insensitive), not site names.
BLOCK_PHRASES: Final = (
    "자동화", "automated", "unusual traffic", "captcha", "접근이 차단", "access denied",
)
_HTTP_BLOCK_TITLE: Final = re.compile(r"^\s*(?:http\s*)?(403|429)\b", re.IGNORECASE)
_HTTP_BLOCK_TEXT: Final = re.compile(
    r"\b403\s+forbidden\b|\b429\s+too\s+many\s+requests\b|\btoo\s+many\s+requests\b",
    re.IGNORECASE,
)
# Generic status lines such as "HTTP 403", "HTTP/1.1 429", "Error 403", "Status code: 429".
_HTTP_STATUS_TEXT: Final = re.compile(
    r"\b(?:http(?:/\d(?:\.\d)?)?(?:\s+(?:error|status))?|error|status(?:\s+code)?)"
    r"\s*[:#]?\s*(403|429)\b",
    re.IGNORECASE,
)
_TRANSIENT_ERRORS: Final = frozenset({"stale_target", "observation_failed"})
LOOP_REPEATS: Final = 3
STALL_REPEATS: Final = 2
FAILURE_REPEATS: Final = 2
SETTLE_TIMEOUT_S: Final = 2.0
SETTLE_OBSERVES: Final = 2

type ActionKey = tuple[str, int | None, str | None, str | None]


def action_key(action: BrowserAction) -> ActionKey:
    target = action.target.backend_node_id if action.target is not None else None
    return (action.kind, target, action.value, action.url)


def _loop_key(key: ActionKey) -> tuple[str, int | None, str | None]:
    """(operation, target) for the loop guard: input values are not part of the target."""
    kind, target, _value, url = key
    return (kind, target, url)


@dataclass(frozen=True, slots=True)
class StepRecord:
    fingerprint: str
    action: ActionKey
    outcome: str
    error_kind: str | None
    fingerprint_after: str | None


def record_step(
    before: Observation, action: BrowserAction, result: ExecutionResult,
    after: Observation | None,
) -> StepRecord:
    return StepRecord(
        fingerprint=before.fingerprint, action=action_key(action), outcome=result.outcome,
        error_kind=result.error_kind,
        fingerprint_after=after.fingerprint if after is not None else None,
    )


@dataclass(frozen=True, slots=True)
class FailureSignals:
    """Facts the loop already has; ``observation`` is the re-observation after the step."""

    observation: Observation | None
    action: BrowserAction | None = None
    brief: TaskBrief | None = None
    observation_stale: bool = False
    page_loading: bool = False
    safety: Safety | None = None


@dataclass(frozen=True, slots=True)
class Failure:
    kind: FailureKind
    evidence: str


@dataclass(frozen=True, slots=True)
class Recovery:
    action: RecoveryAction
    failure: Failure
    terminal_status: Literal["blocked_by_site"] | None = None


def may_execute(action: BrowserAction | None, history: Sequence[StepRecord]) -> bool:
    """False for an action that already ended ``uncertain`` anywhere in this run."""
    if action is None:
        return True
    key = action_key(action)
    return not any(step.outcome == "uncertain" and step.action == key for step in history)


def _blocked_evidence(observation: Observation) -> str | None:
    if _HTTP_BLOCK_TITLE.search(observation.title):
        return f"http_status_title:{observation.title.strip()[:40]}"
    screen = f"{observation.title}\n{observation.page_text}"
    if match := _HTTP_BLOCK_TEXT.search(screen):
        return f"http_status_text:{match.group(0).lower()}"
    if match := _HTTP_STATUS_TEXT.search(screen):
        return f"http_status:{match.group(1)}"
    folded = screen.casefold()
    for phrase in BLOCK_PHRASES:
        if phrase.casefold() in folded:
            return f"block_phrase:{phrase}"
    return None


def _auth_screen(observation: Observation) -> bool:
    return any(e.in_viewport and is_password_input(e) for e in observation.elements)


def _empty(observation: Observation | None) -> bool:
    return observation is None or (not observation.elements and not observation.page_text.strip())


def _plan_error(history: Sequence[StepRecord]) -> str | None:
    if not history:
        return None
    last = history[-1]
    same = sum(
        1 for step in history
        if step.fingerprint == last.fingerprint
        and _loop_key(step.action) == _loop_key(last.action)
    )
    if same >= LOOP_REPEATS:
        return f"same_state_action_repeated:{same}"
    recent = history[-STALL_REPEATS:]
    if len(recent) == STALL_REPEATS and all(
        step.outcome == "executed" and step.fingerprint_after == step.fingerprint
        for step in recent
    ):
        return f"no_progress:{STALL_REPEATS}"
    recent = history[-FAILURE_REPEATS:]
    if len(recent) == FAILURE_REPEATS and all(
        step.outcome != "executed" and step.error_kind == last.error_kind
        for step in recent
    ):
        return f"same_failure_repeated:{last.error_kind}"
    return None


def classify(
    execution_result: ExecutionResult | None,
    verify_result: VerifyResult | None,
    signals: FailureSignals,
    history: Sequence[StepRecord],
) -> Failure | None:
    """Classify the latest step; ``None`` means no failure, continue with Jev.

    ``history`` includes the latest step. Precedence: access block, user need,
    uncertain mutation, plan error, missing value, transient page.
    """
    observation = signals.observation
    if verify_result is not None and verify_result.status == "PASS":
        return None
    if observation is not None and (evidence := _blocked_evidence(observation)):
        return Failure("access_blocked", evidence)
    if signals.brief is not None and signals.brief.missing_inputs:
        return Failure("needs_user", "missing_inputs:" + ",".join(signals.brief.missing_inputs))
    if signals.safety in ("confirm", "handoff"):
        return Failure("needs_user", f"safety_{signals.safety}")
    if observation is not None and _auth_screen(observation):
        return Failure("needs_user", "auth_screen")
    if execution_result is not None and execution_result.outcome == "uncertain":
        return Failure("uncertain_mutation", f"uncertain:{execution_result.error_kind}")
    if evidence := _plan_error(history):
        return Failure("plan_error", evidence)
    action = signals.action
    if action is not None and action.kind == "type_text" and (
        action.value is None
        or (execution_result is not None and execution_result.error_kind == "invalid_value")
    ):
        return Failure("needs_value", "type_text_value_missing")
    if _empty(observation):
        return Failure("transient_page", "empty_observation")
    if signals.observation_stale:
        return Failure("transient_page", "stale_observation")
    if signals.page_loading:
        return Failure("transient_page", "page_loading")
    if execution_result is not None and execution_result.outcome == "failed_not_executed":
        kind = execution_result.error_kind or "not_executed"
        evidence = kind if kind in _TRANSIENT_ERRORS else f"not_executed:{kind}"
        return Failure("transient_page", evidence)
    return None


def plan_recovery(failure: Failure) -> Recovery:
    """Map a failure to its recovery; gate every later action with ``may_execute``."""
    match failure.kind:
        case "transient_page":
            return Recovery("reobserve", failure)
        case "needs_value":
            return Recovery("llm_text", failure)
        case "plan_error":
            return Recovery("llm_replan", failure)
        case "needs_user":
            ask = failure.evidence.startswith("missing_inputs:") or (
                failure.evidence == "safety_confirm"
            )
            return Recovery("ask_user" if ask else "handoff", failure)
        case "access_blocked":
            return Recovery("stop_blocked", failure, terminal_status="blocked_by_site")
        case "uncertain_mutation":
            return Recovery("reobserve_verify", failure)
    assert_never(failure.kind)


async def settle(
    browser: BrowserPort, previous: Observation | None, *,
    timeout_s: float = SETTLE_TIMEOUT_S, observes: int = SETTLE_OBSERVES,
) -> Observation | None:
    """Transient recovery: at most ``observes`` re-observations within ``timeout_s``.

    No fixed sleep: returns the first non-empty observation whose fingerprint
    differs from ``previous``, else the last one obtained (None if none arrived).
    """
    latest: Observation | None = None
    with anyio.move_on_after(timeout_s):
        for _ in range(observes):
            latest = await browser.observe()
            changed = previous is None or latest.fingerprint != previous.fingerprint
            if changed and not _empty(latest):
                break
    return latest
