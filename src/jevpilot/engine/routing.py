"""Multi-signal routing; a Jev done choice never proves task completion."""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevpilot.browser.port import Observation
from jevpilot.engine.brief import AnswerFields, TaskBrief
from jevpilot.engine.jev import StepDecision
from jevpilot.engine.verify import VerifyResult
from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.profile import Profile

RouteAction = Literal[
    "jev_execute", "llm_text", "llm_choose", "llm_replan", "reobserve",
    "ask_user", "llm_extract",
]
_LLM_ACTIONS: Final = frozenset({"llm_text", "llm_choose", "llm_replan", "llm_extract"})
_TARGETED: Final = frozenset({"click", "type_text", "select_option"})


class RoutingPolicy(BaseModel):
    """Initial tuning values, not calibrated probabilities; adjust after evaluation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False,
    )
    policy_version: str = Field(default="bu-1", min_length=1)
    easy_op_conf: float = Field(default=0.85, ge=0, le=1)
    easy_target_conf: float = Field(default=0.85, ge=0, le=1)
    easy_margin: float = Field(default=0.20, ge=0, le=1)
    easy_max_candidates: int = Field(default=150, ge=1)
    stall_fingerprint_repeats: int = Field(default=2, ge=1)
    failure_repeats: int = Field(default=2, ge=1)
    max_llm_calls: int = Field(default=8, ge=0)
    max_steps: int = Field(default=40, ge=1)
    max_wall_s: float = Field(default=300, gt=0)

    @classmethod
    def from_profile(cls, profile: Profile) -> "RoutingPolicy":
        return cls.model_validate(profile.policy_overrides)


@dataclass(frozen=True, slots=True)
class Route:
    level: Literal["easy", "mid", "high"]
    action: RouteAction
    reason: str


@dataclass(frozen=True, slots=True)
class RoutingSignals:
    observation: Observation | None
    decision: StepDecision
    observation_fresh: bool = True
    input_key: str | None = None
    subgoal: str = ""
    verification: VerifyResult | None = None


@dataclass(frozen=True, slots=True)
class RoutingHistory:
    """Run-local history; route appends decisions to reserve one assistance.

    The loop supplies the initial fingerprint followed by post-action fingerprints
    (including the current one), never extra reobservations. Two unchanged actions
    therefore mean three identical fingerprints. It owns the execution/failure
    counters, actual LLM call count (including brief/repair), and monotonic elapsed
    time. Reserved assistance stays consumed even if the provider fails.
    """

    fingerprints: tuple[str, ...] = ()
    consecutive_failures: int = 0
    llm_calls: int = 0
    steps: int = 0
    elapsed_s: float = 0
    routes: list[tuple[tuple[str, str], Route]] = field(default_factory=list)


def _decide(
    signals: RoutingSignals, brief: TaskBrief, history: RoutingHistory,
    policy: RoutingPolicy,
) -> Route:
    observation, decision = signals.observation, signals.decision
    if brief.missing_inputs:
        return Route("high", "ask_user", "missing_inputs")
    if history.steps >= policy.max_steps:
        return Route("high", "ask_user", "step_limit")
    if history.elapsed_s >= policy.max_wall_s:
        return Route("high", "ask_user", "wall_limit")
    if history.llm_calls > policy.max_llm_calls:
        return Route("high", "ask_user", "llm_call_limit")
    if observation is None or not (observation.elements or observation.page_text.strip()):
        return Route("mid", "reobserve", "empty_observation")
    if (
        not signals.observation_fresh
        or decision.bound_to[:2] != (observation.snapshot_id, observation.url)
    ):
        return Route("mid", "reobserve", "stale_observation")
    same = 0
    for fingerprint in reversed(history.fingerprints):
        if fingerprint != observation.fingerprint:
            break
        same += 1
    if same - 1 >= policy.stall_fingerprint_repeats:
        return Route("high", "llm_replan", "stalled_fingerprint")
    if history.consecutive_failures >= policy.failure_repeats:
        return Route("high", "llm_replan", "repeated_failure")
    if brief.multi_step and not signals.subgoal.strip():
        return Route("high", "llm_replan", "subgoal_missing")
    needs_answers = bool(
        brief.required_outputs or brief.comparison
        or any(isinstance(c, AnswerFields) for c in brief.success_conditions)
    )
    verification = signals.verification
    extraction_missing = verification is not None and verification.status != "PASS" and (
        verification.reason in {"evidence_missing", "answer_fields_missing"}
    )
    if (
        needs_answers and observation.page_text.strip()
        and (decision.operation == "done" or extraction_missing)
        and (verification is None or verification.status != "PASS")
    ):
        return Route("mid", "llm_extract", "answer_extraction_needed")
    if decision.operation == "type_text" and signals.input_key not in brief.input_values:
        return Route("mid", "llm_text", "input_value_missing")
    scores = decision.signals
    confidence: list[tuple[float | None, float]] = [
        (scores.op_confidence, policy.easy_op_conf), (scores.op_margin, policy.easy_margin),
    ]
    if decision.operation in _TARGETED:
        confidence.extend([
            (scores.target_confidence, policy.easy_target_conf),
            (scores.target_margin, policy.easy_margin),
        ])
    if scores.option_confidence is not None or scores.option_margin is not None:
        confidence.extend([
            (scores.option_confidence, policy.easy_target_conf),
            (scores.option_margin, policy.easy_margin),
        ])
    if not all(value is not None and value >= threshold for value, threshold in confidence):
        return Route("mid", "llm_choose", "confidence_or_margin")
    if scores.candidate_count > policy.easy_max_candidates or scores.truncated:
        return Route("mid", "llm_choose", "candidate_limit")
    return Route("easy", "jev_execute", "easy_signals")


def route(
    signals: RoutingSignals, brief: TaskBrief, history: RoutingHistory, *,
    ledger: RunLedger, policy: RoutingPolicy | None = None,
) -> Route:
    """Choose and record one route; ledger/policy are explicit run dependencies.

    Limits are checked before the next action/call. Exhausted LLM allowance does
    not block an LLM-free Jev step. A changed URL/fingerprint starts a new state;
    another snapshot ID alone does not. Successful assistance may be followed by
    Jev execution on the same page, but never by a second LLM assistance.
    Missing, empty, or stale observations retain the last established state.
    """
    active = policy or RoutingPolicy()
    result = _decide(signals, brief, history, active)
    observation = signals.observation
    state = history.routes[-1][0] if history.routes else ("", "")
    if (
        observation is not None
        and (observation.elements or observation.page_text.strip())
        and signals.observation_fresh
        and signals.decision.bound_to[:2] == (observation.snapshot_id, observation.url)
    ):
        state = (observation.url, observation.fingerprint)
    previous: list[RouteAction] = []
    for key, earlier in reversed(history.routes):
        if key != state:
            break
        previous.append(earlier.action)
    if result.action == "reobserve" and "reobserve" in previous:
        result = Route("high", "ask_user", "observation_unavailable")
    if result.action in _LLM_ACTIONS:
        if any(action in _LLM_ACTIONS for action in previous):
            reason = "unchanged_after_replan" if "llm_replan" in previous else "unchanged_after_assistance"
            result = Route("high", "ask_user", reason)
        elif history.llm_calls >= active.max_llm_calls:
            result = Route("high", "ask_user", "llm_call_limit")
    scores = signals.decision.signals
    ledger.event("route", {
        "policy_version": active.policy_version,
        "thresholds": active.model_dump(mode="json", exclude={"policy_version"}),
        "signals": {
            "operation": signals.decision.operation,
            "op_confidence": scores.op_confidence, "op_margin": scores.op_margin,
            "target_confidence": scores.target_confidence, "target_margin": scores.target_margin,
            "option_confidence": scores.option_confidence, "option_margin": scores.option_margin,
            "candidate_count": scores.candidate_count, "truncated": scores.truncated,
            "observation_fresh": signals.observation_fresh,
            "observation_present": observation is not None,
            "fingerprint": observation.fingerprint if observation is not None else None,
            "input_available": signals.input_key in brief.input_values,
            "missing_inputs_count": len(brief.missing_inputs),
            "multi_step": brief.multi_step, "subgoal_present": bool(signals.subgoal.strip()),
            "verify_status": signals.verification.status if signals.verification else None,
            "verify_reason": signals.verification.reason if signals.verification else None,
        },
        "history": {
            "fingerprints": list(history.fingerprints),
            "consecutive_failures": history.consecutive_failures,
            "llm_calls": history.llm_calls, "steps": history.steps,
            "elapsed_s": history.elapsed_s, "same_state_routes": list(previous),
        },
        "level": result.level, "action": result.action, "reason": result.reason,
    })
    history.routes.append((state, result))
    return result
