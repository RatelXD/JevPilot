import time
from collections.abc import Sequence

from playwright.async_api import Page

from jevpilot.decision import DecisionInput, DecisionProvider
from jevpilot.executor import ActionExecution, PlaywrightExecutor
from jevpilot.observer import extract_action_candidates
from jevpilot.planner import PlannerProvider
from jevpilot.telemetry import ExecutionMetrics, ExecutionTrace
from jevpilot.verifier import TextEqualsVerifier, VerificationResult


class HybridBrowserAgent:
    def __init__(
        self,
        planner: PlannerProvider,
        decision_provider: DecisionProvider,
        confidence_threshold: float = 0.7,
        max_replans: int = 1,
    ) -> None:
        self.planner = planner
        self.decision_provider = decision_provider
        self.confidence_threshold = confidence_threshold
        self.max_replans = max_replans
        self.executor = PlaywrightExecutor()
        self.verifier = TextEqualsVerifier()

    async def run(self, page: Page, goal: str, selectors: Sequence[str]) -> ExecutionTrace:
        started = time.perf_counter()
        decision_latency_ms = 0.0
        fallbacks = 0
        action_executions: list[ActionExecution] = []
        verification = VerificationResult(success=False, message="not executed")

        for _ in range(self.max_replans + 1):
            plan = await self.planner.plan(goal)
            candidates = await extract_action_candidates(page, selectors)
            try:
                decision = await self.decision_provider.decide(
                    DecisionInput(candidates=candidates, plan_context=plan.model_dump(mode="json"))
                )
            except Exception:
                fallbacks += 1
                continue
            decision_latency_ms += decision.latency_ms

            if decision.confidence < self.confidence_threshold:
                fallbacks += 1
                continue

            selected = next((c for c in candidates if c.id == decision.selected_action_id), None)
            if not selected:
                fallbacks += 1
                continue

            execution = await self.executor.execute(page, selected)
            action_executions.append(execution)
            if not execution.success:
                fallbacks += 1
                continue

            verification = await self.verifier.verify(page, plan)
            if verification.success:
                break
            fallbacks += 1

        llm_calls = self.planner.call_count
        jev_calls = 0
        if self.decision_provider.provider_kind == "llm":
            llm_calls += self.decision_provider.call_count
        elif self.decision_provider.provider_kind == "jev":
            jev_calls += self.decision_provider.call_count

        metrics = ExecutionMetrics(
            task_success=verification.success,
            total_execution_latency_ms=(time.perf_counter() - started) * 1000,
            decision_latency_ms=decision_latency_ms,
            llm_calls=llm_calls,
            jev_calls=jev_calls,
            actions=len(action_executions),
            fallbacks_replans=fallbacks,
            token_cost_usd=getattr(self.decision_provider, "total_cost_usd", None),
        )
        return ExecutionTrace(
            goal=goal,
            planner_provider=self.planner.__class__.__name__,
            decision_provider=self.decision_provider.__class__.__name__,
            metrics=metrics,
            action_executions=action_executions,
            verification=verification,
        )
