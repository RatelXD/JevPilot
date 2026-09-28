from jevpilot.executor import ActionExecution
from jevpilot.telemetry import ExecutionMetrics, ExecutionTrace
from jevpilot.verifier import VerificationResult


def test_execution_trace_contains_required_metrics() -> None:
    metrics = ExecutionMetrics(
        task_success=True,
        total_execution_latency_ms=123.4,
        decision_latency_ms=12.3,
        llm_calls=1,
        jev_calls=0,
        actions=1,
        fallbacks_replans=0,
        token_cost_usd=0.001,
        api_cost_usd=0.001,
    )
    trace = ExecutionTrace(
        goal="demo",
        planner_provider="StaticPlannerProvider",
        decision_provider="MockDecisionProvider",
        metrics=metrics,
        action_executions=[
            ActionExecution(action_id="candidate_0", selector="#run", latency_ms=10.0, success=True)
        ],
        verification=VerificationResult(success=True, observed_value="done", message="ok"),
    )

    assert trace.metrics.llm_calls == 1
    assert trace.metrics.actions == 1
    assert trace.metrics.task_success is True
