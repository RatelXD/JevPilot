from datetime import UTC, datetime

from pydantic import BaseModel, Field

from jevpilot.executor import ActionExecution
from jevpilot.verifier import VerificationResult


class ExecutionMetrics(BaseModel):
    task_success: bool
    total_execution_latency_ms: float
    decision_latency_ms: float
    llm_calls: int
    jev_calls: int
    actions: int
    fallbacks_replans: int
    token_cost_usd: float | None = None
    api_cost_usd: float | None = None


class ExecutionTrace(BaseModel):
    goal: str
    planner_provider: str
    decision_provider: str
    metrics: ExecutionMetrics
    action_executions: list[ActionExecution]
    verification: VerificationResult
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
