from pydantic import BaseModel

from jevpilot.observer import ActionCandidate


class DecisionInput(BaseModel):
    candidates: list[ActionCandidate]
    plan_context: dict[str, str]


class DecisionOutput(BaseModel):
    selected_action_id: str
    confidence: float
    latency_ms: float
    provider_name: str
    reason: str = ""
    token_cost_usd: float | None = None
