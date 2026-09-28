from pydantic import BaseModel


class ActionExecution(BaseModel):
    action_id: str
    selector: str
    latency_ms: float
    success: bool
    error: str | None = None
