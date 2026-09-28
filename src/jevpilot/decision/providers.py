import os
import time
from abc import ABC, abstractmethod

from .models import DecisionInput, DecisionOutput


class DecisionProvider(ABC):
    provider_kind = "mock"

    def __init__(self) -> None:
        self.call_count = 0
        self.total_cost_usd = 0.0

    @abstractmethod
    async def decide(self, payload: DecisionInput) -> DecisionOutput: ...


class MockDecisionProvider(DecisionProvider):
    provider_kind = "mock"

    def __init__(self, preferred_action_id: str | None = None, confidence: float = 1.0) -> None:
        super().__init__()
        self.preferred_action_id = preferred_action_id
        self.confidence = confidence

    async def decide(self, payload: DecisionInput) -> DecisionOutput:
        start = time.perf_counter()
        self.call_count += 1
        if not payload.candidates:
            raise ValueError("No action candidates available")

        action_id = self.preferred_action_id or payload.candidates[0].id
        if self.preferred_action_id and all(c.id != self.preferred_action_id for c in payload.candidates):
            action_id = payload.candidates[0].id

        return DecisionOutput(
            selected_action_id=action_id,
            confidence=self.confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            provider_name="mock",
            reason="Deterministic selection for tests",
        )


class LLMDecisionProvider(DecisionProvider):
    provider_kind = "llm"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self.api_key = os.getenv("JEVPILOT_LLM_API_KEY", "")
        self.model = model or os.getenv("JEVPILOT_LLM_MODEL", "gpt-5-mini")

    async def decide(self, payload: DecisionInput) -> DecisionOutput:
        self.call_count += 1
        if not self.api_key:
            raise RuntimeError(
                "JEVPILOT_LLM_API_KEY is required for LLMDecisionProvider. TODO: implement API integration."
            )
        raise NotImplementedError("TODO: implement LLM selector API call")


class JevDecisionProvider(DecisionProvider):
    provider_kind = "jev"

    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self.api_key = os.getenv("JEVPILOT_JEV_API_KEY", "")
        self.model = model or os.getenv("JEVPILOT_JEV_MODEL", "jev-default")

    async def decide(self, payload: DecisionInput) -> DecisionOutput:
        self.call_count += 1
        if not self.api_key:
            raise RuntimeError(
                "JEVPILOT_JEV_API_KEY is required for JevDecisionProvider. TODO: implement API integration."
            )
        raise NotImplementedError("TODO: implement Jev selector API call")
