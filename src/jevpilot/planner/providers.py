import os
from abc import ABC, abstractmethod

from .models import Plan


class PlannerProvider(ABC):
    def __init__(self) -> None:
        self.call_count = 0

    @abstractmethod
    async def plan(self, goal: str) -> Plan: ...


class StaticPlannerProvider(PlannerProvider):
    def __init__(self, plan: Plan) -> None:
        super().__init__()
        self._plan = plan

    async def plan(self, goal: str) -> Plan:
        self.call_count += 1
        return self._plan


class LLMPlannerProvider(PlannerProvider):
    def __init__(self, model: str | None = None) -> None:
        super().__init__()
        self.api_key = os.getenv("JEVPILOT_LLM_API_KEY", "")
        self.model = model or os.getenv("JEVPILOT_LLM_MODEL", "gpt-5-mini")

    async def plan(self, goal: str) -> Plan:
        self.call_count += 1
        if not self.api_key:
            raise RuntimeError(
                "JEVPILOT_LLM_API_KEY is required for LLMPlannerProvider. TODO: implement API integration."
            )
        raise NotImplementedError("TODO: implement frontier LLM planning API call")
