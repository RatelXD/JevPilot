import pytest

from jevpilot.decision import (
    DecisionInput,
    JevDecisionProvider,
    LLMDecisionProvider,
    MockDecisionProvider,
)
from jevpilot.observer import ActionCandidate, ActionType


@pytest.fixture
def sample_input() -> DecisionInput:
    return DecisionInput(
        candidates=[
            ActionCandidate(
                id="candidate_0",
                action_type=ActionType.CLICK,
                selector="#run",
                text="Run",
                enabled=True,
            )
        ],
        plan_context={"target_selector": "#run"},
    )


@pytest.mark.asyncio
async def test_mock_provider_selects_preferred_candidate(sample_input: DecisionInput) -> None:
    provider = MockDecisionProvider(preferred_action_id="candidate_0", confidence=0.9)

    result = await provider.decide(sample_input)

    assert result.selected_action_id == "candidate_0"
    assert result.confidence == 0.9
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_llm_provider_requires_api_key(sample_input: DecisionInput, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVPILOT_LLM_API_KEY", raising=False)
    provider = LLMDecisionProvider()

    with pytest.raises(RuntimeError):
        await provider.decide(sample_input)


@pytest.mark.asyncio
async def test_jev_provider_requires_api_key(sample_input: DecisionInput, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVPILOT_JEV_API_KEY", raising=False)
    provider = JevDecisionProvider()

    with pytest.raises(RuntimeError):
        await provider.decide(sample_input)
