import pytest

from jevpilot.planner import LLMPlannerProvider, Plan, StaticPlannerProvider


@pytest.mark.asyncio
async def test_static_planner_returns_fixed_plan() -> None:
    expected = Plan(target_selector="#run", verification_selector="#status", expected_text="done")
    provider = StaticPlannerProvider(expected)

    result = await provider.plan("do task")

    assert result == expected
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_llm_planner_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVPILOT_LLM_API_KEY", raising=False)
    provider = LLMPlannerProvider()

    with pytest.raises(RuntimeError):
        await provider.plan("do task")
