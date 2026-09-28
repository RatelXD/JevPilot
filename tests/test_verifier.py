import pytest

from jevpilot.planner import Plan
from jevpilot.verifier import TextEqualsVerifier


class FakeLocator:
    def __init__(self, text: str) -> None:
        self._text = text

    @property
    def first(self) -> "FakeLocator":
        return self

    async def inner_text(self) -> str:
        return self._text


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self._text)


@pytest.mark.asyncio
async def test_text_equals_verifier() -> None:
    verifier = TextEqualsVerifier()
    plan = Plan(target_selector="#run", verification_selector="#status", expected_text="done")

    result = await verifier.verify(FakePage("done"), plan)

    assert result.success is True
    assert result.observed_value == "done"
