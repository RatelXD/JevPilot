from collections.abc import Sequence

from playwright.async_api import Page

from .models import ActionCandidate, ActionType


async def extract_action_candidates(page: Page, selectors: Sequence[str]) -> list[ActionCandidate]:
    candidates: list[ActionCandidate] = []
    for index, selector in enumerate(selectors):
        locator = page.locator(selector).first
        if await locator.count() == 0:
            continue
        if not await locator.is_visible():
            continue
        text = (await locator.inner_text()).strip()
        candidates.append(
            ActionCandidate(
                id=f"candidate_{index}",
                action_type=ActionType.CLICK,
                selector=selector,
                text=text,
                enabled=await locator.is_enabled(),
            )
        )
    return candidates
