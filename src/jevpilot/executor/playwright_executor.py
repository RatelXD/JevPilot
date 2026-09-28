import time

from playwright.async_api import Page

from jevpilot.observer import ActionCandidate

from .models import ActionExecution


class PlaywrightExecutor:
    async def execute(self, page: Page, action: ActionCandidate) -> ActionExecution:
        started = time.perf_counter()
        try:
            await page.locator(action.selector).first.click()
            return ActionExecution(
                action_id=action.id,
                selector=action.selector,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=True,
            )
        except Exception as exc:  # pragma: no cover
            return ActionExecution(
                action_id=action.id,
                selector=action.selector,
                latency_ms=(time.perf_counter() - started) * 1000,
                success=False,
                error=str(exc),
            )
