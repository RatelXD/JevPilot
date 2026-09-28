import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

from jevpilot.decision import MockDecisionProvider
from jevpilot.planner import Plan, StaticPlannerProvider
from jevpilot.runtime import HybridBrowserAgent


async def main() -> None:
    html = Path("benchmarks/tasks/deterministic_page.html").read_text(encoding="utf-8")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(html)

        planner = StaticPlannerProvider(
            Plan(target_selector="#run", verification_selector="#status", expected_text="done")
        )
        decision = MockDecisionProvider(preferred_action_id="candidate_0", confidence=0.99)
        agent = HybridBrowserAgent(planner=planner, decision_provider=decision)

        trace = await agent.run(page, goal="run deterministic action", selectors=["#run"])
        print(trace.model_dump_json(indent=2))

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
