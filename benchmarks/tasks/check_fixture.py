"""Capture the local catalog's real browser surface without model calls."""

import json
from pathlib import Path

import anyio
from playwright.async_api import async_playwright

from benchmarks.tasks.tasks import evaluate_catalog, fixture_server


async def main() -> None:
    output = Path("artifacts/fixture-qa")
    output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, str | int | bool]] = []
    with fixture_server() as origin:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                for name, width, height in (("desktop", 1280, 800), ("mobile", 390, 844)):
                    context = await browser.new_context(viewport={"width": width, "height": height})
                    try:
                        page = await context.new_page()
                        _ = await page.goto(f"{origin}/list.html", wait_until="domcontentloaded")
                        _ = await page.screenshot(path=str(output / f"{name}-list.png"))
                        async with page.expect_navigation(wait_until="domcontentloaded"):
                            await page.get_by_role("link", name="Aurora", exact=True).click()
                        evaluation = await evaluate_catalog(page)
                        _ = await page.screenshot(path=str(output / f"{name}-detail.png"))
                        if evaluation.status != "passed":
                            raise RuntimeError("Catalog URL and item ID verification failed")
                        receipts.append({
                            "viewport": name, "width": width, "height": height,
                            "action": 'get_by_role("link", name="Aurora", exact=True).click()',
                            "url": page.url, "item_id": await page.locator("#item-id").inner_text(),
                            "passed": True, "model_calls": 0,
                        })
                    finally:
                        await context.close()
            finally:
                await browser.close()
    report = {"scenarios": receipts, "cleanup": "browser, contexts, HTTP server and thread closed"}
    _ = (output / "receipt.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    anyio.run(main)
