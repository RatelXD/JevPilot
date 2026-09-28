from playwright.async_api import Page

from jevpilot.planner import Plan

from .models import VerificationResult


class TextEqualsVerifier:
    async def verify(self, page: Page, plan: Plan) -> VerificationResult:
        text = (await page.locator(plan.verification_selector).first.inner_text()).strip()
        success = text == plan.expected_text
        return VerificationResult(
            success=success,
            observed_value=text,
            message="verification passed" if success else "verification failed",
        )
