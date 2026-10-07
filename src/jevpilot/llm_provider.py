"""Select the explicitly configured LLM transport."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never, final

from jevpilot.codex_client import CodexClient
from jevpilot.gateway_client import GatewayModelClient
from jevpilot.model_client import ProviderError
from jevpilot.privacy import PrivacyPolicy


class LLMProvider(StrEnum):
    """Supported LLM account paths."""

    CHOSUN_GATEWAY = "gateway"
    CODEX_SUBSCRIPTION = "chatgpt-subscription"


@final
@dataclass(frozen=True, slots=True)
class LLMClientSelection:
    """Resolved provider and its owned structured client."""

    provider: LLMProvider
    client: GatewayModelClient | CodexClient

    async def aclose(self) -> None:
        """Close only provider transports that own a persistent resource."""
        match self.client:
            case GatewayModelClient():
                await self.client.aclose()
                return
            case CodexClient():
                return
        assert_never(self.client)


async def select_llm_client(
    *, provider: LLMProvider | None = None
) -> LLMClientSelection:
    """Build one selected provider without falling back between accounts."""
    if provider is None:
        raw_provider = os.getenv(
            "JEVPILOT_LLM_PROVIDER", LLMProvider.CHOSUN_GATEWAY
        ).strip()
        try:
            selected_provider = LLMProvider(raw_provider)
        except ValueError:
            raise ProviderError(
                kind="configuration",
                call_id="llm-provider",
                retryable=False,
                message="JEVPILOT_LLM_PROVIDER must be gateway or chatgpt-subscription",
            ) from None
    else:
        selected_provider = provider

    match selected_provider:
        case LLMProvider.CHOSUN_GATEWAY:
            api_key = os.getenv("JEVPILOT_LLM_API_KEY", "").strip()
            if not api_key:
                raise ProviderError(
                    kind="configuration",
                    call_id="llm-provider",
                    retryable=False,
                    message="JEVPILOT_LLM_API_KEY must be set for gateway",
                )
            client = GatewayModelClient(
                api_key=api_key,
                privacy_policy=PrivacyPolicy(),
            )
            return LLMClientSelection(provider=selected_provider, client=client)
        case LLMProvider.CODEX_SUBSCRIPTION:
            client = CodexClient(privacy=PrivacyPolicy())
            await client.verify_subscription_login()
            return LLMClientSelection(provider=selected_provider, client=client)
    assert_never(selected_provider)
