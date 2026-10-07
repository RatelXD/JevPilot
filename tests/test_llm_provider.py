import pytest

from jevpilot.codex_client import CodexClient
from jevpilot.gateway_client import GatewayModelClient
from jevpilot.llm_provider import LLMProvider, select_llm_client
from jevpilot.model_client import ProviderError


@pytest.mark.asyncio
async def test_gateway_provider_uses_shared_api_key_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_LLM_PROVIDER", "gateway")
    monkeypatch.setenv("JEVPILOT_LLM_API_KEY", "test-gateway-key")

    # When
    selection = await select_llm_client()

    # Then
    assert selection.provider is LLMProvider.CHOSUN_GATEWAY
    assert isinstance(selection.client, GatewayModelClient)
    await selection.aclose()


@pytest.mark.asyncio
async def test_gateway_provider_fails_before_use_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_LLM_PROVIDER", "gateway")
    monkeypatch.delenv("JEVPILOT_LLM_API_KEY", raising=False)

    # When / Then
    with pytest.raises(ProviderError, match="JEVPILOT_LLM_API_KEY"):
        _ = await select_llm_client()


@pytest.mark.asyncio
async def test_chatgpt_subscription_can_be_selected_without_gateway_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_LLM_PROVIDER", "chatgpt-subscription")
    monkeypatch.delenv("JEVPILOT_LLM_API_KEY", raising=False)
    verified: list[bool] = []

    async def verify_login(_client: CodexClient) -> None:
        verified.append(True)

    monkeypatch.setattr(CodexClient, "verify_subscription_login", verify_login)

    # When
    selection = await select_llm_client()

    # Then
    assert selection.provider is LLMProvider.CODEX_SUBSCRIPTION
    assert isinstance(selection.client, CodexClient)
    assert verified == [True]
    await selection.aclose()


@pytest.mark.asyncio
async def test_unknown_provider_is_rejected_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_LLM_PROVIDER", "unknown")

    # When / Then
    with pytest.raises(ProviderError, match="JEVPILOT_LLM_PROVIDER"):
        _ = await select_llm_client()


@pytest.mark.asyncio
async def test_legacy_codex_subscription_provider_id_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_LLM_PROVIDER", "codex_subscription")

    # When / Then
    with pytest.raises(ProviderError, match="chatgpt-subscription"):
        _ = await select_llm_client()
