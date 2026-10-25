from dataclasses import replace
from pathlib import Path
from typing import override

import pytest

from jevpilot.llm import registry
from jevpilot.llm.api_providers import ApiKeyProvider, ApiProviderId
from jevpilot.llm.codex_app_server import CodexAppServerProvider
from jevpilot.llm.codex_exec_adapter import CodexExecProvider
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import CodexTransport, Profile, load_profile
from jevpilot.llm.types import Capabilities, ModelRef, ProviderId
from jevpilot.model_catalog import ChatGPTModel
from jevpilot.shell import model_menu as menu


def input_choice(monkeypatch: pytest.MonkeyPatch, choice: str) -> None:
    def answer(_prompt: str) -> str:
        return choice

    monkeypatch.setattr("builtins.input", answer)


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("JEVPILOT_OPENAI_API_KEY", "synthetic-openai")
    monkeypatch.setenv("JEVPILOT_ANTHROPIC_API_KEY", "synthetic-anthropic")
    return tmp_path


def profile() -> Profile:
    return Profile(
        "subscription", ModelRef("chatgpt_subscription", "subscription", "first", "high", True),
        codex_transport="exec", policy_overrides={"max_steps": 7},
    )


async def catalog() -> tuple[ChatGPTModel, ...]:
    return (
        ChatGPTModel("first", "First", True, True, ("low", "high")),
        ChatGPTModel("second", "Second", False, False, ("medium",)),
    )


@pytest.mark.parametrize("provider_id", ["openai", "anthropic"])
async def test_registry_api_uses_existing_factory(
    provider_id: ProviderId,
) -> None:
    # Given
    selected = Profile("api", ModelRef(provider_id, "api", "model"))
    # When
    provider = registry.build_provider(selected)
    # Then
    try:
        assert isinstance(provider, ApiKeyProvider)
        assert provider.ref == selected.model
    finally:
        await provider.aclose()


@pytest.mark.parametrize("transport,expected", [
    ("app_server", CodexAppServerProvider), ("exec", CodexExecProvider),
])
async def test_registry_subscription_uses_selected_transport(
    transport: CodexTransport,
    expected: type[CodexAppServerProvider] | type[CodexExecProvider],
) -> None:
    # Given
    selected = replace(profile(), codex_transport=transport)
    # When
    provider = registry.build_provider(selected)
    # Then
    assert isinstance(provider, expected)
    assert provider.ref == selected.model
    await provider.aclose()


@pytest.mark.parametrize("provider_id", [
    "google", "openrouter", "openai_compatible", "chosun_gateway", "chatgpt_subscription",
])
def test_registry_rejects_unsupported_api_before_factory(
    monkeypatch: pytest.MonkeyPatch, provider_id: ProviderId,
) -> None:
    # Given
    selected = Profile("api", ModelRef(provider_id, "api", "m"))

    def forbidden(_ref: ModelRef):
        raise AssertionError("factory must not run")

    monkeypatch.setattr(registry, "create_api_provider", forbidden)
    # When / Then
    with pytest.raises(LLMError) as raised:
        _ = registry.build_provider(selected)
    assert raised.value.kind == "configuration"


@pytest.mark.parametrize("selected", [
    Profile("subscription", ModelRef("openai", "subscription", "m")),
    Profile("subscription", ModelRef("anthropic", "subscription", "m")),
    Profile("api", ModelRef("openai", "subscription", "m")),
    Profile("subscription", ModelRef("chatgpt_subscription", "api", "m")),
])
def test_registry_mismatched_auth_never_falls_back(selected: Profile) -> None:
    # Given / When / Then
    with pytest.raises(LLMError) as raised:
        _ = registry.build_provider(selected)
    assert raised.value.kind == "configuration"


async def test_subscription_capabilities_are_selected_catalog_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(menu, "list_chatgpt_models", catalog)
    selected = replace(profile(), model=replace(profile().model, model_id="second"))
    # When
    capabilities = await menu.model_capabilities(selected)
    # Then
    assert capabilities.reasoning_efforts == ("medium",)
    assert not capabilities.supports_fast
    assert capabilities.structured_output == "native"
    assert not capabilities.images


async def test_model_choice_preserves_auth_transport_and_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(menu, "list_chatgpt_models", catalog)
    input_choice(monkeypatch, "2")
    current = profile()
    # When
    selected = await menu.select_model(current, emit=lambda _text: None)
    # Then
    assert selected == load_profile()
    assert selected.model.model_id == "second"
    assert selected.model.reasoning_effort is None and not selected.model.fast
    assert selected.auth_mode == current.auth_mode
    assert selected.model.provider == current.model.provider
    assert selected.codex_transport == "exec"
    assert selected.policy_overrides == current.policy_overrides


@pytest.mark.parametrize("choice", ["", "0", "3", "openai/model", "other", "-1"])
async def test_model_cancel_or_invisible_selection_does_not_save(
    home: Path, monkeypatch: pytest.MonkeyPatch, choice: str,
) -> None:
    # Given
    monkeypatch.setattr(menu, "list_chatgpt_models", catalog)
    input_choice(monkeypatch, choice)
    current = profile()
    # When
    result = await menu.select_model(current, emit=lambda _text: None)
    # Then
    assert result == current
    assert not (home / ".jevpilot" / "profile.json").exists()


def test_effort_cycles_supported_values_and_preserves_profile() -> None:
    # Given
    current = profile()
    capabilities = Capabilities("prompted", True, ("low", "high"), True)
    # When
    result = menu.cycle_effort(current, capabilities)
    # Then
    assert result == replace(current, model=replace(current.model, reasoning_effort="low"))
    assert result == load_profile()


@pytest.mark.parametrize("supported", [False, True])
def test_fast_only_changes_supported_subscription(supported: bool) -> None:
    # Given
    current = profile()
    capabilities = Capabilities("prompted", True, (), supported)
    # When
    result = menu.set_fast(current, capabilities, False)
    # Then
    assert result == (
        replace(current, model=replace(current.model, fast=False)) if supported else current
    )


def test_empty_efforts_and_api_fast_are_noop(home: Path) -> None:
    # Given
    current = Profile("api", ModelRef("openai", "api", "model"))
    capabilities = Capabilities("native", True, (), True)
    # When
    effort = menu.cycle_effort(current, capabilities)
    fast = menu.set_fast(current, capabilities)
    # Then
    assert effort == fast == current
    assert not (home / ".jevpilot" / "profile.json").exists()


@pytest.mark.parametrize("action", ["effort", "fast", "select"])
async def test_failed_save_returns_current(
    monkeypatch: pytest.MonkeyPatch, action: str,
) -> None:
    # Given
    current = profile()
    capabilities = Capabilities("prompted", True, ("low", "high"), True)

    def deny(_profile: Profile) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(menu, "save_profile", deny)
    monkeypatch.setattr(menu, "list_chatgpt_models", catalog)
    input_choice(monkeypatch, "2")
    # When
    match action:
        case "effort":
            result = menu.cycle_effort(current, capabilities)
        case "fast":
            result = menu.set_fast(current, capabilities)
        case _:
            result = await menu.select_model(current, emit=lambda _text: None)
    # Then
    assert result == current


@pytest.mark.parametrize("provider_id", ["openai", "anthropic"])
async def test_api_menu_uses_current_provider_and_closes_temporary_clients(
    monkeypatch: pytest.MonkeyPatch, provider_id: ProviderId,
) -> None:
    # Given
    selected = Profile(
        "api", ModelRef(provider_id, "api", "old", "high", base_url="https://local.invalid"),
        codex_transport="exec", policy_overrides={"max_steps": 7},
    )
    created: list[ApiKeyProvider] = []
    closed: list[ModelRef] = []

    class ObservedProvider(ApiKeyProvider):
        @override
        async def aclose(self) -> None:
            await super().aclose()
            closed.append(self.ref)

    def factory(ref: ModelRef) -> ApiKeyProvider:
        provider = ObservedProvider(ref, "synthetic")
        created.append(provider)
        return provider

    async def listed(
        provider: ApiProviderId, key: str, base_url: str | None,
    ) -> list[str]:
        assert provider == provider_id
        assert key == f"synthetic-{provider_id}"
        assert base_url == selected.model.base_url
        return ["visible"]

    monkeypatch.setattr(registry, "create_api_provider", factory)
    monkeypatch.setattr(menu, "list_models", listed)
    input_choice(monkeypatch, "1")
    # When
    result = await menu.select_model(selected, emit=lambda _text: None)
    # Then
    assert result == replace(selected, model=replace(selected.model, model_id="visible"))
    assert result == load_profile()
    assert len(created) == len(closed)
    assert all(provider.ref in closed for provider in created)
