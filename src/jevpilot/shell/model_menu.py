"""Model selection restricted to the current provider and authentication mode."""

from collections.abc import Callable
from dataclasses import replace

import anyio
from anyio.to_thread import run_sync

from jevpilot.llm.api_providers import list_models
from jevpilot.llm.credentials import get_api_key, get_base_url
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import Profile, save_profile
from jevpilot.llm.registry import build_provider
from jevpilot.llm.types import Capabilities
from jevpilot.model_catalog import list_chatgpt_models
from jevpilot.model_client import ProviderError


async def model_capabilities(profile: Profile) -> Capabilities:
    provider = build_provider(profile)
    try:
        match profile.auth_mode:
            case "api":
                return provider.capabilities()
            case "subscription":
                models = await list_chatgpt_models()
                for model in models:
                    if model.model_id == profile.model.model_id:
                        return replace(
                            provider.capabilities(),
                            reasoning_efforts=model.reasoning_efforts,
                            supports_fast=model.supports_fast,
                        )
                raise LLMError(
                    "configuration", retryable=False,
                    message="selected model is not in the subscription catalog",
                )
    finally:
        with anyio.CancelScope(shield=True):
            await provider.aclose()


def _persist(current: Profile, selected: Profile) -> Profile:
    try:
        save_profile(selected)
    except (OSError, ValueError):
        return current
    return selected


def cycle_effort(profile: Profile, capabilities: Capabilities) -> Profile:
    efforts = capabilities.reasoning_efforts
    if not efforts:
        return profile
    effort = profile.model.reasoning_effort
    index = (efforts.index(effort) + 1) % len(efforts) if effort in efforts else 0
    return _persist(
        profile, replace(profile, model=replace(profile.model, reasoning_effort=efforts[index])),
    )


def set_fast(
    profile: Profile, capabilities: Capabilities, enabled: bool | None = None,
) -> Profile:
    if (
        profile.auth_mode != "subscription"
        or profile.model.auth_mode != "subscription"
        or profile.model.provider != "chatgpt_subscription"
        or not capabilities.supports_fast
    ):
        return profile
    return _persist(profile, replace(
        profile, model=replace(
            profile.model, fast=not profile.model.fast if enabled is None else enabled,
        ),
    ))


async def select_model(
    current: Profile, *, emit: Callable[[str], None] = print,
) -> Profile:
    try:
        # Validate the complete profile before any catalog request.
        provider = build_provider(current)
        with anyio.CancelScope(shield=True):
            await provider.aclose()
        emit(f"{current.model.provider} ({current.auth_mode})")
        match current.auth_mode, current.model.provider:
            case "api", ("openai" | "anthropic") as provider_id:
                key = get_api_key(provider_id)
                if not key:
                    raise LLMError("auth_missing", retryable=False, message="API key is missing")
                ids = await list_models(
                    provider_id, key, current.model.base_url or get_base_url(provider_id),
                )
                capabilities = provider.capabilities()
                emit(
                    f"추론: {', '.join(capabilities.reasoning_efforts) or 'unsupported'}"
                    + " · Fast=unsupported",
                )
                for index, model_id in enumerate(ids, 1):
                    emit(f"{index}) {model_id}")
                models = ()
            case "subscription", "chatgpt_subscription":
                models = await list_chatgpt_models()
                ids = [model.model_id for model in models]
                for index, model in enumerate(models, 1):
                    emit(
                        f"{index}) {model.model_id} · {', '.join(model.reasoning_efforts)}"
                        + f" · Fast={'on' if model.supports_fast else 'unsupported'}",
                    )
            case _:
                return current
        choice = (await run_sync(
            input, "모델 번호 (Enter 취소): ",
        )).strip()
        if not choice.isdecimal() or not 1 <= int(choice) <= len(ids):
            return current
        index = int(choice) - 1
        ref = replace(current.model, model_id=ids[index])
        selected = replace(current, model=ref)
        if models:
            model = models[index]
            capabilities = replace(
                provider.capabilities(),
                reasoning_efforts=model.reasoning_efforts,
                supports_fast=model.supports_fast,
            )
        else:
            capabilities = await model_capabilities(selected)
        selected = replace(selected, model=replace(
            ref,
            reasoning_effort=(
                ref.reasoning_effort if ref.reasoning_effort in capabilities.reasoning_efforts
                else None
            ),
            fast=ref.fast and capabilities.supports_fast and current.auth_mode == "subscription",
        ))
        save_profile(selected)
        return selected
    except (EOFError, KeyboardInterrupt):
        return current
    except (LLMError, ProviderError, OSError, ValueError, TimeoutError):
        emit("모델을 선택하지 못했습니다. 인증과 연결을 확인하세요.")
        return current
