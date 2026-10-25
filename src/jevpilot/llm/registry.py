"""Construct exactly the provider selected by the saved profile."""

from jevpilot.llm.api_providers import create_api_provider
from jevpilot.llm.codex_app_server import CodexAppServerProvider
from jevpilot.llm.codex_exec_adapter import CodexExecProvider
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import Profile
from jevpilot.llm.types import LLMProvider


def build_provider(profile: Profile) -> LLMProvider:
    ref = profile.model
    match profile.auth_mode, ref.auth_mode, ref.provider:
        case "api", "api", ("openai" | "anthropic"):
            return create_api_provider(ref)
        case "subscription", "subscription", "chatgpt_subscription":
            match profile.codex_transport:
                case "app_server":
                    return CodexAppServerProvider(ref)
                case "exec":
                    return CodexExecProvider(ref)
        case _:
            raise LLMError(
                "configuration", retryable=False,
                message="unsupported provider or authentication configuration",
            )
