"""Provider/model profile (``~/.jevpilot/profile.json``) with a read-only v1 import."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast, get_args

from jevpilot.codex_auth import ensure_jevpilot_home
from jevpilot.llm.credentials import read_private_json, write_private_json
from jevpilot.llm.types import AuthMode, ModelRef, ProviderId
from jevpilot.model_selection import ModelSelectionError, load_model_selection

CodexTransport = Literal["app_server", "exec"]
_VERSION: Final = 1
_BAD: Final = "profile file has an unsupported format"


@dataclass(frozen=True, slots=True)
class Profile:
    auth_mode: AuthMode
    model: ModelRef
    codex_transport: CodexTransport = "app_server"
    policy_overrides: dict[str, object] = field(default_factory=dict)


def profile_path() -> Path:
    return ensure_jevpilot_home() / "profile.json"


def _opt_str(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(_BAD)
    return value


def _parse(raw: object) -> Profile:
    if not isinstance(raw, dict) or raw.get("version") != _VERSION:
        raise ValueError(_BAD)
    data = cast(dict[str, object], raw)
    auth_mode = data.get("auth_mode")
    transport = data.get("codex_transport", "app_server")
    model = data.get("model")
    overrides = data.get("policy_overrides", {})
    if (
        auth_mode not in get_args(AuthMode)
        or transport not in get_args(CodexTransport)
        or not isinstance(model, dict)
        or not isinstance(overrides, dict)
    ):
        raise ValueError(_BAD)
    m = cast(dict[str, object], model)
    provider, model_id, fast = m.get("provider"), m.get("model_id"), m.get("fast", False)
    if (
        provider not in get_args(ProviderId)
        or m.get("auth_mode") != auth_mode
        or not isinstance(model_id, str)
        or not isinstance(fast, bool)
    ):
        raise ValueError(_BAD)
    ref = ModelRef(
        provider=cast(ProviderId, provider),
        auth_mode=cast(AuthMode, auth_mode),
        model_id=model_id,
        reasoning_effort=_opt_str(m.get("reasoning_effort")),
        fast=fast,
        base_url=_opt_str(m.get("base_url")),
    )
    return Profile(
        auth_mode=cast(AuthMode, auth_mode),
        model=ref,
        codex_transport=cast(CodexTransport, transport),
        policy_overrides=cast(dict[str, object], overrides),
    )


def load_profile() -> Profile | None:
    """Load the saved profile; missing file means none."""
    raw = read_private_json(profile_path(), "profile file")
    return None if raw is None else _parse(raw)


def save_profile(profile: Profile) -> None:
    model = profile.model
    write_private_json(
        profile_path(),
        {
            "version": _VERSION,
            "auth_mode": profile.auth_mode,
            "model": {
                "provider": model.provider,
                "auth_mode": model.auth_mode,
                "model_id": model.model_id,
                "reasoning_effort": model.reasoning_effort,
                "fast": model.fast,
                "base_url": model.base_url,
            },
            "codex_transport": profile.codex_transport,
            "policy_overrides": profile.policy_overrides,
        },
    )


def suggest_profile() -> Profile | None:
    """Saved profile, else a suggestion imported (read-only) from v1 settings.json."""
    saved = load_profile()
    if saved is not None:
        return saved
    try:
        selection = load_model_selection()
    except ModelSelectionError as error:
        if "regular file" in str(error) or "another user" in str(error):
            raise PermissionError(str(error)) from None
        return None
    if selection is None:
        return None
    return Profile(
        auth_mode="subscription",
        model=ModelRef(
            provider="chatgpt_subscription",
            auth_mode="subscription",
            model_id=selection.model_id,
            reasoning_effort=selection.reasoning_effort,
            fast=selection.fast,
        ),
    )


__all__ = ["CodexTransport", "Profile", "load_profile", "profile_path", "save_profile", "suggest_profile"]
