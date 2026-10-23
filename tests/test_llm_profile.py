import hashlib
import json
import stat
from pathlib import Path

import pytest

from jevpilot.llm.profile import (
    Profile,
    load_profile,
    save_profile,
    suggest_profile,
)
from jevpilot.llm.types import ModelRef


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_save_load_roundtrip_0600(home: Path) -> None:
    profile = Profile(
        auth_mode="api",
        model=ModelRef("openai", "api", "gpt-x", "high", False, None),
        codex_transport="exec",
        policy_overrides={"a": 1},
    )
    save_profile(profile)
    path = home / ".jevpilot" / "profile.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert load_profile() == profile


def test_missing_profile_and_settings_is_none() -> None:
    assert load_profile() is None
    assert suggest_profile() is None


def test_symlink_rejected(home: Path) -> None:
    save_profile(Profile("api", ModelRef("openai", "api", "m")))
    path = home / ".jevpilot" / "profile.json"
    target = home / "other.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(PermissionError):
        load_profile()


def test_v1_settings_import_leaves_file_untouched(home: Path) -> None:
    directory = home / ".jevpilot"
    directory.mkdir(mode=0o700)
    settings = directory / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "provider": "chatgpt-subscription",
                "model_id": "gpt-5",
                "reasoning_effort": "high",
                "fast": True,
            }
        )
    )
    before = hashlib.sha256(settings.read_bytes()).hexdigest()
    suggestion = suggest_profile()
    assert suggestion is not None
    assert suggestion.auth_mode == "subscription"
    assert suggestion.model == ModelRef(
        "chatgpt_subscription", "subscription", "gpt-5", "high", True, None
    )
    assert hashlib.sha256(settings.read_bytes()).hexdigest() == before
    assert not (directory / "profile.json").exists()


def test_saved_profile_wins_over_v1_settings(home: Path) -> None:
    save_profile(Profile("api", ModelRef("openai", "api", "m")))
    (home / ".jevpilot" / "settings.json").write_text(
        '{"provider":"chatgpt-subscription","model_id":"gpt-5"}'
    )
    suggestion = suggest_profile()
    assert suggestion is not None and suggestion.auth_mode == "api"


def test_invalid_profile_is_value_error(home: Path) -> None:
    save_profile(Profile("api", ModelRef("openai", "api", "m")))
    (home / ".jevpilot" / "profile.json").write_text('{"version":1}')
    with pytest.raises(ValueError):
        load_profile()
