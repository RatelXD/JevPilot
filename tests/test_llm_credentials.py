import os
import stat
from pathlib import Path

import pytest

from jevpilot.llm import credentials
from jevpilot.llm.credentials import (
    Credential,
    get_api_key,
    get_base_url,
    load_credentials,
    save_credential,
)

SENTINEL = "sk-test-SECRET-abcd"


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in credentials._ENV_KEY.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("JEVPILOT_COMPAT_BASE_URL", raising=False)
    return tmp_path


def test_save_is_0600_and_roundtrips(home: Path) -> None:
    save_credential("openai", SENTINEL)
    save_credential("openai_compatible", "k2", "https://x.example/v1")
    path = home / ".jevpilot" / "credentials.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = load_credentials()
    assert loaded["openai"].api_key == SENTINEL
    assert loaded["openai_compatible"].base_url == "https://x.example/v1"
    assert get_base_url("openai_compatible") == "https://x.example/v1"


def test_env_beats_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    save_credential("openai", "stored")
    assert get_api_key("openai") == "stored"
    monkeypatch.setenv("JEVPILOT_OPENAI_API_KEY", "from-env")
    assert get_api_key("openai") == "from-env"
    monkeypatch.setenv("JEVPILOT_COMPAT_BASE_URL", "https://env.example")
    assert get_base_url("openai_compatible") == "https://env.example"
    assert get_api_key("anthropic") is None


def test_symlink_rejected(home: Path) -> None:
    save_credential("openai", SENTINEL)
    path = home / ".jevpilot" / "credentials.json"
    target = home / "elsewhere.json"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(PermissionError):
        load_credentials()
    with pytest.raises(PermissionError):
        get_api_key("openai")


def test_other_owner_rejected(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    save_credential("openai", SENTINEL)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(home).st_uid + 1)
    with pytest.raises(PermissionError):
        credentials.read_private_json(home / ".jevpilot" / "credentials.json", "x")


def test_repr_masks_key() -> None:
    text = repr(Credential(api_key=SENTINEL, base_url=None))
    assert SENTINEL not in text
    assert text.count("abcd") == 1 and "SECRET" not in text


def test_malformed_file_is_value_error(home: Path) -> None:
    save_credential("openai", SENTINEL)
    (home / ".jevpilot" / "credentials.json").write_text("{}")
    with pytest.raises(ValueError):
        load_credentials()
