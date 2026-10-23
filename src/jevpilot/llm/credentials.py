"""Private API-key store for JevPilot (``~/.jevpilot/credentials.json``, mode 0600)."""

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from jevpilot.codex_auth import ensure_jevpilot_home

CredentialProvider = Literal[
    "openai",
    "anthropic",
    "google",
    "openrouter",
    "openai_compatible",
    "chosun_gateway",
    "jev",
]

_ENV_KEY: Final[dict[str, str]] = {
    "openai": "JEVPILOT_OPENAI_API_KEY",
    "anthropic": "JEVPILOT_ANTHROPIC_API_KEY",
    "google": "JEVPILOT_GOOGLE_API_KEY",
    "openrouter": "JEVPILOT_OPENROUTER_API_KEY",
    "openai_compatible": "JEVPILOT_COMPAT_API_KEY",
    "chosun_gateway": "JEVPILOT_LLM_API_KEY",
    "jev": "JEVPILOT_JEV_API_KEY",
}
_ENV_BASE_URL: Final[dict[str, str]] = {"openai_compatible": "JEVPILOT_COMPAT_BASE_URL"}
_FILE_NAME: Final = "credentials.json"
_VERSION: Final = 1


def mask_key(key: str) -> str:
    """Show only the last four characters of a secret."""
    return f"sk-…{key[-4:]}" if len(key) > 4 else "sk-…"


@dataclass(frozen=True, slots=True)
class Credential:
    api_key: str
    base_url: str | None = None

    def __repr__(self) -> str:
        return f"Credential(api_key={mask_key(self.api_key)!r}, base_url={self.base_url!r})"


def credentials_path() -> Path:
    return ensure_jevpilot_home() / _FILE_NAME


def read_private_json(path: Path, label: str) -> object | None:
    """Read a JSON file only if it is a regular, owner-owned, non-symlink file."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise PermissionError(f"{label} must be a regular file")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise PermissionError(f"{label} is owned by another user")
    try:
        return cast(object, json.loads(path.read_bytes()))
    except ValueError:
        raise ValueError(f"{label} is not valid JSON") from None


def write_private_json(path: Path, payload: object) -> None:
    """Atomically write JSON with mode 0600 (temp file, fsync, os.replace)."""
    directory = path.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=directory
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            _ = stream.write(json.dumps(payload, indent=2).encode("utf-8"))
            _ = stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_credentials() -> dict[str, Credential]:
    """Load stored credentials; missing file means none."""
    raw = read_private_json(credentials_path(), "credentials file")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or raw.get("version") != _VERSION:
        raise ValueError("credentials file has an unsupported format")
    providers = cast(dict[str, object], raw).get("providers")
    if not isinstance(providers, dict):
        raise ValueError("credentials file has an unsupported format")
    result: dict[str, Credential] = {}
    for name, entry in cast(dict[str, object], providers).items():
        if not isinstance(entry, dict):
            raise ValueError("credentials file has an unsupported format")
        data = cast(dict[str, object], entry)
        key = data.get("api_key")
        base_url = data.get("base_url")
        if not isinstance(key, str) or not (base_url is None or isinstance(base_url, str)):
            raise ValueError("credentials file has an unsupported format")
        result[name] = Credential(api_key=key, base_url=base_url)
    return result


def save_credential(
    provider: CredentialProvider, api_key: str, base_url: str | None = None
) -> None:
    """Store one provider's key, keeping the others."""
    if provider not in _ENV_KEY:
        raise ValueError(f"unknown credential provider: {provider}")
    if not api_key.strip():
        raise ValueError("api key must not be empty")
    stored = load_credentials()
    stored[provider] = Credential(api_key=api_key.strip(), base_url=base_url)
    providers: dict[str, dict[str, str]] = {}
    for name, credential in stored.items():
        entry = {"api_key": credential.api_key}
        if credential.base_url is not None:
            entry["base_url"] = credential.base_url
        providers[name] = entry
    write_private_json(credentials_path(), {"version": _VERSION, "providers": providers})


def get_api_key(provider: CredentialProvider) -> str | None:
    """Environment variable first, then the stored value."""
    env = os.environ.get(_ENV_KEY[provider], "").strip()
    if env:
        return env
    stored = load_credentials().get(provider)
    return stored.api_key if stored else None


def get_base_url(provider: CredentialProvider) -> str | None:
    """Base URL for OpenAI-compatible providers: environment first, then stored."""
    name = _ENV_BASE_URL.get(provider)
    if name:
        env = os.environ.get(name, "").strip()
        if env:
            return env
    stored = load_credentials().get(provider)
    return stored.base_url if stored else None
