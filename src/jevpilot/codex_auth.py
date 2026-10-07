"""Use the official Codex CLI login in a JevPilot-owned profile."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final

import anyio

_CODEX_ENV_KEYS: Final = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "TERM",
    "COLORTERM",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "DBUS_SESSION_BUS_ADDRESS",
)
_CHATGPT_LOGIN_MARKER: Final = b"Logged in using ChatGPT"
CODEX_COMMAND: Final = ("codex",)
_BROWSER_OPENERS: Final = (
    "xdg-open",
    "wslview",
    "wsl-open",
    "open",
    "gio",
    "sensible-browser",
    "x-www-browser",
    "firefox",
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "microsoft-edge",
    "microsoft-edge-stable",
)


def jevpilot_home() -> Path:
    """Return the private root used for JevPilot's local state."""
    return Path.home() / ".jevpilot"


def codex_home() -> Path:
    """Return the Codex profile owned by JevPilot."""
    return jevpilot_home() / "codex"


def ensure_private_directory(path: Path) -> Path:
    """Create or validate one owner-controlled directory with mode 0700."""
    if path.is_symlink():
        raise PermissionError("JevPilot state directory must not be a symbolic link")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise PermissionError("JevPilot state path is not a directory")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise PermissionError("JevPilot state directory is owned by another user")
    os.chmod(path, 0o700)
    return path


def ensure_jevpilot_home() -> Path:
    """Create or validate JevPilot's owner-only state root."""
    return ensure_private_directory(jevpilot_home())


def ensure_codex_home() -> Path:
    """Create or validate the isolated Codex profile directory."""
    _ = ensure_jevpilot_home()
    return ensure_private_directory(codex_home())


def _prepare_codex_home(profile: Path | None) -> Path:
    """Validate a custom test profile or the nested default JevPilot profile."""
    if profile is None or profile == codex_home():
        return ensure_codex_home()
    return ensure_private_directory(profile)


def codex_environment(*, profile: Path | None = None) -> dict[str, str]:
    """Build a small environment that excludes API keys and ambient credentials."""
    environment = {
        key: os.environ[key]
        for key in _CODEX_ENV_KEYS
        if key in os.environ
    }
    selected_profile = _prepare_codex_home(profile)
    environment["CODEX_HOME"] = str(selected_profile)
    for key in ("OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"):
        _ = environment.pop(key, None)
    return environment


def codex_command_prefix(command: tuple[str, ...] = CODEX_COMMAND) -> tuple[str, ...]:
    """Force Codex to store authentication inside its isolated profile."""
    return (*command, "-c", 'cli_auth_credentials_store="file"')


def launch_chatgpt_login(
    *, command: tuple[str, ...] = CODEX_COMMAND, profile: Path | None = None
) -> int:
    """Start official Codex OAuth without launching a browser."""
    selected_profile = _prepare_codex_home(profile)
    environment = codex_environment(profile=selected_profile)
    codex_login = (*codex_command_prefix(command), "login")
    if os.name == "nt":
        result = subprocess.run(
            (*codex_login, "--device-auth"),
            env=environment,
            check=False,
        )
        return result.returncode

    browser_stub = shutil.which("true")
    if browser_stub is None:
        raise FileNotFoundError("browserless Codex login requires the system true command")
    with TemporaryDirectory(prefix="jevpilot-browserless-") as opener_directory:
        opener_root = Path(opener_directory)
        for opener_name in _BROWSER_OPENERS:
            (opener_root / opener_name).symlink_to(browser_stub)
        environment["PATH"] = os.pathsep.join(
            (opener_directory, environment.get("PATH", ""))
        )
        environment["BROWSER"] = "xdg-open"
        result = subprocess.run(
            codex_login,
            env=environment,
            check=False,
        )
        return result.returncode


async def codex_login_status(
    *, command: tuple[str, ...] = CODEX_COMMAND, profile: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run the official CLI status check without exposing its output."""
    selected_profile = _prepare_codex_home(profile)
    with anyio.fail_after(10):
        return await anyio.run_process(
            (*codex_command_prefix(command), "login", "status"),
            env=codex_environment(profile=selected_profile),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def is_chatgpt_login_status(result: subprocess.CompletedProcess[bytes]) -> bool:
    """Recognize ChatGPT authentication regardless of the CLI output stream."""
    return (
        result.returncode == 0
        and _CHATGPT_LOGIN_MARKER in result.stdout + result.stderr
    )
