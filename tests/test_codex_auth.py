import shutil
import stat
import sys
from pathlib import Path

import pytest

from jevpilot.codex_auth import (
    codex_environment,
    ensure_codex_home,
    ensure_private_directory,
    launch_chatgpt_login,
)


def test_codex_profile_is_private_and_does_not_inherit_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setenv("OPENAI_API_KEY", "api-key-secret")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "access-token-secret")

    # When
    profile = ensure_codex_home()
    environment = codex_environment()

    # Then
    assert profile == tmp_path / ".jevpilot" / "codex"
    assert stat.S_IMODE((tmp_path / ".jevpilot").stat().st_mode) == 0o700
    assert stat.S_IMODE(profile.stat().st_mode) == 0o700
    assert environment["CODEX_HOME"] == str(profile)
    assert "OPENAI_API_KEY" not in environment
    assert "CODEX_ACCESS_TOKEN" not in environment


def test_private_codex_directory_rejects_symbolic_links(tmp_path: Path) -> None:
    # Given
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)

    # When / Then
    with pytest.raises(PermissionError, match="symbolic link"):
        _ = ensure_private_directory(alias)


def test_official_login_delegates_to_codex_in_selected_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    profile = tmp_path / "codex-profile"
    script = tmp_path / "fake_codex.py"
    opener_dir_file = tmp_path / "opener-dir.txt"
    true_path = shutil.which("true")
    assert true_path is not None
    script_source = "\n".join(
        (
            "import os, shutil, subprocess, sys",
            "from pathlib import Path",
            f"assert os.environ['CODEX_HOME'] == {str(profile)!r}",
            "assert os.environ['BROWSER'] == 'xdg-open'",
            "assert 'OPENAI_API_KEY' not in os.environ",
            "assert sys.argv[1:] == ['-c', 'cli_auth_credentials_store=\"file\"', 'login']",
            f"assert Path(shutil.which('xdg-open')).resolve() == Path({true_path!r})",
            "subprocess.run(['xdg-open', 'https://auth.openai.com/example'], check=True)",
            f"Path({str(opener_dir_file)!r}).write_text(os.environ['PATH'].split(os.pathsep)[0])",
        )
    ) + "\n"
    _ = script.write_text(
        script_source,
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")

    # When
    result = launch_chatgpt_login(
        command=(sys.executable, str(script)),
        profile=profile,
    )

    # Then
    assert result == 0
    assert stat.S_IMODE(profile.stat().st_mode) == 0o700
    assert not Path(opener_dir_file.read_text(encoding="utf-8")).exists()
