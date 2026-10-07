import stat
from pathlib import Path

import pytest

from jevpilot.model_selection import (
    ModelSelection,
    ModelSelectionError,
    load_model_selection,
    save_model_selection,
    select_available_model,
    selection_path,
)


def test_model_selection_is_persisted_without_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    selection = ModelSelection(
        provider="chatgpt-subscription",
        model_id="gpt-6.1-sol",
    )

    # When
    save_model_selection(selection)
    loaded = load_model_selection()

    # Then
    assert loaded == selection
    assert selection.model_ref == "chatgpt-subscription/gpt-6.1-sol"
    assert stat.S_IMODE(selection_path().stat().st_mode) == 0o600
    saved_text = selection_path().read_text(encoding="utf-8")
    assert "access_token" not in saved_text
    assert "api_key" not in saved_text


def test_model_selection_rejects_malformed_and_unavailable_choices() -> None:
    # Given
    available = frozenset({"gpt-6.1-sol"})

    # When / Then
    with pytest.raises(ModelSelectionError, match="provider/model"):
        _ = select_available_model("gpt-6.1-sol", available)
    with pytest.raises(ModelSelectionError, match="malformed"):
        _ = select_available_model("chatgpt-subscription/", available)
    with pytest.raises(ModelSelectionError, match="not available"):
        _ = select_available_model("chatgpt-subscription/unknown", available)
    with pytest.raises(ModelSelectionError, match="only chatgpt-subscription"):
        _ = select_available_model("openai-api/gpt-6.1-sol", available)


def test_model_selection_preserves_model_id_suffix_after_provider_separator() -> None:
    # Given / When
    selection = select_available_model(
        "chatgpt-subscription/vendor/model-v2",
        frozenset({"vendor/model-v2"}),
    )

    # Then
    assert selection.model_id == "vendor/model-v2"
    assert selection.model_ref == "chatgpt-subscription/vendor/model-v2"


def test_model_selection_rejects_symlinked_settings_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _ = selection_path().parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = tmp_path / "outside.json"
    _ = target.write_text(
        '{"provider":"chatgpt-subscription","model_id":"gpt-6.1-sol"}',
        encoding="utf-8",
    )
    selection_path().symlink_to(target)

    # When / Then
    with pytest.raises(ModelSelectionError, match="regular file"):
        _ = load_model_selection()
