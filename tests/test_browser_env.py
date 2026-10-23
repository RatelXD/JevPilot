import os
from importlib import import_module
from typing import Protocol, cast

import pytest

from jevpilot.browser.env import prepare_browser_use_environment


class BrowserSession(Protocol):
    async def start(self) -> None: ...

    async def kill(self) -> None: ...


class BrowserUseModule(Protocol):
    Browser: type[BrowserSession]


def test_prepare_browser_use_environment_sets_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = (
        "ANONYMIZED_TELEMETRY",
        "BROWSER_USE_CLOUD_SYNC",
        "BROWSER_USE_LOGGING_LEVEL",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    prepare_browser_use_environment()

    assert os.environ["ANONYMIZED_TELEMETRY"] == "false"
    assert os.environ["BROWSER_USE_CLOUD_SYNC"] == "false"
    assert os.environ["BROWSER_USE_LOGGING_LEVEL"] == "warning"


def test_prepare_browser_use_environment_preserves_user_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "custom-telemetry")
    monkeypatch.setenv("BROWSER_USE_CLOUD_SYNC", "custom-cloud")
    monkeypatch.setenv("BROWSER_USE_LOGGING_LEVEL", "debug")

    prepare_browser_use_environment()

    assert os.environ["ANONYMIZED_TELEMETRY"] == "custom-telemetry"
    assert os.environ["BROWSER_USE_CLOUD_SYNC"] == "custom-cloud"
    assert os.environ["BROWSER_USE_LOGGING_LEVEL"] == "debug"


@pytest.mark.browser
async def test_browser_use_can_start_a_real_browser() -> None:
    prepare_browser_use_environment()
    browser_use = cast(BrowserUseModule, cast(object, import_module("browser_use")))

    browser = browser_use.Browser()
    try:
        await browser.start()
    finally:
        await browser.kill()
