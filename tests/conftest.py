import os

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if os.environ.get("JEVPILOT_BROWSER_TESTS") == "1":
        return

    skip_browser = pytest.mark.skip(
        reason="set JEVPILOT_BROWSER_TESTS=1 to run browser tests"
    )
    for item in items:
        if "browser" in item.keywords:
            item.add_marker(skip_browser)
