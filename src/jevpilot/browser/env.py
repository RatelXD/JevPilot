"""Configure browser-use before importing it.

browser-use 0.13.11 calls ``load_dotenv()`` from import-time modules including
``agent.service``, ``utils``, ``observability``, ``telemetry.service``, and
``logging_config``.
"""

import os


def prepare_browser_use_environment() -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("BROWSER_USE_CLOUD_SYNC", "false")
    os.environ.setdefault("BROWSER_USE_LOGGING_LEVEL", "warning")
