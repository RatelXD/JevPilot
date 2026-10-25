"""Extend the existing, read-only native IPC adapter with final page evidence."""
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from typing import Final

type Json = str | int | float | bool | None | list[Json] | dict[str, Json]
ADAPTER: Final = Path("/home/jhou8/jevpilot/evaluation/harness/ultrafast_adapter.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("eval_native_adapter", ADAPTER)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)

    class EvidenceIPC(adapter.IPC):
        def emit(self, **fields: Json) -> None:
            if fields.get("type") == "terminal" and fields.get("target_id"):
                from browser_harness.helpers import cdp

                session = cdp(
                    "Target.attachToTarget", targetId=fields["target_id"], flatten=True,
                )["sessionId"]
                try:
                    result = cdp(
                        "Runtime.evaluate", session_id=session, returnByValue=True,
                        expression="({url:location.href,page_text:document.body.innerText,"
                        "item_id:document.querySelector('#item-id')?.innerText||''})",
                    )["result"]["value"]
                    fields.update(result)
                finally:
                    cdp("Target.detachFromTarget", sessionId=session)
            super().emit(**fields)

    def owned_temp(*, prefix: str, dir: str) -> TemporaryDirectory[str]:
        return TemporaryDirectory(prefix=prefix, dir=os.environ["TMPDIR"])

    original_execute = adapter.execute

    def localized_execute(ipc: EvidenceIPC, url: str, goal: str) -> int:
        # The adapter has initialized its isolated Browser Harness environment
        # before this callback imports the untouched comparator.
        from jev_ultrafast import browser

        original_cdp = browser.cdp

        def localized_cdp(method: str, **params: Json) -> dict[str, Json]:
            result = original_cdp(method, **params)
            if method == "Target.attachToTarget":
                # Configure the native session before Browser.__init__ navigates.
                # Keep its existing device metrics and all agent decisions.
                original_cdp(
                    "Emulation.setLocaleOverride",
                    session_id=result["sessionId"], locale="ko_KR",
                )
            return result

        setattr(browser, "cdp", localized_cdp)
        try:
            return original_execute(ipc, url, goal)
        finally:
            setattr(browser, "cdp", original_cdp)

    setattr(adapter, "IPC", EvidenceIPC)
    setattr(adapter, "TemporaryDirectory", owned_temp)
    setattr(adapter, "execute", localized_execute)
    return adapter.main()


if __name__ == "__main__":
    raise SystemExit(main())
