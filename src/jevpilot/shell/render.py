"""Persist ledger events before emitting allowlisted shell progress."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, override

from pydantic import JsonValue

from jevpilot.ledger.sink import LedgerSink
from jevpilot.privacy import PrivacyPolicy


class ShellSink(LedgerSink):
    def __init__(
        self, run_id: str, *, emit: Callable[[str], None],
        root: Path | None = None, privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        super().__init__(run_id, root=root, privacy_policy=privacy_policy)
        self.emit: Callable[[str], None] = emit

    @override
    def event(
        self, kind: Literal["span", "call", "route", "recovery"], payload: JsonValue,
    ) -> None:
        super().event(kind, payload)
        if kind != "span" or not isinstance(payload, dict) or payload.get("end_s") is not None:
            return
        match payload.get("kind"):
            case "observe":
                label = "관찰"
            case "jev":
                label = "Jev 선택"
            case "llm":
                label = "LLM(text)" if payload.get("label") == "llm_text" else "LLM"
            case "execute":
                label = "실행"
            case "verify":
                label = "검증"
            case "recovery":
                label = "복구"
            case _:
                return
        self.emit(f"진행: {label}")
