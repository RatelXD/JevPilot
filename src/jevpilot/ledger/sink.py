"""Redact at persistence time; browser input values are stored as lengths."""

import json
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from jevpilot.ledger.models import RunSummary
from jevpilot.privacy import PrivacyPolicy, redact_json
from jevpilot.task import ContractViolation


def _input_lengths(value: JsonValue) -> JsonValue:
    match value:
        case dict():
            return {
                key: (
                    len(item) if isinstance(item, str) else len(json.dumps(item, ensure_ascii=False))
                ) if key.lower() in {
                    "input", "value", "text", "input_value", "input_values",
                    "user_input", "user_payload", "prompt", "instructions",
                } else _input_lengths(item)
                for key, item in value.items()
            }
        case list():
            return [_input_lengths(item) for item in value]
        case str() | int() | float() | None:
            return value


class LedgerSink:
    def __init__(
        self, run_id: str, *, root: Path | None = None,
        privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ContractViolation("run_id", "must be a single directory name")
        self.directory: Path = (root or Path.home() / ".jevpilot" / "runs") / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.privacy_policy: PrivacyPolicy = privacy_policy or PrivacyPolicy()

    def event(
        self, kind: Literal["span", "call", "route", "recovery"],
        payload: JsonValue,
    ) -> None:
        clean = redact_json(_input_lengths(payload), policy=self.privacy_policy)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps({"type": kind, "data": clean}, ensure_ascii=False) + "\n")

    def write_receipt(self, summary: RunSummary) -> Path:
        clean = redact_json(summary.model_dump(mode="json"), policy=self.privacy_policy)
        path = self.directory / "receipt.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(clean, stream, ensure_ascii=False, indent=2)
            _ = stream.write("\n")
        return path
