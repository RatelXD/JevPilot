from pathlib import Path

from pydantic import JsonValue, TypeAdapter
from jevpilot.ledger.sink import LedgerSink


def test_public_option_is_literal_but_text_and_urls_remain_private(tmp_path: Path) -> None:
    # Given
    sink = LedgerSink("privacy", root=tmp_path)
    # When
    for operation, value, options in [
        ("select_option", "north", ["north"]),
        ("select_option", "private-option", ["north"]),
        ("type_text", "private-text", ["private-text"]),
    ]:
        sink.event("recovery", {
            "event": "execution", "operation": operation, "value": value,
            "observed_options": list[JsonValue](options), "url": "https://example.test/item?secret=private-query",
        })
    # Then
    raw = (sink.directory / "events.jsonl").read_text()
    records = [
        TypeAdapter(dict[str, dict[str, JsonValue] | str]).validate_json(line)["data"]
        for line in raw.splitlines()
    ]
    assert all(isinstance(record, dict) for record in records)
    records = [record for record in records if isinstance(record, dict)]
    assert [record["value"] for record in records] == ["north", 14, 12]
    assert all(record["url"] == "https://example.test/item" for record in records)
    assert "private-" not in raw and "observed_options" not in raw
