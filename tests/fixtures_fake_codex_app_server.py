"""Offline executable Codex wire fixture, including login and exec entry points."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter


def main() -> None:
    scenario = sys.argv[1]
    home = Path(os.environ["CODEX_HOME"])
    assert "OPENAI_API_KEY" not in os.environ
    assert "CODEX_ACCESS_TOKEN" not in os.environ
    assert 'cli_auth_credentials_store="file"' in sys.argv
    if "login" in sys.argv:
        if scenario == "auth_missing":
            sys.exit(1)
        print("Logged in using ChatGPT", file=sys.stderr)
        return

    def log(data: dict[str, JsonValue]) -> None:
        with (home / "wire.jsonl").open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(data) + "\n")

    def emit(data: dict[str, JsonValue]) -> None:
        print(json.dumps(data), flush=True)

    if "exec" in sys.argv:
        assert "--ephemeral" in sys.argv
        assert "--ignore-user-config" in sys.argv
        assert "read-only" in sys.argv
        assert 'web_search="disabled"' in sys.argv
        _ = sys.stdin.read()
        marker = home / "repaired"
        text = '{"answer":"2"}'
        if scenario == "text":
            text = '{"text":"2"}'
        if scenario in {"repair", "invalid"} and not marker.exists():
            marker.touch()
            text = "{}"
        if scenario in {"invalid", "text_invalid"}:
            text = "{}"
        emit({"type": "item.completed", "item": {
            "type": "command_execution" if scenario == "tool" else "agent_message",
            "text": text,
        }})
        emit({"type": "turn.completed", "usage": {
            "input_tokens": 100, "output_tokens": 5, "cached_input_tokens": 10,
        }})
        return

    thread = ""
    turn = ""
    calls = 0
    for line in sys.stdin:
        request = TypeAdapter(dict[str, JsonValue]).validate_json(line)
        log({**request, "pid": os.getpid()})
        method = request["method"]
        params = request.get("params", {})
        assert isinstance(params, dict)

        def reply(result: dict[str, JsonValue]) -> None:
            emit({"id": request["id"], "result": result})

        def notify(method_name: str, payload: dict[str, JsonValue]) -> None:
            emit({"method": method_name, "params": payload})

        if method == "initialize":
            reply({"userAgent": "fake"})
        elif method == "thread/start":
            calls += 1
            thread = f"thread-{calls}"
            assert params["ephemeral"] is True
            assert params["sandbox"] == "read-only"
            assert params["approvalPolicy"] == "never"
            config = params["config"]
            assert isinstance(config, dict)
            assert config["web_search"] == "disabled"
            assert config["features.shell_tool"] is False
            assert config["features.unified_exec"] is False
            assert config["mcp_servers"] == {}
            notify("account/rateLimits/updated", {"rateLimits": {
                "limitId": "codex", "primary": {"usedPercent": 20, "resetsAt": 1234},
            }})
            notify("account/rateLimits/updated", {"rateLimits": {
                "limitId": None, "primary": {"usedPercent": 21},
            }})
            if scenario == "rpc_rate":
                emit({"id": request["id"], "error": {
                    "code": 429, "data": {"codexErrorInfo": "rateLimitExceeded"},
                    "message": "private upstream details",
                }})
            else:
                reply({"thread": {"id": thread}, "model": "gpt-5.3-codex"})
        elif method == "turn/start":
            assert params["threadId"] == thread
            schema = params["outputSchema"]
            if scenario == "text":
                assert schema is None
            else:
                assert isinstance(schema, dict)
                assert schema["type"] == "object"
            assert params["approvalPolicy"] == "never"
            sandbox = params["sandboxPolicy"]
            assert isinstance(sandbox, dict)
            assert sandbox["type"] == "readOnly"
            turn = f"turn-{calls}"
            if scenario == "crash_once" and not (home / "crashed").exists():
                (home / "crashed").touch()
                sys.exit(3)
            notify("turn/started", {"threadId": thread, "turn": {"id": turn}})
            reply({"turn": {"id": turn, "status": "inProgress", "items": []}})
            if scenario == "cancel":
                continue
            if scenario == "notification_rate":
                notify("account/rateLimits/updated", {"rateLimits": {
                    "primary": {"usedPercent": 100},
                    "rateLimitReachedType": "rate_limit_reached",
                }})
                continue
            if scenario == "quota":
                notify("turn/completed", {"threadId": thread, "turn": {
                    "id": turn, "status": "failed",
                    "error": {"codexErrorInfo": "usageLimitExceeded"},
                }})
                continue
            text = "{}" if scenario == "invalid" or (
                scenario == "repair" and calls == 1
            ) else '{"answer":"2"}'
            if scenario == "text":
                text = "2"
            item_type = "commandExecution" if scenario == "tool" else "agentMessage"
            notify("item/completed", {
                "threadId": thread, "turnId": turn,
                "item": {"type": item_type, "id": "item", "text": text},
            })
            if scenario != "no_usage":
                notify("thread/tokenUsage/updated", {
                    "threadId": thread, "turnId": turn, "tokenUsage": {
                        "last": {"inputTokens": 1, "outputTokens": 1},
                        "total": {"inputTokens": 100, "outputTokens": 5,
                                  "cachedInputTokens": 10, "reasoningOutputTokens": 2},
                    },
                })
            notify("turn/completed", {"threadId": thread, "turn": {
                "id": turn, "status": "completed", "items": [], "error": None,
            }})
        elif method == "turn/interrupt":
            assert params == {"threadId": thread, "turnId": turn}
            reply({})
        elif method == "initialized":
            pass
        else:
            raise AssertionError(method)


if __name__ == "__main__":
    main()
