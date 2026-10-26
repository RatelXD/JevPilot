"""ChatGPT subscription calls over a session-owned Codex app-server.

Protocol verified with codex-cli 0.160.0, using
``codex app-server generate-json-schema --out <task-8>/codex-schema``:
initialize(clientInfo) -> initialized; thread/start(ephemeral, sandbox,
approvalPolicy, config); turn/start(threadId, input, outputSchema, effort);
turn/interrupt(threadId, turnId). Notifications are item/started,
item/completed, turn/started, turn/completed, thread/tokenUsage/updated
(tokenUsage.total/last: inputTokens, cachedInputTokens, outputTokens,
reasoningOutputTokens), and account/rateLimits/updated (rateLimits).
turn errors use codexErrorInfo=rateLimitExceeded/usageLimitExceeded.

Each attempt gets a new ephemeral thread. Calls are serialized, not replayed
after a crash; the next call starts one fresh server. No paid API fallback.
Images are unsupported; callers must use text observations/use_vision=False.
"""

from __future__ import annotations

import json
from copy import deepcopy
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar, Final

import anyio
from anyio.abc import Process
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import JsonValue, TypeAdapter, ValidationError

from jevpilot.codex_auth import (
    codex_command_prefix,
    codex_environment,
)
from jevpilot.codex_auth import codex_home as default_codex_home
from jevpilot.ledger.recorder import CallScope as LedgerCallScope
from jevpilot.llm.codex_exec_adapter import (
    CodexExecProvider, SubscriptionScope, check_subscription_login, require_subscription_ref,
)
from jevpilot.llm.errors import LLMError
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.types import (
    CallScope, Capabilities, LLMRequest, LLMResult, LLMUsage, ModelRef,
)

_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_DISABLED: Final = (
    "shell_tool", "unified_exec", "apps", "browser_use", "browser_use_external",
    "computer_use", "multi_agent", "plugins", "remote_plugin", "hooks",
    "view_image", "image_generation", "skill_mcp_dependency_install",
)


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    match value:
        case dict() as result:
            return result
        case _:
            raise LLMError(
                "invalid_output", retryable=False, message="Invalid Codex protocol object"
            )


def _string(value: JsonValue) -> str:
    match value:
        case str() as result:
            return result
        case _:
            raise LLMError(
                "invalid_output", retryable=False, message="Invalid Codex protocol identifier"
            )


def _strict_output_schema(schema: JsonValue) -> JsonValue:
    """Copy schemas into Codex's strict subset without changing the parse type.

    Like browser-use's SchemaOptimizer, require nullable fields too and strip
    defaults/min_items. Keep references rather than expanding recursive models.
    Property/definition names and literal JSON values are not schema keywords.
    """
    result = deepcopy(schema)

    def normalize(node: JsonValue) -> None:
        match node:
            case dict():
                node.pop("default", None)
                node.pop("min_items", None)
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties)
                    node["additionalProperties"] = False
                elif node.get("type") == "object" and "additionalProperties" not in node:
                    node["additionalProperties"] = False
                for key, value in node.items():
                    if key in {"properties", "$defs", "definitions", "patternProperties"}:
                        if isinstance(value, dict):
                            for child in value.values():
                                normalize(child)
                    elif key in {
                        "items", "prefixItems", "anyOf", "oneOf", "allOf",
                        "additionalProperties", "not", "contains",
                    }:
                        normalize(value)
            case list():
                for child in node:
                    normalize(child)
            case _:
                return

    normalize(result)
    return result


def _schema_rejection(error: JsonValue) -> bool:
    """Recognize protocol rejection markers without exposing upstream text."""
    match error:
        case dict():
            if error.get("code") in ("invalid_json_schema", 400) or error.get("status") == 400:
                return True
            for key in ("error", "data"):
                if _schema_rejection(error.get(key)):
                    return True
            message = error.get("message")
            if isinstance(message, str):
                try:
                    return _schema_rejection(_JSON.validate_json(message))
                except ValidationError:
                    return bool(re.search(
                        r"\binvalid_json_schema\b|\bHTTP(?:/\d(?:\.\d)?)?\s+400\b|"
                        r"\b400 Bad Request\b", message,
                    ))
            return False
        case _:
            return False


class CodexAppServerProvider:
    """Mutable owner of one subprocess, reader and serialized RPC conversation."""

    def __init__(
        self, ref: ModelRef, *, codex_home: Path | None = None,
        command: tuple[str, ...] = ("codex",),
    ) -> None:
        require_subscription_ref(ref)
        self.ref: ModelRef = ref
        self.codex_home: Path = codex_home or default_codex_home()
        self.command: tuple[str, ...] = command
        self._process: Process | None = None
        self._reader: BufferedByteReceiveStream | None = None
        self._lock: anyio.Lock = anyio.Lock()
        self._request_id: int = 0
        self._pending: list[dict[str, JsonValue]] = []
        self._thread_id: str | None = None
        self._turn_id: str | None = None
        self.rate_limits: dict[str, JsonValue] = {}

    def capabilities(self) -> Capabilities:
        return Capabilities(
            structured_output="native", images=False,
            reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
            supports_fast=True,
        )

    async def start(self) -> None:
        """Warm up alongside browser startup; repeated starts reuse the server."""
        async with self._lock:
            try:
                with anyio.fail_after(30):
                    await self._start()
            except TimeoutError:
                await self.aclose()
                raise LLMError(
                    "timeout", retryable=True, message="Codex startup timed out"
                ) from None

    async def _start(self) -> None:
        if self._process is not None and self._process.returncode is None:
            return
        await self.aclose()
        await check_subscription_login(self.command, self.codex_home)
        command = [
            *codex_command_prefix(self.command), "app-server", "--listen", "stdio://",
            "-c", 'web_search="disabled"',
            "-c", "features.skip_host_skill_discovery=true",
            "-c", "features.unbounded_connection_retries=false",
            "-c", "mcp_servers={}",
        ]
        for feature in _DISABLED:
            command.extend(("--disable", feature))
        try:
            self._process = await anyio.open_process(
                command, env=codex_environment(profile=self.codex_home),
                cwd=self.codex_home, stderr=subprocess.DEVNULL,
            )
            assert self._process.stdout is not None
            self._reader = BufferedByteReceiveStream(self._process.stdout)
            _ = await self._request("initialize", {
                "clientInfo": {"name": "jevpilot", "version": "0.1.0"},
            })
            await self._send({"method": "initialized"})
        except (OSError, anyio.EndOfStream, anyio.IncompleteRead, anyio.BrokenResourceError):
            await self.aclose()
            raise LLMError(
                "provider_unavailable", retryable=True, message="Codex app-server could not start"
            ) from None
        except (LLMError, anyio.get_cancelled_exc_class()):
            await self.aclose()
            raise

    async def _send(self, message: dict[str, JsonValue]) -> None:
        assert self._process is not None and self._process.stdin is not None
        await self._process.stdin.send(json.dumps(message).encode() + b"\n")

    async def _receive(self) -> dict[str, JsonValue]:
        assert self._reader is not None
        try:
            message = _mapping(_JSON.validate_json(
                await self._reader.receive_until(b"\n", max_bytes=8 * 1024 * 1024)
            ))
        except (ValidationError, anyio.DelimiterNotFound):
            raise LLMError(
                "invalid_output", retryable=False, message="Invalid Codex JSON-RPC message"
            ) from None
        method = message.get("method")
        if method == "account/rateLimits/updated":
            limits = _mapping(_mapping(message.get("params")).get("rateLimits"))
            # Rolling updates are sparse: null metadata does not erase known values.
            for key, value in limits.items():
                if value is not None:
                    match value, self.rate_limits.get(key):
                        case dict() as new, dict() as old:
                            self.rate_limits[key] = {**old, **{
                                k: v for k, v in new.items() if v is not None
                            }}
                        case _:
                            self.rate_limits[key] = value
            if limits.get("rateLimitReachedType") is not None:
                raise self._error({"codexErrorInfo": "rateLimitExceeded"})
            if limits.get("spendControlReached") is True:
                raise self._error({"codexErrorInfo": "usageLimitExceeded"})
        if method == "turn/started":
            params = _mapping(message.get("params"))
            if params.get("threadId") == self._thread_id:
                self._turn_id = _string(_mapping(params.get("turn")).get("id"))
        if method is not None and "id" in message:
            # Never execute an approval/tool request sent by the server.
            raise LLMError(
                "invalid_output", retryable=False, message="Codex requested an unsupported tool"
            )
        return message

    def _error(self, error: dict[str, JsonValue]) -> LLMError:
        if _schema_rejection(error):
            return LLMError(
                "configuration", retryable=False,
                message="Codex rejected the structured-output schema (HTTP 400 / invalid_json_schema)",
            )
        info = error.get("codexErrorInfo")
        if info is None:
            match error.get("data"):
                case dict() as data:
                    info = data.get("codexErrorInfo")
                case _:
                    pass
        message = error.get("message")
        if isinstance(message, str) and re.search(r"\bunauthorized\s+\(401\)", message):
            info = "unauthorized"
        match info:
            case "rateLimitExceeded":
                return LLMError("rate_limited", retryable=True, message="ChatGPT rate limit reached")
            case "usageLimitExceeded":
                return LLMError("quota_exhausted", retryable=False, message="ChatGPT quota exhausted")
            case "unauthorized":
                return LLMError(
                    "auth_invalid", retryable=False,
                    message="ChatGPT authentication rejected; sign in again with /login",
                )
            case _:
                if error.get("code") == 429:
                    return LLMError("rate_limited", retryable=True, message="ChatGPT rate limit reached")
                return LLMError(
                    "provider_unavailable", retryable=True, message="Codex could not complete the turn"
                )

    async def _request(
        self, method: str, params: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        self._request_id += 1
        request_id = self._request_id
        await self._send({"id": request_id, "method": method, "params": params})
        while True:
            message = await self._receive()
            if message.get("id") == request_id:
                if "error" in message:
                    raise self._error(_mapping(message["error"]))
                return _mapping(message.get("result"))
            self._pending.append(message)

    async def _attempt(self, prompt: StructuredPrompt) -> RawCompletion:
        config: dict[str, JsonValue] = {
            "web_search": "disabled", "mcp_servers": {},
            "features.skip_host_skill_discovery": True,
            "features.unbounded_connection_retries": False,
        }
        config.update({f"features.{name}": False for name in _DISABLED})
        thread = await self._request("thread/start", {
            "model": None if self.ref.model_id == "codex-default" else self.ref.model_id,
            "ephemeral": True, "sandbox": "read-only", "approvalPolicy": "never",
            "cwd": str(self.codex_home), "config": config,
            "baseInstructions": prompt.system,
            "developerInstructions": "Return only the requested result. Do not use any tools.",
            "serviceTier": "fast" if self.ref.fast else None,
        })
        self._thread_id = _string(_mapping(thread.get("thread")).get("id"))
        resolved_model = thread.get("model")
        result = await self._request("turn/start", {
            "threadId": self._thread_id,
            "input": [{"type": "text", "text": prompt.user_text, "text_elements": []}],
            "outputSchema": _strict_output_schema(prompt.json_schema),
            "approvalPolicy": "never",
            "sandboxPolicy": {"type": "readOnly"},
            "effort": self.ref.reasoning_effort,
            "serviceTier": "fast" if self.ref.fast else None,
        })
        self._turn_id = _string(_mapping(result.get("turn")).get("id"))
        text: str | None = None
        usage: LLMUsage | None = None
        while True:
            message = self._pending.pop(0) if self._pending else await self._receive()
            params_value = message.get("params")
            if params_value is None:
                continue
            params = _mapping(params_value)
            if params.get("threadId") != self._thread_id:
                continue
            if params.get("turnId", self._turn_id) != self._turn_id:
                continue
            match message.get("method"):
                case "item/started" | "item/completed":
                    item = _mapping(params.get("item"))
                    match item.get("type"):
                        case "agentMessage":
                            if message.get("method") == "item/completed":
                                text = _string(item.get("text"))
                        case "userMessage" | "reasoning":
                            pass
                        case _:
                            raise LLMError(
                                "invalid_output", retryable=False,
                                message="Codex returned an unsupported tool item",
                            )
                case "thread/tokenUsage/updated":
                    totals = _mapping(_mapping(params.get("tokenUsage")).get("total"))
                    usage = LLMUsage(
                        input_tokens=_token(totals.get("inputTokens")),
                        output_tokens=_token(totals.get("outputTokens")),
                        cached_input_tokens=_token(totals.get("cachedInputTokens")),
                        reasoning_tokens=_token(totals.get("reasoningOutputTokens")),
                    )
                case "turn/completed":
                    turn = _mapping(params.get("turn"))
                    if turn.get("id") != self._turn_id:
                        continue
                    match turn.get("status"):
                        case "completed":
                            if text is None:
                                raise LLMError(
                                    "invalid_output", retryable=False,
                                    message="Codex returned no final agent message",
                                )
                            self._thread_id = self._turn_id = None
                            return RawCompletion(
                                text=text, usage=usage,
                                resolved_model=(
                                    resolved_model if isinstance(resolved_model, str) else None
                                ),
                            )
                        case "interrupted":
                            raise LLMError(
                                "cancelled", retryable=False, message="Codex turn interrupted"
                            )
                        case _:
                            raise self._error(_mapping(turn.get("error")))
                case "error":
                    if not params.get("willRetry", False):
                        raise self._error(_mapping(params.get("error")))
                case _:
                    pass

    async def _interrupt(self) -> None:
        if self._thread_id is not None and self._turn_id is not None:
            with anyio.move_on_after(2, shield=True):
                try:
                    _ = await self._request("turn/interrupt", {
                        "threadId": self._thread_id, "turnId": self._turn_id,
                    })
                except (
                    LLMError, OSError, anyio.EndOfStream, anyio.IncompleteRead,
                    anyio.BrokenResourceError,
                ):
                    pass

    async def generate[T](
        self, request: LLMRequest[T], context: CallScope | LedgerCallScope,
    ) -> LLMResult[T]:
        if request.images:
            raise LLMError(
                "configuration", retryable=False, message="Codex provider requires text observations"
            )

        async def send(prompt: StructuredPrompt) -> RawCompletion:
            try:
                with anyio.fail_after(max(0, context.deadline - time.monotonic())):
                    await self._start()
                    match context:
                        case LedgerCallScope() as scope:
                            scope.reserve_attempt()
                        case _:
                            pass
                    return await self._attempt(prompt)
            except anyio.get_cancelled_exc_class():
                await self._interrupt()
                await self.aclose()
                raise
            except TimeoutError:
                await self._interrupt()
                await self.aclose()
                raise LLMError(
                    "timeout", retryable=True, message="Codex turn deadline exceeded"
                ) from None
            except (OSError, anyio.EndOfStream, anyio.IncompleteRead, anyio.BrokenResourceError):
                await self.aclose()
                raise LLMError(
                    "provider_unavailable", retryable=True, message="Codex app-server disconnected"
                ) from None
            except LLMError:
                await self._interrupt()
                await self.aclose()
                raise

        async with self._lock:
            return await generate_structured(
                request, SubscriptionScope(context), ref=self.ref, mode="native", send=send,
            )

    async def aclose(self) -> None:
        """Reap the owned server even during caller cancellation."""
        with anyio.CancelScope(shield=True):
            process, self._process = self._process, None
            if process is not None:
                if process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    with anyio.move_on_after(2):
                        _ = await process.wait()
                    if process.returncode is None:
                        process.kill()
                _ = await process.wait()
                await process.aclose()
            self._reader = None
            self._pending.clear()
            self._thread_id = self._turn_id = None


def _token(value: JsonValue) -> int | None:
    match value:
        case int() if not isinstance(value, bool) and value >= 0:
            return value
        case None:
            return None
        case _:
            raise LLMError(
                "invalid_output", retryable=False, message="Invalid Codex token usage"
            )


async def _smoke(transport: str, home: Path | None, model: str) -> int:
    from pydantic import BaseModel, ConfigDict

    from jevpilot.ledger.recorder import RunLedger

    class Answer(BaseModel):
        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
        answer: str

    ref = ModelRef(
        provider="chatgpt_subscription", auth_mode="subscription",
        model_id=model, reasoning_effort="low",
    )
    chosen: CodexAppServerProvider | CodexExecProvider
    if transport == "exec":
        chosen = CodexExecProvider(ref, codex_home=home)
    else:
        chosen = CodexAppServerProvider(ref, codex_home=home)
    ledger = RunLedger()
    try:
        for index in range(1, 2 if transport == "exec" else 4):
            started = time.monotonic()
            result = await chosen.generate(
                LLMRequest(
                    purpose="text", system="Answer the supplied question briefly.",
                    user_payload="What is 1 + 1?", output_type=Answer,
                ),
                ledger.scope(deadline=started + 120),
            )
            record = ledger.calls[-1]
            print(json.dumps({
                "transport": transport, "call": index,
                "latency_s": round(time.monotonic() - started, 6),
                "tokens": None if result.usage is None else {
                    "input": result.usage.input_tokens, "output": result.usage.output_tokens,
                    "cached": result.usage.cached_input_tokens,
                    "reasoning": result.usage.reasoning_tokens,
                },
                "api_equivalent_usd": (
                    str(record.api_equivalent_usd)
                    if record.api_equivalent_usd is not None else None
                ),
                "billed_usd": None, "resolved_model": result.resolved_model,
            }), flush=True)
    except LLMError as error:
        print(str(error), flush=True)
        return 2 if error.kind == "auth_missing" else 1
    finally:
        await chosen.aclose()
        print("cleanup: provider closed; owned processes reaped", flush=True)
    return 0


def main() -> int:
    """Small explicit QA entry point; never runs as part of ordinary pytest."""
    import argparse

    parser = argparse.ArgumentParser()
    class Arguments(argparse.Namespace):
        smoke: bool = False
        transport: str = "app_server"
        codex_home: Path | None = None
        model: str = "codex-default"

    _ = parser.add_argument("--smoke", action="store_true", required=True)
    _ = parser.add_argument("--transport", choices=("app_server", "exec"), default="app_server")
    _ = parser.add_argument("--codex-home", type=Path)
    _ = parser.add_argument("--model", default="codex-default")
    args = parser.parse_args(namespace=Arguments())
    return anyio.run(_smoke, args.transport, args.codex_home, args.model)


if __name__ == "__main__":
    sys.exit(main())
