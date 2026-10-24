"""Opt-in per-call Codex exec transport; never used as an automatic fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import JsonValue

from jevpilot.codex_auth import codex_home as default_codex_home
from jevpilot.codex_auth import codex_login_status, is_chatgpt_login_status
from jevpilot.codex_client import CodexClient
from jevpilot.ledger.legacy_bridge import legacy_call_context
from jevpilot.ledger.models import CallRecord, CallUsage
from jevpilot.ledger.recorder import CallScope as LedgerCallScope
from jevpilot.ledger.recorder import RunLedger
from jevpilot.llm.errors import LLMError
from jevpilot.llm.structured import RawCompletion, StructuredPrompt, generate_structured
from jevpilot.llm.types import (
    CallScope, Capabilities, LLMCallRecord, LLMRequest, LLMResult, LLMUsage, ModelRef,
)
from jevpilot.model_client import ProviderError
from jevpilot.privacy import PrivacyPolicy


@dataclass(frozen=True, slots=True)
class SubscriptionScope:
    """Shared subscription record translation for either transport."""

    target: CallScope | LedgerCallScope

    @property
    def run_id(self) -> str:
        return self.target.run_id

    @property
    def parent_span_id(self) -> str | None:
        return self.target.parent_span_id

    @property
    def deadline(self) -> float:
        return self.target.deadline

    def record_call(self, record: LLMCallRecord) -> None:
        match self.target:
            case LedgerCallScope() as scope:
                usage = record.usage
                scope.record_call(CallRecord(
                    call_id=record.call_id, span_id=scope.parent_span_id,
                    kind="llm", provider=record.provider, auth_mode="subscription",
                    purpose=record.purpose, request_model=record.model_id,
                    resolved_model=record.resolved_model,
                    outcome="ok" if record.outcome == "ok" else (
                        "cancelled" if record.outcome == "cancelled" else "failed"
                    ),
                    error_kind=None if record.outcome == "ok" else record.outcome,
                    usage=CallUsage(
                        input_tokens=usage.input_tokens if usage else None,
                        output_tokens=usage.output_tokens if usage else None,
                        cached_input_tokens=usage.cached_input_tokens if usage else None,
                        reasoning_tokens=usage.reasoning_tokens if usage else None,
                    ),
                    latency_s=record.ended_monotonic - record.started_monotonic,
                ))
            case _:
                self.target.record_call(record)


async def check_subscription_login(command: tuple[str, ...], profile: Path) -> None:
    """Require official ChatGPT auth, never ambient API credentials."""
    try:
        status = await codex_login_status(command=command, profile=profile)
    except TimeoutError:
        raise LLMError(
            "timeout", retryable=True, message="Codex login status timed out"
        ) from None
    except OSError:
        raise LLMError(
            "provider_unavailable", retryable=False, message="Codex CLI is unavailable"
        ) from None
    if not is_chatgpt_login_status(status):
        raise LLMError(
            "auth_missing", retryable=False,
            message="ChatGPT subscription login is required; run /login",
        )


def require_subscription_ref(ref: ModelRef) -> None:
    if ref.provider != "chatgpt_subscription" or ref.auth_mode != "subscription":
        raise LLMError(
            "configuration", retryable=False, message="A ChatGPT subscription model is required"
        )


class CodexExecProvider:
    """Capture frozen v1 events, then record each structured attempt exactly once."""

    def __init__(
        self, ref: ModelRef, *, codex_home: Path | None = None,
        command: tuple[str, ...] = ("codex",), privacy: PrivacyPolicy | None = None,
    ) -> None:
        require_subscription_ref(ref)
        self.ref: ModelRef = ref
        self.codex_home: Path = codex_home or default_codex_home()
        self.command: tuple[str, ...] = command
        self._client: CodexClient = CodexClient(
            privacy=privacy or PrivacyPolicy(), command=command, codex_home=self.codex_home,
        )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            structured_output="native", images=False,
            reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
            supports_fast=True,
        )

    async def generate[T](
        self, request: LLMRequest[T], context: CallScope | LedgerCallScope,
    ) -> LLMResult[T]:
        if request.images:
            raise LLMError(
                "configuration", retryable=False, message="Codex provider requires text observations"
            )

        async def send(prompt: StructuredPrompt) -> RawCompletion:
            await check_subscription_login(self.command, self.codex_home)
            # The bridge observes v1 usage even when v1's schema validation fails.
            # Do not feed both legacy and structured records into the caller ledger.
            captured = RunLedger(context.run_id)
            legacy = legacy_call_context(
                captured.scope(
                    deadline=context.deadline,
                    reserve_attempt=(
                        context.reserve_attempt if isinstance(context, LedgerCallScope) else lambda: None
                    ),
                ),
                role="selector", max_attempts=1,
            )
            schema: JsonValue = prompt.json_schema
            instructions = prompt.system
            if schema is None:
                schema = {
                    "type": "object", "properties": {"text": {"type": "string"}},
                    "required": ["text"], "additionalProperties": False,
                }
                instructions += "\nReturn the text response in the JSON object's text field."
            text = ""
            resolved: str | None = None
            try:
                result = await self._client.generate_json(
                    instructions=instructions, payload=prompt.user_text, schema=schema,
                    request_model=self.ref.model_id, context=legacy,
                    reasoning_effort=self.ref.reasoning_effort or "medium", fast=self.ref.fast,
                )
                if request.output_type is str:
                    match result.data:
                        case {"text": str() as response}:
                            text = response
                        case _:
                            raise LLMError(
                                "invalid_output", retryable=False,
                                message="Codex exec returned invalid text output",
                            )
                else:
                    text = json.dumps(result.data)
                resolved = result.response_model
            except ProviderError as error:
                match error.kind:
                    case "invalid_response":
                        # Shared validation owns the single repair budget.
                        if request.output_type is str:
                            raise LLMError(
                                "invalid_output", retryable=False,
                                message="Codex exec returned invalid text output",
                            ) from None
                    case "unexpected_tool_use":
                        raise LLMError(
                            "invalid_output", retryable=False, message="Codex returned a tool item"
                        ) from None
                    case "transport_timeout":
                        raise LLMError(
                            "timeout", retryable=True, message="Codex exec deadline exceeded"
                        ) from None
                    case "configuration":
                        raise LLMError(
                            "configuration", retryable=False, message="Codex exec is not configured"
                        ) from None
                    case _:
                        raise LLMError(
                            "provider_unavailable", retryable=True, message="Codex exec failed"
                        ) from None
            usage: LLMUsage | None = None
            if captured.calls:
                recorded = captured.calls[-1].usage
                usage = LLMUsage(
                    input_tokens=recorded.input_tokens, output_tokens=recorded.output_tokens,
                    cached_input_tokens=recorded.cached_input_tokens,
                    reasoning_tokens=recorded.reasoning_tokens,
                )
            return RawCompletion(text=text, resolved_model=resolved, usage=usage)

        return await generate_structured(
            request, SubscriptionScope(context), ref=self.ref, mode="native", send=send,
        )

    async def aclose(self) -> None:
        """CodexClient owns and reaps each per-call process."""
