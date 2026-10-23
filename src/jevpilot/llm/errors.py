"""Safe LLM provider failures shared by every provider adapter.

Messages must never carry API keys, prompt text, or raw model output: callers
construct them from fixed strings and structural details only.
"""

from __future__ import annotations

from typing import Literal, override

LLMErrorKind = Literal[
    "auth_missing",
    "auth_invalid",
    "rate_limited",
    "quota_exhausted",
    "timeout",
    "invalid_output",
    "provider_unavailable",
    "cancelled",
    "configuration",
]


class LLMError(Exception):
    kind: LLMErrorKind
    retryable: bool
    retry_after_s: float | None
    message: str

    def __init__(
        self,
        kind: LLMErrorKind,
        *,
        retryable: bool,
        message: str,
        retry_after_s: float | None = None,
    ) -> None:
        self.kind = kind
        self.retryable = retryable
        self.retry_after_s = retry_after_s
        self.message = message
        super().__init__(message)

    @override
    def __str__(self) -> str:
        return f"{self.message} (kind={self.kind})"

    @override
    def __repr__(self) -> str:
        return (
            f"LLMError(kind={self.kind!r}, retryable={self.retryable!r}, "
            f"retry_after_s={self.retry_after_s!r}, message={self.message!r})"
        )
