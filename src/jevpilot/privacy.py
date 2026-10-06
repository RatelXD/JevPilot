"""Sanitize configured synthetic secrets at model and trace boundaries."""

from dataclasses import dataclass, field
import re
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from pydantic import JsonValue

REDACTED: Final = "[REDACTED]"
SENSITIVE_KEYS: Final = frozenset(
    {"authorization", "api_key", "apikey", "password", "cookie", "cookies", "token", "secret"}
)
_URL: Final = re.compile(r"https?://[^\s<>\"']+")
_URL_TRAILING: Final = ".,;:!?)"


@dataclass(frozen=True, slots=True)
class PrivacyPolicy:
    """Exact configured secrets, not a general-purpose PII detector."""

    secret_values: tuple[str, ...] = field(default=(), repr=False)


def redact_text(value: str, *, policy: PrivacyPolicy) -> str:
    """Remove known secrets and private URL components from text."""
    clean = value
    for secret in sorted(policy.secret_values, key=len, reverse=True):
        if secret:
            clean = clean.replace(secret, REDACTED)

    def strip_url(match: re.Match[str]) -> str:
        raw_url = match.group()
        url = raw_url.rstrip(_URL_TRAILING)
        punctuation = raw_url[len(url):]
        try:
            parsed_url = urlsplit(url)
            host = parsed_url.hostname
            if host is None:
                return REDACTED
            authority = f"[{host}]" if ":" in host else host
            if parsed_url.port is not None:
                authority = f"{authority}:{parsed_url.port}"
            sanitized = urlunsplit(
                (parsed_url.scheme, authority, parsed_url.path, "", "")
            )
        except ValueError:
            return REDACTED
        return f"{sanitized}{punctuation}"

    return _URL.sub(strip_url, clean)


def redact_json(value: JsonValue, *, policy: PrivacyPolicy) -> JsonValue:
    """Return a sanitized copy; never mutate the caller's payload."""
    match value:
        case dict():
            return {
                redact_text(key, policy=policy): (
                    REDACTED
                    if key.lower().replace("-", "_") in SENSITIVE_KEYS
                    else redact_json(item, policy=policy)
                )
                for key, item in value.items()
            }
        case list():
            return [redact_json(item, policy=policy) for item in value]
        case str():
            return redact_text(value, policy=policy)
        case bool() | int() | float() | None:
            return value
