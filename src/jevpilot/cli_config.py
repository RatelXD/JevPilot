"""Parse and validate user-run CLI arguments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal
from urllib.parse import SplitResult, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Mode = Literal["llm_only", "jev_hybrid"]
_MODE_BY_VALUE: dict[str, Mode] = {
    "llm_only": "llm_only",
    "jev_hybrid": "jev_hybrid",
}
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CliError(ValueError):
    """Safe user-facing command or environment configuration failure."""


class UserRunConfig(BaseModel):
    """Validated URL, goal and code-owned completion checks."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    mode: Mode
    url: NonEmptyText
    goal: NonEmptyText
    allowed_origins: tuple[NonEmptyText, ...]
    verify_url_path: NonEmptyText | None = None
    verify_text: NonEmptyText | None = None
    max_steps: Annotated[int, Field(gt=0, le=100)]
    wall_time_ms: Annotated[int, Field(gt=0, le=600_000)]

    @field_validator("allowed_origins")
    @classmethod
    def normalize_allowed_origins(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            try:
                parsed = urlsplit(value)
            except ValueError:
                raise CliError("allowed origin is invalid") from None
            if (
                parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise CliError("allowed origins must not contain credentials or paths")
            normalized.append(_origin(parsed))
        if not normalized or len(set(normalized)) != len(normalized):
            raise CliError("allowed origins must be non-empty and unique")
        return tuple(normalized)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise CliError("URL must be absolute HTTP(S), without embedded credentials")
        return value

    @model_validator(mode="after")
    def validate_completion_conditions(self) -> UserRunConfig:
        if self.verify_url_path is None and self.verify_text is None:
            raise CliError("provide --verify-url-path and/or --verify-text")
        if self.verify_url_path is not None and not self.verify_url_path.startswith("/"):
            raise CliError("--verify-url-path must start with '/'")
        origin = _origin(urlsplit(self.url))
        if origin not in self.allowed_origins:
            raise CliError("the initial URL origin must be explicitly allowed")
        return self


@dataclass(frozen=True, slots=True)
class CliOptions:
    mode: Mode | None
    url: str | None
    goal: str | None
    allowed_origins: tuple[str, ...]
    verify_url_path: str | None
    verify_text: str | None
    max_steps: int
    wall_time_ms: int
    live: bool
    help: bool


USAGE = (
    "usage: jevpilot run --mode llm_only|jev_hybrid --url URL --goal TEXT "
    "(--verify-url-path PATH and/or --verify-text TEXT) "
    "[--allow-origin ORIGIN] [--max-steps N] [--wall-time-ms N] --live"
)


def parse_options(arguments: list[str]) -> CliOptions:
    """Parse one explicit user-run command."""
    if arguments in (["--help"], ["-h"], ["run", "--help"], ["run", "-h"]):
        return CliOptions(None, None, None, (), None, None, 24, 120_000, False, True)
    if not arguments or arguments[0] != "run":
        raise CliError(USAGE)
    fields: dict[str, str] = {}
    origins: list[str] = []
    live = False
    index = 1
    while index < len(arguments):
        key = arguments[index]
        if key == "--live":
            live = True
            index += 1
            continue
        if key not in {
            "--mode",
            "--url",
            "--goal",
            "--allow-origin",
            "--verify-url-path",
            "--verify-text",
            "--max-steps",
            "--wall-time-ms",
        }:
            raise CliError(USAGE)
        if index + 1 >= len(arguments):
            raise CliError(USAGE)
        value = arguments[index + 1]
        if key == "--allow-origin":
            origins.append(value)
        else:
            field = key.removeprefix("--").replace("-", "_")
            if field in fields:
                raise CliError(f"{key} can be supplied only once")
            fields[field] = value
        index += 2
    mode = _MODE_BY_VALUE.get(fields.get("mode", ""))
    if mode is None:
        raise CliError(USAGE)
    try:
        max_steps = int(fields.get("max_steps", "24"))
        wall_time_ms = int(fields.get("wall_time_ms", "120000"))
    except ValueError:
        raise CliError("step and wall-time budgets must be integers") from None
    return CliOptions(
        mode=mode,
        url=fields.get("url"),
        goal=fields.get("goal"),
        allowed_origins=tuple(origins),
        verify_url_path=fields.get("verify_url_path"),
        verify_text=fields.get("verify_text"),
        max_steps=max_steps,
        wall_time_ms=wall_time_ms,
        live=live,
        help=False,
    )


def _origin(parsed: SplitResult) -> str:
    scheme = parsed.scheme
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or hostname is None:
        raise CliError("URL origin is invalid")
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parsed.port
    except ValueError:
        raise CliError("URL origin is invalid") from None
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
    suffix = f":{port}" if port is not None else ""
    return f"{scheme}://{host}{suffix}"


def create_user_config(options: CliOptions) -> UserRunConfig:
    """Require a live user task and derive exact allowed origins."""
    if not options.live:
        raise CliError("real model execution requires the explicit --live option")
    if options.url is None or options.goal is None or options.mode is None:
        raise CliError(USAGE)
    try:
        parsed = urlsplit(options.url)
        initial_origin = _origin(parsed)
    except ValueError:
        raise CliError("URL is invalid") from None
    origins = tuple(dict.fromkeys((initial_origin, *options.allowed_origins)))
    return UserRunConfig(
        mode=options.mode,
        url=options.url,
        goal=options.goal,
        allowed_origins=origins,
        verify_url_path=options.verify_url_path,
        verify_text=options.verify_text,
        max_steps=options.max_steps,
        wall_time_ms=options.wall_time_ms,
    )
