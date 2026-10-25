import getpass
import json
import stat
import subprocess
import threading
import warnings
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import override

import anyio
import httpx
import httpx2
import pytest
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion
from pydantic import JsonValue, TypeAdapter

from jevpilot.ledger.models import CallRecord, CallUsage, RunSummary
from jevpilot.llm.api_providers import ApiProviderId, list_models, validate_key
from jevpilot.llm.credentials import get_api_key, load_credentials, save_credential
from jevpilot.llm.profile import Profile, load_profile, save_profile
from jevpilot.llm.types import ModelRef
from jevpilot.model_catalog import ChatGPTModel
from jevpilot.shell import login as auth

KEY = "synthetic-auth-candidate-9182"
JSON = TypeAdapter(dict[str, JsonValue])


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in (
        "JEVPILOT_OPENAI_API_KEY", "JEVPILOT_ANTHROPIC_API_KEY",
        "JEVPILOT_JEV_API_KEY", "JEVPILOT_OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def inputs(monkeypatch: pytest.MonkeyPatch, choices: list[str], secrets: list[str]) -> None:
    answers = iter(choices)
    hidden = iter(secrets)

    def answer(_prompt: str) -> str:
        return next(answers)

    def secret(_prompt: str) -> str:
        return next(hidden)

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(getpass, "getpass", secret)


@contextmanager
def fake_api(status: int) -> Generator[tuple[str, list[str]]]:
    """Serve models=200 independently of the actual inference response."""
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: str) -> None:
            pass

        def reply(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            _ = self.wfile.write(body)

        def do_GET(self) -> None:
            seen.append(self.path)
            assert self.headers["Authorization"] == f"Bearer {KEY}"
            self.reply(200, b'{"data":[{"id":"fake-model"}]}')

        def do_POST(self) -> None:
            seen.append(self.path)
            assert self.headers["Authorization"] == f"Bearer {KEY}"
            body = JSON.validate_json(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "fake-model"
            assert KEY not in json.dumps(body)
            if status == 200:
                self.reply(200, json.dumps({
                    "id": "probe", "object": "chat.completion", "created": 1,
                    "model": "fake-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                 "finish_reason": "stop"}],
                }).encode())
            else:
                self.reply(status, b'{"error":{"message":"rejected","type":"invalid_api_key"}}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("status", [200, 401])
@pytest.mark.parametrize("existing", [False, True])
async def test_selected_inference_controls_persistence(
    home: Path, monkeypatch: pytest.MonkeyPatch, status: int, existing: bool,
) -> None:
    # Given
    current = Profile("api", ModelRef("anthropic", "api", "old"))
    directory = home / ".jevpilot"
    if existing:
        save_credential("anthropic", "previous")
        save_profile(current)
    before = {
        name: (directory / name).read_bytes() if (directory / name).exists() else None
        for name in ("credentials.json", "profile.json")
    }
    inputs(monkeypatch, ["1", "1", "1"], [KEY, ""])
    messages: list[str] = []
    with fake_api(status) as (base_url, seen):
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)
        # An old/env key must never replace the candidate for validation.
        monkeypatch.setenv("JEVPILOT_OPENAI_API_KEY", "old-environment-key")
        # When
        result = await auth.login(current if existing else None, emit=messages.append)
    # Then
    assert seen == ["/v1/models", "/v1/models", "/v1/chat/completions"]
    if status == 200:
        assert result == load_profile()
        assert result is not None and result.model.model_id == "fake-model"
        assert load_credentials()["openai"].api_key == KEY
        assert stat.S_IMODE((directory / "credentials.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((directory / "profile.json").stat().st_mode) == 0o600
    else:
        assert result == (current if existing else None)
        for name, content in before.items():
            path = directory / name
            assert (path.read_bytes() if path.exists() else None) == content
    events_path, = directory.glob("runs/auth-*/events.jsonl")
    events = [JSON.validate_json(line) for line in events_path.read_text().splitlines()]
    call, = [CallRecord.model_validate(event["data"]) for event in events if event["type"] == "call"]
    assert call.outcome == ("ok" if status == 200 else "failed")
    assert call.error_kind == (None if status == 200 else "auth_invalid")
    assert call.usage == CallUsage()
    assert call.billed_usd is None
    assert call.api_estimate_usd is None
    assert KEY not in "\n".join(messages) + events_path.read_text()
    assert (events_path.parent / "receipt.json").is_file()


@pytest.mark.parametrize("choice", ["3", "4", "google", "openrouter", "openai_compatible", "chosun_gateway"])
async def test_removed_provider_choice_never_requests_or_saves(
    home: Path, monkeypatch: pytest.MonkeyPatch, choice: str,
) -> None:
    # Given
    inputs(monkeypatch, ["1", choice], [])
    # When
    result = await auth.login(emit=lambda _text: None)
    # Then
    assert result is None
    assert not (home / ".jevpilot" / "credentials.json").exists()
    assert not (home / ".jevpilot" / "profile.json").exists()


@pytest.mark.parametrize("authorized", [True, False])
async def test_subscription_status_login_recheck_and_catalog(
    home: Path, monkeypatch: pytest.MonkeyPatch, authorized: bool,
) -> None:
    # Given
    inputs(monkeypatch, ["2", "1"], [""])
    statuses = iter([authorized, True])
    launches: list[int] = []

    async def status() -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            [], 0, b"Logged in using ChatGPT" if next(statuses) else b"", b"",
        )

    async def catalog() -> tuple[ChatGPTModel, ...]:
        return (ChatGPTModel("catalog-model", "Catalog", True),)

    monkeypatch.setattr(auth, "codex_login_status", status)
    monkeypatch.setattr(auth, "list_chatgpt_models", catalog)
    monkeypatch.setattr(auth, "launch_chatgpt_login", lambda: launches.append(1) or 0)
    # When
    result = await auth.login(emit=lambda _text: None)
    # Then
    assert result == load_profile()
    assert result is not None and result.auth_mode == "subscription"
    assert result.codex_transport == "app_server"
    assert result.model.provider == "chatgpt_subscription"
    assert launches == ([] if authorized else [1])
    assert not (home / ".jevpilot" / "credentials.json").exists()


async def test_subscription_failed_recheck_preserves_profile(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    current = Profile("api", ModelRef("openai", "api", "old"))
    save_profile(current)
    before = (home / ".jevpilot" / "profile.json").read_bytes()
    inputs(monkeypatch, ["2"], [])

    async def status() -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess([], 1, b"", b"")

    monkeypatch.setattr(auth, "codex_login_status", status)
    monkeypatch.setattr(auth, "launch_chatgpt_login", lambda: 0)
    # When
    result = await auth.login(current, emit=lambda _text: None)
    # Then
    assert result == current
    assert (home / ".jevpilot" / "profile.json").read_bytes() == before
    assert not (home / ".jevpilot" / "credentials.json").exists()


@pytest.mark.parametrize("status", [200, 204, 401, 403, 500, None])
async def test_jev_real_validation_accounts_only_observed_http(
    home: Path, monkeypatch: pytest.MonkeyPatch, status: int | None,
) -> None:
    # Given
    inputs(monkeypatch, ["3"], [KEY])
    original = httpx2.AsyncClient
    sent: list[int] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(1)
        if status is None:
            raise httpx2.ConnectError("offline", request=request)
        return httpx2.Response(status, content=b"")

    def client(
        *, event_hooks: Mapping[str, list[Callable[[httpx2.Response], Awaitable[None]]]],
    ) -> httpx2.AsyncClient:
        return original(transport=httpx2.MockTransport(handle), event_hooks=event_hooks)

    monkeypatch.setattr(httpx2, "AsyncClient", client)
    messages: list[str] = []
    # When
    assert await auth.login(emit=messages.append) is None
    # Then
    assert len(sent) == 1
    assert get_api_key("jev") == (KEY if status in (200, 204) else None)
    events_path, = (home / ".jevpilot").glob("runs/auth-*/events.jsonl")
    events = [JSON.validate_json(line) for line in events_path.read_text().splitlines()]
    calls = [CallRecord.model_validate(event["data"]) for event in events if event["type"] == "call"]
    if status is None:
        assert not calls
        assert any(event["type"] == "recovery" for event in events)
    else:
        call, = calls
        assert call.outcome == ("ok" if status in (200, 204) else "failed")
        assert call.usage.input_tokens is None and call.billed_usd is None
    assert KEY not in events_path.read_text() + "\n".join(messages)
    assert not (home / ".jevpilot" / "profile.json").exists()


async def test_jev_environment_precedence_never_prompts_or_saves(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv("JEVPILOT_JEV_API_KEY", KEY)
    inputs(monkeypatch, ["3"], [])
    messages: list[str] = []
    # When
    _ = await auth.login(emit=messages.append)
    # Then
    assert messages and KEY not in "\n".join(messages)
    assert not (home / ".jevpilot" / "credentials.json").exists()


async def test_no_echo_fallback_refuses_secret_input(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    inputs(monkeypatch, ["1", "1"], [])

    def unsafe_getpass(_prompt: str) -> str:
        warnings.warn("echo unavailable", getpass.GetPassWarning)
        raise AssertionError("must refuse before fallback")

    monkeypatch.setattr(getpass, "getpass", unsafe_getpass)
    # When
    result = await auth.login(emit=lambda _text: None)
    # Then
    assert result is None
    assert not (home / ".jevpilot" / "credentials.json").exists()


async def test_probe_cancelled_before_response_does_not_persist(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from jevpilot.llm import api_providers

    entered = anyio.Event()
    cancellation = anyio.CancelScope()

    class Chat:
        def __init__(self, **_kwargs: JsonValue | httpx.AsyncClient) -> None:
            pass

        async def ainvoke(self, _messages: list[BaseMessage]) -> ChatInvokeCompletion[str]:
            entered.set()
            await anyio.Event().wait()
            raise AssertionError("cancel did not interrupt")

    monkeypatch.setattr(api_providers, "ChatOpenAI", Chat)
    inputs(monkeypatch, ["1", "1", "1"], [KEY])
    with fake_api(200) as (base_url, _seen):
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)

        async def run() -> None:
            with cancellation:
                _ = await auth.login(emit=lambda _text: None)

        # When
        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(run)
                await entered.wait()
                cancellation.cancel()
    # Then
    assert not (home / ".jevpilot" / "credentials.json").exists()
    assert not (home / ".jevpilot" / "profile.json").exists()
    receipt, = (home / ".jevpilot").glob("runs/auth-*/receipt.json")
    assert RunSummary.model_validate_json(receipt.read_text()).by_kind["llm"].calls == 1


@pytest.mark.parametrize("stage", ["list", "validate"])
async def test_auth_failure_before_inference_preserves_existing_files(
    home: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    # Given
    from jevpilot.llm.errors import LLMError

    current = Profile("api", ModelRef("openai", "api", "old"))
    save_credential("openai", "previous")
    save_profile(current)
    paths = [home / ".jevpilot" / name for name in ("credentials.json", "profile.json")]
    before = [path.read_bytes() for path in paths]
    inputs(monkeypatch, ["1", "1", "1"], [KEY])

    async def reject(
        _provider: ApiProviderId, _key: str, _base_url: str | None,
        _model_id: str | None = None,
    ) -> None:
        raise LLMError("auth_invalid", retryable=False, message="rejected")

    with fake_api(200) as (base_url, seen):
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)
        monkeypatch.setattr(auth, "list_models" if stage == "list" else "validate_key", reject)
        # When
        result = await auth.login(current, emit=lambda _text: None)
    # Then
    assert result == current
    assert [path.read_bytes() for path in paths] == before
    assert seen == ([] if stage == "list" else ["/v1/models"])
    assert not list((home / ".jevpilot").glob("runs/auth-*"))


async def test_partial_storage_failure_does_not_claim_new_profile(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    current = Profile("api", ModelRef("anthropic", "api", "old"))
    save_profile(current)
    inputs(monkeypatch, ["1", "1", "1"], [KEY])

    def deny(_profile: Profile) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(auth, "save_profile", deny)
    with fake_api(200) as (base_url, _seen):
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)
        # When
        result = await auth.login(current, emit=lambda _text: None)
    # Then
    assert result == current == load_profile()
    assert get_api_key("openai") == KEY
    assert (home / ".jevpilot" / "credentials.json").exists()


async def test_anthropic_login_probes_candidate_before_persistence(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    from jevpilot.llm import api_providers

    inputs(monkeypatch, ["1", "2", "1"], [KEY, ""])
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == KEY
        seen.append(request.url.path)
        return httpx.Response(200, json={"data": [{"id": "claude-test"}]})

    async def listed(
        provider: ApiProviderId, key: str, base_url: str | None,
    ) -> list[str]:
        return await list_models(provider, key, base_url, transport=httpx.MockTransport(handle))

    async def validated(
        provider: ApiProviderId, key: str, base_url: str | None, model_id: str,
    ) -> None:
        await validate_key(
            provider, key, base_url, model_id, transport=httpx.MockTransport(handle),
        )

    class Chat:
        def __init__(self, **kwargs: JsonValue | httpx.AsyncClient) -> None:
            assert kwargs["api_key"] == KEY
            assert kwargs["model"] == "claude-test"

        async def ainvoke(self, _messages: list[BaseMessage]) -> ChatInvokeCompletion[str]:
            assert not (home / ".jevpilot" / "credentials.json").exists()
            assert not (home / ".jevpilot" / "profile.json").exists()
            seen.append("inference")
            return ChatInvokeCompletion(completion="ok", usage=None)

    monkeypatch.setattr(auth, "list_models", listed)
    monkeypatch.setattr(auth, "validate_key", validated)
    monkeypatch.setattr(api_providers, "ChatAnthropic", Chat)
    # When
    result = await auth.login(emit=lambda _text: None)
    # Then
    assert result == load_profile()
    assert result is not None and result.model.provider == "anthropic"
    assert get_api_key("anthropic") == KEY
    assert seen == ["/v1/models", "/v1/models", "inference"]
