import base64
import os
import functools
import threading
import time
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from jevpilot.browser.env import prepare_browser_use_environment

prepare_browser_use_environment()

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from pydantic import BaseModel

from jevpilot.llm import (
    CallScope,
    Capabilities,
    LLMCallRecord,
    LLMError,
    LLMRequest,
    LLMResult,
    LLMUsage,
    ModelRef,
)
from jevpilot.llm.bridge import JevPilotChatModel
from jevpilot.llm.structured import parse_structured

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_FIXTURE = REPO_ROOT / "benchmarks" / "tasks" / "fixtures" / "search.html"
PNG = b"\x89PNG\r\n\x1a\nfake"


class _Scope:
    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    @property
    def run_id(self) -> str:
        return "run-1"

    @property
    def parent_span_id(self) -> str | None:
        return None

    @property
    def deadline(self) -> float:
        return time.monotonic() + 60

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class _FakeProvider:
    def __init__(
        self,
        reply: str = "hello",
        *,
        images: bool = True,
        usage: LLMUsage | None = None,
        error: LLMError | None = None,
    ) -> None:
        self._ref = ModelRef(provider="openai", auth_mode="api", model_id="fake-1")
        self._reply = reply
        self._images = images
        self._usage = usage
        self._error = error
        self.requests: list[LLMRequest[Any]] = []

    @property
    def ref(self) -> ModelRef:
        return self._ref

    def capabilities(self) -> Capabilities:
        return Capabilities(
            structured_output="prompted",
            images=self._images,
            reasoning_efforts=(),
            supports_fast=False,
        )

    async def generate[T](self, request: LLMRequest[T], context: CallScope) -> LLMResult[T]:
        _ = context
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMResult(
            value=parse_structured(self._reply, request.output_type),
            resolved_model="fake-1",
            usage=self._usage,
            attempts=1,
            call_ids=["c1"],
        )

    async def aclose(self) -> None:
        return None


class _Answer(BaseModel):
    city: str
    count: int


def _png_message() -> UserMessage:
    url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    return UserMessage(
        content=[
            ContentPartTextParam(text="look"),
            ContentPartImageParam(image_url=ImageURL(url=url)),
        ]
    )


def test_is_base_chat_model() -> None:
    model = JevPilotChatModel(_FakeProvider())
    assert isinstance(model, BaseChatModel)
    assert model.model == "fake-1"
    assert model.model_name == "fake-1"
    assert model.provider == "openai"
    assert model.name == "openai/fake-1"


async def test_string_output_and_role_preservation() -> None:
    provider = _FakeProvider("hello")
    model = JevPilotChatModel(provider)
    result = await model.ainvoke(
        [
            SystemMessage(content="be brief"),
            UserMessage(content="q1"),
            AssistantMessage(content="a1"),
            UserMessage(content="q2"),
        ]
    )
    assert result.completion == "hello"
    request = provider.requests[0]
    assert request.purpose == "agent_step"
    assert request.output_type is str
    assert request.system == "be brief"
    assert request.user_payload == "[user]\nq1\n\n[assistant]\na1\n\n[user]\nq2"


async def test_structured_output_round_trip() -> None:
    provider = _FakeProvider('{"city": "Seoul", "count": 3}')
    model = JevPilotChatModel(provider)
    result = await model.ainvoke([UserMessage(content="go")], _Answer)
    assert result.completion == _Answer(city="Seoul", count=3)
    assert provider.requests[0].output_type is _Answer


async def test_images_forwarded_when_supported() -> None:
    provider = _FakeProvider(images=True)
    model = JevPilotChatModel(provider)
    assert model.supports_vision is True
    _ = await model.ainvoke([_png_message()])
    assert provider.requests[0].images == [PNG]
    assert provider.requests[0].user_payload == "[user]\nlook"


async def test_images_dropped_without_vision_support() -> None:
    provider = _FakeProvider(images=False)
    model = JevPilotChatModel(provider)
    assert model.supports_vision is False
    _ = await model.ainvoke([_png_message()])
    assert provider.requests[0].images == []


async def test_full_usage_maps_to_chat_invoke_usage() -> None:
    usage = LLMUsage(
        input_tokens=100, output_tokens=20, cached_input_tokens=40, reasoning_tokens=None
    )
    model = JevPilotChatModel(_FakeProvider(usage=usage))
    result = await model.ainvoke([UserMessage(content="x")])
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.prompt_cached_tokens == 40
    assert result.usage.completion_tokens == 20
    assert result.usage.total_tokens == 120


@pytest.mark.parametrize(
    "usage",
    [
        None,
        LLMUsage(
            input_tokens=None, output_tokens=5, cached_input_tokens=None, reasoning_tokens=None
        ),
        LLMUsage(
            input_tokens=5, output_tokens=None, cached_input_tokens=None, reasoning_tokens=None
        ),
    ],
)
async def test_incomplete_usage_becomes_none(usage: LLMUsage | None) -> None:
    model = JevPilotChatModel(_FakeProvider(usage=usage))
    result = await model.ainvoke([UserMessage(content="x")])
    assert result.usage is None


async def test_invalid_output_raises_browser_use_provider_error() -> None:
    error = LLMError("invalid_output", retryable=False, message="bad json")
    model = JevPilotChatModel(_FakeProvider(error=error))
    with pytest.raises(ModelProviderError) as caught:
        _ = await model.ainvoke([UserMessage(content="x")], _Answer)
    assert not isinstance(caught.value, ModelRateLimitError)
    assert "invalid_output" in caught.value.message
    assert caught.value.model == "fake-1"


async def test_rate_limit_maps_to_rate_limit_error() -> None:
    error = LLMError("rate_limited", retryable=True, message="slow down")
    model = JevPilotChatModel(_FakeProvider(error=error))
    with pytest.raises(ModelRateLimitError):
        _ = await model.ainvoke([UserMessage(content="x")])


async def test_scope_factory_is_used_per_call() -> None:
    scopes: list[_Scope] = []

    def factory() -> _Scope:
        scopes.append(_Scope())
        return scopes[-1]

    model = JevPilotChatModel(_FakeProvider(), scope_factory=factory)
    _ = await model.ainvoke([UserMessage(content="x")])
    _ = await model.ainvoke([UserMessage(content="y")])
    assert len(scopes) == 2


_DONE = (
    '{"evaluation_previous_goal": "start", "memory": "m", "next_goal": "finish", '
    '"action": [{"done": {"text": "ok", "success": true}}]}'
)


def _attempt_file() -> Path:
    base = Path(os.environ.get("JEVPILOT_ATTEMPT_DIR", str(REPO_ROOT / ".task-9-out")))
    base.mkdir(parents=True, exist_ok=True)
    return base / "task-9-jevpilot-browser-use.txt"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)


@pytest.fixture(autouse=True)
def _no_version_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROWSER_USE_VERSION_CHECK", "false")


@pytest.fixture
def search_url() -> Iterator[str]:
    handler = functools.partial(_QuietHandler, directory=str(SEARCH_FIXTURE.parent))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{SEARCH_FIXTURE.name}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.browser
async def test_browser_use_agent_runs_one_step_on_fixture(search_url: str) -> None:
    from browser_use import Agent, Browser

    provider = _FakeProvider(_DONE, images=False)
    model = JevPilotChatModel(provider)
    browser = Browser(headless=True, enable_default_extensions=False)
    agent = Agent(
        task=f"Open {search_url} and finish.",
        llm=cast(Any, model),
        browser=browser,
        use_vision=model.supports_vision,
    )
    try:
        history = await agent.run(max_steps=3)
    finally:
        await browser.kill()
    steps = history.number_of_steps()
    with _attempt_file().open("w", encoding="utf-8") as out:
        _ = out.write(f"happy: steps={steps} llm_calls={len(provider.requests)}\n")
    assert steps >= 1
    assert provider.requests[0].purpose == "agent_step"


@pytest.mark.browser
async def test_browser_use_agent_survives_provider_invalid_output(search_url: str) -> None:
    from browser_use import Agent, Browser

    error = LLMError("invalid_output", retryable=False, message="bad json")
    provider = _FakeProvider(error=error, images=False)
    model = JevPilotChatModel(provider)
    browser = Browser(headless=True, enable_default_extensions=False)
    agent = Agent(
        task=f"Open {search_url} and finish.",
        llm=cast(Any, model),
        browser=browser,
        use_vision=False,
        max_failures=1,
    )
    try:
        history = await agent.run(max_steps=2)
    finally:
        await browser.kill()
    errors = [e for e in history.errors() if e]
    with _attempt_file().open("a", encoding="utf-8") as out:
        _ = out.write(f"failure: errors={len(errors)} first={errors[0][:120]!r}\n")
    assert errors
    assert "invalid_output" in errors[0]
    assert not history.is_successful()
