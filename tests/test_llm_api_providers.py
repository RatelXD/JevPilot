import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import ContentPartImageParam, SystemMessage, UserMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from pydantic import BaseModel

from benchmarks.browser_use_eval.fake_openai_server import start_in_thread
from jevpilot.ledger import LedgerSink, RunLedger
from jevpilot.llm import api_providers
from jevpilot.llm.api_providers import (
    ApiKeyProvider,
    create_api_provider,
    list_models,
    validate_key,
)
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import LLMCallRecord, LLMRequest, LLMUsage, ModelRef, ProviderId
from jevpilot.task import ContractViolation

SECRET = "sk-test-SECRET-key-9876"


class Brief(BaseModel):
    goal: str
    start_url: str | None
    done_when: str


class _Scope:
    def __init__(self, deadline_s: float = 30) -> None:
        self.records: list[LLMCallRecord] = []
        self._deadline = time.monotonic() + deadline_s

    @property
    def run_id(self) -> str:
        return "run-1"

    @property
    def parent_span_id(self) -> str | None:
        return None

    @property
    def deadline(self) -> float:
        return self._deadline

    def record_call(self, record: LLMCallRecord) -> None:
        self.records.append(record)


class _FakeChat:
    instances: list["_FakeChat"] = []
    result: Any = None
    error: Exception | None = None
    results: list[ChatInvokeCompletion[Brief]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[list[Any], Any]] = []
        _FakeChat.instances.append(self)

    async def ainvoke(self, messages: list[Any], output_format: Any = None) -> Any:
        self.calls.append((messages, output_format))
        if _FakeChat.error is not None:
            raise _FakeChat.error
        if _FakeChat.results:
            return _FakeChat.results.pop(0)
        return _FakeChat.result


@pytest.fixture
def fake_chat(monkeypatch: pytest.MonkeyPatch) -> type[_FakeChat]:
    _FakeChat.instances = []
    _FakeChat.error = None
    _FakeChat.result = None
    _FakeChat.results = []
    monkeypatch.setattr(api_providers, "ChatOpenAI", _FakeChat)
    monkeypatch.setattr(api_providers, "ChatAnthropic", _FakeChat)
    return _FakeChat


def _usage(prompt: int, completion: int, cached: int | None) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt,
        prompt_cached_tokens=cached,
        prompt_cache_creation_tokens=None,
        prompt_image_tokens=None,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


BRIEF = Brief(goal="g", start_url=None, done_when="d")


def _request(**overrides: Any) -> LLMRequest[Any]:
    values: dict[str, Any] = {
        "purpose": "brief", "system": "SYS", "user_payload": {"task": "t"}, "output_type": Brief,
    }
    values.update(overrides)
    return LLMRequest(**values)


async def test_openai_generate_maps_messages_usage_and_cost(fake_chat: type[_FakeChat]) -> None:
    fake_chat.result = ChatInvokeCompletion(completion=BRIEF, usage=_usage(1_000_000, 1_000_000, 0))
    ref = ModelRef(provider="openai", auth_mode="api", model_id="gpt-5.3-codex", reasoning_effort="high")
    provider = ApiKeyProvider(ref, SECRET)
    scope = _Scope()

    result = await provider.generate(_request(), scope)

    assert result.value == BRIEF
    assert result.usage == LLMUsage(1_000_000, 1_000_000, 0, None)
    chat = fake_chat.instances[0]
    assert chat.kwargs["reasoning_effort"] == "high"
    assert chat.kwargs["reasoning_models"] == ["gpt-5.3-codex"]
    assert chat.kwargs["max_retries"] == 0
    messages, output_format = chat.calls[0]
    assert output_format is Brief
    assert isinstance(messages[0], SystemMessage) and messages[0].content == "SYS"
    assert isinstance(messages[1], UserMessage) and json.loads(str(messages[1].content)) == {"task": "t"}
    assert [r.outcome for r in scope.records] == ["ok"]
    assert provider.estimate_usd(result.usage) == Decimal("15.75")
    await provider.aclose()


async def test_openai_without_effort_sends_no_reasoning(fake_chat: type[_FakeChat]) -> None:
    fake_chat.result = ChatInvokeCompletion(completion="hello", usage=None)
    provider = ApiKeyProvider(ModelRef(provider="openai", auth_mode="api", model_id="gpt-x"), SECRET)

    result = await provider.generate(_request(output_type=str, purpose="text"), _Scope())

    assert result.value == "hello" and result.usage is None
    assert fake_chat.instances[0].kwargs["reasoning_models"] == []
    assert fake_chat.instances[0].calls[0][1] is None
    assert provider.estimate_usd(result.usage) is None


async def test_anthropic_thinking_budget_images_and_unknown_cost(fake_chat: type[_FakeChat]) -> None:
    fake_chat.result = ChatInvokeCompletion(completion=BRIEF, usage=_usage(500, 40, 100))
    ref = ModelRef(provider="anthropic", auth_mode="api", model_id="claude-x", reasoning_effort="medium")
    provider = ApiKeyProvider(ref, SECRET)

    result = await provider.generate(_request(images=[b"\x89PNG....", b"\xff\xd8jpeg"]), _Scope())

    chat = fake_chat.instances[0]
    assert chat.kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    assert chat.kwargs["max_tokens"] == 8192 + 8192
    parts = chat.calls[0][0][1].content
    images = [p for p in parts if isinstance(p, ContentPartImageParam)]
    assert [i.image_url.media_type for i in images] == ["image/png", "image/jpeg"]
    assert result.usage == LLMUsage(500, 40, 100, None)
    assert provider.estimate_usd(result.usage) is None
    assert provider.capabilities().reasoning_efforts == ("low", "medium", "high")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ModelProviderError(message=f"bad key {SECRET}", status_code=401), "auth_invalid"),
        (ModelRateLimitError(message="slow"), "rate_limited"),
        (ModelRateLimitError(message="insufficient_quota"), "quota_exhausted"),
        (ModelProviderError(message="boom", status_code=503), "provider_unavailable"),
    ],
)
async def test_provider_errors_are_normalized_without_secrets(
    fake_chat: type[_FakeChat], error: Exception, kind: str
) -> None:
    fake_chat.error = error
    provider = ApiKeyProvider(ModelRef(provider="openai", auth_mode="api", model_id="m"), SECRET)
    scope = _Scope()

    with pytest.raises(LLMError) as caught:
        _ = await provider.generate(_request(), scope)

    assert caught.value.kind == kind
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)
    assert caught.value.__cause__ is None
    assert [r.outcome for r in scope.records] == [kind]
    assert SECRET not in repr(provider)


async def test_validation_failure_maps_to_invalid_output(fake_chat: type[_FakeChat]) -> None:
    try:
        _ = Brief.model_validate_json("{}")
    except Exception as exc:
        wrapped = ModelProviderError(message="raw model text", status_code=502)
        wrapped.__cause__ = exc
        fake_chat.error = wrapped
    provider = ApiKeyProvider(ModelRef(provider="anthropic", auth_mode="api", model_id="m"), SECRET)

    with pytest.raises(LLMError) as caught:
        _ = await provider.generate(_request(), _Scope())

    assert caught.value.kind == "invalid_output"
    assert "raw model text" not in str(caught.value)


def test_configuration_errors() -> None:
    with pytest.raises(LLMError) as effort:
        _ = ApiKeyProvider(
            ModelRef(provider="anthropic", auth_mode="api", model_id="m", reasoning_effort="xhigh"), SECRET
        )
    assert effort.value.kind == "configuration"
    with pytest.raises(LLMError) as subscription:
        _ = ApiKeyProvider(
            ModelRef(provider="chatgpt_subscription", auth_mode="subscription", model_id="m"), SECRET
        )
    assert subscription.value.kind == "configuration"


def test_create_api_provider_requires_key(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("JEVPILOT_ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError) as caught:
        _ = create_api_provider(ModelRef(provider="anthropic", auth_mode="api", model_id="m"))
    assert caught.value.kind == "auth_missing"
    monkeypatch.setenv("JEVPILOT_ANTHROPIC_API_KEY", SECRET)
    provider = create_api_provider(ModelRef(provider="anthropic", auth_mode="api", model_id="m"))
    assert SECRET not in repr(provider)


def _transport(status: int, seen: list[httpx.Request], body: Any = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        payload = body if body is not None else {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_list_models_success_and_auth_headers(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVPILOT_OPENAI_BASE_URL", raising=False)
    seen: list[httpx.Request] = []

    models = await list_models(provider, SECRET, transport=_transport(200, seen))

    assert models == ["model-a", "model-b"]
    request = seen[0]
    if provider == "openai":
        assert str(request.url) == "https://api.openai.com/v1/models"
        assert request.headers["authorization"] == f"Bearer {SECRET}"
    else:
        assert request.url.path == "/v1/models" and request.url.host == "api.anthropic.com"
        assert request.headers["x-api-key"] == SECRET
        assert request.headers["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (401, {"error": {"message": "no"}}, "auth_invalid"),
        (403, {"error": {"message": "no"}}, "auth_invalid"),
        (429, {"error": {"message": "slow"}}, "rate_limited"),
        (429, {"error": {"code": "insufficient_quota"}}, "quota_exhausted"),
    ],
)
async def test_list_models_errors(provider: Any, status: int, body: Any, kind: str) -> None:
    with pytest.raises(LLMError) as caught:
        _ = await list_models(provider, SECRET, transport=_transport(status, [], body))
    assert caught.value.kind == kind
    assert SECRET not in str(caught.value) and SECRET not in repr(caught.value)


async def test_validate_key_checks_model_and_fails_on_401() -> None:
    await validate_key("openai", SECRET, model_id="model-a", transport=_transport(200, []))
    with pytest.raises(LLMError) as missing:
        await validate_key("openai", SECRET, model_id="model-z", transport=_transport(200, []))
    assert missing.value.kind == "configuration"
    with pytest.raises(LLMError) as rejected:
        await validate_key("anthropic", SECRET, transport=_transport(401, [], {"error": {}}))
    assert rejected.value.kind == "auth_invalid"


async def test_list_models_rejects_empty_key() -> None:
    with pytest.raises(LLMError) as caught:
        _ = await list_models("openai", "  ")
    assert caught.value.kind == "auth_missing"


@pytest.mark.parametrize("provider_id", ["openai", "anthropic"])
async def test_generate_records_each_attempt_in_real_ledger(
    provider_id: ProviderId, fake_chat: type[_FakeChat],
) -> None:
    fake_chat.results = [
        ChatInvokeCompletion(completion=Brief.model_construct(goal="incomplete"), usage=_usage(100, 10, 20)),
        ChatInvokeCompletion(completion=BRIEF, usage=_usage(200, 20, 30)),
    ]
    provider = ApiKeyProvider(ModelRef(provider=provider_id, auth_mode="api", model_id="unknown"), SECRET)
    observed = []
    ledger = RunLedger("api-attempts", on_call=observed.append)
    reserved: list[int] = []
    with ledger.span("brief") as parent:
        scope = ledger.scope(
            deadline=time.monotonic() + 30, step_index=7,
            reserve_attempt=lambda: reserved.append(len(observed)),
        )
        try:
            result = await provider.generate(_request(), scope)
        finally:
            await provider.aclose()

    assert result.value == BRIEF and result.attempts == 2
    assert result.usage == LLMUsage(300, 30, 50, None)
    assert reserved == [0, 1]
    assert [r.call_id for r in ledger.calls] == result.call_ids
    assert observed == ledger.calls and len(observed) == 2
    assert [r.attempts for r in ledger.calls] == [1, 1]
    assert [r.outcome for r in ledger.calls] == ["failed", "ok"]
    assert [r.error_kind for r in ledger.calls] == ["invalid_output", None]
    assert [r.usage.input_tokens for r in ledger.calls] == [100, 200]
    assert [r.usage.output_tokens for r in ledger.calls] == [10, 20]
    assert [r.usage.cached_input_tokens for r in ledger.calls] == [20, 30]
    assert scope.run_id == ledger.run_id and scope.parent_span_id == parent.span_id
    assert all(r.span_id == parent.span_id and r.provider == provider_id for r in ledger.calls)
    assert all(r.purpose == "brief" and r.auth_mode == "api" for r in ledger.calls)
    assert all(r.latency_s >= 0 and r.billed_usd is None for r in ledger.calls)
    assert all(r.api_estimate_usd is None and r.api_equivalent_usd is None for r in ledger.calls)


@pytest.mark.parametrize("provider_id", ["openai", "anthropic"])
async def test_real_ledger_preserves_error_and_unknown_usage(
    provider_id: ProviderId, fake_chat: type[_FakeChat],
) -> None:
    fake_chat.error = ModelProviderError(message=SECRET, status_code=401)
    provider = ApiKeyProvider(ModelRef(provider=provider_id, auth_mode="api", model_id="unknown"), SECRET)
    ledger = RunLedger()
    try:
        with pytest.raises(LLMError) as caught:
            await provider.generate(_request(), ledger.scope(deadline=time.monotonic() + 30))
    finally:
        await provider.aclose()
    assert caught.value.kind == "auth_invalid"
    assert len(ledger.calls) == 1
    record = ledger.calls[0]
    assert record.outcome == "failed" and record.error_kind == "auth_invalid"
    assert all(value is None for value in record.usage.model_dump().values())
    assert record.billed_usd is None and record.api_estimate_usd is None
    assert SECRET not in record.model_dump_json()


@pytest.mark.parametrize("expired", [False, True])
async def test_real_ledger_deadline_and_budget_prevent_transmission(
    expired: bool, fake_chat: type[_FakeChat],
) -> None:
    provider = ApiKeyProvider(ModelRef(provider="openai", auth_mode="api", model_id="unknown"), SECRET)
    ledger = RunLedger()
    reserved: list[int] = []

    def deny_attempt() -> None:
        reserved.append(1)
        raise ContractViolation("budget", "attempt budget exceeded")

    scope = ledger.scope(
        deadline=time.monotonic() + (-1 if expired else 30),
        reserve_attempt=deny_attempt,
    )
    try:
        with pytest.raises(LLMError if expired else ContractViolation):
            await provider.generate(_request(), scope)
    finally:
        await provider.aclose()
    assert reserved == ([] if expired else [1])
    assert not ledger.calls
    assert all(not chat.calls for chat in fake_chat.instances)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("status", [199, 200, 204, 299, 300, 302, 399])
async def test_model_list_authentication_status_boundaries(
    provider: api_providers.ApiProviderId, status: int,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status, content=b"" if status == 204 else b'{"data":[{"id":"model-a"}]}',
            headers={"location": "https://untrusted.invalid/models"},
        )

    transport = httpx.MockTransport(handler)
    if 200 <= status < 300:
        assert await list_models(provider, SECRET, transport=transport) == (
            [] if status == 204 else ["model-a"]
        )
        await validate_key(provider, SECRET, transport=transport)
        if status == 204:
            with pytest.raises(LLMError) as missing:
                await validate_key(provider, SECRET, model_id="model-a", transport=transport)
            assert missing.value.kind == "configuration"
    else:
        for operation in (list_models, validate_key):
            with pytest.raises(LLMError) as rejected:
                await operation(provider, SECRET, transport=transport)
            assert rejected.value.kind == "provider_unavailable"
    assert len(seen) == (3 if status == 204 else 2)
    assert all(request.url.host != "untrusted.invalid" for request in seen)


async def test_openai_over_real_http_to_fake_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    server, base_url = start_in_thread()
    try:
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)
        assert await list_models("openai", SECRET) == ["fake-model"]
        provider = ApiKeyProvider(ModelRef(provider="openai", auth_mode="api", model_id="gpt-5.3-codex"), SECRET)
        sink = LedgerSink("api-http", root=tmp_path)
        ledger = RunLedger("api-http", sink=sink)
        scope = ledger.scope(deadline=time.monotonic() + 30)
        result = await provider.generate(_request(), scope)
        await provider.aclose()
    finally:
        server.shutdown()
        server.server_close()
    assert result.value.goal == "find the north region price"
    assert result.usage == LLMUsage(120, 30, 20, None)
    assert [r.outcome for r in ledger.calls] == ["ok"]
    record = ledger.calls[0]
    assert record.call_id == result.call_ids[0] and record.span_id == ledger.root_span_id
    assert record.usage.model_dump() == {
        "input_tokens": 120, "output_tokens": 30, "cached_input_tokens": 20, "reasoning_tokens": None,
    }
    assert record.api_estimate_usd == Decimal("0.0005985")
    assert record.billed_usd is None and record.api_equivalent_usd is None
    assert record.price_source is not None and record.price_checked_on == "2026-10-23"
    _ = ledger.finish()
    receipt = json.loads((sink.directory / "receipt.json").read_text())
    totals = receipt["by_kind"]["llm"]
    assert totals["calls"] == 1
    assert totals["input_tokens"]["known_sum"] == "120"
    assert totals["api_estimate_usd"]["known_sum"] == "0.0005985"
    assert totals["billed_usd"]["unknown_count"] == 1
    events = [json.loads(line) for line in (sink.directory / "events.jsonl").read_text().splitlines()]
    assert [event["data"]["call_id"] for event in events if event["type"] == "call"] == result.call_ids


async def test_fake_server_401_is_auth_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    server, base_url = start_in_thread(status=401)
    try:
        monkeypatch.setenv("JEVPILOT_OPENAI_BASE_URL", base_url)
        provider = ApiKeyProvider(ModelRef(provider="openai", auth_mode="api", model_id="fake-model"), SECRET)
        ledger = RunLedger()
        with pytest.raises(LLMError) as caught:
            _ = await provider.generate(_request(), ledger.scope(deadline=time.monotonic() + 30))
        await provider.aclose()
        with pytest.raises(LLMError) as listed:
            _ = await list_models("openai", SECRET)
    finally:
        server.shutdown()
        server.server_close()
    assert caught.value.kind == "auth_invalid"
    assert listed.value.kind == "auth_invalid"
    assert len(ledger.calls) == 1
    assert ledger.calls[0].outcome == "failed" and ledger.calls[0].error_kind == "auth_invalid"
    assert ledger.calls[0].usage.input_tokens is None
    assert ledger.calls[0].billed_usd is None and ledger.calls[0].api_estimate_usd is None
