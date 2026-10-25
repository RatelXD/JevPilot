"""Validate candidate credentials before persisting a login."""

import getpass
import os
import time
import warnings
from collections.abc import Callable, Sequence
from uuid import uuid4

import anyio
import httpx2
from anyio.to_thread import run_sync

from jevpilot.codex_auth import (
    codex_login_status,
    ensure_jevpilot_home,
    is_chatgpt_login_status,
    launch_chatgpt_login,
)
from jevpilot.engine.jev import JEV_MODEL, JevUnavailable, validate_key as validate_jev_key
from jevpilot.ledger.models import CallRecord, Outcome
from jevpilot.ledger.recorder import RunLedger
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.api_providers import ApiKeyProvider, list_models, validate_key
from jevpilot.llm.credentials import get_api_key, get_base_url, mask_key, save_credential
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import Profile, save_profile
from jevpilot.llm.types import LLMRequest, ModelRef
from jevpilot.model_catalog import list_chatgpt_models
from jevpilot.model_client import ProviderError


def _secret(prompt: str) -> str:
    # getpass's fallback can echo secrets. Refuse it instead.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        return getpass.getpass(prompt).strip()


async def _choose(ids: Sequence[str], emit: Callable[[str], None]) -> str | None:
    for index, model_id in enumerate(ids, 1):
        emit(f"{index}) {model_id}")
    choice = (await run_sync(input, "모델 번호 (Enter 취소): ")).strip()
    if not choice.isdecimal() or not 1 <= int(choice) <= len(ids):
        return None
    return ids[int(choice) - 1]


def _validation_ledger() -> RunLedger:
    run_id = f"auth-{uuid4().hex}"
    return RunLedger(run_id, sink=LedgerSink(
        run_id, root=ensure_jevpilot_home() / "runs",
    ))


async def _probe(ref: ModelRef, candidate_key: str) -> None:
    ledger = _validation_ledger()
    outcome: Outcome = "failed"
    provider = None
    try:
        with ledger.span("llm", "authentication"):
            provider = ApiKeyProvider(ref, candidate_key)
            with anyio.fail_after(30):
                _ = await provider.generate(
                    LLMRequest(
                        purpose="text", system="Return ok.", user_payload={}, output_type=str,
                    ),
                    ledger.scope(deadline=time.monotonic() + 30),
                )
            outcome = "ok"
    except anyio.get_cancelled_exc_class():
        outcome = "cancelled"
        raise
    finally:
        try:
            if provider is not None:
                with anyio.CancelScope(shield=True):
                    await provider.aclose()
        finally:
            _ = ledger.finish(outcome)


async def _check_jev(key: str) -> None:
    """Observe response status without inventing usage absent from validate_key."""
    ledger = _validation_ledger()
    outcome: Outcome = "failed"
    statuses: list[int] = []

    async def received(response: httpx2.Response) -> None:
        statuses.append(response.status_code)

    try:
        with ledger.span("jev", "authentication") as span:
            started = time.monotonic()
            error_kind: str | None = None
            try:
                async with httpx2.AsyncClient(event_hooks={"response": [received]}) as client:
                    with anyio.fail_after(30):
                        await validate_jev_key(key, http_client=client)
                outcome = "ok"
            except JevUnavailable as error:
                error_kind = error.kind
                raise
            except TimeoutError:
                error_kind = "timeout"
                raise
            except anyio.get_cancelled_exc_class():
                outcome = "cancelled"
                error_kind = "cancelled"
                raise
            finally:
                if statuses:
                    ledger.record_call(CallRecord(
                        call_id=uuid4().hex, span_id=span.span_id,
                        kind="jev", provider="jev", auth_mode="api",
                        purpose="authentication",
                        request_model=os.environ.get("JEVPILOT_JEV_MODEL", "").strip() or JEV_MODEL,
                        outcome=outcome, error_kind=error_kind,
                        latency_s=time.monotonic() - started,
                    ))
                else:
                    ledger.event("recovery", {
                        "reason": "validation_transmission_unconfirmed",
                        "error_kind": error_kind or "transport_unknown",
                    })
    finally:
        _ = ledger.finish(outcome)


async def _jev_login(*, force: bool, emit: Callable[[str], None]) -> None:
    if os.environ.get("JEVPILOT_JEV_API_KEY", "").strip():
        emit("Jev 키: 환경변수 사용 중")
        return
    try:
        existing = get_api_key("jev")
        if existing and not force:
            try:
                await _check_jev(existing)
                return
            except (JevUnavailable, TimeoutError):
                emit("Jev 키를 확인하세요.")
        key = await run_sync(
            _secret, "Jev(TypeSafe) API 키를 입력하세요(Enter 건너뛰기): ",
        )
        if not key:
            return
        await _check_jev(key)
        save_credential("jev", key)
        emit(f"Jev 키를 저장했습니다: {mask_key(key)}")
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        return
    except (JevUnavailable, TimeoutError):
        emit("Jev 키를 확인하세요. 인증 또는 연결에 실패했습니다.")
    except (OSError, ValueError):
        emit("Jev 키를 저장하지 못했습니다.")


async def login(
    current: Profile | None = None, *, emit: Callable[[str], None] = print,
) -> Profile | None:
    emit("로그인 방식을 선택하세요.\n  1) API 키\n  2) 구독 로그인 (ChatGPT)\n  3) Jev API 키 등록/변경")
    selected = current
    try:
        choice = (await run_sync(input, "번호: ")).strip()
        match choice:
            case "1":
                emit("공급자: 1) OpenAI 2) Anthropic")
                provider_choice = (await run_sync(input, "번호: ")).strip()
                match provider_choice:
                    case "1":
                        provider_id = "openai"
                    case "2":
                        provider_id = "anthropic"
                    case _:
                        return current
                key = await run_sync(_secret, "API 키: ")
                base_url = get_base_url(provider_id)
                ids = await list_models(provider_id, key, base_url)
                model_id = await _choose(ids, emit)
                if model_id is None:
                    return current
                ref = ModelRef(provider_id, "api", model_id, base_url=base_url)
                await validate_key(provider_id, key, base_url, model_id)
                await _probe(ref, key)
                selected = Profile("api", ref)
                # Each writer is atomic, not a transaction across these two files.
                save_credential(provider_id, key, base_url)
                save_profile(selected)
                emit(f"API 키로 로그인했습니다: {provider_id} · {mask_key(key)}")
            case "2":
                if not is_chatgpt_login_status(await codex_login_status()):
                    emit("ChatGPT 구독 로그인을 진행합니다.")
                    if await run_sync(launch_chatgpt_login) != 0:
                        emit("구독 로그인에 실패했습니다.")
                        return current
                    if not is_chatgpt_login_status(await codex_login_status()):
                        emit("구독 로그인을 확인하지 못했습니다.")
                        return current
                models = await list_chatgpt_models()
                model_id = await _choose([model.model_id for model in models], emit)
                if model_id is None:
                    return current
                selected = Profile(
                    "subscription", ModelRef("chatgpt_subscription", "subscription", model_id),
                    codex_transport="app_server",
                )
                save_profile(selected)
                emit("ChatGPT 구독으로 로그인했습니다.")
            case "3":
                await _jev_login(force=True, emit=emit)
                return current
            case _:
                return current
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        return current
    except LLMError as error:
        emit(f"로그인하지 못했습니다. 키를 확인하세요. ({error.kind})")
        return current
    except (ProviderError, TimeoutError):
        emit("로그인하지 못했습니다. 인증과 연결을 확인하세요.")
        return current
    except (OSError, ValueError):
        emit("로그인 정보를 저장하지 못했습니다. 일부 파일은 저장되었을 수 있습니다.")
        return current
    await _jev_login(force=False, emit=emit)
    return selected
