"""Real comparison arms. The campaign parent owns the trial process deadline."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Final, Literal
from uuid import uuid4

import anyio
from anyio.streams.buffered import BufferedByteReceiveStream
from pydantic import BaseModel, ConfigDict, JsonValue, PrivateAttr, TypeAdapter
from playwright.async_api import Browser, Page, async_playwright

from benchmarks.browser_use_eval.oracle import PageEvidence, grade
from benchmarks.browser_use_eval.scenarios import Scenario
from jevpilot.browser.browser_use_adapter import BrowserUseAdapter
from jevpilot.browser.port import BrowserConfig, Observation
from jevpilot.engine.assist import llm_extract
from jevpilot.engine.brief import AnswerFields, OutputField, TaskBrief
from jevpilot.engine.control import ScriptedControl
from jevpilot.engine.jev import JevSelector
from jevpilot.engine.loop import _LLMScope, run_task
from jevpilot.engine.routing import RoutingPolicy
from jevpilot.engine.verify import verify
from jevpilot.ledger.models import CallRecord, CallUsage
from jevpilot.ledger.recorder import RunLedger
from jevpilot.ledger.sink import LedgerSink
from jevpilot.llm.bridge import JevPilotChatModel
from jevpilot.llm.codex_app_server import CodexAppServerProvider
from jevpilot.llm.errors import LLMError
from jevpilot.llm.types import LLMRequest, ModelRef

from browser_use import Agent, BrowserSession
from browser_use.browser.session import CDPSession
from cdp_use import CDPClient

Arm = Literal["A", "B", "C", "D"]
NATIVE_PYTHON: Final = "/home/jhou8/jevpilot/comparators/jev-ultrafast/.venv/bin/python"
ARM_NAMES: Final = {
    "A": "llm-centric", "B": "jev-centric", "C": "jev-adaptive", "D": "jev-ultrafast-ref",
}
EVIDENCE_JS: Final = (
    "({url:location.href,page_text:document.body.innerText,"
    "item_id:document.querySelector('#item-id')?.innerText||'',"
    "http_status:performance.getEntriesByType('navigation')[0]?.responseStatus||null,"
    "language:navigator.language,languages:navigator.languages,"
    "intl_locale:Intl.DateTimeFormat().resolvedOptions().locale,"
    "viewport:[innerWidth,innerHeight]})"
)


class EvalSession(BrowserSession):
    """The browser-use session attached to the harness's localized Chrome."""

    _localized_sessions: set[str] = PrivateAttr(default_factory=set)
    _locale_lock: anyio.Lock = PrivateAttr(default_factory=anyio.Lock)

    def __init__(self, endpoint: str) -> None:
        super().__init__(
            cdp_url=endpoint, keep_alive=True, viewport={"width": 1280, "height": 800},
            enable_default_extensions=False, captcha_solver=False,
        )

    async def get_or_create_cdp_session(
        self, target_id: str | None = None, focus: bool = True,
    ) -> CDPSession:
        session = await super().get_or_create_cdp_session(target_id, focus)
        async with self._locale_lock:
            if session.session_id not in self._localized_sessions:
                current = await session.cdp_client.send.Runtime.evaluate(
                    params={"expression": "Intl.DateTimeFormat().resolvedOptions().locale",
                            "returnByValue": True}, session_id=session.session_id,
                )
                # Targets sharing a renderer share ICU's override. Chrome rejects
                # competing owners, even when both request the same locale.
                if current["result"].get("value") != "ko-KR":
                    await session.cdp_client.send.Emulation.setLocaleOverride(
                        params={"locale": "ko_KR"}, session_id=session.session_id,
                    )
                self._localized_sessions.add(session.session_id)
        return session

class NativeText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str


@dataclass(frozen=True, slots=True)
class ArmResult:
    status: str
    answer: str
    page: PageEvidence
    extraction_verified: bool | None = None
    error_kind: str | None = None


async def page_evidence(page: Page) -> PageEvidence:
    """Read the full body directly from the browser, never a product observation."""
    return TypeAdapter(PageEvidence).validate_python(await page.evaluate("() => " + EVIDENCE_JS))


async def exact_page(browser: Browser, target_id: str) -> Page:
    for context in browser.contexts:
        for page in context.pages:
            session = await context.new_cdp_session(page)
            try:
                identity = (await session.send("Target.getTargetInfo"))["targetInfo"]["targetId"]
                if identity == target_id:
                    return page
            finally:
                await session.detach()
    raise RuntimeError("Advertised native target is unavailable")


async def extract(
    scenario: Scenario, page: PageEvidence, provider: CodexAppServerProvider,
    ledger: RunLedger, deadline: float,
) -> tuple[str, bool]:
    """A and D use exactly the engine's extraction and evidence verification."""
    observation = Observation(
        snapshot_id="final", url=page.url, title="", tabs=[], elements=[],
        page_text=page.page_text, fingerprint="final", observed_at_monotonic=time.monotonic(),
    )
    brief = TaskBrief(
        goal_restated=scenario.goal, start_url=scenario.start_url, input_values={},
        required_outputs=[OutputField(name=name, description=name) for name in scenario.outputs],
        success_conditions=[AnswerFields(names=scenario.outputs)], missing_inputs=[],
        mutating_intent=False, multi_step=False, comparison=None,
    )
    answer = await llm_extract(
        observation, brief, provider=provider,
        context=_LLMScope(ledger, ledger.root_span_id, deadline),
    )
    return (
        "; ".join(f"{name}: {value.value}" for name, value in answer.fields.items()),
        verify(observation, brief, answer).status == "PASS",
    )


class EvalBrowser(BrowserUseAdapter):
    """Attach the product adapter to the harness-owned Chrome and capture on close."""

    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.evidence = PageEvidence("", "")

    async def start(self, config: BrowserConfig) -> None:
        self._config = config.model_copy(update={"headless": True})
        self._session = EvalSession(self.endpoint)
        await self._session.start()
        self._navigation_client = CDPClient(self._session.cdp_client.url)
        await self._navigation_client.start()
        self._navigation_client.register.Page.frameStartedNavigating(self._navigation_started)
        self._navigation_client.register.Page.lifecycleEvent(self._navigation_finished)

    async def close(self) -> None:
        try:
            if self._session is not None:
                current = await self._session.get_or_create_cdp_session()
                response = await current.cdp_client.send.Runtime.evaluate(
                    params={
                        "expression": EVIDENCE_JS,
                        "returnByValue": True,
                    }, session_id=current.session_id,
                )
                self.evidence = TypeAdapter(PageEvidence).validate_python(response["result"].get("value"))
        finally:
            await super().close()


async def native(
    scenario: Scenario, endpoint: str, browser: Browser, provider: CodexAppServerProvider,
    ledger: RunLedger, deadline: float, output: Path,
) -> ArmResult:
    environment = {**os.environ, "TYPESAFE_API_KEY": os.environ["JEVPILOT_JEV_API_KEY"]}
    evidence = PageEvidence("", "")
    with (output / "native-stderr.txt").open("wb") as errors:
        process = await anyio.open_process(
            [NATIVE_PYTHON, str(Path(__file__).with_name("ultrafast_worker.py")),
             "--cdp", endpoint, "--url", scenario.start_url, "--goal", scenario.goal,
             "--trial-id", ledger.run_id],
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
        )
        assert process.stdin is not None and process.stdout is not None
        incoming = process.stdin
        stream = BufferedByteReceiveStream(process.stdout)
        status = "failed"
        helper_error: LLMError | None = None

        async def send(message: dict[str, JsonValue]) -> None:
            await incoming.send(json.dumps(message).encode() + b"\n")

        try:
            while True:
                event = TypeAdapter(dict[str, JsonValue]).validate_json(
                    await stream.receive_until(b"\n", 10_000_000),
                )
                match event["type"]:
                    case "ready":
                        await send({"type": "start"})
                    case "checkpoint":
                        target = event.get("target_id")
                        passed = False
                        if isinstance(target, str):
                            evidence = await page_evidence(await exact_page(browser, target))
                            if not scenario.outputs:
                                passed = grade(scenario.scenario_id, "", evidence).status == "passed"
                        await send({"type": "stop" if passed else "continue"})
                    case "text_request":
                        try:
                            result = await provider.generate(LLMRequest(
                                purpose="text", system=str(event["instructions"]),
                                user_payload=event["payload"], output_type=NativeText,
                            ), _LLMScope(ledger, ledger.root_span_id, deadline))
                        except LLMError as error:
                            helper_error = error
                            await send({"type": "stop"})
                            continue
                        await send({
                            "type": "text_result", "text": result.value.model_dump_json(),
                            "metadata": {"model": result.resolved_model,
                                         "usage": None if result.usage is None else {
                                             "input_tokens": result.usage.input_tokens,
                                             "output_tokens": result.usage.output_tokens,
                                         }},
                        })
                    case "call_event":
                        code = event.get("status_code")
                        ledger.record_call(CallRecord(
                            call_id=uuid4().hex, span_id=ledger.root_span_id, kind="jev",
                            provider="jev", auth_mode="api", purpose="native",
                            request_model=str(event.get("request_model") or "unknown"),
                            resolved_model=str(event.get("response_model") or "unknown"),
                            outcome="ok" if code == 200 else "failed",
                            latency_s=float(str(event["elapsed_ms"])) / 1000,
                            usage=CallUsage.model_validate({
                                "input_tokens": event.get("input_tokens"),
                                "output_tokens": event.get("output_tokens"),
                            }),
                        ))
                    case "terminal":
                        evidence = PageEvidence(
                            str(event.get("url") or evidence.url),
                            str(event.get("page_text") or evidence.page_text),
                            str(event.get("item_id") or evidence.item_id),
                        )
                        target = event.get("target_id")
                        if isinstance(target, str):
                            evidence = await page_evidence(await exact_page(browser, target))
                        status = str(event["status"])
                        await send({"type": "close"})
                        await process.wait()
                        break
                    case _:
                        raise RuntimeError("Invalid native worker event")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            await process.aclose()
    if helper_error is not None:
        return ArmResult("failed", "", evidence, error_kind=helper_error.kind)
    answer, verified = await extract(scenario, evidence, provider, ledger, deadline) if scenario.outputs else ("", None)
    return ArmResult(status, answer, evidence, verified)


async def execute(
    arm: Arm, scenario: Scenario, ref: ModelRef, output: Path, ledger: RunLedger,
) -> ArmResult:
    """Run once; Chrome inherits the worker's process group for hard deadlines."""
    deadline = time.monotonic() + scenario.deadline_s
    profile = Path(os.environ["TMPDIR"]) / "chrome-profile"
    provider = CodexAppServerProvider(ref, codex_home=Path.home() / ".jevpilot/codex")
    chrome = await anyio.open_process(
        ["/usr/bin/google-chrome", "--headless=new", "--no-first-run",
         "--no-default-browser-check", "--disable-background-networking", "--disable-sync",
         "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
         "--accept-lang=ko-KR,ko",
         f"--user-data-dir={profile}", "--lang=ko-KR", "--window-size=1280,800", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        env={**os.environ, "LANG": "ko_KR.UTF-8", "LC_ALL": "ko_KR.UTF-8"},
    )
    assert chrome.stderr is not None
    stream = BufferedByteReceiveStream(chrome.stderr)
    try:
        with anyio.fail_after(20):
            while True:
                line = (await stream.receive_until(b"\n", 65536)).decode()
                if "DevTools listening on " in line:
                    endpoint = line.split("DevTools listening on ", 1)[1].strip()
                    break
        match arm:
            case "B" | "C":
                adapter = EvalBrowser(endpoint)
                outcome = await run_task(
                    scenario.goal, browser=adapter, llm=provider, jev=JevSelector(),
                    control=ScriptedControl(approve_confirm=scenario.scenario_id == "F-03"),
                    ledger=ledger, policy=RoutingPolicy(max_wall_s=float(scenario.deadline_s)),
                    start_url=scenario.start_url, mode="jev_only" if arm == "B" else "adaptive",
                )
                answer = "" if outcome.extraction is None else "; ".join(
                    f"{name}: {field.value}" for name, field in outcome.extraction.fields.items()
                )
                return ArmResult(outcome.status, answer, adapter.evidence)
            case "A" | "D":
                async with async_playwright() as pw:
                    browser = await pw.chromium.connect_over_cdp(endpoint)
                    try:
                        if arm == "D":
                            return await native(scenario, endpoint, browser, provider, ledger, deadline, output)
                        session = EvalSession(endpoint)
                        await session.start()
                        try:
                            page = browser.contexts[0].pages[0]
                            await page.set_viewport_size({"width": 1280, "height": 800})
                            await page.goto(scenario.start_url, wait_until="domcontentloaded")
                            history = await Agent(
                                task=scenario.goal,
                                llm=JevPilotChatModel(provider, scope_factory=lambda: _LLMScope(
                                    ledger, ledger.root_span_id, deadline,
                                )),
                                browser_session=session, use_vision=False,
                                max_failures=1, final_response_after_failure=False,
                                file_system_path=str(Path(os.environ["TMPDIR"]) / "agent-files"),
                            ).run(max_steps=40)
                            target = (await session.get_or_create_cdp_session()).target_id
                            evidence = await page_evidence(await exact_page(browser, target))
                            answer, verified = await extract(
                                scenario, evidence, provider, ledger, deadline,
                            ) if scenario.outputs else ("", None)
                            return ArmResult(
                                "completed" if history.is_done() else "incomplete",
                                answer, evidence, verified,
                            )
                        finally:
                            await session.kill()
                    finally:
                        await browser.close()
    finally:
        with anyio.CancelScope(shield=True):
            await provider.aclose()
            if chrome.returncode is None:
                chrome.terminate()
                with anyio.move_on_after(5):
                    await chrome.wait()
                if chrome.returncode is None:
                    chrome.kill()
                    await chrome.wait()
            await chrome.aclose()


def make_ledger(output: Path) -> RunLedger:
    def called(record: CallRecord) -> None:
        with (output / "calls.jsonl").open("a") as stream:
            stream.write(record.model_dump_json() + "\n")
    return RunLedger(output.name, sink=LedgerSink("ledger", root=output), on_call=called)
