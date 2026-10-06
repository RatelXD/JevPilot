from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, final

import anyio
from playwright.async_api import Error as BrowserError, Page

from jevpilot.observer.service import Observer
from jevpilot.task import ActionType, TaskSpec

from .models import ActionExecution, ActionRequest

_SUPPORTED_OPERATIONS = frozenset(
    {
        ActionType.CLICK,
        ActionType.TYPE_TEXT,
        ActionType.SELECT,
        ActionType.SCROLL_UP,
        ActionType.SCROLL_DOWN,
        ActionType.WAIT,
        ActionType.DONE,
        ActionType.BLOCKED,
    }
)
ExecutionStatus = Literal["executed", "failed_not_executed", "uncertain"]


@final
class PlaywrightExecutor:
    def __init__(
        self,
        page: Page,
        observer: Observer,
        *,
        task: TaskSpec,
        record_execution: Callable[[ActionExecution], None],
    ) -> None:
        unsupported = set(task.allowed_operations) - _SUPPORTED_OPERATIONS
        if unsupported:
            names = ", ".join(sorted(operation.value for operation in unsupported))
            raise ValueError(f"unsupported first-slice operations: {names}")
        self._page = page
        self._observer = observer
        self._task = task
        self._record_execution = record_execution
        self._consumed_decision_ids: set[str] = set()
        self._decision_lock = asyncio.Lock()

    async def execute(
        self,
        request: ActionRequest,
        *,
        deadline_monotonic: float,
    ) -> ActionExecution:
        started_at = datetime.now(UTC)
        async with self._decision_lock:
            already_consumed = request.decision_id in self._consumed_decision_ids
            self._consumed_decision_ids.add(request.decision_id)

        if already_consumed:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("decision id was already consumed",),
                error_kind="decision_already_consumed",
            )
        if deadline_monotonic <= time.monotonic():
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("execution deadline expired before dispatch",),
                error_kind="deadline_expired",
            )
        if request.operation not in self._task.allowed_operations:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("operation is not allowed by task policy",),
                error_kind="operation_not_allowed",
            )
        if request.operation not in _SUPPORTED_OPERATIONS:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("operation is outside the first slice",),
                error_kind="unsupported_operation",
            )
        if not self._observer.matches_active_snapshot(request.snapshot_id):
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("request does not match the active snapshot",),
                error_kind="snapshot_mismatch",
            )

        if request.operation in {ActionType.DONE, ActionType.BLOCKED}:
            return self._finalize(
                request,
                started_at=started_at,
                status="executed",
                dispatched=True,
                evidence=(f"control operation accepted: {request.operation.value}",),
                error_kind=None,
            )

        if request.operation in {ActionType.SCROLL_UP, ActionType.SCROLL_DOWN}:
            try:
                fresh = await self._observer.scroll_is_fresh(
                    snapshot_id=request.snapshot_id,
                    operation=request.operation,
                    deadline_monotonic=deadline_monotonic,
                )
            except (BrowserError, TimeoutError, ValueError):
                fresh = False
            delta = self._observer.scroll_delta(
                snapshot_id=request.snapshot_id,
                operation=request.operation,
            )
            start_y = self._observer.scroll_position(
                snapshot_id=request.snapshot_id
            )
            if not fresh or delta is None or start_y is None:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="failed_not_executed",
                    dispatched=False,
                    evidence=("scroll candidate is stale or unavailable",),
                    error_kind="stale_before_execution",
                )
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="failed_not_executed",
                    dispatched=False,
                    evidence=("execution deadline expired before scroll",),
                    error_kind="deadline_expired",
                )
            try:
                with anyio.fail_after(remaining):
                    await self._page.mouse.wheel(0, delta)
                    _ = await self._page.wait_for_function(
                        "(previousY) => window.scrollY !== previousY",
                        arg=start_y,
                        timeout=max(1, int(remaining * 1000)),
                    )
            except asyncio.CancelledError:
                _ = self._finalize(
                    request,
                    started_at=started_at,
                    status="uncertain",
                    dispatched=True,
                    evidence=("scroll was interrupted after dispatch",),
                    error_kind="scroll_cancelled",
                )
                raise
            except (BrowserError, TimeoutError):
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="uncertain",
                    dispatched=True,
                    evidence=("scroll result is uncertain",),
                    error_kind="scroll_interrupted",
                )
            return self._finalize(
                request,
                started_at=started_at,
                status="executed",
                dispatched=True,
                evidence=("bounded viewport scroll completed",),
                error_kind=None,
            )

        if request.operation is ActionType.WAIT:
            if request.candidate_id is None or request.wait_condition is None:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="failed_not_executed",
                    dispatched=False,
                    evidence=("bounded wait condition is missing",),
                    error_kind="invalid_request",
                )
            try:
                changed = await self._observer.wait_for_state_change(
                    snapshot_id=request.snapshot_id,
                    candidate_id=request.candidate_id,
                    wait_condition=request.wait_condition,
                    deadline_monotonic=deadline_monotonic,
                )
            except ValueError:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="failed_not_executed",
                    dispatched=False,
                    evidence=("wait snapshot or condition is stale",),
                    error_kind="stale_before_execution",
                )
            except TimeoutError:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="executed",
                    dispatched=True,
                    evidence=("wall-time deadline ended the bounded wait",),
                    error_kind="wait_wall_deadline",
                )
            except asyncio.CancelledError:
                _ = self._finalize(
                    request,
                    started_at=started_at,
                    status="executed",
                    dispatched=True,
                    evidence=("wait was cancelled",),
                    error_kind="wait_cancelled",
                )
                raise
            return self._finalize(
                request,
                started_at=started_at,
                status="executed",
                dispatched=True,
                evidence=(
                    ("observed page state changed",)
                    if changed
                    else ("bounded wait ended without a state change",)
                ),
                error_kind=None if changed else "wait_condition_timeout",
            )

        if request.candidate_id is None or request.target_ref is None:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("click request lacks registry identity",),
                error_kind="invalid_request",
            )
        prepared = await self._observer.prepare_action(
            snapshot_id=request.snapshot_id,
            candidate_id=request.candidate_id,
            target_ref=request.target_ref,
            operation=request.operation,
            option_ref=request.option_ref,
            deadline_monotonic=deadline_monotonic,
        )
        if prepared.element is None:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=prepared.evidence,
                error_kind=prepared.error_kind or "preflight_rejected",
            )

        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            return self._finalize(
                request,
                started_at=started_at,
                status="failed_not_executed",
                dispatched=False,
                evidence=("execution deadline expired before dispatch",),
                error_kind="deadline_expired",
            )
        remaining_ms = max(1, int(remaining * 1000))
        try:
            if request.operation is ActionType.CLICK:
                await prepared.element.click(timeout=remaining_ms)
            elif request.operation is ActionType.TYPE_TEXT:
                if request.text is None:
                    return self._finalize(
                        request,
                        started_at=started_at,
                        status="failed_not_executed",
                        dispatched=False,
                        evidence=("text value is missing",),
                        error_kind="invalid_request",
                    )
                await prepared.element.fill(request.text, timeout=remaining_ms)
            elif request.operation is ActionType.SELECT:
                if request.option_ref is None:
                    return self._finalize(
                        request,
                        started_at=started_at,
                        status="failed_not_executed",
                        dispatched=False,
                        evidence=("native option reference is missing",),
                        error_kind="invalid_request",
                    )
                try:
                    option_index = int(request.option_ref)
                except ValueError:
                    return self._finalize(
                        request,
                        started_at=started_at,
                        status="failed_not_executed",
                        dispatched=False,
                        evidence=("native option reference is invalid",),
                        error_kind="invalid_request",
                    )
                _ = await prepared.element.select_option(
                    index=option_index,
                    timeout=remaining_ms,
                )
            else:
                return self._finalize(
                    request,
                    started_at=started_at,
                    status="failed_not_executed",
                    dispatched=False,
                    evidence=("operation is not implemented by this executor slice",),
                    error_kind="unsupported_operation",
                )
        except asyncio.CancelledError:
            _ = self._finalize(
                request,
                started_at=started_at,
                status="uncertain",
                dispatched=True,
                evidence=(
                    *prepared.evidence,
                    f"{request.operation.value} dispatch was cancelled before confirmation",
                ),
                error_kind="dispatch_cancelled",
            )
            raise
        except (BrowserError, TimeoutError):
            return self._finalize(
                request,
                started_at=started_at,
                status="uncertain",
                dispatched=True,
                evidence=(
                    *prepared.evidence,
                    f"{request.operation.value} dispatch was interrupted before confirmation",
                ),
                error_kind="dispatch_interrupted",
            )
        return self._finalize(
            request,
            started_at=started_at,
            status="executed",
            dispatched=True,
            evidence=(
                *prepared.evidence,
                f"{request.operation.value} returned successfully",
            ),
            error_kind=None,
        )

    def _finalize(
        self,
        request: ActionRequest,
        *,
        started_at: datetime,
        status: ExecutionStatus,
        dispatched: bool,
        evidence: tuple[str, ...],
        error_kind: str | None,
    ) -> ActionExecution:
        execution = ActionExecution(
            action_id=request.action_id,
            decision_id=request.decision_id,
            snapshot_id=request.snapshot_id,
            operation=request.operation,
            candidate_id=request.candidate_id,
            target_ref=request.target_ref,
            option_ref=request.option_ref,
            status=status,
            dispatched=dispatched,
            started_at=started_at,
            ended_at=datetime.now(UTC),
            evidence=evidence,
            error_kind=error_kind,
        )
        self._record_execution(execution)
        return execution
