"""Persist ledger events before emitting allowlisted shell progress."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal, override

from pydantic import JsonValue

from jevpilot.ledger.sink import LedgerSink
from jevpilot.privacy import PrivacyPolicy


class ShellSink(LedgerSink):
    def __init__(
        self, run_id: str, *, emit: Callable[[str], None],
        root: Path | None = None, privacy_policy: PrivacyPolicy | None = None,
    ) -> None:
        super().__init__(run_id, root=root, privacy_policy=privacy_policy)
        self.emit: Callable[[str], None] = emit
        self._origin: float | None = None
        self._ordinals: dict[str, int] = {}

    @override
    def event(
        self, kind: Literal["span", "call", "route", "recovery"], payload: JsonValue,
    ) -> None:
        super().event(kind, payload)
        if kind != "span" or not isinstance(payload, dict):
            return
        match payload.get("kind"):
            case "run":
                start = payload.get("start_s")
                if isinstance(start, (int, float)):
                    self._origin = float(start)
                return
            case "brief":
                label = "작업 해석"
            case "route":
                label = "경로 판단"
            case "observe":
                label = "관찰"
            case "jev":
                label = "Jev 선택"
            case "llm":
                label = "LLM(text)" if payload.get("label") == "llm_text" else "LLM"
            case "execute":
                label = "실행"
            case "verify":
                label = "검증"
            case "recovery":
                label = "복구"
            case "wait":
                label = "대기"
            case "handoff":
                label = "사용자 인계"
            case "confirm":
                label = "실행 확인"
            case _:
                return
        span_id, start, end = payload.get("span_id"), payload.get("start_s"), payload.get("end_s")
        if self._origin is None or not isinstance(span_id, str) or not isinstance(start, (int, float)):
            self.emit(f"진행: {label}")
            return
        ordinal = self._ordinals.setdefault(span_id, len(self._ordinals) + 1)
        # Only public, code-owned operation labels belong in terminal output.
        detail = payload.get("label")
        if detail in {
            "browser_start", "provider_start", "navigate", "click", "type_text",
            "select_option", "scroll_down", "scroll_up", "go_back", "wait",
            "llm_text", "llm_choose", "llm_replan", "llm_extract",
            "missing_input", "ask_user", "click_navigation", "cleanup",
        }:
            label += f" [{detail}]"
        if isinstance(end, (int, float)):
            self.emit(
                f"진행 #{ordinal}: {label} 완료 +{end - self._origin:.2f}초"
                + f" · 소요 {end - start:.2f}초 ({payload.get('outcome')})"
            )
        else:
            self.emit(f"진행 #{ordinal}: {label} 시작 +{start - self._origin:.2f}초")
