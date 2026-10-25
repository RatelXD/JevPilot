"""New shell/task entrypoint, preserving every legacy CLI argument."""

import argparse
import json
import sys
from functools import partial

import anyio
from pydantic import ValidationError

from jevpilot.engine.brief import (
    AnswerFields, Condition, OutputField, StructuredTask, UrlContains, VisibleText,
)
from jevpilot.engine.control import ScriptedControl
from jevpilot.engine.jev import JevUnavailable
from jevpilot.llm.errors import LLMError
from jevpilot.llm.profile import suggest_profile
from jevpilot.shell.app_shell import execute_controlled, execute_task, run_shell


class _Arguments(argparse.Namespace):
    goal: str = ""
    start_url: str | None = None
    verify_url_contains: list[str] = []
    verify_text: list[str] = []
    value: list[str] = []
    output: list[str] = []
    approve_submit: bool = False
    headless: bool = False


def _task(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="jevpilot task")
    _ = parser.add_argument("goal")
    _ = parser.add_argument("--start-url")
    _ = parser.add_argument("--verify-url-contains", action="append", default=[])
    _ = parser.add_argument("--verify-text", action="append", default=[])
    _ = parser.add_argument("--value", action="append", default=[])
    _ = parser.add_argument("--output", action="append", default=[])
    _ = parser.add_argument("--approve-submit", action="store_true")
    _ = parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(arguments, namespace=_Arguments())
    goal = sys.stdin.read() if args.goal == "-" else args.goal
    if not goal.strip():
        parser.error("goal must not be empty")
    values: dict[str, str] = {}
    for entry in args.value:
        name, separator, value = entry.partition("=")
        if not separator or not name.strip() or not value.strip():
            parser.error("--value requires a nonempty NAME=VALUE")
        values[name] = value
    conditions: list[Condition] = [
        *[UrlContains(text=text) for text in args.verify_url_contains],
        *[VisibleText(text=text) for text in args.verify_text],
    ]
    outputs = [OutputField(name=name, description=name) for name in args.output]
    if outputs:
        conditions.append(AnswerFields(names=[output.name for output in outputs]))
    task: str | StructuredTask = goal
    if args.start_url and conditions:
        task = StructuredTask(
            start_url=args.start_url, success_conditions=conditions,
            input_values=values, required_outputs=outputs,
        )
    elif values or conditions:
        # Partial caller hints still belong to natural-language interpretation.
        task = goal + "\n" + json.dumps({
            "input_values": values,
            "success_conditions": [condition.model_dump() for condition in conditions],
            "required_outputs": [output.model_dump() for output in outputs],
        }, ensure_ascii=False)
    try:
        profile = suggest_profile()
        if profile is None:
            print("먼저 /login으로 API 키 또는 구독 로그인을 선택하세요.")
            return 2
        if args.goal == "-":
            outcome = anyio.run(partial(
                execute_controlled, task, profile, ScriptedControl(approve_confirm=args.approve_submit),
                start_url=args.start_url, headless=args.headless,
            ))
        else:
            outcome = anyio.run(partial(
                execute_task, task, profile, start_url=args.start_url,
                headless=args.headless, approve_submit=args.approve_submit,
            ))
    except JevUnavailable as error:
        if error.kind == "auth_missing":
            print("Jev 키가 없습니다. /login에서 3번을 선택하세요.")
        else:
            print(f"Jev 인증과 연결을 확인하세요. /login ({error.kind})")
        return 2
    except LLMError as error:
        print(f"LLM 인증을 확인하세요. /login ({error.kind})")
        return 2
    except (OSError, ValueError):
        print("설정을 읽지 못했습니다. /login 인증을 확인하세요.")
        return 2
    return 0 if outcome.status == "verified" else 1


def entrypoint() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments == ["shell"]:
        raise SystemExit(run_shell())
    if arguments[0] == "task":
        try:
            raise SystemExit(_task(arguments[1:]))
        except ValidationError:
            print("작업 옵션은 비어 있지 않은 값을 사용해야 합니다.", file=sys.stderr)
            raise SystemExit(2) from None
    from jevpilot.cli import entrypoint as legacy_entrypoint

    legacy_entrypoint()
