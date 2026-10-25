"""Campaign inputs, with holdout integrity checked before loading either split."""
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path
from typing import Final, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

HOLDOUT: Final = Path(__file__).with_name("holdout.json")
DEFAULT_FIXTURE_URL: Final = "http://127.0.0.1:8775"


class HoldoutMismatch(RuntimeError):
    """The campaign's sealed scenario bytes have changed."""


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    scenario_id: str
    goal: str
    start_url: str
    split: Literal["dev", "holdout"] = "holdout"
    outputs: list[str] = Field(default_factory=list)

    @property
    def deadline_s(self) -> int:
        return 60 if self.scenario_id.startswith("F-") else 180


def holdout_sha256(path: Path = HOLDOUT) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_holdout(expected: str, path: Path = HOLDOUT) -> None:
    if holdout_sha256(path) != expected:
        raise HoldoutMismatch("holdout sha256 mismatch; refusing evaluation")


def campaign_date(clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> str:
    """Resolve once at campaign creation; resume reads the saved manifest date."""
    return (clock().astimezone(ZoneInfo("Asia/Seoul")).date() + timedelta(days=1)).isoformat()


def load_scenarios(
    split: Literal["dev", "holdout"], *, expected_hash: str, date: str,
    fixture_base_url: str = DEFAULT_FIXTURE_URL, holdout_path: Path = HOLDOUT,
) -> list[Scenario]:
    check_holdout(expected_hash, holdout_path)
    match split:
        case "holdout":
            return [
                scenario.model_copy(update={"goal": scenario.goal.format(date=date)})
                for scenario in TypeAdapter(list[Scenario]).validate_json(holdout_path.read_bytes())
            ]
        case "dev":
            # Reuse the trusted task definitions verbatim, not a rewritten recipe.
            from benchmarks.tasks.tasks import (
                autocomplete_task, filter_task, form_task, multi_filter_task,
            )
            from jevpilot.task import Budget

            budget = Budget(
                max_steps=40, max_mutations=40, max_replans=2, max_model_attempts=40,
                max_attempts_per_call=1, max_stale_observations=3, max_waits=5,
                max_wait_ms=1000, wall_time_ms=60000, cost_limit_usd=None,
            )
            fixtures = []
            for number, factory in enumerate(
                (filter_task, form_task, autocomplete_task, multi_filter_task), start=2,
            ):
                task, _binding = factory(fixture_base_url.rstrip("/"), "jev_hybrid", budgets=budget)
                fixtures.append(Scenario(
                    scenario_id=f"F-{number:02}", goal=task.goal,
                    start_url=task.initial_url, split="dev",
                ))
            return [*fixtures, Scenario(
                scenario_id="D-1", goal="Find the Python version that added asyncio.TaskGroup.",
                start_url="https://docs.python.org/3/", split="dev", outputs=["version"],
            ), Scenario(
                scenario_id="D-2", goal="세종대왕 문서의 출생 연도를 알려 주세요.",
                start_url="https://ko.wikipedia.org/", split="dev", outputs=["year"],
            ), Scenario(
                scenario_id="D-3", goal="Find the most expensive book in the Travel category and its title and price.",
                start_url="https://books.toscrape.com/", split="dev", outputs=["title", "price"],
            )]
