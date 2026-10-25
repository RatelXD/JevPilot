"""Code-owned completion checks; model assertions alone cannot pass."""

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Final, Literal, assert_never
import unicodedata

from urllib.parse import urljoin

from jevpilot.browser.port import BrowserAction, ExecutionResult, Observation
from jevpilot.engine.brief import (
    AnswerFields, Comparison, Extraction, FieldValue, TaskBrief, UrlContains,
    VisibleText, OpenedTarget,
)

VerifyStatus = Literal["PASS", "FAIL", "UNKNOWN"]
_NUMBER: Final = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_NO_RESULT: Final = (
    "no results", "no matching results", "nothing found", "no matches",
    "not found", "결과 없음", "결과가 없습니다", "결과가 없음", "결과 없습니다",
)


@dataclass(frozen=True, slots=True)
class VerifyResult:
    status: VerifyStatus
    reason: str


@dataclass(frozen=True, slots=True)
class OpenEvidence:
    before: Observation
    action: BrowserAction
    after: Observation
    result: ExecutionResult = ExecutionResult(outcome="executed")


def _opened(text: str, observation: Observation, evidence: tuple[OpenEvidence, ...]) -> bool:
    wanted = _normalize(text).casefold()
    for item in evidence:
        target = item.action.target
        if item.action.kind != "click" or target is None:
            continue
        if item.result.outcome != "executed" or target.snapshot_id != item.before.snapshot_id:
            continue
        if wanted not in _normalize(target.text).casefold():
            continue
        if item.result.navigation_id is not None:
            if observation.document_id != item.result.navigation_id:
                continue
        elif (
            observation.url != item.after.url
            or observation.document_id != item.after.document_id
        ):
            continue
        if wanted not in _normalize(observation.page_text).casefold():
            continue
        href = target.attributes.get("href")
        if href:
            if observation.url == urljoin(item.before.url, href) and observation.url != item.before.url:
                return True
        elif observation.url != item.before.url:
            return True
        elif (
            observation.page_text != item.before.page_text
            and any(e.role == "dialog" for e in observation.elements)
        ):
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _grounded(field: FieldValue, page_text: str) -> bool:
    quote, value = _normalize(field.quote), _normalize(field.value)
    return bool(quote and value and quote in page_text and value in quote)


def _number(value: str) -> Decimal | None:
    cleaned = "".join(
        char for char in _normalize(value)
        if char != "," and not char.isspace() and unicodedata.category(char) != "Sc"
    )
    return Decimal(cleaned) if _NUMBER.fullmatch(cleaned) else None


def _compare(
    comparison: Comparison, extraction: Extraction, page_text: str,
) -> VerifyResult:
    chosen = extraction.fields.get(comparison.field)
    if chosen is None:
        return VerifyResult("FAIL", "comparison_field_missing")
    if not extraction.compared_items:
        return VerifyResult("FAIL", "compared_items_missing")
    numbers: list[Decimal] = []
    for item in extraction.compared_items:
        if not _grounded(item, page_text):
            return VerifyResult("FAIL", "compared_item_not_grounded")
        number = _number(item.value)
        if number is None:
            return VerifyResult("FAIL", "comparison_not_numeric")
        numbers.append(number)
    selected = _number(chosen.value)
    if selected is None:
        return VerifyResult("FAIL", "comparison_not_numeric")
    match comparison.kind:
        case "max":
            matches = selected == max(numbers)
        case "min":
            matches = selected == min(numbers)
    if not matches:
        return VerifyResult("FAIL", "comparison_extremum_mismatch")
    return VerifyResult("PASS", "comparison_grounded")


def verify(
    observation: Observation | None,
    brief: TaskBrief | None,
    extraction: Extraction | None = None,
    *,
    opened: tuple[OpenEvidence, ...] = (),
) -> VerifyResult:
    """Verify every condition and required answer against the current snapshot.

    Missing evidence is UNKNOWN, contradicted evidence is FAIL, and both are
    non-success. No condition or model-supplied ``done`` flag implies completion.
    """
    if observation is None or brief is None:
        return VerifyResult("UNKNOWN", "observation_or_brief_missing")
    if brief.missing_inputs:
        return VerifyResult("UNKNOWN", "needs_user")
    names = {field.name for field in brief.required_outputs}
    unknown = False
    page_text = _normalize(observation.page_text)
    for condition in brief.success_conditions:
        match condition:
            case OpenedTarget(text=text):
                if not _opened(text, observation, opened):
                    return VerifyResult("FAIL", "target_not_opened")
                continue
            case UrlContains(text=text):
                if not observation.url:
                    unknown = True
                elif text not in observation.url:
                    return VerifyResult("FAIL", "url_condition_mismatch")
                continue
            case VisibleText(text=text):
                if not page_text:
                    unknown = True
                elif _normalize(text) not in page_text:
                    return VerifyResult("FAIL", "visible_text_mismatch")
                continue
            case AnswerFields(names=required):
                names.update(required)
                continue
        assert_never(condition)
    if brief.comparison is not None:
        names.add(brief.comparison.field)
    if not brief.success_conditions and not names:
        return VerifyResult("UNKNOWN", "success_conditions_missing")
    if names and (extraction is None or not page_text):
        unknown = True
    elif extraction is not None and extraction.no_result:
        # Any no-result claim must be grounded, whether or not answers are required.
        quote = _normalize(extraction.no_result_quote or "")
        if (
            not quote or quote not in page_text
            or not any(phrase in quote.casefold() for phrase in _NO_RESULT)
        ):
            return VerifyResult("FAIL", "no_result_not_grounded")
    elif names and extraction is not None:
        if not names.issubset(extraction.fields):
            return VerifyResult("FAIL", "answer_fields_missing")
        if not all(_grounded(field, page_text) for field in extraction.fields.values()):
            return VerifyResult("FAIL", "answer_not_grounded")
        if brief.comparison is not None:
            result = _compare(brief.comparison, extraction, page_text)
            if result.status != "PASS":
                return result
    if unknown:
        return VerifyResult("UNKNOWN", "evidence_missing")
    return VerifyResult("PASS", "conditions_met")
