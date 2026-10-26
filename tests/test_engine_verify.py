from typing import Literal

import pytest

from jevpilot.browser.port import BrowserAction, ElementRef, Observation
from jevpilot.engine.brief import (
    AnswerFields, Comparison, Condition, Extraction, FieldValue, OutputField,
    TaskBrief, UrlContains, VisibleText, OpenedTarget,
)
from jevpilot.engine.verify import OpenEvidence, verify


def observation(text: str = "1397년 출생", url: str = "https://example.test/done") -> Observation:
    return Observation(
        snapshot_id="s", url=url, title="", tabs=[], elements=[], page_text=text,
        fingerprint="f", observed_at_monotonic=0,
    )


def brief(
    conditions: list[Condition] | None = None, comparison: Comparison | None = None,
    outputs: list[OutputField] | None = None,
) -> TaskBrief:
    return TaskBrief(
        goal_restated="Read the answer", start_url="https://example.test",
        input_values={}, required_outputs=outputs or [],
        success_conditions=(
            [AnswerFields(names=["year"])] if conditions is None else conditions
        ),
        missing_inputs=[], mutating_intent=False, multi_step=False, comparison=comparison,
    )


def extraction(
    value: str = "1397", quote: str = "1397년 출생",
    compared: list[FieldValue] | None = None,
) -> Extraction:
    return Extraction(
        fields={"year": FieldValue(value=value, quote=quote)},
        no_result=False, no_result_quote=None, compared_items=compared or [],
    )


@pytest.mark.parametrize("value,quote", [
    ("1397", "1397년 출생"),
    ("1397", "１３９７년\u3000출생"),
    ("1397년 출생", "1397년 \n\t 출생"),
])
def test_answer_when_value_and_quote_are_grounded_passes(value: str, quote: str) -> None:
    # Given
    page, task, answer = observation(), brief(), extraction(value, quote)
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "PASS"


def test_answer_when_quote_is_absent_from_page_fails() -> None:
    # Given
    page, task, answer = observation(), brief(), extraction("1900", "1900년 출생")
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "FAIL"
    assert result.reason == "answer_not_grounded"
    print(f"absent page quote: {result.status} ({result.reason})")


@pytest.mark.parametrize("source,expected", [
    ("https://example.test/done", "PASS"),
    ("https://invented.test", "FAIL"),
])
def test_answer_source_must_be_the_observed_url(source: str, expected: str) -> None:
    # Given
    task = brief(conditions=[AnswerFields(names=["source"])])
    answer = Extraction(
        fields={"source": FieldValue(value=source, quote=source)},
        no_result=False, no_result_quote=None, compared_items=[],
    )
    # When
    result = verify(observation(), task, answer)
    # Then
    assert result.status == expected


@pytest.mark.parametrize("value,quote", [
    ("1900", "1397년 출생"), ("", "1397년 출생"), ("1397", ""),
    (" ", "1397년 출생"), ("1397", "\n\t"),
])
def test_answer_when_value_is_not_in_quote_or_evidence_empty_fails(
    value: str, quote: str,
) -> None:
    # Given
    answer = extraction(value, quote)
    # When
    result = verify(observation(), brief(), answer)
    # Then
    assert result.status == "FAIL"


@pytest.mark.parametrize("required", ["condition", "output"])
def test_answer_when_required_field_is_missing_fails(required: str) -> None:
    # Given
    task = brief(
        conditions=[AnswerFields(names=["year", "name"])] if required == "condition" else None,
        outputs=[OutputField(name="name", description="Name")] if required == "output" else [],
    )
    # When
    result = verify(observation(), task, extraction())
    # Then
    assert result.status == "FAIL"
    assert result.reason == "answer_fields_missing"


@pytest.mark.parametrize("conditions,text,url,status", [
    ([UrlContains(text="/done"), VisibleText(text="Ready now")], "Ready\n now",
     "https://example.test/done", "PASS"),
    ([UrlContains(text="/done")], "done", "https://example.test/start", "FAIL"),
    ([VisibleText(text="Done")], "Still working", "https://example.test/done", "FAIL"),
    ([VisibleText(text="Done")], "", "https://example.test/done", "UNKNOWN"),
    ([UrlContains(text="/done")], "Done", "", "UNKNOWN"),
])
def test_conditions_when_observed_values_differ_are_checked_in_code(
    conditions: list[Condition], text: str, url: str, status: str,
) -> None:
    # Given
    page, task = observation(text, url), brief(conditions=conditions)
    # When
    result = verify(page, task)
    # Then
    assert result.status == status


@pytest.mark.parametrize("kind,chosen,status", [
    ("max", "$1,900", "FAIL"), ("max", "$2,000", "PASS"),
    ("min", "$1,900", "PASS"), ("min", "$2,000", "FAIL"),
])
def test_comparison_when_another_grounded_item_is_extreme_checks_all_items(
    kind: Literal["max", "min"], chosen: str, status: str,
) -> None:
    # Given
    page = observation("A $1,900\nB $2,000")
    task = brief(comparison=Comparison(field="year", kind=kind))
    answer = extraction(chosen, f"{'A' if chosen == '$1,900' else 'B'} {chosen}", [
        FieldValue(value="$1,900", quote="A $1,900"),
        FieldValue(value="$2,000", quote="B $2,000"),
    ])
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == status


@pytest.mark.parametrize("value,quote", [
    ("$3,000", "C $3,000"), ("$3,000", "B $2,000"), ("NaN", "NaN"),
    ("Infinity", "Infinity"), ("not numeric", "not numeric"),
])
def test_comparison_when_compared_evidence_is_invalid_fails(value: str, quote: str) -> None:
    # Given
    page = observation("A $1,900 B $2,000 NaN Infinity not numeric")
    task = brief(comparison=Comparison(field="year", kind="max"))
    answer = extraction("$1,900", "A $1,900", [FieldValue(value=value, quote=quote)])
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "FAIL"


@pytest.mark.parametrize("chosen", ["$2,000", "not numeric"])
def test_comparison_when_items_are_missing_or_chosen_is_not_numeric_fails(chosen: str) -> None:
    # Given
    page = observation("A $2,000 not numeric")
    task = brief(comparison=Comparison(field="year", kind="max"))
    answer = extraction(
        chosen, f"A {chosen}",
        [] if chosen == "$2,000" else [FieldValue(value="$2,000", quote="A $2,000")],
    )
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "FAIL"


def test_comparison_when_fullwidth_currency_and_decimals_are_present_passes() -> None:
    # Given
    page = observation("A ￥１，９００．５０ B ￥２，０００．００")
    task = brief(comparison=Comparison(field="year", kind="min"))
    answer = extraction("￥１，９００．５０", "A ￥１，９００．５０", [
        FieldValue(value="￥１，９００．５０", quote="A ￥１，９００．５０"),
        FieldValue(value="￥２，０００．００", quote="B ￥２，０００．００"),
    ])
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "PASS"


@pytest.mark.parametrize("page_text,quote,status", [
    ("검색 결과가 없습니다.", "검색 결과가 없습니다.", "PASS"),
    ("No results found", "No results found", "PASS"),
    ("Items shown", "No results found", "FAIL"),
    ("Items shown", "Items shown", "FAIL"),
    ("No results found", None, "FAIL"),
    ("No results found", "", "FAIL"),
])
def test_no_result_when_page_does_not_support_the_claim_fails(
    page_text: str, quote: str | None, status: str,
) -> None:
    # Given
    answer = Extraction(fields={}, no_result=True, no_result_quote=quote, compared_items=[])
    # When
    result = verify(observation(page_text), brief(), answer)
    # Then
    assert result.status == status


@pytest.mark.parametrize("condition", [
    VisibleText(text="Results loaded"), UrlContains(text="/results"),
])
@pytest.mark.parametrize("page_text,quote,status", [
    ("Results loaded. Item A is available.", "No results found", "FAIL"),
    ("Results loaded. Item A is available.", None, "FAIL"),
    ("Results loaded. Item A is available.", "", "FAIL"),
    ("Results loaded. Item A is available.", "Item A is available", "FAIL"),
    ("Results loaded. No results found.", "No results found", "PASS"),
])
def test_no_result_when_brief_has_only_conditions_must_still_be_grounded(
    condition: Condition, page_text: str, quote: str | None, status: str,
) -> None:
    # Given
    page = observation(page_text, "https://example.test/results")
    answer = Extraction(fields={}, no_result=True, no_result_quote=quote, compared_items=[])
    # When
    result = verify(page, brief(conditions=[condition]), answer)
    # Then
    assert result.status == status
    if status == "FAIL":
        assert result.reason == "no_result_not_grounded"


@pytest.mark.parametrize("condition", [
    VisibleText(text="Results loaded"), UrlContains(text="/results"),
])
def test_conditions_only_when_no_extraction_is_given_still_pass(condition: Condition) -> None:
    # Given
    page = observation("Results loaded. Item A is available.", "https://example.test/results")
    # When
    result = verify(page, brief(conditions=[condition]))
    # Then
    assert result.status == "PASS"


def test_no_result_when_url_condition_fails_cannot_override_it() -> None:
    # Given
    task = brief(conditions=[UrlContains(text="/target"), AnswerFields(names=["year"])])
    answer = Extraction(
        fields={}, no_result=True, no_result_quote="No results", compared_items=[],
    )
    # When
    result = verify(observation("No results"), task, answer)
    # Then
    assert result.status == "FAIL"


@pytest.mark.parametrize("missing", ["observation", "brief", "extraction", "conditions", "inputs"])
def test_unknown_when_evidence_is_missing_is_never_success(missing: str) -> None:
    # Given
    page = None if missing == "observation" else observation()
    task = None if missing == "brief" else brief(conditions=[] if missing == "conditions" else None)
    if missing == "inputs" and task is not None:
        task = TaskBrief.model_validate({**task.model_dump(), "missing_inputs": ["date"]})
    answer = None if missing == "extraction" else extraction()
    # When
    result = verify(page, task, answer)
    # Then
    assert result.status == "UNKNOWN"
    assert result.status != "PASS"


def test_done_when_only_model_declaration_is_visible_cannot_finish() -> None:
    # Given
    page, task = observation("done"), brief()
    # When
    result = verify(page, task)
    # Then
    assert result.status == "UNKNOWN"


@pytest.mark.parametrize("change", ["typed", "initial", "wrong_destination", "opened", "dialog"])
def test_opened_target_requires_browser_transition_not_a_name(change: str) -> None:
    # Given
    before = observation("Widget", "https://example.test/list")
    target = ElementRef(
        snapshot_id="s", index=1, backend_node_id=1, tag="a", text="Widget",
        attributes={"href": "/detail?region=north"}, in_viewport=True,
    )
    after = observation("Widget details", "https://example.test/detail?region=north")
    action = BrowserAction(kind="click", target=target)
    if change in {"initial", "typed"}:
        after = before
        action = BrowserAction(kind="type_text", target=target, value="Widget")
    if change == "wrong_destination":
        after = after.model_copy(update={"url": "https://example.test/detail?region=south"})
    if change == "dialog":
        target = target.model_copy(update={"attributes": {}})
        action = BrowserAction(kind="click", target=target)
        after = after.model_copy(update={
            "url": before.url,
            "elements": [target.model_copy(update={"role": "dialog"})],
        })
    # When
    result = verify(after, brief([OpenedTarget(text="Widget")]), opened=(
        OpenEvidence(before, action, after),
    ))
    # Then
    assert result.status == ("PASS" if change in {"opened", "dialog"} else "FAIL")


@pytest.mark.parametrize("change", [
    "delayed", "unrelated", "wrong_target", "wrong_destination", "listing",
    "stale", "uncertain", "unexecuted", "no_navigation",
])
def test_delayed_open_is_bound_to_executed_target_and_document(change: str) -> None:
    # Given: the first post-click observation predates the destination commit.
    before = observation("Widget", "https://example.test/list")
    target = ElementRef(
        snapshot_id="s", index=1, backend_node_id=1, tag="a", text="Widget",
        attributes={"href": "/detail"}, in_viewport=True,
    )
    current = observation("Widget details", "https://example.test/detail").model_copy(
        update={"document_id": "click-loader"},
    )
    from jevpilot.browser.port import ExecutionResult
    result = ExecutionResult(outcome="executed", navigation_id="click-loader")
    if change == "unrelated":
        current = current.model_copy(update={"document_id": "later-loader"})
    if change == "wrong_target":
        target = target.model_copy(update={"text": "Other"})
    if change == "wrong_destination":
        current = current.model_copy(update={"url": "https://example.test/other"})
    if change == "listing":
        current = before.model_copy(update={"document_id": "click-loader"})
    if change == "stale":
        target = target.model_copy(update={"snapshot_id": "old"})
    if change in {"uncertain", "unexecuted"}:
        result = result.model_copy(update={
            "outcome": "uncertain" if change == "uncertain" else "failed_not_executed",
        })
    if change == "no_navigation":
        result = result.model_copy(update={"navigation_id": None})
    # When
    checked = verify(current, brief([OpenedTarget(text="Widget")]), opened=(
        OpenEvidence(before, BrowserAction(kind="click", target=target), before, result),
    ))
    # Then
    assert checked.status == ("PASS" if change == "delayed" else "FAIL")
