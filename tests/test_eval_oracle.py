"""Independent answer and fixture grading; no network or paid providers."""
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.browser_use_eval.oracle import PageEvidence, grade, normalize


@pytest.mark.parametrize("text,answer,http_status", [
    ("Access denied", "", None),
    ("403 Forbidden", "", None),
    ("403 Forbidden", "3.11", None),
    ("접근이 차단되었습니다.", "", None),
    ("429 Too Many Requests", "", None),
    ("Version 3.11", "3.11", 403),
    ("Version 3.11", "3.11", 429),
    ("접속을 제한합니다", "3.11", 200),
    ("요청이 너무 많습니다", "3.11", 200),
    ("CAPTCHA", "3.11", 200),
    ("Verify that you are a human", "3.11", 200),
    ("Rate limit exceeded", "3.11", 200),
    ("자동입력 방지 보안문자", "3.11", 200),
])
def test_blocked_when_independent_page_denies_access_precedes_correct_answer(
    text: str, answer: str, http_status: int | None,
) -> None:
    # Given: the answer may look correct; only the harness page is trusted.
    page = PageEvidence("https://example.test", text, http_status=http_status)
    # When
    result = grade("D-1", answer, page)
    # Then
    assert result.status == "blocked"


@pytest.mark.parametrize("http_status", [None, 200])
def test_normal_page_when_status_is_available_or_unknown_is_still_graded(
    http_status: int | None,
) -> None:
    # Given
    page = PageEvidence("https://example.test", "Added in version 3.11", http_status=http_status)
    # When
    result = grade("D-1", "3.11", page)
    # Then
    assert result.status == "passed"


def test_fixture_when_blocked_cannot_pass_using_cached_dom_and_url() -> None:
    # Given
    page = PageEvidence("http://fixture.test/items/aurora.html", "Aurora", "aurora", 403)
    # When
    result = grade("F-04", "", page)
    # Then
    assert result.status == "blocked"


def test_oracle_import_when_product_is_forbidden() -> None:
    # Given: a fresh interpreter rejects even transitive product imports.
    script = """
import importlib.abc
import sys
class BlockProduct(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'jevpilot' or fullname.startswith('jevpilot.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockProduct())
from benchmarks.browser_use_eval.oracle import grade, PageEvidence
assert grade('D-1', '3.11', PageEvidence('', '')).status == 'passed'
assert not any(n == 'jevpilot' or n.startswith('jevpilot.') for n in sys.modules)
"""
    # When
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1],
        capture_output=True, text=True, timeout=10, check=False,
    )
    # Then
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("answer", ["  Ｍａｒｃｈ　１４, １８７９ ", "march 14,\n1879"])
def test_normalization_when_spacing_case_and_width_differ(answer: str) -> None:
    # Given / When
    result = grade("H-1", answer, PageEvidence("", "Albert Einstein"))
    # Then
    assert result.status == "passed"
    assert normalize("Ａ\t B") == "a b"


@pytest.mark.parametrize("scenario,url,item,status", [
    ("F-02", "/items/aurora.html?region=north", "aurora", "passed"),
    ("F-02", "/items/aurora.html?region=south", "aurora", "failed"),
    ("F-04", "/items/aurora.html", "decoy", "failed"),
    ("F-04", "/items/aurora.html", "aurora", "passed"),
    ("F-03", "/submitted.html?name=Riley&city=Reykjavik&region=north", "", "passed"),
    ("F-03", "/submitted.html?name=Riley&city=Wrong&region=north", "", "failed"),
    ("F-05", "/items/aurora.html?region=north&free=true", "", "passed"),
    ("F-05", "/items/aurora.html?region=north", "", "failed"),
])
def test_fixture_when_url_and_dom_are_independently_observed(
    scenario: str, url: str, item: str, status: str,
) -> None:
    # Given
    page = PageEvidence("http://fixture.test" + url, "Aurora", item)
    # When
    result = grade(scenario, "agent says done", page)
    # Then
    assert result.status == status


@pytest.mark.parametrize("text,answer,status", [
    ("KTX 101 서울 09:20 부산 12:00 일반실 예약가능", "09:20 일반실 예약가능", "unverifiable"),
    ("표의 시작\nKTX 101 서울 09:20 부산 12:00 일반실 예약가능", "09:20 일반실 예약가능", "passed"),
    ("KTX 99 서울 08:50 부산 11:00 일반실 매진\nKTX 101 서울 09:20 부산 12:00 일반실 예약가능",
     "09:20 일반실 예약가능", "passed"),
    ("KTX 99 서울 08:50 부산 11:00 일반실 매진\nKTX 101 서울 09:10 부산 12:00 일반실 매진\n"
     "KTX 103 서울 09:20 부산 12:30 일반실 예약가능", "09:20 일반실 예약가능", "failed"),
    ("표의 시작\nKTX 99 서울 09:00 부산 11:00 일반실 예약가능\n"
     "KTX 101 서울 09:10 부산 12:00 일반실 매진", "09:10 일반실 예약불가", "passed"),
    ("표의 시작\nKTX 101 서울 09:20 부산 12:00 특실 예약가능 일반실 매진",
     "09:20 일반실 예약가능", "failed"),
    ("표의 시작\nKTX 101 서울 09:20 부산 12:00 특실 예약가능", "09:20 예약가능", "unverifiable"),
    ("Access denied", "09:20 예약가능", "blocked"),
    ("구분 열차번호 출발 도착 특실 일반실\n"
     "KTX 99 서울 08:50 부산 11:00 예약하기 매진\n"
     "KTX 101 서울 09:20 부산 12:00 매진 예약하기",
     "09:20 일반실 예약가능", "passed"),
])
def test_train_when_only_complete_table_proves_first_departure(
    text: str, answer: str, status: str,
) -> None:
    # Given
    page = PageEvidence("https://www.letskorail.com", text)
    # When
    result = grade("H-4", answer, page)
    # Then
    assert result.status == status


def test_catalogue_when_independent_fetch_differs_from_agent() -> None:
    # Given
    fetched: list[str] = []
    html = """
    <article><h3><a title="Cheap">x</a></h3><p class="price_color">£12.00</p></article>
    <article><h3><a title="Costly &amp; Good">y</a></h3><p class="price_color">£55.99</p></article>
    """

    def fetch(url: str) -> str:
        fetched.append(url)
        return html
    # When
    result = grade("D-3", "Costly & Good £55.99", PageEvidence("", "Cheap"), fetch=fetch)
    # Then
    assert result.status == "passed"
    assert fetched == ["https://books.toscrape.com/catalogue/category/books/travel_2/index.html"]


def test_mystery_when_first_title_comes_from_page_two() -> None:
    # Given
    fetched: list[str] = []

    def fetch(url: str) -> str:
        fetched.append(url)
        return '<h3><a title="First">x</a></h3><h3><a title="Second">y</a></h3>'
    # When
    result = grade("H-3", "First", PageEvidence("", ""), fetch=fetch)
    # Then
    assert result.status == "passed"
    assert fetched == ["https://books.toscrape.com/catalogue/category/books/mystery_3/page-2.html"]


@pytest.mark.parametrize("scenario,answer", [("H-2", "submitted"), ("D-2", "13970"), ("D-1", "3.110")])
def test_answer_when_expected_is_only_a_substring_fails(scenario: str, answer: str) -> None:
    # Given / When
    result = grade(scenario, answer, PageEvidence("", ""))
    # Then
    assert result.status == "failed"


def test_korean_year_suffix_when_answer_is_correct_passes() -> None:
    # Given / When
    result = grade("D-2", "1397년", PageEvidence("", ""))
    # Then
    assert result.status == "passed"
