"""Independent grading: no product imports, observations, or success declarations."""
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from html.parser import HTMLParser
import re
import unicodedata
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen
from typing import Literal


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True, slots=True)
class Verdict:
    status: Literal["passed", "failed", "blocked", "unverifiable"]
    reason: str


@dataclass(frozen=True, slots=True)
class PageEvidence:
    url: str
    page_text: str
    item_id: str = ""
    http_status: int | None = None
    language: str = ""
    languages: tuple[str, ...] = ()
    intl_locale: str = ""
    viewport: tuple[int, int] | None = None


class BookParser(HTMLParser):
    """Read titles and prices from the independent catalogue response."""

    def __init__(self) -> None:
        super().__init__()
        self.books: list[tuple[str, Decimal]] = []
        self.titles: list[str] = []
        self._title: str | None = None
        self._price = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a" and attributes.get("title"):
            self._title = attributes["title"]
            assert self._title is not None
            self.titles.append(self._title)
        if tag == "p" and "price_color" in (attributes.get("class") or "").split():
            self._price = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._price = False

    def handle_data(self, data: str) -> None:
        if self._price and self._title:
            amount = re.search(r"\d+\.\d{2}", data)
            if amount:
                self.books.append((self._title, Decimal(amount[0])))


def fetch_html(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8")


def train_verdict(answer: str, text: str) -> Verdict:
    """Full body text must establish the beginning of the searched timetable."""
    normalized = normalize(text)
    # A row starts at a KTX service and ends before the next service.
    rows = re.findall(
        r"\bKTX(?:-산천|-이음)?\s*(?:\d+)?\s+(.+?)(?=\bKTX|\Z)",
        unicodedata.normalize("NFKC", text), flags=re.S | re.I,
    )
    heading = re.split(r"\bKTX", text, maxsplit=1, flags=re.I)[0]
    seat_columns = re.findall(r"특실|일반실", heading)
    parsed: list[tuple[int, str, str]] = []
    for row in rows:
        times = re.findall(r"(?<!\d)([0-2]\d:[0-5]\d)(?!\d)", row)
        if not times:
            continue
        departure = times[0]
        hour, minute = map(int, departure.split(":"))
        if hour > 23:
            continue
        # Use the explicit general-seat column, never a special-seat status.
        available = re.search(
            r"일반실\s*[:：]?\s*(예약\s*가능|예약하기|매진|입석|예약대기|불가)",
            row,
        )
        availability: str | None = available[1] if available else None
        if availability is None and "일반실" in seat_columns:
            statuses = re.findall(r"예약\s*가능|예약하기|매진|입석|예약대기|불가", row)
            index = seat_columns.index("일반실")
            if len(statuses) >= len(seat_columns):
                availability = statuses[index]
        if availability is None:
            return Verdict("unverifiable", "general-seat column missing")
        parsed.append((hour * 60 + minute, departure, normalize(availability)))
    has_beginning = any(minutes < 540 for minutes, _, _ in parsed) or bool(
        re.search(r"(?:표의 시작|첫 페이지|조회 시작|전체 조회|전체 열차|1\s*/\s*\d+)", normalized)
    )
    if not parsed or not has_beginning:
        return Verdict("unverifiable", "partial timetable cannot prove first departure")
    candidates = sorted(row for row in parsed if row[0] > 540)
    if not candidates:
        return Verdict("unverifiable", "no evidenced departure after 09:00")
    _, departure, availability = candidates[0]
    claim = normalize(answer)
    claimed_times = re.findall(r"(?<!\d)[0-2]\d:[0-5]\d(?!\d)", claim)
    if claimed_times != [departure]:
        return Verdict("failed", "answer is not the first departure")
    available = availability in {"예약 가능", "예약가능", "예약하기"}
    positive = bool(re.search(r"예약\s*가능|예약하기", claim))
    negative = bool(re.search(r"매진|예약\s*불가|예약\s*불가능|예약할 수 없", claim))
    passed = positive and not negative if available else negative and not positive
    return Verdict("passed" if passed else "failed", "first departure and general seats")


def grade(
    scenario_id: str, answer: str, page: PageEvidence, *,
    fetch: Callable[[str], str] = fetch_html,
) -> Verdict:
    normalized = normalize(page.page_text)
    if page.http_status in {403, 429} or re.search(
        r"access (?:denied|blocked|restricted)|forbidden|"
        r"too many requests|rate[\s-]?limit(?:ed| exceeded)|"
        r"captcha|verify (?:that )?you are (?:a )?human|"
        r"접(?:근|속)(?:이|을)?\s*(?:차단|제한|거부)|비정상적인 접근|"
        r"요청(?:이|을)?\s*(?:너무 많|제한)|자동\s*입력\s*방지|"
        r"보안\s*문자|로봇이 아님",
        normalized,
    ):
        return Verdict("blocked", "site blocked independent observation")
    if scenario_id.startswith("F-"):
        parts = urlsplit(page.url)
        query = parse_qs(parts.query)
        expected = {"F-02": {"region": ["north"]}, "F-03": {
            "name": ["Riley"], "city": ["Reykjavik"], "region": ["north"],
        }, "F-04": {}, "F-05": {"region": ["north"], "free": ["true"]}}[scenario_id]
        path = "/submitted.html" if scenario_id == "F-03" else "/items/aurora.html"
        passed = parts.path == path and all(query.get(k) == v for k, v in expected.items())
        if scenario_id in {"F-02", "F-04"}:
            passed = passed and page.item_id == "aurora"
        return Verdict("passed" if passed else "failed", "trusted fixture URL/DOM conditions")
    if scenario_id == "H-4":
        return train_verdict(answer, page.page_text)
    expected_values = {
        "D-1": ["3.11"], "D-2": ["1397"],
        "H-1": ["March 14, 1879"], "H-2": ["MIT"],
    }
    if scenario_id in {"D-3", "H-3"}:
        url = (
            "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
            if scenario_id == "D-3" else
            "https://books.toscrape.com/catalogue/category/books/mystery_3/page-2.html"
        )
        try:
            parser = BookParser()
            parser.feed(fetch(url))
        except (OSError, TimeoutError, ValueError):
            return Verdict("unverifiable", "independent source unavailable")
        if scenario_id == "D-3":
            if not parser.books:
                return Verdict("unverifiable", "independent catalogue empty")
            title, price = max(parser.books, key=lambda book: book[1])
            expected_values[scenario_id] = [title, f"{price:.2f}"]
        else:
            if not parser.titles:
                return Verdict("unverifiable", "independent page has no titles")
            expected_values[scenario_id] = [parser.titles[0]]
    claim = normalize(answer)
    # Boundaries prevent MIT matching "submitted", 1397 matching 13970, etc.
    passed = all(
        re.search(r"(?<!\w)" + re.escape(normalize(value)) + r"(?![a-z0-9_])", claim)
        for value in expected_values[scenario_id]
    )
    return Verdict("passed" if passed else "failed", "independent answer comparison")
