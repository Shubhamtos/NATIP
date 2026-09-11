"""Screener.in fundamentals collector."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, ConfigDict, Field

FetchHtml = Callable[[str], str]
FetchBinary = Callable[[str], bytes]
SCREENER_TIMEOUT_SECONDS = 30
SCREENER_RETRIES = 3
SCREENER_RETRY_BACKOFF_SECONDS = 1.5

CONCALL_SUMMARY_FOCUS = {
    "Revenue growth": (
        "revenue",
        "sales",
        "growth",
        "yoy",
        "qoq",
        "segment",
    ),
    "Profitability": (
        "ebitda",
        "margin",
        "pat",
        "profit",
        "profitability",
    ),
    "Management guidance": (
        "guidance",
        "target",
        "outlook",
        "capacity",
        "capex",
        "demand",
    ),
    "Segment performance": (
        "segment",
        "product",
        "geography",
        "business",
        "division",
    ),
    "Order book / demand visibility": (
        "order book",
        "pipeline",
        "utilisation",
        "utilization",
        "booking",
        "demand",
    ),
    "Capacity expansion": (
        "capacity",
        "expansion",
        "commission",
        "plant",
        "utilisation",
        "capex",
    ),
    "Cash flow and debt": (
        "cash flow",
        "operating cash",
        "net debt",
        "debt",
        "working capital",
        "funding",
    ),
    "Market share and competitive position": (
        "market share",
        "customer",
        "launch",
        "pricing",
        "competitive",
        "advantage",
    ),
    "Risks and weak signals": (
        "risk",
        "receivable",
        "volume",
        "pressure",
        "concentration",
        "regulatory",
        "delay",
    ),
    "New sector entry": (
        "new sector",
        "new business",
        "adjacent",
        "diversification",
        "foray",
        "entered",
    ),
    "What changed from previous concall": (
        "previous",
        "compared",
        "change",
        "improved",
        "declined",
        "increase",
        "decrease",
    ),
}


class ScreenerFundamentalReport(BaseModel):
    """Parsed Screener fundamentals for one company."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    company_name: str
    source_url: str
    fetched_at: datetime
    ratios: dict[str, str] = Field(default_factory=dict)
    shareholding_latest: dict[str, str] = Field(default_factory=dict)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    tables: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    missing_data: list[str] = Field(default_factory=list)


class ConcallTranscriptLink(BaseModel):
    """Concall transcript link discovered from Screener."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str


class ConcallTranscriptSummary(BaseModel):
    """Summary for one concall focus area."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str
    summary: str
    evidence: list[str] = Field(default_factory=list)


class ConcallTranscriptReport(BaseModel):
    """Parsed concall transcript report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    source_url: str
    fetched_at: datetime
    transcripts: list[ConcallTranscriptLink] = Field(default_factory=list)
    summaries: list[ConcallTranscriptSummary] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class ScreenerFundamentalProvider:
    """Best-effort public-page collector for Screener fundamentals."""

    def __init__(
        self,
        *,
        fetch_html: FetchHtml | None = None,
        fetch_binary: FetchBinary | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize provider.

        Args:
            fetch_html: Optional HTML fetch dependency for tests.
            fetch_binary: Optional binary fetch dependency for PDF tests.
            clock: Optional UTC clock dependency.
        """

        self._fetch_html = fetch_html or self._request_html
        self._fetch_binary = fetch_binary or self._request_binary
        self._clock = clock or (lambda: datetime.now(UTC))

    def fetch(self, symbol: str) -> ScreenerFundamentalReport:
        """Fetch and parse fundamentals for a company.

        Args:
            symbol: NSE or Screener company symbol.

        Returns:
            Parsed Screener fundamentals report.
        """

        cleaned_symbol = symbol.strip().upper().replace(".NS", "")
        source_url = f"https://www.screener.in/company/{cleaned_symbol}/"
        html = self._fetch_html(source_url)
        return parse_screener_fundamentals_page(
            html,
            symbol=cleaned_symbol,
            source_url=source_url,
            fetched_at=self._clock(),
        )

    def fetch_concall_transcripts(
        self,
        symbol: str,
        *,
        max_transcripts: int = 1,
    ) -> ConcallTranscriptReport:
        """Fetch and summarize latest concall transcripts from Screener.

        Args:
            symbol: NSE or Screener company symbol.
            max_transcripts: Maximum latest PDF transcripts to read.

        Returns:
            Concall transcript summary report.
        """

        cleaned_symbol = symbol.strip().upper().replace(".NS", "")
        source_url = f"https://www.screener.in/company/{cleaned_symbol}/"
        html = self._fetch_html(source_url)
        links = discover_concall_transcript_links(html, source_url=source_url)
        missing_data: list[str] = []
        if not links:
            missing_data.append("No concall transcript PDF links were visible on Screener.")
        used_links = links[:max_transcripts]
        extracted_texts: list[str] = []
        for link in used_links:
            try:
                pdf_bytes = self._fetch_binary(link.url)
                extracted_texts.append(_extract_pdf_text(pdf_bytes))
            except Exception as exc:
                missing_data.append(f"Could not read concall transcript '{link.title}': {exc}")
        summaries = summarize_concall_transcript_texts(extracted_texts)
        if not extracted_texts:
            missing_data.append("Concall transcript text could not be extracted.")
        return ConcallTranscriptReport(
            symbol=cleaned_symbol,
            source_url=source_url,
            fetched_at=self._clock(),
            transcripts=used_links,
            summaries=summaries,
            missing_data=missing_data,
        )

    @staticmethod
    def _request_html(url: str) -> str:
        """Request a Screener company page."""

        return _request_with_retries(
            url,
            purpose="one-company fundamental review",
        ).text

    @staticmethod
    def _request_binary(url: str) -> bytes:
        """Request a binary document."""

        return _request_with_retries(
            url,
            purpose="one-company concall transcript review",
        ).content


def _request_with_retries(url: str, *, purpose: str) -> requests.Response:
    """Request Screener with retry/backoff for slow public pages."""

    last_error: Exception | None = None
    headers = {
        "User-Agent": (f"NATIP local research tool; contact: local-user; purpose: {purpose}")
    }
    for attempt in range(1, SCREENER_RETRIES + 1):
        try:
            response = requests.get(
                url,
                timeout=SCREENER_TIMEOUT_SECONDS,
                headers=headers,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == SCREENER_RETRIES:
                break
            time.sleep(SCREENER_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def parse_screener_fundamentals_page(
    html: str,
    *,
    symbol: str,
    source_url: str,
    fetched_at: datetime,
) -> ScreenerFundamentalReport:
    """Parse key Screener fundamentals from a company HTML page."""

    soup = BeautifulSoup(html, "html.parser")
    ratios = _top_ratios(soup)
    tables = _financial_tables(soup)
    shareholding_latest = _latest_shareholding(tables.get("Shareholding", []))
    pros, cons = _pros_cons(soup)
    missing_data: list[str] = []
    if not ratios:
        missing_data.append("Screener top ratios were not visible.")
    for table_name in ["Quarters", "Profit & Loss", "Balance Sheet", "Cash Flow", "Ratios"]:
        if table_name not in tables:
            missing_data.append(f"Screener {table_name} table was not visible.")
    if not shareholding_latest:
        missing_data.append("Screener latest shareholding percentages were not visible.")

    return ScreenerFundamentalReport(
        symbol=symbol,
        company_name=_company_name(soup) or symbol,
        source_url=source_url,
        fetched_at=fetched_at,
        ratios=ratios,
        shareholding_latest=shareholding_latest,
        pros=pros,
        cons=cons,
        tables=tables,
        missing_data=missing_data,
    )


def discover_concall_transcript_links(
    html: str,
    *,
    source_url: str,
) -> list[ConcallTranscriptLink]:
    """Discover concall transcript PDF links from a Screener company page."""

    soup = BeautifulSoup(html, "html.parser")
    links: list[ConcallTranscriptLink] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "")
        title = _clean_text(anchor.get_text(" ", strip=True)) or href.rsplit("/", maxsplit=1)[-1]
        haystack = f"{title} {href}".casefold()
        if ".pdf" not in haystack and "transcript" not in haystack and "concall" not in haystack:
            continue
        if not any(
            marker in haystack
            for marker in (
                "concall",
                "con-call",
                "conference call",
                "earnings call",
                "transcript",
            )
        ):
            continue
        url = urljoin(source_url, href)
        if url in seen:
            continue
        seen.add(url)
        links.append(ConcallTranscriptLink(title=title, url=url))
    return links


def summarize_concall_transcript_texts(
    texts: list[str],
) -> list[ConcallTranscriptSummary]:
    """Summarize extracted concall transcript text by investor checklist topic."""

    combined = "\n".join(texts)
    if not combined.strip():
        return [
            ConcallTranscriptSummary(
                topic=topic,
                summary="Concall transcript text was not available for this checklist item.",
                evidence=[],
            )
            for topic in CONCALL_SUMMARY_FOCUS
        ]
    sentences = _transcript_sentences(combined)
    summaries: list[ConcallTranscriptSummary] = []
    for topic, keywords in CONCALL_SUMMARY_FOCUS.items():
        evidence = _matching_sentences(sentences, keywords, limit=4)
        if evidence:
            summary = " ".join(evidence[:2])
        else:
            summary = (
                "No clear disclosure found in the extracted concall transcript text for this "
                "checklist item."
            )
        summaries.append(
            ConcallTranscriptSummary(
                topic=topic,
                summary=summary,
                evidence=evidence,
            )
        )
    return summaries


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes with optional pypdf dependency."""

    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("Install pypdf to read concall transcript PDFs.") from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages[:40]]
    return "\n".join(page_texts)


def _transcript_sentences(text: str) -> list[str]:
    """Split concall transcript text into compact sentence-like snippets."""

    normalized = " ".join(text.replace("\x00", " ").split())
    raw_parts = normalized.replace(";", ".").split(".")
    return [
        part.strip()
        for part in raw_parts
        if 30 <= len(part.strip()) <= 280 and any(character.isalpha() for character in part)
    ]


def _matching_sentences(
    sentences: list[str],
    keywords: tuple[str, ...],
    *,
    limit: int,
) -> list[str]:
    """Return best matching snippets for a topic."""

    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        lowered = sentence.casefold()
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score:
            scored.append((score, sentence))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence in scored:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected


def _company_name(soup: BeautifulSoup) -> str | None:
    """Extract company name from page title or heading."""

    heading = soup.find("h1")
    if heading:
        return _clean_text(heading.get_text(" ", strip=True))
    title = soup.find("title")
    if title:
        return _clean_text(title.get_text(" ", strip=True).split("|", maxsplit=1)[0])
    return None


def _top_ratios(soup: BeautifulSoup) -> dict[str, str]:
    """Extract Screener top-ratio cards."""

    ratios: dict[str, str] = {}
    container = soup.find(id="top-ratios") or soup.find("ul", class_="top-ratios")
    if container is None:
        return ratios
    for item in container.find_all("li"):
        label_node = item.find(class_="name")
        value_node = item.find(class_="number")
        label = _clean_text(label_node.get_text(" ", strip=True)) if label_node else ""
        value = _clean_text(value_node.get_text(" ", strip=True)) if value_node else ""
        if not label or not value:
            texts = [
                _clean_text(text)
                for text in item.stripped_strings
                if _clean_text(text) and _clean_text(text) not in {"/", ":"}
            ]
            if len(texts) >= 2:
                label, value = texts[0], " ".join(texts[1:])
        if label and value:
            ratios[label] = value
    return ratios


def _financial_tables(soup: BeautifulSoup) -> dict[str, list[dict[str, str]]]:
    """Extract main Screener financial tables."""

    table_specs = {
        "Quarters": "quarters",
        "Profit & Loss": "profit-loss",
        "Balance Sheet": "balance-sheet",
        "Cash Flow": "cash-flow",
        "Ratios": "ratios",
        "Shareholding": "shareholding",
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for label, section_id in table_specs.items():
        section = soup.find(id=section_id)
        table = section.find("table") if isinstance(section, Tag) else None
        if table is None:
            table = _table_after_heading(soup, label)
        rows = _table_rows(table) if isinstance(table, Tag) else []
        if rows:
            tables[label] = rows
    return tables


def _table_after_heading(soup: BeautifulSoup, heading_text: str) -> Tag | None:
    """Find a table following a heading with matching text."""

    target = heading_text.lower()
    for heading in soup.find_all(["h2", "h3"]):
        if target not in _clean_text(heading.get_text(" ", strip=True)).lower():
            continue
        table = heading.find_next("table")
        if isinstance(table, Tag):
            return table
    return None


def _table_rows(table: Tag | None) -> list[dict[str, str]]:
    """Convert an HTML table to row dictionaries."""

    if table is None:
        return []
    rows = table.find_all("tr")
    if not rows:
        return []
    headers = [
        _clean_text(cell.get_text(" ", strip=True)) for cell in rows[0].find_all(["th", "td"])
    ]
    if not headers:
        return []

    parsed: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = [_clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
        if not any(cells):
            continue
        record: dict[str, str] = {}
        for index, value in enumerate(cells):
            header = (
                headers[index] if index < len(headers) and headers[index] else f"Column {index}"
            )
            record[header] = value
        parsed.append(record)
    return parsed


def _pros_cons(soup: BeautifulSoup) -> tuple[list[str], list[str]]:
    """Extract Screener pros and cons lists."""

    return _list_near_heading(soup, "Pros"), _list_near_heading(soup, "Cons")


def _latest_shareholding(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return latest available holding percentage by holder category."""

    latest: dict[str, str] = {}
    for row in rows:
        category = _first_non_empty_value(row)
        period = _latest_period_key(row)
        if category and period and row.get(period):
            latest[category.rstrip("+").strip()] = row[period]
    return latest


def _first_non_empty_value(row: dict[str, str]) -> str | None:
    """Return first non-empty value in a parsed Screener row."""

    for value in row.values():
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _latest_period_key(row: dict[str, str]) -> str | None:
    """Return the latest period column from a Screener table row."""

    keys = list(row)
    for key in reversed(keys[1:]):
        value = _clean_text(row.get(key, ""))
        if value and value not in {"-", "--"}:
            return key
    return None


def _list_near_heading(soup: BeautifulSoup, heading_text: str) -> list[str]:
    """Return list items near a heading."""

    target = heading_text.lower()
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if target not in _clean_text(heading.get_text(" ", strip=True)).lower():
            continue
        list_node = heading.find_next(["ul", "ol"])
        if not isinstance(list_node, Tag):
            return []
        return [
            _clean_text(item.get_text(" ", strip=True))
            for item in list_node.find_all("li", recursive=False)
            if _clean_text(item.get_text(" ", strip=True))
        ]
    return []


def _clean_text(value: Any) -> str:
    """Normalize page text."""

    return " ".join(str(value).replace("\xa0", " ").split())
