"""Screener latest quarterly-result collector."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

from app.intelligence.quarterly_results.analysis import analyze_quarterly_result, parse_number
from app.intelligence.quarterly_results.models import QuarterlyResultRecord
from app.intelligence.quarterly_results.pdf import PDF_MAX_BYTES, extract_pdf_facts
from app.intelligence.quarterly_results.storage import QuarterlyResultsStore
from app.providers.fundamentals.screener import parse_screener_fundamentals_page

SCREENER_LATEST_RESULTS_URL = "https://www.screener.in/results/latest/"
DEFAULT_USER_AGENT = "NATIP local research tool; purpose: quarterly-result analysis"

FetchText = Callable[[str], str]
FetchBinary = Callable[[str], bytes]


class ScreenerQuarterlyResultsCollector:
    """Collect latest quarterly results from Screener where access is permitted."""

    def __init__(
        self,
        *,
        store: QuarterlyResultsStore | None = None,
        fetch_text: FetchText | None = None,
        fetch_binary: FetchBinary | None = None,
        clock: Callable[[], datetime] | None = None,
        request_delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        """Create collector with injectable network dependencies for tests."""

        self.store = store or QuarterlyResultsStore()
        self._fetch_text = fetch_text or self._request_text
        self._fetch_binary = fetch_binary or self._request_binary
        self._clock = clock or (lambda: datetime.now(UTC))
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max(1, max_retries)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    def collect_latest(
        self,
        *,
        max_pages: int = 1,
        max_results: int = 10,
    ) -> list[QuarterlyResultRecord]:
        """Collect, validate, analyze and save latest result records."""

        records: list[QuarterlyResultRecord] = []
        for page in range(1, max(1, max_pages) + 1):
            page_url = _url_with_page(SCREENER_LATEST_RESULTS_URL, page)
            html = self._fetch_text(page_url)
            if _looks_access_blocked(html):
                blocked = QuarterlyResultRecord(
                    filing_id=f"access_blocked_{self._clock():%Y%m%d%H%M%S}",
                    company_name="Screener latest results",
                    screener_company_url=SCREENER_LATEST_RESULTS_URL,
                    source_url=page_url,
                    status="access_blocked",
                    status_message=(
                        "Screener latest results page requires login or permitted account access. "
                        "NATIP did not bypass access controls."
                    ),
                    validation_warnings=[
                        "Access blocked: sign in through an approved workflow or provide permitted cached HTML."
                    ],
                )
                self.store.save_record(blocked)
                return [blocked]
            discovered = parse_latest_results_page(html, source_url=page_url, fetched_at=self._clock())
            if not discovered:
                empty = QuarterlyResultRecord(
                    filing_id=f"no_results_{self._clock():%Y%m%d%H%M%S}",
                    company_name="Screener latest results",
                    screener_company_url=SCREENER_LATEST_RESULTS_URL,
                    source_url=page_url,
                    status="failed",
                    status_message="No result rows were parsed from the latest-results page.",
                    validation_warnings=["Screener page layout may have changed or rows were hidden."],
                )
                self.store.save_record(empty)
                return [empty]
            for record in discovered:
                if len(records) >= max_results:
                    return records
                records.append(self.process_record(record))
                time.sleep(max(0.0, self.request_delay_seconds))
        return records

    def collect_from_latest_results_html(
        self,
        html: str,
        *,
        source_url: str = SCREENER_LATEST_RESULTS_URL,
        max_results: int = 10,
        enrich: bool = False,
    ) -> list[QuarterlyResultRecord]:
        """Collect latest results from user-provided permitted Screener HTML.

        This supports a safe workflow where the user logs into Screener in their
        browser, saves the latest-results page HTML, and imports it locally without
        sharing or storing credentials in NATIP.
        """

        if _looks_access_blocked(html):
            blocked = QuarterlyResultRecord(
                filing_id=f"access_blocked_import_{self._clock():%Y%m%d%H%M%S}",
                company_name="Screener latest results import",
                screener_company_url=source_url,
                source_url=source_url,
                status="access_blocked",
                status_message="Imported HTML is still a login/register/access-blocked page.",
                validation_warnings=[
                    "Open Screener after logging in, then save/import the actual latest-results page."
                ],
            )
            return [self.store.save_record(blocked)]
        discovered = parse_latest_results_page(html, source_url=source_url, fetched_at=self._clock())
        if not discovered:
            empty = QuarterlyResultRecord(
                filing_id=f"no_results_import_{self._clock():%Y%m%d%H%M%S}",
                company_name="Screener latest results import",
                screener_company_url=source_url,
                source_url=source_url,
                status="failed",
                status_message="No result rows were parsed from the imported HTML.",
                validation_warnings=["Imported HTML may not be the Screener latest-results page."],
            )
            return [self.store.save_record(empty)]
        records: list[QuarterlyResultRecord] = []
        for record in discovered[: max(1, max_results)]:
            if enrich:
                records.append(self.process_record(record))
                time.sleep(max(0.0, self.request_delay_seconds))
            else:
                validated = validate_record(record.model_copy(update={"status": "validated"}))
                analysis = analyze_quarterly_result(validated)
                published = validated.model_copy(
                    update={
                        "status": "published",
                        "status_message": (
                            "Fast import from logged-in latest-results HTML. "
                            "Company-page/PDF enrichment was not run."
                        ),
                        "analysis": analysis,
                    }
                )
                records.append(self.store.save_record(published))
        return records

    def collect_from_latest_results_text(
        self,
        page_text: str,
        *,
        source_url: str = SCREENER_LATEST_RESULTS_URL,
        max_results: int = 10,
    ) -> list[QuarterlyResultRecord]:
        """Collect latest results from copied visible Screener page text.

        This is the fastest local workflow when the user is already logged into
        Screener in Chrome: open the latest-results page, select all visible
        content, copy, and paste it into NATIP. No credentials are stored and no
        extra company/PDF requests are made.
        """

        discovered = parse_latest_results_text(page_text, source_url=source_url, fetched_at=self._clock())
        if not discovered:
            empty = QuarterlyResultRecord(
                filing_id=f"no_results_text_import_{self._clock():%Y%m%d%H%M%S}",
                company_name="Screener latest results text import",
                screener_company_url=source_url,
                source_url=source_url,
                status="failed",
                status_message="No result rows were parsed from the pasted Screener page text.",
                validation_warnings=[
                    "Paste copied visible text from https://www.screener.in/results/latest/ after logging in."
                ],
            )
            return [self.store.save_record(empty)]

        records: list[QuarterlyResultRecord] = []
        for record in discovered[: max(1, max_results)]:
            validated = validate_record(record.model_copy(update={"status": "validated"}))
            analysis = analyze_quarterly_result(validated)
            published = validated.model_copy(
                update={
                    "status": "published",
                    "status_message": (
                        "Fast import from copied logged-in latest-results page text. "
                        "Company-page/PDF enrichment was not run."
                    ),
                    "analysis": analysis,
                }
            )
            records.append(self.store.save_record(published))
        return records

    def process_record(self, record: QuarterlyResultRecord) -> QuarterlyResultRecord:
        """Process one discovered result through download/extract/validate/analyze."""

        working = record
        try:
            company_html = self._fetch_text(record.screener_company_url)
            report = parse_screener_fundamentals_page(
                company_html,
                symbol=record.symbol or _symbol_from_company_url(record.screener_company_url) or record.company_name,
                source_url=record.screener_company_url,
                fetched_at=self._clock(),
            )
            quarterly_history = report.tables.get("Quarters", [])[:]
            top_ratios = report.ratios
            history_unavailable = len(quarterly_history) < 4
            source_values = dict(record.source_values)
            source_values.update(_latest_quarter_values(quarterly_history))
            normalized = dict(record.normalized_values)
            normalized.update({key: parse_number(value) for key, value in source_values.items()})
            working = working.model_copy(
                update={
                    "status": "downloaded",
                    "top_ratios": top_ratios,
                    "quarterly_history": quarterly_history[-8:],
                    "historical_unavailable": history_unavailable,
                    "source_values": source_values,
                    "normalized_values": normalized,
                }
            )
        except Exception as exc:
            working = working.model_copy(
                update={
                    "status": "failed",
                    "status_message": f"Company history fetch failed: {exc}",
                    "extraction_errors": [*working.extraction_errors, f"Company history fetch failed: {exc}"],
                }
            )

        if working.pdf_url:
            try:
                pdf_bytes = self._fetch_binary(working.pdf_url)
                pdf_path, digest = self.store.save_pdf(working.filing_id, pdf_bytes)
                facts = extract_pdf_facts(pdf_bytes, document_url=working.pdf_url)
                working = working.model_copy(
                    update={
                        "status": "extracted",
                        "pdf_path": str(pdf_path),
                        "pdf_sha256": digest,
                        "document_facts": facts,
                    }
                )
            except Exception as exc:
                working = working.model_copy(
                    update={
                        "extraction_errors": [*working.extraction_errors, f"PDF extraction failed: {exc}"],
                    }
                )

        validated = validate_record(working)
        analysis = analyze_quarterly_result(validated)
        published = validated.model_copy(update={"status": "published", "analysis": analysis})
        return self.store.save_record(published)

    def _request_text(self, url: str) -> str:
        """Request text with bounded retries."""

        return self._request(url).text

    def _request_binary(self, url: str) -> bytes:
        """Request binary PDF with bounded retries and size limits."""

        response = self._request(url, stream=True)
        content = response.content
        if len(content) > PDF_MAX_BYTES:
            raise ValueError("PDF exceeded NATIP download safety limit.")
        return content

    def _request(self, url: str, *, stream: bool = False) -> requests.Response:
        """Request a URL with retry/backoff."""

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=30, stream=stream)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(attempt * 1.5)
        assert last_error is not None
        raise last_error


def parse_latest_results_page(
    html: str,
    *,
    source_url: str,
    fetched_at: datetime,
) -> list[QuarterlyResultRecord]:
    """Parse latest-results rows from Screener HTML."""

    soup = BeautifulSoup(html, "html.parser")
    records: list[QuarterlyResultRecord] = []
    for row in _candidate_rows(soup):
        record = _record_from_row(row, source_url=source_url, fetched_at=fetched_at)
        if record is not None:
            records.append(record)
    return _deduplicate(records)


def parse_latest_results_text(
    page_text: str,
    *,
    source_url: str,
    fetched_at: datetime,
) -> list[QuarterlyResultRecord]:
    """Parse copied visible text from Screener latest-results page."""

    lines = _copied_text_lines(page_text)
    records: list[QuarterlyResultRecord] = []
    for start, end in _text_result_blocks(lines):
        record = _record_from_text_block(lines[start:end], source_url=source_url, fetched_at=fetched_at)
        if record is not None:
            records.append(record)
    return _deduplicate(records)


def validate_record(record: QuarterlyResultRecord) -> QuarterlyResultRecord:
    """Validate identity, period, basis and source consistency."""

    warnings = list(record.validation_warnings)
    if not record.symbol:
        warnings.append("Stock identifier was not visible; company identity requires manual review.")
    if not record.reporting_quarter:
        warnings.append("Reporting quarter was not visible.")
    if not record.reporting_basis:
        warnings.append("Consolidated/standalone basis was not visible.")
    if record.historical_unavailable:
        warnings.append("Eight-quarter history was not fully available.")
    if record.pdf_url and not record.pdf_sha256:
        warnings.append("PDF link was visible but document could not be hashed/extracted.")
    if not record.quarterly_history:
        warnings.append("Screener quarterly history was unavailable; calculations are limited.")
    return record.model_copy(update={"status": "validated", "validation_warnings": warnings})


def _candidate_rows(soup: BeautifulSoup) -> list[Tag]:
    """Return rows that look like latest-result entries."""

    rows: list[Tag] = []
    for row in soup.find_all("tr"):
        if row.find("a", href=re.compile(r"/company/")):
            rows.append(row)
    if rows:
        return rows
    for item in soup.find_all(["li", "article", "div"]):
        text = item.get_text(" ", strip=True)
        if len(text) < 20:
            continue
        if item.find("a", href=re.compile(r"/company/")) and re.search(r"result|quarter|profit|sales", text, re.I):
            rows.append(item)
    return rows


def _record_from_row(
    row: Tag,
    *,
    source_url: str,
    fetched_at: datetime,
) -> QuarterlyResultRecord | None:
    """Parse one latest-results row into a record."""

    company_anchor = row.find("a", href=re.compile(r"/company/"))
    if not isinstance(company_anchor, Tag):
        return None
    company_url = urljoin(source_url, str(company_anchor.get("href") or ""))
    company_name = _clean(company_anchor.get_text(" ", strip=True)) or _symbol_from_company_url(company_url) or "Unknown"
    symbol = _symbol_from_company_url(company_url)
    text = _clean(row.get_text(" ", strip=True))
    pdf_url = _pdf_link(row, source_url=source_url)
    quarter = _quarter_from_text(text)
    basis = _basis_from_text(text)
    announcement_date = _date_from_text(text)
    source_values = _row_source_values(row)
    quarterly_history = _quarterly_history_from_latest_result(row)
    if quarterly_history:
        source_values.update(_latest_quarter_values(quarterly_history))
    top_ratios = _top_ratios_from_result_text(text)
    filing_id = _filing_id(company_url, quarter, announcement_date, pdf_url, text)
    return QuarterlyResultRecord(
        filing_id=filing_id,
        company_name=company_name,
        symbol=symbol,
        screener_company_url=company_url,
        source_url=source_url,
        reporting_quarter=quarter,
        announcement_date=announcement_date,
        reporting_basis=basis,
        collection_timestamp=fetched_at,
        status="discovered",
        source_values=source_values,
        normalized_values={key: parse_number(value) for key, value in source_values.items()},
        top_ratios=top_ratios,
        quarterly_history=quarterly_history,
        historical_unavailable=len(quarterly_history) < 4,
        pdf_url=pdf_url,
    )


def _quarterly_history_from_latest_result(row: Tag) -> list[dict[str, str]]:
    """Parse compact Screener latest-results financial table from one result card."""

    table = row.find("table")
    if not isinstance(table, Tag):
        return []
    table_rows = table.find_all("tr")
    if len(table_rows) < 2:
        return []
    headers = [
        _clean(cell.get_text(" ", strip=True))
        for cell in table_rows[0].find_all(["th", "td"])
        if _clean(cell.get_text(" ", strip=True))
    ]
    periods = [header for header in headers if re.search(r"\b(?:Mar|Jun|Sep|Dec)\s+20\d{2}\b", header, re.I)]
    if not periods:
        return []
    output: list[dict[str, str]] = []
    for metric_row in table_rows[1:]:
        cells = [
            _clean(cell.get_text(" ", strip=True))
            for cell in metric_row.find_all(["th", "td"])
        ]
        cells = [cell for cell in cells if cell]
        if len(cells) < 2:
            continue
        metric = cells[0]
        values = cells[2:] if len(cells) >= len(periods) + 2 else cells[1:]
        if len(values) < len(periods):
            values = [*values, *[""] * (len(periods) - len(values))]
        record: dict[str, str] = {"": _normalize_metric_label(metric)}
        # Store periods oldest-to-newest so analysis offset 0 is latest.
        for period, value in zip(reversed(periods), reversed(values[: len(periods)]), strict=False):
            record[period] = value
        if len(cells) >= 2 and not re.search(r"\b(?:Mar|Jun|Sep|Dec)\s+20\d{2}\b", cells[1], re.I):
            record["YoY"] = cells[1]
        output.append(record)
    return output


def _normalize_metric_label(metric: str) -> str:
    """Normalize compact Screener result-card metric labels."""

    normalized = metric.strip()
    if normalized.casefold() in {"ebidt", "ebitda"}:
        return "Operating Profit"
    return normalized


def _text_result_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Return start/end indexes for copied latest-result text blocks."""

    starts: list[int] = []
    for index, line in enumerate(lines):
        if line in {"PDF", "YOY", "Sales", "EBIDT", "EBITDA", "Operating Profit", "Net profit", "EPS"}:
            continue
        if "PDF" not in line:
            continue
        window = " ".join(lines[index + 1 : index + 16])
        if "YOY" not in window:
            continue
        if not re.search(r"\b(?:Price|M\.Cap|PE)\b", window, re.I):
            continue
        if re.search(r"\b(?:Mar|Jun|Sep|Dec)\s+20\d{2}\b", window, re.I):
            starts.append(index)
    blocks: list[tuple[int, int]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        blocks.append((start, end))
    return blocks


def _record_from_text_block(
    block: list[str],
    *,
    source_url: str,
    fetched_at: datetime,
) -> QuarterlyResultRecord | None:
    """Parse one copied visible latest-result block into a record."""

    if len(block) < 8:
        return None
    company_name = _clean(re.sub(r"\s+PDF\s*$", "", block[0], flags=re.I))
    if not company_name or company_name.casefold() in {"pdf", "yoy"}:
        return None
    text = _clean(" ".join(block))
    quarter = _quarter_from_text(text)
    quarterly_history = _quarterly_history_from_text_block(block)
    if not quarterly_history:
        return None
    source_values = {"Copied Page Text": text[:500]}
    source_values.update(_latest_quarter_values(quarterly_history))
    top_ratios = _top_ratios_from_result_text(text)
    basis = _basis_from_text(text)
    if basis is None and "* standalone numbers" in text.casefold():
        basis = "Standalone"
    symbol = _symbol_from_text_company_name(company_name)
    company_url = (
        urljoin(source_url, f"/company/{symbol}/")
        if symbol
        else f"{source_url}#company={_slugify(company_name)}"
    )
    filing_id = _filing_id(company_url, quarter, None, None, text)
    return QuarterlyResultRecord(
        filing_id=filing_id,
        company_name=company_name,
        symbol=symbol,
        screener_company_url=company_url,
        source_url=source_url,
        reporting_quarter=quarter,
        reporting_basis=basis,
        collection_timestamp=fetched_at,
        status="discovered",
        source_values=source_values,
        normalized_values={key: parse_number(value) for key, value in source_values.items()},
        top_ratios=top_ratios,
        quarterly_history=quarterly_history,
        historical_unavailable=len(quarterly_history) < 4,
        validation_warnings=[
            "Imported from copied Screener page text; company URL/PDF evidence requires optional enrichment."
        ],
    )


def _quarterly_history_from_text_block(block: list[str]) -> list[dict[str, str]]:
    """Parse compact latest-result values from copied visible text."""

    try:
        yoy_index = next(index for index, line in enumerate(block) if line.casefold() == "yoy")
    except StopIteration:
        return []
    periods: list[str] = []
    cursor = yoy_index + 1
    while cursor < len(block) and len(periods) < 4:
        line = block[cursor]
        if re.fullmatch(r"(?:Mar|Jun|Sep|Dec)\s+20\d{2}", line, re.I):
            periods.append(line.title())
            cursor += 1
            continue
        break
    if not periods:
        return []

    output: list[dict[str, str]] = []
    metrics = {"sales", "ebidt", "ebitda", "operating profit", "net profit", "eps"}
    index = cursor
    while index < len(block):
        metric = block[index]
        if metric.casefold() not in metrics:
            index += 1
            continue
        raw_values: list[str] = []
        value_index = index + 1
        while value_index < len(block) and len(raw_values) < len(periods) + 1:
            value = block[value_index]
            if value.casefold() in metrics:
                break
            if _looks_metric_value(value):
                raw_values.append(value)
            value_index += 1
        if len(raw_values) >= len(periods):
            yoy_value = raw_values[0] if len(raw_values) > len(periods) else ""
            period_values = raw_values[-len(periods) :]
            record: dict[str, str] = {"": _normalize_metric_label(metric)}
            for period, value in zip(reversed(periods), reversed(period_values), strict=False):
                record[period] = value
            if yoy_value:
                record["YoY"] = yoy_value
            output.append(record)
        index = max(value_index, index + 1)
    return output


def _copied_text_lines(page_text: str) -> list[str]:
    """Normalize copied Screener text into one logical token per line.

    Browser copy often returns table rows as tab-separated cells. Splitting those
    cells makes the text parser behave like the accessibility tree and saved HTML
    parser.
    """

    lines: list[str] = []
    for raw_line in page_text.splitlines():
        if "\t" in raw_line:
            parts = [_clean(part) for part in raw_line.split("\t")]
            lines.extend(part for part in parts if part)
            continue
        cleaned = _clean(raw_line)
        if cleaned:
            lines.append(cleaned)
    return lines


def _looks_metric_value(value: str) -> bool:
    """Return whether a copied text line looks like a result-table value."""

    cleaned = value.replace(",", "").replace("₹", "").replace("%", "").strip()
    cleaned = cleaned.replace("⇡", "").replace("⇣", "").replace("+", "").strip()
    return bool(re.search(r"^-?\d+(?:\.\d+)?$", cleaned))


def _symbol_from_text_company_name(company_name: str) -> str | None:
    """Conservatively infer a Screener-style symbol from a copied company name."""

    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", company_name).strip()
    if not cleaned:
        return None
    token = "".join(part.upper() for part in cleaned.split())
    return token[:24] if token else None


def _slugify(value: str) -> str:
    """Create a stable local anchor slug."""

    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "unknown"


def _top_ratios_from_result_text(text: str) -> dict[str, str]:
    """Extract visible price/market-cap/PE values from result-card text."""

    ratios: dict[str, str] = {}
    market_cap = re.search(r"M\.Cap\s+₹\s+([0-9,.]+)\s+Cr", text, re.I)
    pe = re.search(r"\bPE\s+([0-9,.]+)", text, re.I)
    price = re.search(r"Price\s+₹\s+([0-9,.]+)", text, re.I)
    if market_cap:
        ratios["Market Cap"] = f"₹ {market_cap.group(1)} Cr"
    if pe:
        ratios["Stock P/E"] = pe.group(1)
    if price:
        ratios["Current Price"] = f"₹ {price.group(1)}"
    return ratios


def _row_source_values(row: Tag) -> dict[str, str]:
    """Preserve original labels/values visible in the latest-results row."""

    values: dict[str, str] = {}
    headers = []
    table = row.find_parent("table")
    if isinstance(table, Tag):
        first_row = table.find("tr")
        if isinstance(first_row, Tag):
            headers = [_clean(cell.get_text(" ", strip=True)) for cell in first_row.find_all(["th", "td"])]
    cells = [_clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
    for index, cell in enumerate(cells):
        label = headers[index] if index < len(headers) and headers[index] else f"Column {index + 1}"
        if cell:
            values[label] = cell
    if not values:
        values["Row Text"] = _clean(row.get_text(" ", strip=True))
    return values


def _latest_quarter_values(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return original latest-quarter row-label values."""

    values: dict[str, str] = {}
    if not rows:
        return values
    period_keys = [
        key
        for key in list(rows[0])[1:]
        if re.search(r"\b(?:Mar|Jun|Sep|Dec)\s+20\d{2}\b", key, re.I)
    ]
    if not period_keys:
        return values
    latest_period = period_keys[-1]
    for row in rows:
        label = _first_value(row)
        if label and row.get(latest_period):
            values[label] = row[latest_period]
    return values


def _first_value(row: dict[str, str]) -> str:
    """Return first non-empty row value."""

    for value in row.values():
        cleaned = _clean(value)
        if cleaned:
            return cleaned.rstrip("+").strip()
    return ""


def _pdf_link(row: Tag, *, source_url: str) -> str | None:
    """Return attached PDF URL when visible."""

    for anchor in row.find_all("a", href=True):
        href = str(anchor.get("href") or "")
        text = _clean(anchor.get_text(" ", strip=True)).casefold()
        if ".pdf" in href.casefold() or "pdf" in text or "result" in text:
            return urljoin(source_url, href)
    return None


def _symbol_from_company_url(url: str) -> str | None:
    """Extract Screener symbol from company URL."""

    match = re.search(r"/company/([^/]+)/?", url)
    if not match:
        return None
    symbol = match.group(1).strip().upper()
    if symbol in {"CONSOLIDATED", ""}:
        return None
    return symbol


def _quarter_from_text(text: str) -> str | None:
    """Extract reporting quarter/month label."""

    match = re.search(r"\b(?:Mar|Jun|Sep|Dec)\s+20\d{2}\b", text, re.I)
    return match.group(0).title() if match else None


def _date_from_text(text: str) -> str | None:
    """Extract announcement date-ish text."""

    match = re.search(r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+20\d{2}\b", text, re.I)
    return match.group(0).title() if match else None


def _basis_from_text(text: str) -> str | None:
    """Extract reporting basis."""

    lowered = text.casefold()
    if "consolidated" in lowered:
        return "Consolidated"
    if "standalone" in lowered:
        return "Standalone"
    return None


def _filing_id(
    company_url: str,
    quarter: str | None,
    announcement_date: str | None,
    pdf_url: str | None,
    text: str,
) -> str:
    """Return stable filing id."""

    raw = "|".join([company_url, quarter or "", announcement_date or "", pdf_url or "", text[:120]])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    symbol = _symbol_from_company_url(company_url) or "UNKNOWN"
    return f"{symbol}_{quarter or 'UNKNOWN'}_{digest}".replace(" ", "_")


def _url_with_page(url: str, page: int) -> str:
    """Return URL with page query parameter."""

    if page <= 1:
        return url
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query))
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _looks_access_blocked(html: str) -> bool:
    """Return whether Screener redirected to login/register or blocked access."""

    lowered = html.casefold()
    return (
        "get a free account" in lowered
        or "/login/" in lowered
        or "/register/" in lowered
        or "already registered" in lowered
    )


def _deduplicate(records: list[QuarterlyResultRecord]) -> list[QuarterlyResultRecord]:
    """Deduplicate records by filing id."""

    seen: set[str] = set()
    output: list[QuarterlyResultRecord] = []
    for record in records:
        if record.filing_id in seen:
            continue
        seen.add(record.filing_id)
        output.append(record)
    return output


def _clean(value: Any) -> str:
    """Normalize display text."""

    return " ".join(str(value).replace("\xa0", " ").split())
