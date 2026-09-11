"""Shareholding-pattern scanner for Buying Agent."""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

import requests
from bs4 import BeautifulSoup, Tag

from app.providers.fundamentals import ScreenerFundamentalReport

TIJORI_CONNECT_TIMEOUT_SECONDS = 4
TIJORI_READ_TIMEOUT_SECONDS = 8
TIJORI_BASE_URLS = (
    "https://www.tijorifinance.com/company/{slug}/shareholding/",
    "https://new-platform.tijorifinance.com/company/{slug}/shareholding/",
)
TIJORI_CACHE_TTL_SECONDS = 60 * 60 * 6
_TIJORI_REPORT_CACHE: dict[tuple[str, str], tuple[float, ScreenerFundamentalReport]] = {}
_TIJORI_CACHE_LOCK = threading.Lock()
_SHAREHOLDING_SCAN_JOBS: dict[str, "ShareholdingScanJob"] = {}
_SHAREHOLDING_JOB_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ShareholdingScanResult:
    """Institutional shareholding score for one stock."""

    symbol: str
    company: str
    institutional_score: float
    latest_period: str | None
    fii_latest: float | None
    dii_latest: float | None
    promoter_latest: float | None
    public_latest: float | None
    pledge_latest: float | None
    reasons: tuple[str, ...]
    buying_rule_status: str = "NOT RECOMMENDED"
    buying_rule_reasons: tuple[str, ...] = ()
    missing_data: tuple[str, ...] = ()
    error: str | None = None

    @property
    def combined_institutional_latest(self) -> float | None:
        """Return latest FII+DII holding when both are available."""

        if self.fii_latest is None or self.dii_latest is None:
            return None
        return self.fii_latest + self.dii_latest

    @property
    def status(self) -> str:
        """Return shareholding qualification status."""

        if self.error:
            return "DATA UNAVAILABLE"
        if self.institutional_score >= 70:
            return "QUALIFIED"
        return "NOT QUALIFIED"


@dataclass(slots=True)
class ShareholdingScanJob:
    """Mutable background shareholding scan job."""

    job_id: str
    total: int
    completed: int
    results: list[ShareholdingScanResult]
    errors: list[ShareholdingScanResult]
    status: str
    message: str
    started_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ShareholdingScanSnapshot:
    """Read-only shareholding scan job snapshot for the dashboard."""

    job_id: str
    total: int
    completed: int
    results: tuple[ShareholdingScanResult, ...]
    errors: tuple[ShareholdingScanResult, ...]
    status: str
    message: str
    started_at: datetime
    updated_at: datetime

    @property
    def progress(self) -> float:
        """Return completed fraction from 0 to 1."""

        if self.total <= 0:
            return 1.0
        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def is_running(self) -> bool:
        """Return whether the background job is still active."""

        return self.status == "running"


async def scan_shareholding_patterns(
    *,
    symbols: list[tuple[str, str]],
    min_score: float = 70.0,
    max_concurrency: int = 4,
    fetch_report: Callable[..., ScreenerFundamentalReport] | None = None,
) -> tuple[list[ShareholdingScanResult], list[ShareholdingScanResult]]:
    """Scan symbols for institutional shareholding accumulation.

    Args:
        symbols: Symbol and company-name pairs.
        min_score: Deprecated; filtering is applied by the dashboard after all rows are scored.
        max_concurrency: Maximum parallel Tijori Finance fetches.
        fetch_report: Optional cached report fetcher. It may accept either symbol or
            symbol and company name.

    Returns:
        Scored results and skipped/error results.
    """

    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def scan_one(symbol: str, company: str) -> ShareholdingScanResult:
        async with semaphore:
            try:
                report_fetcher = fetch_report or fetch_tijori_shareholding_report
                report = await asyncio.to_thread(
                    _fetch_report_for_symbol,
                    report_fetcher,
                    symbol,
                    company,
                )
            except Exception as exc:
                return _shareholding_error(symbol, company, str(exc))
            return score_shareholding_report(report, company=company)

    results = await asyncio.gather(*(scan_one(symbol, company) for symbol, company in symbols))
    errors = [result for result in results if result.error]
    matches = list(results)
    matches.sort(key=lambda item: (item.institutional_score, item.symbol), reverse=True)
    return matches, errors


def start_shareholding_scan_job(
    *,
    symbols: list[tuple[str, str]],
    max_concurrency: int = 8,
    batch_size: int = 25,
    fetch_report: Callable[..., ScreenerFundamentalReport] | None = None,
) -> str:
    """Start a background Shareholding Pattern scan.

    The job is independent from Streamlit's page reruns, so opening another browser tab
    will not cancel the scan.
    """

    now = datetime.now(UTC)
    job_id = uuid4().hex
    job = ShareholdingScanJob(
        job_id=job_id,
        total=len(symbols),
        completed=0,
        results=[],
        errors=[],
        status="running",
        message="Starting Tijori Finance scan...",
        started_at=now,
        updated_at=now,
    )
    with _SHAREHOLDING_JOB_LOCK:
        _SHAREHOLDING_SCAN_JOBS[job_id] = job

    thread = threading.Thread(
        target=_run_shareholding_scan_job,
        args=(
            job_id,
            list(symbols),
            max(1, max_concurrency),
            max(1, batch_size),
            fetch_report or fetch_cached_tijori_shareholding_report,
        ),
        daemon=True,
    )
    thread.start()
    return job_id


def get_shareholding_scan_job(job_id: str) -> ShareholdingScanSnapshot | None:
    """Return a read-only snapshot for a background shareholding scan job."""

    with _SHAREHOLDING_JOB_LOCK:
        job = _SHAREHOLDING_SCAN_JOBS.get(job_id)
        if job is None:
            return None
        return ShareholdingScanSnapshot(
            job_id=job.job_id,
            total=job.total,
            completed=job.completed,
            results=tuple(job.results),
            errors=tuple(job.errors),
            status=job.status,
            message=job.message,
            started_at=job.started_at,
            updated_at=job.updated_at,
        )


def _run_shareholding_scan_job(
    job_id: str,
    symbols: list[tuple[str, str]],
    max_concurrency: int,
    batch_size: int,
    fetch_report: Callable[..., ScreenerFundamentalReport],
) -> None:
    """Run a Shareholding Pattern scan outside Streamlit's render lifecycle."""

    try:
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            batch_results, batch_errors = asyncio.run(
                scan_shareholding_patterns(
                    symbols=batch,
                    max_concurrency=max_concurrency,
                    fetch_report=fetch_report,
                )
            )
            completed = min(start + len(batch), len(symbols))
            with _SHAREHOLDING_JOB_LOCK:
                job = _SHAREHOLDING_SCAN_JOBS[job_id]
                job.results.extend(batch_results)
                job.results.sort(
                    key=lambda item: (item.institutional_score, item.symbol),
                    reverse=True,
                )
                job.errors.extend(batch_errors)
                job.completed = completed
                job.message = f"Scanned {completed}/{len(symbols)} stocks."
                job.updated_at = datetime.now(UTC)
        with _SHAREHOLDING_JOB_LOCK:
            job = _SHAREHOLDING_SCAN_JOBS[job_id]
            job.status = "completed"
            job.message = f"Completed {job.completed}/{job.total} stocks."
            job.updated_at = datetime.now(UTC)
    except Exception as exc:
        with _SHAREHOLDING_JOB_LOCK:
            job = _SHAREHOLDING_SCAN_JOBS[job_id]
            job.status = "failed"
            job.message = str(exc)
            job.updated_at = datetime.now(UTC)


def score_shareholding_report(
    report: ScreenerFundamentalReport,
    *,
    company: str | None = None,
) -> ShareholdingScanResult:
    """Score a shareholding report.

    Args:
        report: Parsed shareholding report.
        company: Optional company display name.

    Returns:
        Institutional shareholding score result.
    """

    rows = report.tables.get("Shareholding", [])
    if not rows:
        return _shareholding_error(
            report.symbol,
            company or report.company_name,
            "Shareholding table was not visible.",
        )

    periods = _period_columns(rows)
    reasons: list[str] = []
    missing: list[str] = []
    score = 0.0

    fii = _series_for(rows, ("FII", "FIIs", "FPI", "Foreign"))
    dii = _series_for(rows, ("DIIs", "DII +", "DII Holding"))
    if not dii:
        dii = _sum_series_for(rows, ("Mutual Funds", "Insurance Companies", "DII-Others"))
    promoter = _series_for(rows, ("Promoter", "Promoters"))
    public = _series_for(
        rows,
        ("Public Shareholding", "Public", "Retail", "Individual <", "Non-Institutions"),
    )
    pledge = _series_for(rows, ("Pledge", "Pledged"))
    latest_period = periods[-1] if periods else None

    fii_latest = _latest(fii)
    dii_latest = _latest(dii)
    promoter_latest = _latest(promoter)
    public_latest = _latest(public)
    pledge_latest = _latest(pledge)

    if _qoq_up(fii):
        score += 12
        reasons.append("FII/FPI holding increased versus previous quarter.")
    else:
        missing_or_reason(fii, missing, reasons, "FII/FPI holding did not increase QoQ.")
    if _consecutive_up(fii):
        score += 8
        reasons.append("FII/FPI holding increased for 2 consecutive quarters.")

    if _qoq_up(dii):
        score += 12
        reasons.append("DII/MF holding increased versus previous quarter.")
    else:
        missing_or_reason(dii, missing, reasons, "DII/MF holding did not increase QoQ.")
    if _consecutive_up(dii):
        score += 8
        reasons.append("DII/MF holding increased for 2 consecutive quarters.")

    combined = _combine_series(fii, dii)
    if _qoq_up(combined):
        score += 15
        reasons.append("FII+DII combined holding increased QoQ, a strong institutional signal.")
    if _two_quarter_decline(combined):
        score -= 20
        reasons.append("Flag: FII+DII holding declined for 2 consecutive quarters.")

    if _qoq_non_down(promoter):
        score += 10
        reasons.append("Promoter holding is stable or increasing.")
    else:
        missing_or_reason(promoter, missing, reasons, "Promoter holding declined QoQ.")
    promoter_yoy_drop = _yoy_drop(promoter)
    if promoter_yoy_drop is not None and promoter_yoy_drop > 2:
        score -= 15
        reasons.append(
            f"Penalty: promoter holding declined {promoter_yoy_drop:.1f}% YoY; verify QIP/dilution."
        )

    if pledge_latest is None:
        missing.append("Promoter pledge data")
    elif pledge_latest == 0:
        score += 8
        reasons.append("Promoter pledge is 0%.")
    elif pledge_latest > 10 or _qoq_up(pledge):
        score -= 20
        reasons.append("Reject/penalty: promoter pledge is rising or above 10%.")
    else:
        reasons.append("Promoter pledge exists but is not above 10% or rising.")

    if _qoq_up(combined) and _qoq_down(public):
        score += 12
        reasons.append("Institutional ownership increased while public/retail holding decreased.")
    else:
        missing_or_reason(public, missing, reasons, "Public/retail holding did not decrease QoQ.")

    missing.append("Number of institutional investors")
    reasons.append(
        "Increasing number of institutional investors needs detailed holder-count data; "
        "category percentages alone cannot confirm breadth."
    )

    score = round(max(0.0, min(100.0, score)), 1)
    buying_rule_status, buying_rule_reasons = _shareholding_buying_rule_check(
        institutional_score=score,
        error=None,
        fii=fii,
        dii=dii,
        promoter=promoter,
        pledge=pledge,
        combined=combined,
        missing_data=missing,
        reasons=reasons,
    )
    return ShareholdingScanResult(
        symbol=report.symbol,
        company=company or report.company_name,
        institutional_score=score,
        latest_period=latest_period,
        fii_latest=fii_latest,
        dii_latest=dii_latest,
        promoter_latest=promoter_latest,
        public_latest=public_latest,
        pledge_latest=pledge_latest,
        reasons=tuple(reasons),
        buying_rule_status=buying_rule_status,
        buying_rule_reasons=buying_rule_reasons,
        missing_data=tuple(sorted(set(missing))),
    )


def missing_or_reason(
    values: list[float | None],
    missing: list[str],
    reasons: list[str],
    reason: str,
) -> None:
    """Append missing-data or negative reason for a series."""

    if len([value for value in values if value is not None]) < 2:
        missing.append(reason.removesuffix("."))
    else:
        reasons.append(reason)


def _shareholding_error(symbol: str, company: str, error: str) -> ShareholdingScanResult:
    """Return a shareholding scanner error."""

    return ShareholdingScanResult(
        symbol=symbol,
        company=company,
        institutional_score=0.0,
        latest_period=None,
        fii_latest=None,
        dii_latest=None,
        promoter_latest=None,
        public_latest=None,
        pledge_latest=None,
        reasons=(error,),
        buying_rule_status="DATA UNAVAILABLE",
        buying_rule_reasons=(
            "Shareholding data was unavailable, so buying rules cannot be checked.",
        ),
        missing_data=("Shareholding table",),
        error=error,
    )


def _shareholding_buying_rule_check(
    *,
    institutional_score: float,
    error: str | None,
    fii: list[float | None],
    dii: list[float | None],
    promoter: list[float | None],
    pledge: list[float | None],
    combined: list[float | None],
    missing_data: list[str],
    reasons: list[str],
) -> tuple[str, tuple[str, ...]]:
    """Apply buying-agent gate checks available from shareholding data."""

    if error:
        return (
            "DATA UNAVAILABLE",
            ("Shareholding data was unavailable, so buying rules cannot be checked.",),
        )

    failures: list[str] = []
    positives: list[str] = []
    if institutional_score < 70:
        failures.append("Institutional Score is below the buying-agent threshold of 70.")
    else:
        positives.append("Institutional Score is above the buying-agent threshold of 70.")

    if _qoq_up(combined):
        positives.append("FII+DII combined holding increased QoQ.")
    else:
        failures.append("FII+DII combined holding is not increasing QoQ.")

    if _two_quarter_decline(combined):
        failures.append("FII+DII holding declined for 2 consecutive quarters.")

    if _qoq_non_down(promoter):
        positives.append("Promoter holding is stable or increasing.")
    else:
        failures.append("Promoter holding is not stable/increasing.")

    pledge_latest = _latest(pledge)
    if pledge_latest is None:
        failures.append("Promoter pledge data is missing.")
    elif pledge_latest > 10 or _qoq_up(pledge):
        failures.append("Promoter pledge is rising or above 10%.")
    else:
        positives.append("Promoter pledge rule is acceptable.")

    if missing_data:
        positives.append("Full buying-agent Stage 2 still needs price, liquidity and fundamentals.")

    if failures:
        return "NOT RECOMMENDED", tuple(failures[:4])
    if institutional_score >= 80 and _consecutive_up(fii) and _consecutive_up(dii):
        return "STRONG STAGE-2", tuple(positives[:4])
    return "STAGE-2 READY", tuple(positives[:4] or reasons[:4])


def fetch_tijori_shareholding_report(symbol: str, company: str) -> ScreenerFundamentalReport:
    """Fetch and parse Tijori Finance shareholding tables.

    Args:
        symbol: NSE symbol.
        company: Company display name used for slug discovery.

    Returns:
        A typed report compatible with the shareholding scorer.
    """

    errors: list[str] = []
    for url_template in TIJORI_BASE_URLS:
        for slug in _tijori_slug_candidates(company):
            url = url_template.format(slug=slug)
            try:
                response = _request_tijori_page(url)
                response.raise_for_status()
                report = parse_tijori_shareholding_page(
                    response.text,
                    symbol=symbol,
                    company=company,
                    source_url=response.url,
                    fetched_at=datetime.now(UTC),
                )
                if report.tables.get("Shareholding"):
                    return report
            except Exception as exc:
                errors.append(f"{url}: {exc}")
    raise RuntimeError("No Tijori shareholding table found. " + " | ".join(errors[:3]))


def fetch_cached_tijori_shareholding_report(
    symbol: str,
    company: str,
) -> ScreenerFundamentalReport:
    """Fetch a Tijori report using a thread-safe in-process cache."""

    key = (symbol.strip().upper().replace(".NS", ""), company.strip().casefold())
    now = time.monotonic()
    with _TIJORI_CACHE_LOCK:
        cached = _TIJORI_REPORT_CACHE.get(key)
        if cached and now - cached[0] <= TIJORI_CACHE_TTL_SECONDS:
            return cached[1]

    report = fetch_tijori_shareholding_report(symbol, company)
    with _TIJORI_CACHE_LOCK:
        _TIJORI_REPORT_CACHE[key] = (now, report)
    return report


def _request_tijori_page(url: str) -> requests.Response:
    """Request a Tijori page and skip dashboard redirects quickly."""

    headers = {"User-Agent": "Mozilla/5.0 NATIP local shareholding research"}
    timeout = (TIJORI_CONNECT_TIMEOUT_SECONDS, TIJORI_READ_TIMEOUT_SECONDS)
    response = requests.get(
        url,
        timeout=timeout,
        headers=headers,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        location = response.headers.get("Location", "")
        if "dashboard" in location.casefold():
            raise RuntimeError("Tijori redirected this slug to dashboard.")
        redirect_url = urljoin(url, location)
        response = requests.get(
            redirect_url,
            timeout=timeout,
            headers=headers,
            allow_redirects=False,
        )
    if "dashboard" in response.url.casefold():
        raise RuntimeError("Tijori redirected this slug to dashboard.")
    return response


def parse_tijori_shareholding_page(
    html: str,
    *,
    symbol: str,
    company: str,
    source_url: str,
    fetched_at: datetime,
) -> ScreenerFundamentalReport:
    """Parse a Tijori shareholding page into the common fundamentals report shape."""

    soup = BeautifulSoup(html, "html.parser")
    shareholding_rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        if not isinstance(table, Tag):
            continue
        rows = _html_table_rows(table)
        if _looks_like_shareholding_rows(rows):
            shareholding_rows = rows
            break
    trend_rows = _tijori_trend_rows(html)
    if trend_rows:
        shareholding_rows = trend_rows + shareholding_rows
    if not shareholding_rows:
        raise RuntimeError("Tijori page did not expose a readable shareholding table.")

    shareholding_latest = _latest_shareholding_from_rows(shareholding_rows)
    return ScreenerFundamentalReport(
        symbol=symbol.strip().upper().replace(".NS", ""),
        company_name=company,
        source_url=source_url,
        fetched_at=fetched_at,
        shareholding_latest=shareholding_latest,
        tables={"Shareholding": shareholding_rows},
        missing_data=[
            "Tijori Finance provides shareholding percentages only; fundamentals tables are not included."
        ],
    )


def _fetch_report_for_symbol(
    report_fetcher: Callable[..., ScreenerFundamentalReport],
    symbol: str,
    company: str,
) -> ScreenerFundamentalReport:
    """Call a report fetcher that may accept symbol or symbol and company."""

    try:
        return report_fetcher(symbol, company)
    except TypeError as exc:
        try:
            return report_fetcher(symbol)
        except TypeError:
            raise exc


def _tijori_slug_candidates(company: str) -> tuple[str, ...]:
    """Return likely Tijori company slugs for a display name."""

    base = company.casefold()
    replacements = {
        "&": " and ",
        ".": " ",
        ",": " ",
        "'": " ",
        "(": " ",
        ")": " ",
    }
    for source, target in replacements.items():
        base = base.replace(source, target)
    base = re.sub(r"\bco\b", "company", base)
    base = re.sub(r"\bcorp\b", "corporation", base)
    base = re.sub(r"[^a-z0-9]+", " ", base).strip()
    variants: list[str] = []
    if re.search(r"\bltd\b$", base):
        variants.extend([re.sub(r"\bltd\b$", "limited", base), base])
    elif re.search(r"\blimited\b$", base):
        variants.extend([base, re.sub(r"\blimited\b$", "ltd", base)])
    else:
        variants.extend([f"{base} limited", base, f"{base} ltd"])

    slugs: list[str] = []
    for variant in variants:
        slug = re.sub(r"\s+", "-", variant.strip("- "))
        if slug and slug not in slugs:
            slugs.append(slug)
    return tuple(slugs)


def _html_table_rows(table: Tag) -> list[dict[str, str]]:
    """Convert an HTML table to row dictionaries."""

    parsed_rows: list[list[str]] = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):
            continue
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["th", "td"])
            if isinstance(cell, Tag)
        ]
        if cells:
            parsed_rows.append(cells)
    if len(parsed_rows) < 2:
        return []

    headers = parsed_rows[0]
    normalized_headers = [header or "" for header in headers]
    rows: list[dict[str, str]] = []
    for raw_cells in parsed_rows[1:]:
        values = raw_cells[: len(normalized_headers)]
        if len(values) < len(normalized_headers):
            values.extend([""] * (len(normalized_headers) - len(values)))
        row_dict = dict(zip(normalized_headers, values, strict=False))
        if any(value.strip() for value in row_dict.values()):
            rows.append(row_dict)
    return rows


def _looks_like_shareholding_rows(rows: list[dict[str, str]]) -> bool:
    """Return whether rows contain recognizable shareholding categories."""

    labels = [next(iter(row.values()), "").casefold() for row in rows]
    joined = " | ".join(labels)
    return "promoter" in joined and ("fii" in joined or "mutual fund" in joined)


def _tijori_trend_rows(html: str) -> list[dict[str, str]]:
    """Parse Tijori aggregate trendData JavaScript rows when available."""

    match = re.search(r"var\s+trendData\s*=\s*(\[.*?\]);", html, flags=re.DOTALL)
    if not match:
        return []
    trend_block = match.group(1)
    rows: list[dict[str, str]] = []
    for series_match in re.finditer(
        r"\{'name':\s*'([^']+)'\s*,\s*'data':\s*\[(.*?)\]\}",
        trend_block,
        flags=re.DOTALL,
    ):
        label = series_match.group(1).strip()
        points = re.findall(
            r"\[(\d{10,})\s*,\s*([-+]?\d+(?:\.\d+)?)\]",
            series_match.group(2),
        )
        if not points:
            continue
        row: dict[str, str] = {"in %": label}
        for timestamp_ms, value in points[-12:]:
            period = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=UTC).strftime("%b'%y")
            row[period] = value
        rows.append(row)
    return rows


def _latest_shareholding_from_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return latest shareholding values from parsed table rows."""

    latest: dict[str, str] = {}
    for row in rows:
        holder = next(iter(row.values()), "").strip()
        if not holder:
            continue
        for column in reversed(list(row.keys())[1:]):
            value = str(row.get(column, "")).strip()
            if value and value not in {"-", "--", "—", "NA", "N/A"}:
                latest[holder.rstrip("+").strip()] = value
                break
    return latest


def _period_columns(rows: list[dict[str, str]]) -> list[str]:
    """Return period columns from a shareholding table."""

    if not rows:
        return []
    return list(rows[0].keys())[1:]


def _series_for(rows: list[dict[str, str]], labels: tuple[str, ...]) -> list[float | None]:
    """Return row numeric series for a matching holder label."""

    for label in labels:
        normalized_label = label.casefold()
        for row in rows:
            first_value = next(iter(row.values()), "")
            normalized = first_value.casefold()
            if normalized_label in normalized:
                return [_parse_percent(row[column]) for column in list(row.keys())[1:]]
    return []


def _sum_series_for(rows: list[dict[str, str]], labels: tuple[str, ...]) -> list[float | None]:
    """Return summed numeric series across matching holder labels."""

    matched = [
        [_parse_percent(row[column]) for column in list(row.keys())[1:]]
        for row in rows
        if any(
            label.casefold() == next(iter(row.values()), "").casefold().strip() for label in labels
        )
    ]
    if not matched:
        return []
    length = max(len(series) for series in matched)
    totals: list[float | None] = []
    for index in range(length):
        values = [series[index] for series in matched if index < len(series)]
        if not values or any(value is None for value in values):
            totals.append(None)
        else:
            totals.append(sum(value for value in values if value is not None))
    return totals


def _parse_percent(value: Any) -> float | None:
    """Parse a percentage-like table value."""

    cleaned = str(value).replace("%", "").replace(",", "").strip()
    if not cleaned or cleaned in {"-", "--", "—", "NA", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _latest(values: list[float | None]) -> float | None:
    """Return latest non-empty value."""

    for value in reversed(values):
        if value is not None:
            return value
    return None


def _qoq_up(values: list[float | None]) -> bool:
    """Return whether latest value is greater than previous value."""

    clean = [value for value in values if value is not None]
    return len(clean) >= 2 and clean[-1] > clean[-2]


def _qoq_down(values: list[float | None]) -> bool:
    """Return whether latest value is lower than previous value."""

    clean = [value for value in values if value is not None]
    return len(clean) >= 2 and clean[-1] < clean[-2]


def _qoq_non_down(values: list[float | None]) -> bool:
    """Return whether latest value is stable or increasing."""

    clean = [value for value in values if value is not None]
    return len(clean) >= 2 and clean[-1] >= clean[-2]


def _consecutive_up(values: list[float | None]) -> bool:
    """Return whether the latest two quarter moves are positive."""

    clean = [value for value in values if value is not None]
    return len(clean) >= 3 and clean[-1] > clean[-2] > clean[-3]


def _two_quarter_decline(values: list[float | None]) -> bool:
    """Return whether the latest two quarter moves are negative."""

    clean = [value for value in values if value is not None]
    return len(clean) >= 3 and clean[-1] < clean[-2] < clean[-3]


def _combine_series(
    first: list[float | None],
    second: list[float | None],
) -> list[float | None]:
    """Combine two percentage series by aligned position."""

    length = max(len(first), len(second))
    combined: list[float | None] = []
    for index in range(length):
        first_value = first[index] if index < len(first) else None
        second_value = second[index] if index < len(second) else None
        if first_value is None or second_value is None:
            combined.append(None)
        else:
            combined.append(first_value + second_value)
    return combined


def _yoy_drop(values: list[float | None]) -> float | None:
    """Return latest YoY drop using four-quarter lookback when available."""

    clean = [value for value in values if value is not None]
    if len(clean) < 5:
        return None
    return clean[-5] - clean[-1]
