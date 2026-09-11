"""Screener acquisition modes and HTML table parsing."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.fundamentals.config import (
    OUTPUT_DIR,
    RAW_HTML_DIR,
    RAW_TABLES,
    SCREENER_EXPORT_DIR,
    ScreenerIngestionConfig,
    ensure_fundamental_dirs,
)
from app.fundamentals.screener_exports import (
    RAW_COLUMNS,
    infer_financial_type,
    normalize_field_name,
    parse_numeric,
    parse_period_end,
    process_screener_exports,
)
from app.fundamentals.symbol_map import create_symbol_map, load_symbol_map

TEST_TICKERS = ("RELIANCE.NS", "TCS.NS", "INFY.NS", "LT.NS", "SUNPHARMA.NS")
TABLE_IDS = {
    "quarterly": ("quarters", "quarterly", "quarterly-results"),
    "annual_pnl": ("profit-loss", "profit", "profit-loss"),
    "balance_sheet": ("balance-sheet", "balance", "balance-sheet"),
    "cashflow": ("cash-flow", "cash", "cash-flow"),
    "ratios": ("ratios", "ratio", "ratios"),
}
ACQUISITION_TEST_OUTPUT = OUTPUT_DIR / "screener_acquisition_test.csv"
TABLE_PARSING_AUDIT_OUTPUT = OUTPUT_DIR / "screener_table_parsing_audit.csv"
ACQUISITION_REPORT_OUTPUT = OUTPUT_DIR / "screener_acquisition_report.txt"


def run_acquisition_workflow(
    *,
    permitted_http_mode: bool = False,
    tickers: tuple[str, ...] = TEST_TICKERS,
    limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Run local-export ingestion, optional HTTP acquisition, and reporting.

    Args:
        permitted_http_mode: Whether permitted Screener HTTP access may be attempted.
        tickers: Tickers to audit before scaling.
        limit: Optional max tickers for HTTP mode.

    Returns:
        Acquisition and table parsing audit frames.
    """

    ensure_fundamental_dirs()
    create_symbol_map(overwrite=False)
    exports = sorted(SCREENER_EXPORT_DIR.glob("*.xls*"))
    local_tables = _empty_raw_table_frames()
    local_exports_processed = 0
    if exports:
        local_tables = process_screener_exports()
        local_exports_processed = len(exports)

    acquisition_rows: list[dict[str, Any]] = []
    table_audit_rows: list[dict[str, Any]] = []
    http_tables = _empty_raw_table_frames()
    blocked_or_failed = 0
    accessible = 0

    if not exports and permitted_http_mode:
        result = acquire_permitted_http_tables(tickers=tickers, limit=limit)
        http_tables = result["tables"]
        acquisition_rows = result["acquisition_rows"]
        table_audit_rows = result["table_audit_rows"]
        accessible = int(sum(row["page_accessible"] for row in acquisition_rows))
        blocked_or_failed = len(acquisition_rows) - accessible
        _write_raw_tables(http_tables)
    else:
        acquisition_rows = _disabled_or_local_acquisition_rows(
            tickers=tickers,
            permitted_http_mode=permitted_http_mode,
            local_exports_processed=local_exports_processed,
        )

    acquisition = pd.DataFrame(acquisition_rows)
    table_audit = pd.DataFrame(table_audit_rows)
    if table_audit.empty:
        table_audit = pd.DataFrame(
            columns=[
                "ticker",
                "screener_symbol",
                "source_type",
                "table_type",
                "table_found",
                "raw_rows_extracted",
                "notes",
            ]
        )
    acquisition.to_csv(ACQUISITION_TEST_OUTPUT, index=False)
    table_audit.to_csv(TABLE_PARSING_AUDIT_OUTPUT, index=False)
    _write_acquisition_report(
        local_exports=len(exports),
        permitted_http_mode=permitted_http_mode,
        accessible=accessible,
        blocked_or_failed=blocked_or_failed,
        acquisition=acquisition,
    )
    return {
        "acquisition": acquisition,
        "table_audit": table_audit,
        "tables": http_tables or local_tables,
    }


def acquire_permitted_http_tables(
    *,
    tickers: tuple[str, ...] = TEST_TICKERS,
    limit: int | None = None,
    config: ScreenerIngestionConfig | None = None,
) -> dict[str, Any]:
    """Attempt permitted company-page access and parse required tables."""

    cfg = config or ScreenerIngestionConfig(permitted_http_mode=True)
    symbol_map = load_symbol_map()
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})
    rows_by_table: dict[str, list[dict[str, Any]]] = {key: [] for key in RAW_TABLES}
    acquisition_rows: list[dict[str, Any]] = []
    table_audit_rows: list[dict[str, Any]] = []
    selected = list(tickers[: limit or len(tickers)])

    for ticker in selected:
        mapped = _mapping_for_ticker(symbol_map, ticker)
        if mapped is None:
            acquisition_rows.append(
                _acquisition_row(ticker=ticker, notes="No unambiguous mapping.")
            )
            continue
        screener_symbol = str(mapped["screener_symbol"]).strip().upper()
        consolidated_url = f"https://www.screener.in/company/{screener_symbol}/consolidated/"
        normal_url = f"https://www.screener.in/company/{screener_symbol}/"
        fetched = _fetch_cached_or_http(
            session=session,
            symbol=screener_symbol,
            url=consolidated_url,
            statement_basis="CONSOLIDATED",
            delay_seconds=cfg.request_delay_seconds,
        )
        consolidated_accessible = fetched["accessible"]
        if not fetched["accessible"]:
            fetched = _fetch_cached_or_http(
                session=session,
                symbol=screener_symbol,
                url=normal_url,
                statement_basis="STANDALONE_OR_DEFAULT",
                delay_seconds=cfg.request_delay_seconds,
            )
        html = fetched.get("html") or ""
        parsed = parse_screener_company_html(
            html=html,
            mapping=mapped,
            source=str(fetched.get("source", "")),
            source_type=str(fetched.get("source_type", "SCREENER_HTTP_PAGE")),
            statement_basis=str(fetched.get("statement_basis", "UNKNOWN")),
        )
        for table_type, rows in parsed["rows_by_table"].items():
            rows_by_table[table_type].extend(rows)
        table_audit_rows.extend(parsed["table_audit_rows"])
        acquisition_rows.append(
            _acquisition_row(
                ticker=ticker,
                mapped=mapped,
                page_accessible=bool(fetched["accessible"]),
                consolidated_page=bool(consolidated_accessible),
                quarterly_table=parsed["table_presence"].get("quarterly", False),
                balance_sheet=parsed["table_presence"].get("balance_sheet", False),
                cashflow=parsed["table_presence"].get("cashflow", False),
                ratios=parsed["table_presence"].get("ratios", False),
                raw_rows_extracted=sum(len(rows) for rows in parsed["rows_by_table"].values()),
                notes=str(fetched.get("notes", "")),
            )
        )
    frames = _frames_from_rows(rows_by_table)
    return {
        "tables": frames,
        "acquisition_rows": acquisition_rows,
        "table_audit_rows": table_audit_rows,
    }


def parse_screener_company_html(
    *,
    html: str,
    mapping: pd.Series,
    source: str,
    source_type: str,
    statement_basis: str,
) -> dict[str, Any]:
    """Parse required Screener financial tables from one company page."""

    rows_by_table: dict[str, list[dict[str, Any]]] = {key: [] for key in RAW_TABLES}
    table_audit_rows: list[dict[str, Any]] = []
    table_presence: dict[str, bool] = {}
    soup = BeautifulSoup(html or "", "html.parser")
    timestamp = datetime.now(UTC).isoformat()
    for table_type in RAW_TABLES:
        table = _find_financial_table(soup, table_type)
        table_presence[table_type] = table is not None
        rows = (
            _parse_html_statement_table(
                table=table,
                table_type=table_type,
                mapping=mapping,
                statement_basis=statement_basis,
                source=source,
                source_type=source_type,
                timestamp=timestamp,
            )
            if table is not None
            else []
        )
        rows_by_table[table_type].extend(rows)
        table_audit_rows.append(
            {
                "ticker": mapping["nse_ticker"],
                "screener_symbol": mapping["screener_symbol"],
                "source_type": source_type,
                "table_type": table_type,
                "table_found": table is not None,
                "raw_rows_extracted": len(rows),
                "notes": "" if rows else "Required table missing or no mapped fields found.",
            }
        )
    return {
        "rows_by_table": rows_by_table,
        "table_audit_rows": table_audit_rows,
        "table_presence": table_presence,
    }


def _parse_html_statement_table(
    *,
    table: Any,
    table_type: str,
    mapping: pd.Series,
    statement_basis: str,
    source: str,
    source_type: str,
    timestamp: str,
) -> list[dict[str, Any]]:
    header_cells = table.find_all("th")
    period_labels = [cell.get_text(" ", strip=True) for cell in header_cells][1:]
    output: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        raw_field = cells[0].get_text(" ", strip=True)
        normalized = normalize_field_name(raw_field)
        if normalized is None:
            continue
        for index, cell in enumerate(cells[1:]):
            raw_label = period_labels[index] if index < len(period_labels) else ""
            period_end = parse_period_end(raw_label, table_type)
            if pd.isna(period_end):
                continue
            output.append(
                {
                    "ticker": mapping["nse_ticker"],
                    "screener_symbol": mapping["screener_symbol"],
                    "company_name": mapping["company_name"],
                    "company_financial_type": infer_financial_type(mapping.get("sector", "")),
                    "statement_basis": statement_basis,
                    "period_end": period_end,
                    "publication_date": pd.NaT,
                    "source_publication_date": pd.NaT,
                    "publication_date_quality": "MISSING",
                    "import_timestamp": timestamp,
                    "source": source,
                    "source_type": source_type,
                    "table_type": table_type,
                    "raw_field_name": raw_field,
                    "normalized_field_name": normalized,
                    "raw_period_label": raw_label,
                    "raw_value": cell.get_text(" ", strip=True),
                    "numeric_value": parse_numeric(cell.get_text(" ", strip=True)),
                    "restatement_risk_flag": True,
                }
            )
    return output


def _find_financial_table(soup: BeautifulSoup, table_type: str) -> Any | None:
    patterns = TABLE_IDS[table_type]
    for pattern in patterns:
        container = soup.find(id=pattern)
        if container is not None:
            table = container.find("table") if hasattr(container, "find") else None
            if table is not None:
                return table
    for section in soup.find_all(["section", "div"]):
        section_id = str(section.get("id", "")).lower()
        heading = section.find(["h2", "h3"])
        heading_text = heading.get_text(" ", strip=True).lower() if heading else ""
        if any(pattern in section_id or pattern in heading_text for pattern in patterns):
            table = section.find("table")
            if table is not None:
                return table
    return None


def _fetch_cached_or_http(
    *,
    session: requests.Session,
    symbol: str,
    url: str,
    statement_basis: str,
    delay_seconds: float,
) -> dict[str, Any]:
    cache_path = _latest_recent_cache(symbol, statement_basis)
    if cache_path is not None:
        return {
            "accessible": True,
            "html": cache_path.read_text(encoding="utf-8"),
            "source": str(cache_path),
            "source_type": "SCREENER_HTTP_CACHE",
            "statement_basis": statement_basis,
            "notes": "Used recent local raw HTML cache.",
        }
    try:
        response = session.get(url, timeout=30)
    except requests.RequestException as exc:
        return {
            "accessible": False,
            "html": "",
            "source": url,
            "source_type": "SCREENER_HTTP_PAGE",
            "statement_basis": statement_basis,
            "notes": f"HTTP request failed: {exc}",
        }
    if response.status_code >= 500:
        time.sleep(delay_seconds)
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException as exc:
            return {
                "accessible": False,
                "html": "",
                "source": url,
                "source_type": "SCREENER_HTTP_PAGE",
                "statement_basis": statement_basis,
                "notes": f"HTTP retry failed: {exc}",
            }
    if response.status_code != 200:
        return {
            "accessible": False,
            "html": "",
            "source": url,
            "source_type": "SCREENER_HTTP_PAGE",
            "statement_basis": statement_basis,
            "notes": f"HTTP {response.status_code}; login/export permission may be required.",
        }
    cache_path = RAW_HTML_DIR / f"{symbol}_{statement_basis}_{datetime.now(UTC):%Y%m%d}.html"
    cache_path.write_text(response.text, encoding="utf-8")
    time.sleep(delay_seconds)
    return {
        "accessible": True,
        "html": response.text,
        "source": str(cache_path),
        "source_type": "SCREENER_HTTP_PAGE",
        "statement_basis": statement_basis,
        "notes": "",
    }


def _latest_recent_cache(symbol: str, statement_basis: str, max_age_days: int = 7) -> Path | None:
    candidates = sorted(RAW_HTML_DIR.glob(f"{symbol}_{statement_basis}_*.html"), reverse=True)
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    for path in candidates:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if modified >= cutoff:
            return path
    return None


def _frames_from_rows(rows_by_table: dict[str, list[dict[str, Any]]]) -> dict[str, pd.DataFrame]:
    return {
        table_type: pd.DataFrame(rows, columns=RAW_COLUMNS)
        for table_type, rows in rows_by_table.items()
    }


def _write_raw_tables(tables: dict[str, pd.DataFrame]) -> None:
    for table_type, path in RAW_TABLES.items():
        frame = tables.get(table_type, pd.DataFrame(columns=RAW_COLUMNS))
        frame.to_parquet(path, index=False)


def _empty_raw_table_frames() -> dict[str, pd.DataFrame]:
    return {table_type: pd.DataFrame(columns=RAW_COLUMNS) for table_type in RAW_TABLES}


def _mapping_for_ticker(symbol_map: pd.DataFrame, ticker: str) -> pd.Series | None:
    matches = symbol_map[symbol_map["nse_ticker"].astype(str).str.upper() == ticker.upper()]
    if len(matches) != 1:
        return None
    row = matches.iloc[0]
    if str(row.get("status", "")).upper() != "OK":
        return None
    return row


def _acquisition_row(
    *,
    ticker: str,
    mapped: pd.Series | None = None,
    page_accessible: bool = False,
    consolidated_page: bool = False,
    quarterly_table: bool = False,
    balance_sheet: bool = False,
    cashflow: bool = False,
    ratios: bool = False,
    raw_rows_extracted: int = 0,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "screener_symbol": "" if mapped is None else mapped["screener_symbol"],
        "company_name": "" if mapped is None else mapped["company_name"],
        "symbol_mapping": "OK" if mapped is not None else "REVIEW",
        "page_accessible": page_accessible,
        "consolidated_page": consolidated_page,
        "quarterly_table_found": quarterly_table,
        "balance_sheet_found": balance_sheet,
        "cashflow_found": cashflow,
        "ratios_found": ratios,
        "raw_rows_extracted": raw_rows_extracted,
        "notes": notes,
    }


def _disabled_or_local_acquisition_rows(
    *,
    tickers: tuple[str, ...],
    permitted_http_mode: bool,
    local_exports_processed: int,
) -> list[dict[str, Any]]:
    symbol_map = load_symbol_map()
    rows = []
    notes = (
        f"LOCAL_EXPORT processed {local_exports_processed} files."
        if local_exports_processed
        else "No local exports found; PERMITTED_HTTP disabled."
    )
    if permitted_http_mode and local_exports_processed:
        notes = "Local exports were found, so HTTP fallback was not needed."
    for ticker in tickers:
        rows.append(
            _acquisition_row(
                ticker=ticker,
                mapped=_mapping_for_ticker(symbol_map, ticker),
                notes=notes,
            )
        )
    return rows


def _write_acquisition_report(
    *,
    local_exports: int,
    permitted_http_mode: bool,
    accessible: int,
    blocked_or_failed: int,
    acquisition: pd.DataFrame,
) -> None:
    recommended = (
        "Download company Excel exports from Screener and place them in: "
        "data/fundamentals/screener_exports/"
    )
    lines = [
        "Screener acquisition report",
        "",
        f"Number of local exports: {local_exports}",
        f"Permitted HTTP mode enabled: {permitted_http_mode}",
        f"HTTP company pages accessible: {accessible}",
        f"HTTP blocked/failed: {blocked_or_failed}",
        f"Acquisition rows: {len(acquisition)}",
        "",
        "Recommended next action:",
        recommended,
        "",
        "No publication dates are inferred from Screener period columns.",
        "publication_date remains NaT and publication_date_quality remains MISSING.",
    ]
    ACQUISITION_REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
