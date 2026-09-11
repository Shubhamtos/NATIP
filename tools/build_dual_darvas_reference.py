"""Build reference CSVs for the dual-listed non-F&O Darvas screener.

The Streamlit screener intentionally fails closed unless it has three local
reference files:

* config/reference/nse_securities.csv
* config/reference/bse_securities.csv
* config/reference/nse_fo_stocks.csv

This script downloads public exchange reference files where available, caches
the raw files locally, and normalizes them into the minimal schema consumed by
the deterministic screener. It does not scrape login-protected data.
"""

from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "config" / "reference"
RAW_DIR = PROJECT_ROOT / "data" / "reference" / "raw"
SCREENER_HTML_DIR = PROJECT_ROOT / "data" / "fundamentals" / "raw_html"

NSE_EQUITY_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_FO_MARKET_LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
BSE_BHAVCOPY_URL = "https://www.bseindia.com/download/BhavCopy/Equity/EQ{date_text}_CSV.ZIP"

OUTPUT_COLUMNS = [
    "symbol",
    "isin",
    "active",
    "mainboard",
    "series",
    "instrument_type",
    "surveillance",
    "trade_to_trade",
    "suspended",
    "reference_as_of",
    "source",
]


@dataclass(frozen=True)
class BuildSummary:
    """Reference build summary."""

    nse_rows: int
    bse_rows: int
    fo_rows: int
    bse_source_date: date | None


def main() -> None:
    """Build the three reference CSV files."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--max-bse-lookback-days", type=int, default=10)
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date()
    summary = build_references(as_of=as_of, max_bse_lookback_days=args.max_bse_lookback_days)
    print(
        "Built dual Darvas references: "
        f"NSE={summary.nse_rows}, BSE={summary.bse_rows}, F&O={summary.fo_rows}, "
        f"BSE source date={summary.bse_source_date}"
    )


def build_references(*, as_of: date, max_bse_lookback_days: int = 10) -> BuildSummary:
    """Download/cache and normalize NSE, BSE, and F&O reference files."""

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = _session()
    nse_raw = _download_text(
        session,
        NSE_EQUITY_URL,
        RAW_DIR / f"nse_equity_{as_of:%Y%m%d}.csv",
    )
    fo_raw = _download_text(
        session,
        NSE_FO_MARKET_LOTS_URL,
        RAW_DIR / f"nse_fo_mktlots_{as_of:%Y%m%d}.csv",
    )

    nse = _normalize_nse_equity(pd.read_csv(io.StringIO(nse_raw)), as_of=as_of)
    symbol_to_isin = dict(zip(nse["symbol"], nse["isin"], strict=False))
    bse_raw, bse_source_date = _download_latest_bse_bhavcopy(
        session=session, as_of=as_of, max_lookback_days=max_bse_lookback_days
    )
    if bse_raw:
        bse = _normalize_bse_equity(
            pd.read_csv(io.StringIO(bse_raw)),
            nse_by_isin=nse.set_index("isin")["symbol"].to_dict(),
            as_of=bse_source_date or as_of,
        )
    else:
        bse = _normalize_bse_from_screener_cache(nse, as_of=as_of)
    fo = _normalize_fo_underlyings(pd.read_csv(io.StringIO(fo_raw)), symbol_to_isin=symbol_to_isin)

    nse.to_csv(REFERENCE_DIR / "nse_securities.csv", index=False)
    bse.to_csv(REFERENCE_DIR / "bse_securities.csv", index=False)
    fo.to_csv(REFERENCE_DIR / "nse_fo_stocks.csv", index=False)

    return BuildSummary(
        nse_rows=len(nse),
        bse_rows=len(bse),
        fo_rows=len(fo),
        bse_source_date=bse_source_date,
    )


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36 "
                "NATIP reference builder"
            ),
            "Accept": "text/csv,application/zip,application/octet-stream,*/*",
            "Referer": "https://www.nseindia.com/",
        }
    )
    return session


def _download_text(session: requests.Session, url: str, cache_path: Path) -> str:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8", errors="replace")
    response = _get_with_retry(session, url)
    text = response.content.decode("utf-8", errors="replace")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _download_latest_bse_bhavcopy(
    *,
    session: requests.Session,
    as_of: date,
    max_lookback_days: int,
) -> tuple[str | None, date | None]:
    last_error: Exception | None = None
    for offset in range(max_lookback_days + 1):
        source_date = as_of - timedelta(days=offset)
        date_text = source_date.strftime("%d%m%y")
        cache_path = RAW_DIR / f"bse_equity_bhavcopy_{source_date:%Y%m%d}.csv"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path.read_text(encoding="utf-8", errors="replace"), source_date
        try:
            response = _get_with_retry(session, BSE_BHAVCOPY_URL.format(date_text=date_text))
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
                text = archive.read(member).decode("utf-8", errors="replace")
            cache_path.write_text(text, encoding="utf-8")
            return text, source_date
        except Exception as exc:  # noqa: BLE001 - keep trying previous trading days.
            last_error = exc
            continue
    print(
        "BSE bhavcopy download unavailable; falling back to cached Screener BSE links: "
        f"{last_error}"
    )
    return None, None


def _get_with_retry(session: requests.Session, url: str) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200 and response.content:
                return response
            last_error = RuntimeError(f"HTTP {response.status_code} for {url}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def _normalize_nse_equity(frame: pd.DataFrame, *, as_of: date) -> pd.DataFrame:
    symbol = _column(frame, ["SYMBOL"])
    series = _column(frame, ["SERIES"])
    isin = _column(frame, ["ISIN NUMBER", "ISIN"])
    if symbol is None or isin is None:
        raise ValueError(f"Unexpected NSE equity columns: {list(frame.columns)}")
    output = pd.DataFrame(
        {
            "symbol": frame[symbol].astype(str).str.strip().str.upper(),
            "isin": frame[isin].astype(str).str.strip().str.upper(),
            "active": True,
            "mainboard": True,
            "series": frame[series].astype(str).str.strip().str.upper() if series else "EQ",
            "instrument_type": "EQUITY",
            "surveillance": "",
            "trade_to_trade": False,
            "suspended": False,
            "reference_as_of": as_of.isoformat(),
            "source": NSE_EQUITY_URL,
        }
    )
    output = output[output["series"].eq("EQ") & output["isin"].str.startswith("IN")]
    return output.drop_duplicates(subset=["isin"]).sort_values("symbol")[OUTPUT_COLUMNS]


def _normalize_bse_equity(
    frame: pd.DataFrame,
    *,
    nse_by_isin: dict[str, str],
    as_of: date,
) -> pd.DataFrame:
    isin = _column(frame, ["ISIN_CODE", "ISIN", "ISIN NUMBER"])
    status = _column(frame, ["TRADING_STATUS", "STATUS"])
    group = _column(frame, ["GROUP", "SC_GROUP"])
    if isin is None:
        raise ValueError(f"Unexpected BSE bhavcopy columns: {list(frame.columns)}")
    normalized_isin = frame[isin].astype(str).str.strip().str.upper()
    output = pd.DataFrame(
        {
            "symbol": normalized_isin.map(nse_by_isin).fillna(""),
            "isin": normalized_isin,
            "active": _active_series(frame[status]) if status else True,
            "mainboard": _mainboard_series(frame[group]) if group else True,
            "series": "EQ",
            "instrument_type": "EQUITY",
            "surveillance": "",
            "trade_to_trade": False,
            "suspended": False,
            "reference_as_of": as_of.isoformat(),
            "source": BSE_BHAVCOPY_URL.format(date_text=as_of.strftime("%d%m%y")),
        }
    )
    output = output[output["isin"].str.startswith("IN")]
    return output.drop_duplicates(subset=["isin"]).sort_values("isin")[OUTPUT_COLUMNS]


def _normalize_bse_from_screener_cache(nse: pd.DataFrame, *, as_of: date) -> pd.DataFrame:
    """Build BSE listing rows from locally cached Screener pages.

    Screener pages include BSE quote links of the form:
    https://www.bseindia.com/stock-share-price/.../{symbol}/{scrip_code}/
    This fallback is used only when the BSE public bhavcopy route is unavailable.
    """

    rows: list[dict[str, object]] = []
    for record in nse.itertuples(index=False):
        html_path = _latest_screener_html_path(str(record.symbol))
        if html_path is None:
            continue
        text = html_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"https://www\.bseindia\.com/stock-share-price/", text):
            continue
        rows.append(
            {
                "symbol": str(record.symbol),
                "isin": str(record.isin),
                "active": True,
                "mainboard": True,
                "series": "EQ",
                "instrument_type": "EQUITY",
                "surveillance": "",
                "trade_to_trade": False,
                "suspended": False,
                "reference_as_of": _date_from_screener_html_path(html_path, as_of).isoformat(),
                "source": str(html_path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS).drop_duplicates(subset=["isin"])


def _latest_screener_html_path(symbol: str) -> Path | None:
    if not SCREENER_HTML_DIR.exists():
        return None
    candidates = sorted(
        SCREENER_HTML_DIR.glob(f"{symbol}_CONSOLIDATED_*.html"),
        key=lambda path: path.name,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            SCREENER_HTML_DIR.glob(f"{symbol}_*.html"),
            key=lambda path: path.name,
            reverse=True,
        )
    return candidates[0] if candidates else None


def _date_from_screener_html_path(path: Path, default: date) -> date:
    match = re.search(r"_(\d{8})\.html$", path.name)
    if not match:
        return default
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return default


def _normalize_fo_underlyings(
    frame: pd.DataFrame,
    *,
    symbol_to_isin: dict[str, str],
) -> pd.DataFrame:
    symbol = _column(frame, ["SYMBOL", "Symbol", "Underlying"])
    if symbol is None:
        raise ValueError(f"Unexpected F&O market-lot columns: {list(frame.columns)}")
    symbols = frame[symbol].astype(str).str.strip().str.upper()
    output = pd.DataFrame(
        {
            "symbol": symbols,
            "isin": symbols.map(symbol_to_isin),
            "source": NSE_FO_MARKET_LOTS_URL,
        }
    )
    output = output.dropna(subset=["isin"])
    output = output[output["isin"].astype(str).str.startswith("IN")]
    return output.drop_duplicates(subset=["isin"]).sort_values("symbol")


def _column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(column).strip().upper(): str(column) for column in frame.columns}
    for candidate in candidates:
        column = normalized.get(candidate.strip().upper())
        if column:
            return column
    return None


def _active_series(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.upper().isin({"S", "SUSPENDED", "INACTIVE"})


def _mainboard_series(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.upper().isin({"M", "MT", "XT", "ST", "P"})


if __name__ == "__main__":
    main()
