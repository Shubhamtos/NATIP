"""US-market inputs for the existing deterministic Market Strength workspace."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.backtesting.data.yahoo import YahooDataStore
from app.dashboard.kronos_universe import US_POPULAR
from app.intelligence.kronos_forecast import normalize_stock_symbol
from app.intelligence.market_strength import (
    MarketStrengthSnapshot,
    build_market_strength_snapshot,
    load_cached_market_histories,
)

US_BENCHMARK = "^GSPC"


def latest_completed_us_day(now: datetime | None = None) -> pd.Timestamp:
    """Bound US daily bars at completed New York sessions, with a publication buffer."""
    current = now or datetime.now(ZoneInfo("America/New_York"))
    current = (
        current.replace(tzinfo=ZoneInfo("America/New_York"))
        if current.tzinfo is None
        else current.astimezone(ZoneInfo("America/New_York"))
    )
    candidate = current.date()
    if current.time() < time(16, 15):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return pd.Timestamp(candidate)


def us_market_universe(root: Path, symbols: tuple[str, ...]) -> pd.DataFrame:
    """Build US metadata without adding NSE suffixes or guessing unknown sectors."""
    names = dict(US_POPULAR)
    directory = root / "data/kronos/universes/us_stocks.csv"
    if directory.exists():
        for label in pd.read_csv(directory)["label"].dropna():
            ticker, _, name = str(label).partition(" - ")
            names[ticker] = name or ticker
    metadata_path = root / "data/market_strength/us_company_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    records = []
    for symbol in dict.fromkeys(symbols):
        ticker = normalize_stock_symbol(symbol, "US")
        record = metadata.get(ticker, {})
        records.append(
            {
                "symbol": ticker,
                "company": record.get("company") or names.get(ticker, ticker),
                "sector": record.get("sector") or "Unclassified",
            }
        )
    return pd.DataFrame(records, columns=["symbol", "company", "sector"])


def update_us_market_strength(
    root: Path,
    symbols: tuple[str, ...],
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Update existing adjusted/raw Yahoo caches with a US-session date bound.

    Args:
        root: NATIP project root.
        symbols: Selected US stocks; benchmark is added automatically.
        on_progress: Optional per-ticker UI progress callback.

    Returns:
        Per-symbol success/failure report; one failure does not stop other stocks.
    """
    import yfinance as yf

    stocks = tuple(dict.fromkeys(normalize_stock_symbol(s, "US") for s in symbols))
    store = YahooDataStore(root / "data/backtesting/cache", retries=2)
    end = latest_completed_us_day().date().isoformat()
    metadata_path = root / "data/market_strength/us_company_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    audits = []
    for position, symbol in enumerate((*stocks, US_BENCHMARK), start=1):
        try:
            frame, source = store.load_symbol(symbol, start="2018-01-01", end=end, update=True)
            audits.append(
                {
                    "Ticker": symbol,
                    "Status": "PASS",
                    "Rows": len(frame),
                    "Source": source,
                    "Issue": "",
                }
            )
            if symbol != US_BENCHMARK:
                saved = metadata.get(symbol, {})
                saved_at = pd.to_datetime(saved.get("updated_at"), utc=True, errors="coerce")
                age = (
                    pd.Timestamp.now(tz="UTC") - saved_at
                    if pd.notna(saved_at)
                    else pd.Timedelta(days=999)
                )
                if not saved.get("sector") or age > pd.Timedelta(days=30):
                    try:
                        info = yf.Ticker(symbol).info or {}
                        metadata[symbol] = {
                            "company": info.get("longName") or US_POPULAR.get(symbol, symbol),
                            "sector": info.get("sector") or "Unclassified",
                            "updated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
                            "source": "Yahoo Finance company profile",
                        }
                    except Exception:
                        pass
        except Exception as exc:
            audits.append(
                {
                    "Ticker": symbol,
                    "Status": "FAIL",
                    "Rows": 0,
                    "Source": "unavailable",
                    "Issue": str(exc)[:300],
                }
            )
        if on_progress:
            on_progress(position, len(stocks) + 1, symbol)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2))
    temporary.replace(metadata_path)
    audit = pd.DataFrame(audits)
    audit.to_csv(metadata_path.parent / "us_update_quality.csv", index=False)
    return audit


def build_us_market_strength(root: Path, symbols: tuple[str, ...]) -> MarketStrengthSnapshot:
    """Build existing ranks/scans on US stocks against S&P 500 without network calls."""
    universe = us_market_universe(root, symbols)
    if len(universe) < 2:
        raise ValueError("Select at least two US companies for cross-sectional rankings.")
    histories, quality = load_cached_market_histories(
        universe.symbol.tolist() + [US_BENCHMARK],
        cache_dir=root / "data/backtesting/cache",
        legacy_cache_dir=root / "data/probability/clean_cache_adjusted",
    )
    cutoff = latest_completed_us_day()
    histories = {ticker: frame.loc[frame.Date <= cutoff] for ticker, frame in histories.items()}
    covered = sum(len(histories.get(symbol, [])) >= 253 for symbol in universe.symbol)
    if covered < max(2, int(len(universe) * 0.60 + 0.999)):
        raise ValueError(
            f"US price data is not ready: {covered}/{len(universe)} stocks have at least "
            "253 daily candles. Click Update latest data to fetch the selected companies."
        )
    return build_market_strength_snapshot(
        histories,
        universe,
        benchmark_symbol=US_BENCHMARK,
        quality=quality,
        market="US",
    )
