"""Screener symbol-map generation and loading."""

from __future__ import annotations

import pandas as pd

from app.fundamentals.config import (
    FALLBACK_STOCKS_CSV,
    STOCKS_UNIVERSE_CSV,
    SYMBOL_MAP_PATH,
    ensure_fundamental_dirs,
)

SYMBOL_MAP_COLUMNS = [
    "nse_ticker",
    "screener_symbol",
    "company_name",
    "sector",
    "status",
    "notes",
]


def create_symbol_map(overwrite: bool = False) -> pd.DataFrame:
    """Create the Screener symbol map from the existing NSE universe.

    Args:
        overwrite: Replace the existing map when True.

    Returns:
        Symbol-map DataFrame.
    """

    ensure_fundamental_dirs()
    if SYMBOL_MAP_PATH.exists() and not overwrite:
        return pd.read_csv(SYMBOL_MAP_PATH)

    universe = _load_reference_universe()
    rows: list[dict[str, str]] = []
    for _, row in universe.iterrows():
        ticker = str(row.get("Ticker", row.get("ticker", ""))).strip().upper()
        if not ticker:
            continue
        screener_symbol = ticker.removesuffix(".NS")
        rows.append(
            {
                "nse_ticker": ticker,
                "screener_symbol": screener_symbol,
                "company_name": str(row.get("Company", row.get("company", screener_symbol))).strip(),
                "sector": str(row.get("Sector", row.get("sector", "Unknown"))).strip() or "Unknown",
                "status": "OK",
                "notes": "",
            }
        )

    symbol_map = pd.DataFrame(rows, columns=SYMBOL_MAP_COLUMNS).drop_duplicates("nse_ticker")
    ambiguous = symbol_map["screener_symbol"].duplicated(keep=False)
    symbol_map.loc[ambiguous, "status"] = "REVIEW"
    symbol_map.loc[ambiguous, "notes"] = "Ambiguous duplicate Screener symbol; verify manually."
    symbol_map.to_csv(SYMBOL_MAP_PATH, index=False)
    return symbol_map


def load_symbol_map() -> pd.DataFrame:
    """Load the Screener symbol map, creating it if missing."""

    if not SYMBOL_MAP_PATH.exists():
        return create_symbol_map()
    return pd.read_csv(SYMBOL_MAP_PATH)


def _load_reference_universe() -> pd.DataFrame:
    """Load the current NSE reference universe."""

    if STOCKS_UNIVERSE_CSV.exists():
        return pd.read_csv(STOCKS_UNIVERSE_CSV)
    if FALLBACK_STOCKS_CSV.exists():
        frame = pd.read_csv(FALLBACK_STOCKS_CSV)
        if "Ticker" not in frame.columns and "ticker" in frame.columns:
            frame = frame.rename(columns={"ticker": "Ticker"})
        return frame
    raise FileNotFoundError("No NSE reference universe CSV found.")

