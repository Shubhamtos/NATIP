"""Market-breadth acquisition reuses NATIP's adjusted caches and Yahoo downloader."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from app.backtesting.data.yahoo import YahooDataStore, latest_completed_nse_day, load_nifty_universe
from app.intelligence.market_breadth import normalize_history
from app.probability.config import START_DATE

UNIVERSES = ("Nifty 50", "Nifty 100", "Nifty 250", "Nifty 500", "Nifty Smallcap 250")


def load_breadth_universe(root: Path, name: str) -> tuple[pd.DataFrame, str]:
    """Prefer saved official membership; do not equate Nifty 250 with Smallcap 250."""
    if name not in UNIVERSES:
        raise ValueError("Unsupported breadth universe")
    saved = root / "data/kronos/universes" / (name.replace(" ", "_") + ".csv")
    if name == "Nifty Smallcap 250" and not saved.exists():
        saved = root / "data/nifty_smallcap250_constituents.csv"
    if saved.exists():
        raw = pd.read_csv(saved)
        frame = raw.rename(
            columns={"Symbol": "symbol", "Company Name": "company", "Industry": "sector"}
        )
        source = f"Saved constituent snapshot: {saved.name}"
    elif name == "Nifty 500":
        from app.dashboard.kronos_universe import load_kronos_universe

        _, source = load_kronos_universe(root, name)
        frame = pd.read_csv(saved).rename(
            columns={"Symbol": "symbol", "Company Name": "company", "Industry": "sector"}
        )
    else:
        frame = load_nifty_universe(name)
        source = "NATIP configured constituent catalogue"
    if not {"symbol", "company"}.issubset(frame):
        raise ValueError("Constituent snapshot has no usable company/symbol columns")
    frame = frame.dropna(subset=["symbol"]).copy()
    frame["symbol"] = (
        frame.symbol.astype(str)
        .str.strip()
        .str.upper()
        .map(lambda s: s if s.endswith(".NS") else s + ".NS")
    )
    if not frame.symbol.str.fullmatch(r"[A-Z0-9&._-]+\.NS").all():
        raise ValueError("Constituent file contains invalid NSE ticker symbols")
    frame["company"] = frame.company.fillna(frame.symbol)
    if "sector" not in frame:
        frame["sector"] = "Unclassified"
    mapping = root / "config/screener_symbol_map.csv"
    if mapping.exists():
        known = pd.read_csv(mapping)
        if {"nse_ticker", "sector"}.issubset(known):
            sectors = known.drop_duplicates("nse_ticker").set_index("nse_ticker").sector
            unknown = frame.sector.isna() | frame.sector.isin(["Unknown", "Unclassified", ""])
            frame.loc[unknown, "sector"] = frame.loc[unknown, "symbol"].map(sectors)
    frame["sector"] = frame.sector.fillna("Unclassified")
    return frame[["symbol", "company", "sector"]].drop_duplicates("symbol"), source


def load_breadth_histories(
    root: Path, symbols: list[str], cutoff: pd.Timestamp
) -> tuple[dict, pd.DataFrame]:
    """Pick the freshest valid adjusted source, not the first existing stale file."""
    histories, audit = {}, []
    for ticker in dict.fromkeys([*symbols, "^NSEI"]):
        safe = ticker.replace("^", "INDEX_").replace(".", "_").replace("/", "_")
        paths = (
            root / "data/probability/clean_cache_adjusted" / f"{safe}.csv",
            root / "data/backtesting/cache" / f"{safe}.parquet",
            root / "data/kronos/price_cache" / f"{ticker.replace('.', '_')}.parquet",
        )
        candidates, errors = [], []
        for path in paths:
            if not path.exists():
                continue
            try:
                raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
                normalized = normalize_history(raw)
                normalized = normalized.loc[normalized.Date.le(cutoff)]
                usable = normalized.loc[normalized.Close.gt(0) & normalized.Volume.ge(0)]
                if not usable.empty:
                    candidates.append((usable.Date.max(), len(normalized), raw, path))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"{path.name}: {exc}")
        if candidates:
            # Keep a whole snapshot; never splice differently adjusted histories.
            latest, _, raw, path = max(candidates, key=lambda item: (item[0], item[1]))
            histories[ticker] = raw
            audit.append(
                {
                    "Ticker": ticker,
                    "Cache Source": str(path.relative_to(root)),
                    "Cache Last Date": latest,
                    "Cache Issue": "; ".join(errors),
                }
            )
        else:
            audit.append(
                {
                    "Ticker": ticker,
                    "Cache Source": "Missing",
                    "Cache Last Date": pd.NaT,
                    "Cache Issue": "; ".join(errors) or "No adjusted cache",
                }
            )
    return histories, pd.DataFrame(audit)


def update_breadth_data(
    root: Path,
    symbols: list[str],
    *,
    on_progress: Callable | None = None,
    store_factory: Callable = YahooDataStore,
) -> pd.DataFrame:
    """Bounded per-ticker incremental downloads; one failure never stops the universe."""
    store = store_factory(root / "data/backtesting/cache", retries=2)
    cutoff = latest_completed_nse_day()
    tickers = list(dict.fromkeys([*symbols, "^NSEI"]))
    rows = []
    for number, ticker in enumerate(tickers, start=1):
        try:
            frame, source = store.load_symbol(
                ticker, start=START_DATE, end=cutoff.date().isoformat(), update=True
            )
            last = pd.to_datetime(frame.date).max()
            status = "UPDATED" if source == "yfinance_incremental" else "CACHED"
            if last < cutoff:
                status = "REVIEW"
            rows.append(
                {
                    "Ticker": ticker,
                    "Status": status,
                    "Last Date": last,
                    "Source": source,
                    "Issue": ""
                    if last >= cutoff
                    else "No bar through requested cutoff; check holiday/staleness",
                }
            )
        except Exception as exc:
            rows.append({"Ticker": ticker, "Status": "FAILED", "Issue": str(exc)})
        if on_progress:
            on_progress(number, len(tickers), ticker)
    return pd.DataFrame(rows)
