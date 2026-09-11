"""Download and cache daily OHLCV data for stock probability models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    CACHE_DIR,
    REPORT_DIR,
    START_DATE,
    STOCKS_CSV,
    STOCKS_UNIVERSE_2026_08_CSV,
)


def load_stock_universe(path: Path = STOCKS_UNIVERSE_2026_08_CSV) -> list[str]:
    """Load NSE stock symbols from a CSV file.

    Args:
        path: CSV containing a `symbol` column.

    Returns:
        Ordered list of unique symbols.
    """

    metadata = load_universe_metadata(path)
    symbols = metadata["Ticker"].dropna().astype(str).str.strip().tolist()
    return list(dict.fromkeys(symbol for symbol in symbols if symbol))


def load_universe_metadata(path: Path = STOCKS_UNIVERSE_2026_08_CSV) -> pd.DataFrame:
    """Load stock universe metadata from the new or legacy CSV format."""

    frame = pd.read_csv(path)
    if "Ticker" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "Ticker"})
    if "Ticker" not in frame.columns:
        raise ValueError(f"{path} must contain a Ticker or symbol column.")

    output = frame.copy()
    output["Ticker"] = output["Ticker"].astype(str).str.strip()
    for column in ["Company", "Sector", "MarketCapCategory", "Group"]:
        if column not in output.columns:
            output[column] = "Unknown"
        output[column] = output[column].fillna("Unknown").astype(str).str.strip()
    output = output[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]]
    return output.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)


def download_ohlcv(
    symbol: str,
    *,
    start: str = START_DATE,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Download or read cached daily OHLCV data from yfinance.

    Args:
        symbol: Yahoo Finance symbol.
        start: Start date for daily history.
        force: If true, refresh cached data.
        cache_dir: Local cache directory.

    Returns:
        DataFrame indexed by date with OHLCV columns.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_symbol(symbol)}.csv"
    if cache_path.exists() and not force:
        cached = _normalize_ohlcv(pd.read_csv(cache_path, parse_dates=["Date"]))
        updated = _incremental_update(symbol, cached, cache_path)
        if updated is not None:
            return updated
        return cached

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in real environment.
        raise RuntimeError("Install yfinance before downloading market data.") from exc

    downloaded = yf.download(
        symbol,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}.")
    downloaded = downloaded.reset_index()
    downloaded.to_csv(cache_path, index=False)
    return _normalize_ohlcv(downloaded)


def download_universe(
    symbols: list[str] | None = None,
    *,
    include_benchmark: bool = True,
    force: bool = False,
    update_cached: bool = True,
    batch_size: int = 25,
    retries: int = 2,
) -> dict[str, pd.DataFrame]:
    """Download/cache data for the configured universe.

    Args:
        symbols: Optional symbols. Defaults to `stocks.csv`.
        include_benchmark: Whether to include Nifty 50 benchmark data.
        force: If true, refresh all cached files.
        update_cached: If true, incrementally update existing cache files.

    Returns:
        Mapping of symbol to OHLCV DataFrame.
    """

    universe = symbols or load_stock_universe()
    if include_benchmark and BENCHMARK_SYMBOL not in universe:
        universe = [*universe, BENCHMARK_SYMBOL]
    data: dict[str, pd.DataFrame] = {}
    pending: list[str] = []
    failures: list[dict[str, str]] = []
    for symbol in universe:
        try:
            cache_path = CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
            if cache_path.exists() and not force:
                cached = _normalize_ohlcv(pd.read_csv(cache_path, parse_dates=["Date"]))
                if update_cached:
                    updated = _incremental_update(symbol, cached, cache_path)
                    data[symbol] = updated if updated is not None else cached
                else:
                    data[symbol] = cached
            else:
                pending.append(symbol)
        except Exception as exc:
            failures.append({"Ticker": symbol, "Error": str(exc)})

    if pending:
        for attempt in range(retries + 1):
            remaining: list[str] = []
            for batch in _chunks(pending, batch_size):
                downloaded = _download_batch(batch, start=START_DATE)
                for symbol in batch:
                    try:
                        frame = _frame_from_batch(downloaded, symbol)
                        if frame.empty:
                            raise ValueError("No OHLCV data returned.")
                        normalized = _normalize_ohlcv(frame.reset_index())
                        cache_path = CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
                        normalized.to_csv(cache_path, index=False)
                        data[symbol] = normalized
                    except Exception as exc:
                        if attempt < retries:
                            remaining.append(symbol)
                        else:
                            failures.append({"Ticker": symbol, "Error": str(exc)})
            pending = remaining
            if not pending:
                break

    if failures:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).drop_duplicates("Ticker").to_csv(
            REPORT_DIR / "stock_probability_failed_tickers.csv",
            index=False,
        )
    else:
        failed_path = REPORT_DIR / "stock_probability_failed_tickers.csv"
        if failed_path.exists():
            failed_path.unlink()
    return data


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to a typed daily OHLCV frame."""

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [str(column[0]) for column in normalized.columns]
    if "Date" not in normalized.columns and "Datetime" in normalized.columns:
        normalized = normalized.rename(columns={"Datetime": "Date"})
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    normalized = normalized[required].copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"]).dt.tz_localize(None)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["Date", "Close"]).sort_values("Date")
    normalized = normalized.drop_duplicates(subset=["Date"], keep="last")
    return normalized.reset_index(drop=True)


def _safe_symbol(symbol: str) -> str:
    """Return a filesystem-safe symbol name."""

    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")


def _incremental_update(
    symbol: str,
    cached: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame | None:
    """Append newly available OHLCV rows to an existing cache file."""

    if cached.empty:
        return None
    last_date = pd.to_datetime(cached["Date"]).max()
    today = pd.Timestamp.today().normalize()
    if last_date >= today - pd.Timedelta(days=1):
        return cached
    try:
        downloaded = _download_symbol(
            symbol, start=(last_date - pd.Timedelta(days=5)).date().isoformat()
        )
        if downloaded.empty:
            return cached
        combined = pd.concat([cached, downloaded], ignore_index=True)
        normalized = _normalize_ohlcv(combined)
        normalized.to_csv(cache_path, index=False)
        return normalized
    except Exception:
        return cached


def _download_symbol(symbol: str, *, start: str) -> pd.DataFrame:
    """Download one symbol from yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in real environment.
        raise RuntimeError("Install yfinance before downloading market data.") from exc
    downloaded = yf.download(
        symbol,
        start=start,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        return pd.DataFrame()
    return _normalize_ohlcv(downloaded.reset_index())


def _download_batch(symbols: list[str], *, start: str) -> pd.DataFrame:
    """Download a batch of symbols from yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - exercised in real environment.
        raise RuntimeError("Install yfinance before downloading market data.") from exc
    return yf.download(
        tickers=" ".join(symbols),
        start=start,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )


def _frame_from_batch(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract one symbol from a yfinance batch download."""

    if downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if symbol in downloaded.columns.get_level_values(0):
            frame = downloaded[symbol].copy()
        elif symbol in downloaded.columns.get_level_values(1):
            frame = downloaded.xs(symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = downloaded.copy()
    return frame.dropna(how="all")


def _chunks(items: list[str], size: int) -> list[list[str]]:
    """Split items into fixed-size chunks."""

    return [items[index : index + size] for index in range(0, len(items), max(1, size))]
