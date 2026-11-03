"""Incremental Yahoo Finance data store with adjusted and raw prices."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import datetime, timedelta
from datetime import time as wall_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from app.backtesting.config import PROJECT_ROOT
from app.dashboard.nifty250_universe import (
    fast_nifty50_universe,
    fast_nifty100_universe,
    fast_nifty250_universe,
    fast_nifty_smallcap250_universe,
)

LOGGER = logging.getLogger(__name__)
LEGACY_CLEAN_CACHE = PROJECT_ROOT / "data" / "probability" / "clean_cache_adjusted"


def load_nifty_universe(name: str) -> pd.DataFrame:
    """Load current constituent metadata for a configured NSE universe.

    Current membership is a survivorship-bias limitation. Historical constituent
    files can be supplied by replacing this table before a run.
    """

    key = name.lower().replace("_", "").replace(" ", "")
    loaders = {
        "nifty50": fast_nifty50_universe,
        "nifty100": fast_nifty100_universe,
        "nifty250": fast_nifty250_universe,
        "niftysmallcap250": fast_nifty_smallcap250_universe,
    }
    if key not in loaders:
        path = Path(name)
        if not path.exists():
            raise ValueError(f"Unknown universe {name!r}; use Nifty 50/100/250 or a CSV path")
        frame = pd.read_csv(path)
        ticker_column = "Ticker" if "Ticker" in frame.columns else "symbol"
        if ticker_column not in frame.columns:
            raise ValueError("Universe CSV must contain Ticker or symbol")
        result = frame.rename(columns={ticker_column: "symbol"}).copy()
        result["symbol"] = result["symbol"].astype(str).map(_to_yahoo_symbol)
        for column in ("company", "sector"):
            if column not in result.columns:
                title = column.title()
                result[column] = result.get(title, "Unknown")
        return result[["symbol", "company", "sector"]].drop_duplicates("symbol")

    records = loaders[key]().symbols
    return pd.DataFrame(
        {
            "symbol": [_to_yahoo_symbol(item.symbol) for item in records],
            "company": [item.name for item in records],
            "sector": [item.sector for item in records],
        }
    ).drop_duplicates("symbol")


class YahooDataStore:
    """Read, incrementally update, validate, and cache daily Yahoo OHLCV."""

    def __init__(self, cache_dir: str | Path, *, retries: int = 3) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retries = retries

    def load(
        self,
        symbols: Iterable[str],
        *,
        start: str,
        end: str | None = None,
        update: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return a long-format panel plus per-symbol quality report."""

        frames: list[pd.DataFrame] = []
        audits: list[dict[str, object]] = []
        for symbol in dict.fromkeys(symbols):
            try:
                frame, source = self.load_symbol(symbol, start=start, end=end, update=update)
                frames.append(frame.assign(symbol=symbol))
                audits.append(_audit_symbol(symbol, frame, source=source, error=""))
            except Exception as exc:
                LOGGER.exception("Could not load %s", symbol)
                audits.append(
                    _audit_symbol(symbol, pd.DataFrame(), source="failed", error=str(exc))
                )
        panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not panel.empty:
            panel = panel.sort_values(["date", "symbol"]).drop_duplicates(
                ["date", "symbol"], keep="last"
            )
        return panel, pd.DataFrame(audits)

    def load_symbol(
        self,
        symbol: str,
        *,
        start: str,
        end: str | None,
        update: bool,
    ) -> tuple[pd.DataFrame, str]:
        """Load one symbol, preferring the dedicated Parquet cache."""

        cache_path = self.cache_dir / f"{_safe_symbol(symbol)}.parquet"
        cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
        source = "parquet_cache"
        if cached.empty:
            legacy = _read_legacy_clean_cache(symbol)
            if not legacy.empty:
                cached = legacy
                source = "natip_clean_cache"

        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end) if end else latest_completed_nse_day()
        needs_backfill = not cached.empty and pd.to_datetime(cached["date"]).min() > requested_start
        needs_update = not cached.empty and pd.to_datetime(cached["date"]).max() < requested_end
        if update and (cached.empty or needs_backfill or needs_update):
            fetch_start = pd.Timestamp(start)
            if not cached.empty and not needs_backfill:
                fetch_start = max(
                    fetch_start, pd.to_datetime(cached["date"]).max() - pd.Timedelta(days=7)
                )
            try:
                downloaded = self._download(symbol, fetch_start.date().isoformat(), end)
                if not downloaded.empty:
                    cached = pd.concat([cached, downloaded], ignore_index=True)
                    source = "yfinance_incremental"
            except Exception as exc:
                if cached.empty:
                    raise
                LOGGER.warning("Using cached %s data after update failed: %s", symbol, exc)
                source = f"{source}_update_failed"

        if cached.empty:
            raise ValueError("No market data available")
        normalized = _normalize(cached)
        normalized = normalized[
            (normalized["date"] >= pd.Timestamp(start)) & (normalized["date"] <= requested_end)
        ].reset_index(drop=True)
        normalized.to_parquet(cache_path, index=False)
        return normalized, source

    def _download(self, symbol: str, start: str, end: str | None) -> pd.DataFrame:
        """Download one symbol with bounded exponential backoff."""

        import yfinance as yf

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                yahoo_end = (
                    (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
                    if end
                    else None
                )
                raw = yf.download(
                    symbol,
                    start=start,
                    end=yahoo_end,
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=False,
                )
                if raw.empty:
                    raise ValueError("Yahoo returned no rows")
                return _from_yfinance(raw)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Yahoo download failed for {symbol}: {last_error}")


def add_tradability_fields(panel: pd.DataFrame, *, lookback: int = 20) -> pd.DataFrame:
    """Add backward-looking liquidity, stale-price, and tradability fields."""

    output = panel.sort_values(["symbol", "date"]).copy()
    grouped = output.groupby("symbol", group_keys=False)
    output["traded_value"] = output["adj_close"] * output["volume"]
    output["avg_traded_value_20d"] = grouped["traded_value"].transform(
        lambda values: values.rolling(lookback, min_periods=lookback).mean()
    )
    output["is_zero_volume"] = output["volume"].fillna(0).le(0)
    unchanged = grouped["adj_close"].transform(lambda values: values.diff().eq(0))
    output["is_stale"] = output["is_zero_volume"] & unchanged
    valid_ohlc = (
        output[["adj_open", "adj_high", "adj_low", "adj_close"]].gt(0).all(axis=1)
        & output["adj_high"].ge(output[["adj_open", "adj_close", "adj_low"]].max(axis=1))
        & output["adj_low"].le(output[["adj_open", "adj_close", "adj_high"]].min(axis=1))
    )
    output["is_tradable"] = valid_ohlc & ~output["is_stale"] & ~output["is_zero_volume"]
    return output


def latest_completed_nse_day(now: datetime | None = None) -> pd.Timestamp:
    """Return the latest completed weekday in Asia/Kolkata.

    Exchange holidays are naturally removed when downloaded data is intersected
    with this upper bound. A conservative 16:00 IST cutoff avoids partial daily bars.
    """

    current = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo("Asia/Kolkata"))
    else:
        current = current.astimezone(ZoneInfo("Asia/Kolkata"))
    candidate = current.date()
    if current.time() < wall_time(16, 0):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return pd.Timestamp(candidate)


def merge_point_in_time_fundamentals(
    market: pd.DataFrame,
    fundamentals: pd.DataFrame,
    *,
    reporting_lag_days: int = 45,
) -> pd.DataFrame:
    """Backward as-of merge fundamentals after their true availability date.

    Rows lacking an explicit publication date use a conservative reporting lag and
    are marked as estimated. They must not be presented as verified point-in-time data.
    """

    required = {"symbol", "period_end"}
    if missing := required.difference(fundamentals.columns):
        raise ValueError(f"Fundamentals missing columns: {sorted(missing)}")
    facts = fundamentals.copy()
    facts["period_end"] = pd.to_datetime(facts["period_end"])
    if "publication_date" not in facts:
        facts["publication_date"] = pd.NaT
    facts["publication_date"] = pd.to_datetime(facts["publication_date"])
    facts["availability_estimated"] = facts["publication_date"].isna()
    facts["available_from"] = facts["publication_date"].fillna(
        facts["period_end"] + pd.to_timedelta(reporting_lag_days, unit="D")
    )
    left = market.sort_values(["symbol", "date"]).copy()
    right = facts.sort_values(["symbol", "available_from"]).copy()
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="available_from",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )
    violations = merged["available_from"].notna() & (merged["available_from"] > merged["date"])
    if violations.any():
        raise AssertionError("Point-in-time fundamental merge introduced future information")
    return merged


def _from_yfinance(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [str(item[0]) for item in output.columns]
    output = output.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    factor = pd.to_numeric(output.get("Adj Close"), errors="coerce") / pd.to_numeric(
        output["Close"], errors="coerce"
    )
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(output["date"]).dt.tz_localize(None),
            "raw_open": output["Open"],
            "raw_high": output["High"],
            "raw_low": output["Low"],
            "raw_close": output["Close"],
            "adj_open": output["Open"] * factor,
            "adj_high": output["High"] * factor,
            "adj_low": output["Low"] * factor,
            "adj_close": output["Adj Close"],
            "volume": output["Volume"],
        }
    )
    return _normalize(result)


def _read_legacy_clean_cache(symbol: str) -> pd.DataFrame:
    path = LEGACY_CLEAN_CACHE / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    mapping = {
        "Date": "date",
        "raw_Open": "raw_open",
        "raw_High": "raw_high",
        "raw_Low": "raw_low",
        "raw_Close": "raw_close",
        "adj_Open": "adj_open",
        "adj_High": "adj_high",
        "adj_Low": "adj_low",
        "adj_Close": "adj_close",
        "Volume": "volume",
    }
    available = {old: new for old, new in mapping.items() if old in frame.columns}
    return _normalize(frame.rename(columns=available)[list(available.values())])


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "volume",
    ]
    output = frame.copy()
    for column in required:
        if column not in output:
            output[column] = pd.NA
    output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.tz_localize(None)
    for column in required[1:]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["date", "adj_close"]).sort_values("date")
    output = output.drop_duplicates("date", keep="last")
    output["adjustment_factor"] = output["adj_close"] / output["raw_close"].replace(0, pd.NA)
    output["corporate_action_adjustment"] = output["adjustment_factor"].pct_change().abs().gt(1e-6)
    output["adjusted_return_1d"] = output["adj_close"].pct_change()
    output["is_abnormal_move"] = output["adjusted_return_1d"].abs().gt(0.20)
    normalized = output.reset_index(drop=True)
    normalized["symbol"] = "__single_symbol__"
    normalized = add_tradability_fields(normalized)
    return normalized.drop(columns="symbol")


def _audit_symbol(
    symbol: str, frame: pd.DataFrame, *, source: str, error: str
) -> dict[str, object]:
    if frame.empty:
        return {"symbol": symbol, "status": "FAIL", "source": source, "error": error}
    return {
        "symbol": symbol,
        "status": "PASS" if frame["is_tradable"].mean() >= 0.98 else "REVIEW",
        "source": source,
        "first_date": frame["date"].min(),
        "last_date": frame["date"].max(),
        "rows": len(frame),
        "tradable_rows": int(frame["is_tradable"].sum()),
        "zero_volume_rows": int(frame["is_zero_volume"].sum()),
        "stale_rows": int(frame["is_stale"].sum()),
        "missing_ohlcv_pct": float(
            frame[["adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
            .isna()
            .mean()
            .mean()
        ),
        "corporate_action_adjustments": int(frame["corporate_action_adjustment"].sum()),
        "abnormal_moves_over_20pct": int(frame["is_abnormal_move"].sum()),
        "error": error,
    }


def _to_yahoo_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if value.startswith("^") or value.endswith(".NS"):
        return value
    return f"{value}.NS"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")
