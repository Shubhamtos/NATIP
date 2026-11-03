"""Cross-sectional market-strength research built from NATIP adjusted price caches.

The module is deterministic and end-of-day only. It does not generate orders or
investment advice. All ranks are calculated across the eligible same-date universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketStrengthConfig:
    """Configuration for NATIP relative-strength and scan calculations."""

    return_windows: tuple[int, int, int, int] = (63, 126, 189, 252)
    return_weights: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)
    minimum_history: int = 253
    minimum_universe_coverage: float = 0.60
    max_staleness_days: int = 7
    near_high_threshold: float = -0.05
    near_breakout_threshold: float = -0.03
    volume_surge_ratio: float = 1.50


@dataclass(slots=True)
class MarketStrengthSnapshot:
    """Complete same-date market-strength research snapshot."""

    as_of_date: pd.Timestamp
    market_regime: str
    breadth: dict[str, float]
    stocks: pd.DataFrame
    sectors: pd.DataFrame
    quality: pd.DataFrame
    benchmark: pd.DataFrame
    histories: dict[str, pd.DataFrame]
    market: str = "NSE"
    currency: str = "INR"
    benchmark_symbol: str = "^NSEI"
    benchmark_name: str = "Nifty 50"


def load_cached_market_histories(
    symbols: list[str],
    *,
    cache_dir: Path,
    legacy_cache_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load adjusted OHLCV without network calls or cache mutation."""

    histories: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for symbol in dict.fromkeys(symbols):
        safe = _safe_symbol(symbol)
        parquet_path = cache_dir / f"{safe}.parquet"
        csv_path = legacy_cache_dir / f"{safe}.csv"
        source = "missing"
        frame = pd.DataFrame()
        try:
            if parquet_path.exists():
                frame = _normalize_cached_frame(pd.read_parquet(parquet_path))
                source = "backtesting_cache"
            elif csv_path.exists():
                frame = _normalize_cached_frame(pd.read_csv(csv_path))
                source = "clean_probability_cache"
        except Exception as exc:
            audits.append(
                {
                    "Ticker": symbol,
                    "Status": "FAIL",
                    "Source": source,
                    "Rows": 0,
                    "Last Date": pd.NaT,
                    "Issue": str(exc),
                }
            )
            continue
        if frame.empty:
            audits.append(
                {
                    "Ticker": symbol,
                    "Status": "FAIL",
                    "Source": source,
                    "Rows": 0,
                    "Last Date": pd.NaT,
                    "Issue": "No cached adjusted OHLCV",
                }
            )
            continue
        histories[symbol] = frame
        audits.append(
            {
                "Ticker": symbol,
                "Status": "PASS" if len(frame) >= 253 else "REVIEW",
                "Source": source,
                "Rows": len(frame),
                "Last Date": frame["Date"].max(),
                "Issue": "" if len(frame) >= 253 else "Less than 253 trading sessions",
            }
        )
    return histories, pd.DataFrame(audits)


def build_market_strength_snapshot(
    histories: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    *,
    benchmark_symbol: str = "^NSEI",
    config: MarketStrengthConfig | None = None,
    quality: pd.DataFrame | None = None,
    market: str = "NSE",
) -> MarketStrengthSnapshot:
    """Calculate same-date stock ranks, breadth, sectors and market regime."""

    settings = config or MarketStrengthConfig()
    benchmark = histories.get(benchmark_symbol, pd.DataFrame())
    if benchmark.empty:
        raise ValueError(f"Benchmark cache is unavailable for {benchmark_symbol}")
    eligible_histories = {
        symbol: frame
        for symbol, frame in histories.items()
        if symbol != benchmark_symbol and len(frame) >= settings.minimum_history
    }
    if not eligible_histories:
        raise ValueError("No stocks have enough adjusted price history for market-strength ranks")
    as_of = _select_common_as_of(eligible_histories, benchmark, settings.minimum_universe_coverage)
    current = _cross_section(eligible_histories, as_of, settings)
    if current.empty:
        raise ValueError(f"No same-date stock observations were available for {as_of.date()}")

    benchmark_dates = benchmark.loc[benchmark["Date"] <= as_of, "Date"].drop_duplicates()
    rank_dates = {
        "rs_rating_5d_ago": _date_offset(benchmark_dates, 5),
        "rs_rating_20d_ago": _date_offset(benchmark_dates, 20),
    }
    for column, rank_date in rank_dates.items():
        historical = _cross_section(eligible_histories, rank_date, settings)
        mapping = historical.set_index("Ticker")["RS Rating"] if not historical.empty else {}
        current[column] = current["Ticker"].map(mapping)
        if column == "rs_rating_5d_ago":
            stage_mapping = historical.set_index("Ticker")["Stage"] if not historical.empty else {}
            current["stage_5d_ago"] = current["Ticker"].map(stage_mapping)
    current["RS Change 5D"] = current["RS Rating"] - current["rs_rating_5d_ago"]
    current["RS Change 20D"] = current["RS Rating"] - current["rs_rating_20d_ago"]
    current["RS Line New High"] = current["Ticker"].map(
        {
            ticker: _rs_line_new_high(history, benchmark, as_of)
            for ticker, history in eligible_histories.items()
        }
    ).fillna(False)

    metadata = _normalize_universe(universe)
    current = current.merge(metadata, on="Ticker", how="left")
    current["Company"] = current["Company"].fillna(current["Ticker"])
    current["Sector"] = current["Sector"].fillna("Unknown")
    current["Freshness"] = (as_of - pd.to_datetime(current["Data Date"])).dt.days
    current["Data Status"] = np.where(
        current["Freshness"].le(settings.max_staleness_days), "PASS", "STALE"
    )
    current = current.sort_values(["RS Rating", "RS Composite"], ascending=False).reset_index(
        drop=True
    )
    current.insert(0, "Rank", range(1, len(current) + 1))

    breadth = _market_breadth(current)
    market_regime = _market_regime(benchmark, as_of, breadth)
    sectors = _sector_summary(current)
    return MarketStrengthSnapshot(
        as_of_date=as_of,
        market_regime=market_regime,
        breadth=breadth,
        stocks=current,
        sectors=sectors,
        quality=quality.copy() if quality is not None else pd.DataFrame(),
        benchmark=benchmark.loc[benchmark["Date"] <= as_of].copy(),
        histories=eligible_histories,
        market=market,
        currency="USD" if market == "US" else "INR",
        benchmark_symbol=benchmark_symbol,
        benchmark_name="S&P 500" if benchmark_symbol == "^GSPC" else
                       "Nifty 50" if benchmark_symbol == "^NSEI" else benchmark_symbol,
    )


def apply_ready_scan(stocks: pd.DataFrame, scan_name: str) -> pd.DataFrame:
    """Apply a transparent predefined scan to a market-strength table."""

    if stocks.empty:
        return stocks.copy()
    rules = {
        "RS Leaders": stocks["RS Rating"].ge(90),
        "Stage 2 Leaders": stocks["Stage"].eq("Stage 2") & stocks["RS Rating"].ge(70),
        "Trend Template Leaders": stocks.get(
            "Trend Template", pd.Series(False, index=stocks.index)
        ).fillna(False),
        "Just Entered Stage 2": stocks["Stage"].eq("Stage 2")
        & stocks.get("stage_5d_ago", pd.Series("", index=stocks.index)).ne("Stage 2"),
        "RS Line New Highs": stocks.get(
            "RS Line New High", pd.Series(False, index=stocks.index)
        ).fillna(False),
        "Near 52-Week High": stocks["Distance 52W High"].ge(-0.05),
        "Near Breakout": stocks["Near Breakout"],
        "Confirmed Breakout": stocks["Breakout"],
        "Pocket Pivot": stocks["Pocket Pivot"],
        "Volume Surge": stocks["Volume Ratio"].ge(1.50) & stocks["1D Return"].gt(0),
        "Strong in Weak Market": stocks["RS Rating"].ge(85) & stocks["1M Return"].gt(0),
        "Laggards": stocks["RS Rating"].le(20) & stocks["Stage"].eq("Stage 4"),
    }
    if scan_name not in rules:
        raise ValueError(f"Unknown ready scan: {scan_name}")
    return stocks.loc[rules[scan_name]].copy().sort_values(
        ["RS Rating", "RS Composite"], ascending=False
    )


def breakout_candidates(stocks: pd.DataFrame) -> pd.DataFrame:
    """Return one row per breakout candidate with all matching setup labels."""

    if stocks.empty:
        return stocks.copy()
    setup_masks = {
        "Confirmed breakout": stocks["Breakout"].fillna(False),
        "Near breakout": stocks["Near Breakout"].fillna(False),
        "Pocket pivot": stocks["Pocket Pivot"].fillna(False),
        "Volume surge": stocks["Volume Ratio"].ge(1.50) & stocks["1D Return"].gt(0),
    }
    labels: dict[str, list[str]] = {}
    for setup, mask in setup_masks.items():
        for ticker in stocks.loc[mask, "Ticker"].astype(str):
            labels.setdefault(ticker, []).append(setup)
    if not labels:
        return stocks.iloc[0:0].assign(Setup=pd.Series(dtype="object"))
    result = stocks[stocks["Ticker"].astype(str).isin(labels)].copy()
    result["Setup"] = result["Ticker"].astype(str).map(
        lambda ticker: " · ".join(labels.get(ticker, []))
    )
    return result.sort_values(
        ["Breakout", "Near Breakout", "RS Rating", "Volume Ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def market_strength_alerts(stocks: pd.DataFrame, watchlist: set[str]) -> pd.DataFrame:
    """Return deterministic current alerts for watched tickers."""

    if stocks.empty or not watchlist:
        return pd.DataFrame(columns=["Ticker", "Alert", "Evidence", "Severity"])
    rows: list[dict[str, str]] = []
    selected = stocks[stocks["Ticker"].isin(watchlist)]
    for row in selected.to_dict(orient="records"):
        ticker = str(row["Ticker"])
        if float(row["RS Rating"]) >= 90:
            rows.append(
                {
                    "Ticker": ticker,
                    "Alert": "RS leader",
                    "Evidence": f"RS rating {float(row['RS Rating']):.0f}/99",
                    "Severity": "Positive",
                }
            )
        if bool(row["Near Breakout"]):
            rows.append(
                {
                    "Ticker": ticker,
                    "Alert": "Near breakout",
                    "Evidence": f"{float(row['Distance to Pivot']):+.2%} from prior 20D high",
                    "Severity": "Watch",
                }
            )
        if bool(row["Breakout"]):
            rows.append(
                {
                    "Ticker": ticker,
                    "Alert": "Price-volume breakout",
                    "Evidence": f"Volume {float(row['Volume Ratio']):.2f}x 20D average",
                    "Severity": "Positive",
                }
            )
        if float(row["RS Change 5D"] if pd.notna(row["RS Change 5D"]) else 0) <= -15:
            rows.append(
                {
                    "Ticker": ticker,
                    "Alert": "RS deterioration",
                    "Evidence": f"RS fell {abs(float(row['RS Change 5D'])):.0f} points in 5D",
                    "Severity": "Caution",
                }
            )
    return pd.DataFrame(rows)


def normalized_comparison(
    histories: dict[str, pd.DataFrame], symbols: list[str], *, sessions: int = 126
) -> pd.DataFrame:
    """Return a normalized comparison series for selected symbols."""

    series: list[pd.DataFrame] = []
    for symbol in symbols:
        frame = histories.get(symbol)
        if frame is None or frame.empty:
            continue
        values = frame[["Date", "Close"]].dropna().tail(sessions).copy()
        if values.empty or float(values["Close"].iloc[0]) == 0:
            continue
        values["Normalized"] = values["Close"] / float(values["Close"].iloc[0]) * 100.0
        values["Ticker"] = symbol
        series.append(values[["Date", "Ticker", "Normalized"]])
    return pd.concat(series, ignore_index=True) if series else pd.DataFrame()


def _cross_section(
    histories: dict[str, pd.DataFrame], as_of: pd.Timestamp, config: MarketStrengthConfig
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for symbol, history in histories.items():
        frame = history.loc[history["Date"] <= as_of].copy()
        if len(frame) < config.minimum_history or as_of not in set(frame["Date"]):
            continue
        record = _stock_record(symbol, frame, config)
        if record is not None:
            records.append(record)
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["RS Rating"] = (
        result["RS Composite"].rank(method="average", pct=True).mul(98).add(1).round()
    ).clip(1, 99)
    result["Trend Checks"] = (
        result[
            [
                "Price Above Long MAs",
                "SMA150 Above SMA200",
                "SMA200 Rising",
                "SMA50 Above Long MAs",
                "Price Above SMA50",
                "Above 52W Low Threshold",
                "Near 52W High Threshold",
            ]
        ]
        .fillna(False)
        .sum(axis=1)
        .add(result["RS Rating"].ge(70).astype(int))
        .astype(int)
    )
    return result


def _stock_record(
    symbol: str, frame: pd.DataFrame, config: MarketStrengthConfig
) -> dict[str, Any] | None:
    close = pd.to_numeric(frame["Close"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce")
    if close.isna().iloc[-1] or len(close.dropna()) < config.minimum_history:
        return None
    windows = config.return_windows
    returns = [_lag_return(close, window) for window in windows]
    if any(pd.isna(value) for value in returns):
        return None
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma150 = float(close.rolling(150).mean().iloc[-1])
    sma200_series = close.rolling(200).mean()
    sma200 = float(sma200_series.iloc[-1])
    sma200_prior = float(sma200_series.iloc[-21]) if len(sma200_series) >= 221 else np.nan
    latest = float(close.iloc[-1])
    prior_high20 = float(high.shift(1).rolling(20).max().iloc[-1])
    high52 = float(high.rolling(min(252, len(high))).max().iloc[-1])
    low52 = float(low.rolling(min(252, len(low))).min().iloc[-1])
    avg_volume20 = float(volume.rolling(20).mean().iloc[-1])
    latest_volume = float(volume.iloc[-1]) if pd.notna(volume.iloc[-1]) else np.nan
    volume_ratio = latest_volume / avg_volume20 if avg_volume20 > 0 else np.nan
    down_days = close.diff().lt(0)
    max_down_volume10 = float(volume.where(down_days).shift(1).rolling(10).max().iloc[-1])
    one_day_return = float(close.pct_change().iloc[-1])
    trend_template = bool(
        latest > sma50 > sma150 > sma200
        and pd.notna(sma200_prior)
        and sma200 > sma200_prior
        and latest >= high52 * 0.75
        and latest >= low52 * 1.25
    )
    price_above_long_mas = bool(latest > sma150 and latest > sma200)
    sma150_above_sma200 = bool(sma150 > sma200)
    sma200_rising = bool(pd.notna(sma200_prior) and sma200 > sma200_prior)
    sma50_above_long_mas = bool(sma50 > sma150 and sma50 > sma200)
    price_above_sma50 = bool(latest > sma50)
    above_52w_low_threshold = bool(latest >= low52 * 1.30)
    near_52w_high_threshold = bool(latest >= high52 * 0.75)
    stage = _stage(latest, sma50, sma150, sma200, sma200_prior)
    breakout = bool(latest > prior_high20 and volume_ratio >= config.volume_surge_ratio)
    distance_to_pivot = latest / prior_high20 - 1 if prior_high20 > 0 else np.nan
    near_breakout = bool(-abs(config.near_breakout_threshold) <= distance_to_pivot <= 0)
    pocket_pivot = bool(
        one_day_return > 0
        and latest > sma50
        and pd.notna(max_down_volume10)
        and latest_volume > max_down_volume10
    )
    composite = float(
        sum(
            value * weight
            for value, weight in zip(returns, config.return_weights, strict=True)
        )
    )
    return {
        "Ticker": symbol,
        "Data Date": frame["Date"].iloc[-1],
        "Close": latest,
        "1D Return": one_day_return,
        "1M Return": _lag_return(close, 21),
        "3M Return": returns[0],
        "6M Return": returns[1],
        "9M Return": returns[2],
        "12M Return": returns[3],
        "RS Composite": composite,
        "Stage": stage,
        "Trend Template": trend_template,
        "Distance 52W High": latest / high52 - 1 if high52 > 0 else np.nan,
        "Distance 52W Low": latest / low52 - 1 if low52 > 0 else np.nan,
        "Above SMA20": latest > sma20,
        "Above SMA50": latest > sma50,
        "Above SMA200": latest > sma200,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA150": sma150,
        "SMA200": sma200,
        "Price Above Long MAs": price_above_long_mas,
        "SMA150 Above SMA200": sma150_above_sma200,
        "SMA200 Rising": sma200_rising,
        "SMA50 Above Long MAs": sma50_above_long_mas,
        "Price Above SMA50": price_above_sma50,
        "Above 52W Low Threshold": above_52w_low_threshold,
        "Near 52W High Threshold": near_52w_high_threshold,
        "Pivot": prior_high20,
        "Distance to Pivot": distance_to_pivot,
        "Near Breakout": near_breakout,
        "Breakout": breakout,
        "Pocket Pivot": pocket_pivot,
        "Volume Ratio": volume_ratio,
        "Avg Traded Value 20D": float((close * volume).rolling(20).mean().iloc[-1]),
    }


def _normalize_cached_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "date": "Date",
        "adj_open": "Open",
        "adj_high": "High",
        "adj_low": "Low",
        "adj_close": "Close",
        "volume": "Volume",
        "adj_Open": "Open",
        "adj_High": "High",
        "adj_Low": "Low",
        "adj_Close": "Close",
    }
    output = frame.rename(columns={key: value for key, value in mapping.items() if key in frame})
    for column in ["Date", "Open", "High", "Low", "Close", "Volume"]:
        if column not in output:
            output[column] = pd.NA
    output = output[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    output["Date"] = pd.to_datetime(output["Date"], errors="coerce").dt.tz_localize(None)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return (
        output.dropna(subset=["Date", "Close"])
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    rename = {
        "symbol": "Ticker",
        "ticker": "Ticker",
        "Ticker": "Ticker",
        "company": "Company",
        "name": "Company",
        "sector": "Sector",
    }
    frame = frame.rename(columns={column: rename[column] for column in frame if column in rename})
    if "Ticker" not in frame:
        raise ValueError("Universe must contain a ticker/symbol column")
    if "Company" not in frame:
        frame["Company"] = frame["Ticker"]
    if "Sector" not in frame:
        frame["Sector"] = "Unknown"
    return frame[["Ticker", "Company", "Sector"]].drop_duplicates("Ticker")


def _select_common_as_of(
    histories: dict[str, pd.DataFrame], benchmark: pd.DataFrame, minimum_coverage: float
) -> pd.Timestamp:
    benchmark_dates = set(pd.to_datetime(benchmark["Date"]))
    counts: dict[pd.Timestamp, int] = {}
    for frame in histories.values():
        for value in pd.to_datetime(frame["Date"].tail(10)):
            if value in benchmark_dates:
                counts[value] = counts.get(value, 0) + 1
    required = max(1, int(np.ceil(len(histories) * minimum_coverage)))
    candidates = [date for date, count in counts.items() if count >= required]
    if not candidates:
        raise ValueError("No recent date meets minimum same-date universe coverage")
    return max(candidates)


def _market_breadth(stocks: pd.DataFrame) -> dict[str, float]:
    return {
        "eligible_stocks": float(len(stocks)),
        "above_sma20_pct": float(stocks["Above SMA20"].mean() * 100),
        "above_sma50_pct": float(stocks["Above SMA50"].mean() * 100),
        "above_sma200_pct": float(stocks["Above SMA200"].mean() * 100),
        "advancing_pct": float(stocks["1D Return"].gt(0).mean() * 100),
        "stage2_pct": float(stocks["Stage"].eq("Stage 2").mean() * 100),
        "near_high_pct": float(stocks["Distance 52W High"].ge(-0.05).mean() * 100),
    }


def _market_regime(
    benchmark: pd.DataFrame, as_of: pd.Timestamp, breadth: dict[str, float]
) -> str:
    frame = benchmark.loc[benchmark["Date"] <= as_of]
    close = pd.to_numeric(frame["Close"], errors="coerce")
    latest = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1])
    breadth50 = breadth["above_sma50_pct"]
    if latest > sma50 > sma200 and breadth50 >= 55:
        return "BULLISH PARTICIPATION"
    if latest < sma50 < sma200 and breadth50 <= 40:
        return "BEARISH PARTICIPATION"
    if latest > sma200 and breadth50 < 45:
        return "NARROW / CAUTIOUS"
    return "SIDEWAYS / MIXED"


def _sector_summary(stocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sector, group in stocks.groupby("Sector", dropna=False):
        leaders = group[group["RS Rating"] >= 80]
        rows.append(
            {
                "Sector": sector,
                "Stocks": len(group),
                "Sector RS": float(group["RS Rating"].mean()),
                "Median RS": float(group["RS Rating"].median()),
                "Breadth50": float(group["Above SMA50"].mean() * 100),
                "Stage 2 %": float(group["Stage"].eq("Stage 2").mean() * 100),
                "1M Return": float(group["1M Return"].median()),
                "3M Return": float(group["3M Return"].median()),
                "RS Momentum": float(group["RS Change 20D"].median()),
                "Leaders": len(leaders),
                "Top Stock": str(group.nlargest(1, "RS Rating")["Ticker"].iloc[0]),
            }
        )
    result = pd.DataFrame(rows).sort_values(["Sector RS", "Breadth50"], ascending=False)
    result.insert(0, "Sector Rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def _stage(close: float, sma50: float, sma150: float, sma200: float, prior200: float) -> str:
    rising200 = pd.notna(prior200) and sma200 > prior200
    falling200 = pd.notna(prior200) and sma200 < prior200
    if close > sma50 > sma150 > sma200 and rising200:
        return "Stage 2"
    if close < sma50 < sma150 < sma200 and falling200:
        return "Stage 4"
    if close >= sma200 and sma50 >= sma200:
        return "Stage 1"
    return "Stage 3"


def _lag_return(values: pd.Series, periods: int) -> float:
    if len(values) <= periods:
        return np.nan
    prior = float(values.iloc[-periods - 1])
    return float(values.iloc[-1] / prior - 1) if prior else np.nan


def _date_offset(dates: pd.Series, sessions: int) -> pd.Timestamp:
    ordered = (
        pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values().reset_index(drop=True)
    )
    if len(ordered) <= sessions:
        return pd.Timestamp(ordered.iloc[0])
    return pd.Timestamp(ordered.iloc[-sessions - 1])


def _rs_line_new_high(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    lookback: int = 252,
) -> bool:
    """Return whether the stock/Nifty relative-strength line makes a trailing high."""

    stock_close = stock.loc[stock["Date"] <= as_of, ["Date", "Close"]].rename(
        columns={"Close": "stock_close"}
    )
    benchmark_close = benchmark.loc[
        benchmark["Date"] <= as_of, ["Date", "Close"]
    ].rename(columns={"Close": "benchmark_close"})
    aligned = stock_close.merge(benchmark_close, on="Date", how="inner").tail(lookback + 1)
    if len(aligned) < 127:
        return False
    ratio = pd.to_numeric(aligned["stock_close"], errors="coerce") / pd.to_numeric(
        aligned["benchmark_close"], errors="coerce"
    )
    prior_high = ratio.shift(1).max()
    return bool(pd.notna(prior_high) and float(ratio.iloc[-1]) >= float(prior_high))


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")
