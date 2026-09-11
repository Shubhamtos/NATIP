"""Backward-looking feature engineering for stock outperformance prediction."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "ret_252d",
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "price_to_sma_20",
    "price_to_sma_50",
    "price_to_sma_100",
    "price_to_sma_200",
    "ema_20",
    "ema_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    "atr_14",
    "volatility_20d",
    "volatility_60d",
    "bb_width_20",
    "bb_percent_b_20",
    "volume_to_avg20",
    "distance_52w_high",
    "distance_52w_low",
    "rs_vs_nifty_20d",
    "rs_vs_nifty_60d",
    "rs_vs_nifty_120d",
    "rs_vs_nifty_252d",
]

SECTOR_RELATIVE_FEATURE_COLUMNS = [
    "stock_vs_sector_ret_20d",
    "stock_vs_sector_ret_60d",
    "stock_vs_sector_ret_120d",
    "stock_vs_sector_ret_252d",
    "stock_volatility_relative_sector_20d",
    "stock_volatility_relative_sector_60d",
    "stock_distance_52w_high_relative_sector",
    "sector_momentum_20d",
    "sector_momentum_60d",
    "sector_momentum_120d",
    "sector_strength_vs_nifty_20d",
    "sector_strength_vs_nifty_60d",
    "sector_strength_vs_nifty_120d",
]

MARKET_REGIME_FEATURE_COLUMNS = [
    "nifty_above_sma20",
    "nifty_above_sma50",
    "nifty_above_sma200",
    "nifty_sma50_sma200_trend",
    "nifty_ret_20d",
    "nifty_ret_60d",
    "nifty_ret_120d",
    "nifty_rsi_14",
    "nifty_adx_14",
    "nifty_atr_14",
    "nifty_atr_pct_14",
    "nifty_volatility_20d",
    "nifty_volatility_60d",
    "nifty_distance_52w_high",
    "market_regime_bull_trend",
    "market_regime_bear_trend",
    "market_regime_sideways",
    "market_regime_high_volatility",
    "market_breadth_above_sma50",
    "market_breadth_above_sma200",
    "market_breadth_positive_20d_return",
    "market_breadth_advance_decline_ratio",
]

ADVANCED_FEATURE_COLUMNS = [
    "momentum_20d_vs_60d",
    "momentum_60d_vs_120d",
    "momentum_20d_change",
    "relative_momentum_20d_change",
    "relative_momentum_20d_vs_60d",
    "up_down_volume_ratio_20d",
    "price_return_x_volume_ratio",
    "volume_trend_20d",
    "positive_price_volume_confirmation",
    "volatility_20d_to_60d",
    "atr_14_to_atr_50",
    "bb_width_to_avg_120d",
    "volatility_20d_change",
    "distance_prev_20d_high",
    "distance_prev_50d_high",
    "distance_prev_100d_high",
    "breakout_prev_20d_high",
    "breakout_prev_50d_high",
    "breakout_prev_100d_high",
    "sector_above_sma50",
    "sector_above_sma200",
    "sector_rsi_14",
    "sector_volatility_120d",
    "sector_regime_bull",
    "sector_regime_bear",
    "sector_regime_sideways",
    "beta_60d",
    "beta_120d",
    "beta_252d",
    "corr_nifty_20d",
    "corr_nifty_60d",
    "corr_nifty_120d",
]

ALL_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *SECTOR_RELATIVE_FEATURE_COLUMNS,
    *MARKET_REGIME_FEATURE_COLUMNS,
    *ADVANCED_FEATURE_COLUMNS,
]


def build_features_for_symbol(
    symbol: str,
    stock_frame: pd.DataFrame,
    nifty_frame: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    sector_frames: dict[str, pd.DataFrame] | None = None,
    market_regime_frame: pd.DataFrame | None = None,
    sector_fallback_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build backward-looking features for one stock.

    Args:
        symbol: Stock symbol.
        stock_frame: Daily OHLCV data.
        nifty_frame: Nifty benchmark OHLCV data.

    Returns:
        DataFrame with one row per stock/date and no future-looking columns.
    """

    frame = stock_frame.copy().sort_values("Date")
    frame["symbol"] = symbol
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    volume = frame["Volume"]

    for window in (1, 5, 10, 20, 60, 120, 252):
        frame[f"ret_{window}d"] = close.pct_change(window)

    for window in (20, 50, 100, 200):
        frame[f"sma_{window}"] = close.rolling(window).mean()
        frame[f"price_to_sma_{window}"] = close / frame[f"sma_{window}"] - 1

    frame["ema_20"] = close.ewm(span=20, adjust=False).mean()
    frame["ema_50"] = close.ewm(span=50, adjust=False).mean()
    frame["rsi_14"] = _rsi(close, 14)
    frame["macd"], frame["macd_signal"] = _macd(close)
    frame["atr_14"] = _atr(high, low, close, 14)
    frame["atr_50"] = _atr(high, low, close, 50)
    frame["plus_di_14"], frame["minus_di_14"], frame["adx_14"] = _adx(high, low, close, 14)

    daily_returns = close.pct_change()
    frame["volatility_20d"] = daily_returns.rolling(20).std()
    frame["volatility_60d"] = daily_returns.rolling(60).std()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    frame["bb_width_20"] = (bb_upper - bb_lower) / bb_mid
    frame["bb_percent_b_20"] = (close - bb_lower) / (bb_upper - bb_lower)
    frame["volume_to_avg20"] = volume / volume.rolling(20).mean()

    frame["momentum_20d_vs_60d"] = frame["ret_20d"] - frame["ret_60d"]
    frame["momentum_60d_vs_120d"] = frame["ret_60d"] - frame["ret_120d"]
    frame["momentum_20d_change"] = frame["ret_20d"] - frame["ret_20d"].shift(20)
    up_volume = volume.where(daily_returns > 0, 0).rolling(20).sum()
    down_volume = volume.where(daily_returns < 0, 0).rolling(20).sum()
    frame["up_down_volume_ratio_20d"] = up_volume / down_volume.replace(0, pd.NA)
    frame["price_return_x_volume_ratio"] = frame["ret_20d"] * frame["volume_to_avg20"]
    frame["volume_trend_20d"] = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
    frame["positive_price_volume_confirmation"] = (
        (frame["ret_20d"] > 0) & (frame["volume_to_avg20"] > 1)
    ).astype(float)
    frame["volatility_20d_to_60d"] = frame["volatility_20d"] / frame["volatility_60d"]
    frame["atr_14_to_atr_50"] = frame["atr_14"] / frame["atr_50"]
    frame["bb_width_to_avg_120d"] = frame["bb_width_20"] / frame["bb_width_20"].rolling(120).mean()
    frame["volatility_20d_change"] = frame["volatility_20d"] - frame["volatility_20d"].shift(20)
    for window in (20, 50, 100):
        prior_high = high.shift(1).rolling(window).max()
        frame[f"distance_prev_{window}d_high"] = close / prior_high - 1
        frame[f"breakout_prev_{window}d_high"] = (close > prior_high).astype(float)

    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    frame["distance_52w_high"] = close / high_52w - 1
    frame["distance_52w_low"] = close / low_52w - 1

    nifty = nifty_frame[["Date", "Close"]].rename(columns={"Close": "nifty_close"})
    frame = frame.merge(nifty, on="Date", how="left")
    for window in (20, 60, 120, 252):
        stock_return = close.pct_change(window)
        nifty_return = frame["nifty_close"].pct_change(window)
        frame[f"rs_vs_nifty_{window}d"] = stock_return - nifty_return
    frame["relative_momentum_20d_change"] = (
        frame["rs_vs_nifty_20d"] - frame["rs_vs_nifty_20d"].shift(20)
    )
    frame["relative_momentum_20d_vs_60d"] = (
        frame["rs_vs_nifty_20d"] - frame["rs_vs_nifty_60d"]
    )
    nifty_returns = frame["nifty_close"].pct_change()
    for window in (60, 120, 252):
        covariance = daily_returns.rolling(window).cov(nifty_returns)
        benchmark_variance = nifty_returns.rolling(window).var()
        frame[f"beta_{window}d"] = covariance / benchmark_variance.replace(0, pd.NA)
    for window in (20, 60, 120):
        frame[f"corr_nifty_{window}d"] = daily_returns.rolling(window).corr(nifty_returns)

    sector = _sector_for_symbol(symbol, metadata)
    frame = _add_sector_relative_features(
        frame=frame,
        sector=sector,
        sector_frames=sector_frames or {},
        sector_fallback_frame=sector_fallback_frame,
    )
    if market_regime_frame is not None and not market_regime_frame.empty:
        frame = frame.merge(market_regime_frame, on="Date", how="left")

    for column in ALL_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame[["Date", "symbol", "Close", *ALL_FEATURE_COLUMNS]]
        .dropna(subset=FEATURE_COLUMNS)
        .reset_index(drop=True)
    )


def build_feature_dataset(
    data: dict[str, pd.DataFrame],
    *,
    benchmark_symbol: str = "^NSEI",
    metadata: pd.DataFrame | None = None,
    sector_frames: dict[str, pd.DataFrame] | None = None,
    include_sector_features: bool = False,
    include_market_regime_features: bool = False,
) -> pd.DataFrame:
    """Build features for every stock in a universe."""

    if benchmark_symbol not in data:
        raise ValueError(f"Benchmark data missing: {benchmark_symbol}")
    nifty_frame = data[benchmark_symbol]
    sector_fallback_frame = (
        _build_sector_fallback_frame(data, metadata) if metadata is not None else None
    )
    market_regime_frame = (
        build_market_regime_features(nifty_frame, data, benchmark_symbol=benchmark_symbol)
        if include_market_regime_features
        else None
    )
    frames = [
        build_features_for_symbol(
            symbol,
            frame,
            nifty_frame,
            metadata=metadata,
            sector_frames=sector_frames if include_sector_features else None,
            market_regime_frame=market_regime_frame,
            sector_fallback_frame=sector_fallback_frame if include_sector_features else None,
        )
        for symbol, frame in data.items()
        if symbol != benchmark_symbol and not frame.empty
    ]
    if not frames:
        return pd.DataFrame(columns=["Date", "symbol", "Close", *FEATURE_COLUMNS])
    return pd.concat(frames, ignore_index=True).sort_values(["Date", "symbol"])


def feature_columns_for_variant(variant: str) -> list[str]:
    """Return model feature columns for a feature variant."""

    if variant == "baseline":
        return FEATURE_COLUMNS
    if variant == "sector":
        return [*FEATURE_COLUMNS, *SECTOR_RELATIVE_FEATURE_COLUMNS]
    if variant == "sector_regime":
        return [*FEATURE_COLUMNS, *SECTOR_RELATIVE_FEATURE_COLUMNS, *MARKET_REGIME_FEATURE_COLUMNS]
    if variant == "advanced":
        return [
            *FEATURE_COLUMNS,
            *SECTOR_RELATIVE_FEATURE_COLUMNS,
            *MARKET_REGIME_FEATURE_COLUMNS,
            *ADVANCED_FEATURE_COLUMNS,
        ]
    raise ValueError(f"Unknown feature variant: {variant}")


def build_market_regime_features(
    nifty_frame: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    *,
    benchmark_symbol: str = "^NSEI",
) -> pd.DataFrame:
    """Build backward-looking Nifty market-regime features."""

    frame = nifty_frame.copy().sort_values("Date")
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    frame["nifty_above_sma20"] = (close > sma20).astype(float)
    frame["nifty_above_sma50"] = (close > sma50).astype(float)
    frame["nifty_above_sma200"] = (close > sma200).astype(float)
    frame["nifty_sma50_sma200_trend"] = sma50 / sma200 - 1
    for window in (20, 60, 120):
        frame[f"nifty_ret_{window}d"] = close.pct_change(window)
    frame["nifty_rsi_14"] = _rsi(close, 14)
    frame["nifty_atr_14"] = _atr(high, low, close, 14)
    frame["nifty_atr_pct_14"] = frame["nifty_atr_14"] / close
    _, _, frame["nifty_adx_14"] = _adx(high, low, close, 14)
    returns = close.pct_change()
    frame["nifty_volatility_20d"] = returns.rolling(20).std()
    frame["nifty_volatility_60d"] = returns.rolling(60).std()
    frame["nifty_distance_52w_high"] = close / high.rolling(252).max() - 1
    high_vol_threshold = frame["nifty_volatility_60d"].rolling(252).quantile(0.75)
    frame["market_regime_bull_trend"] = (
        (close > sma50) & (sma50 > sma200) & (frame["nifty_ret_60d"] > 0)
    ).astype(float)
    frame["market_regime_bear_trend"] = (
        (close < sma50) & (sma50 < sma200) & (frame["nifty_ret_60d"] < 0)
    ).astype(float)
    frame["market_regime_high_volatility"] = (
        frame["nifty_volatility_60d"] > high_vol_threshold
    ).astype(float)
    frame["market_regime_sideways"] = (
        (frame["market_regime_bull_trend"] == 0)
        & (frame["market_regime_bear_trend"] == 0)
        & (frame["market_regime_high_volatility"] == 0)
    ).astype(float)
    breadth = _market_breadth(data, benchmark_symbol=benchmark_symbol)
    frame = frame.merge(breadth, on="Date", how="left")
    for column in MARKET_REGIME_FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["Date", *MARKET_REGIME_FEATURE_COLUMNS]]


def _add_sector_relative_features(
    *,
    frame: pd.DataFrame,
    sector: str | None,
    sector_frames: dict[str, pd.DataFrame],
    sector_fallback_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add sector-relative features to a stock feature frame."""

    output = frame.copy()
    sector_frame = _sector_frame_for(sector, sector_frames, sector_fallback_frame)
    if sector_frame is None or sector_frame.empty:
        return output
    output = output.merge(sector_frame, on="Date", how="left")
    for window in (20, 60, 120, 252):
        output[f"stock_vs_sector_ret_{window}d"] = (
            output[f"ret_{window}d"] - output[f"sector_ret_{window}d"]
        )
    output["stock_volatility_relative_sector_20d"] = (
        output["volatility_20d"] / output["sector_volatility_20d"] - 1
    )
    output["stock_volatility_relative_sector_60d"] = (
        output["volatility_60d"] / output["sector_volatility_60d"] - 1
    )
    output["stock_distance_52w_high_relative_sector"] = (
        output["distance_52w_high"] - output["sector_distance_52w_high"]
    )
    output["sector_momentum_20d"] = output["sector_ret_20d"]
    output["sector_momentum_60d"] = output["sector_ret_60d"]
    output["sector_momentum_120d"] = output["sector_ret_120d"]
    output["sector_strength_vs_nifty_20d"] = output["sector_ret_20d"] - output["ret_20d"].sub(
        output["rs_vs_nifty_20d"]
    )
    output["sector_strength_vs_nifty_60d"] = output["sector_ret_60d"] - output["ret_60d"].sub(
        output["rs_vs_nifty_60d"]
    )
    output["sector_strength_vs_nifty_120d"] = output["sector_ret_120d"] - output["ret_120d"].sub(
        output["rs_vs_nifty_120d"]
    )
    return output


def _sector_frame_for(
    sector: str | None,
    sector_frames: dict[str, pd.DataFrame],
    sector_fallback_frame: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return prepared sector feature frame."""

    if sector is None:
        return None
    if sector in sector_frames and not sector_frames[sector].empty:
        return _sector_features_from_ohlcv(sector_frames[sector])
    if sector_fallback_frame is None or sector_fallback_frame.empty:
        return None
    fallback = sector_fallback_frame[sector_fallback_frame["Sector"] == sector].copy()
    if fallback.empty:
        return None
    return _sector_features_from_close(fallback[["Date", "sector_close"]])


def _sector_features_from_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Build sector feature frame from OHLCV benchmark data."""

    clean = frame.copy().sort_values("Date")
    clean = clean.rename(
        columns={"Close": "sector_close", "High": "sector_high", "Low": "sector_low"}
    )
    return _sector_features_from_close(clean[["Date", "sector_close", "sector_high", "sector_low"]])


def _sector_features_from_close(frame: pd.DataFrame) -> pd.DataFrame:
    """Build sector feature frame from a sector close series."""

    output = frame.copy().sort_values("Date")
    close = output["sector_close"]
    high = output["sector_high"] if "sector_high" in output else close
    returns = close.pct_change()
    for window in (20, 60, 120, 252):
        output[f"sector_ret_{window}d"] = close.pct_change(window)
    output["sector_volatility_20d"] = returns.rolling(20).std()
    output["sector_volatility_60d"] = returns.rolling(60).std()
    output["sector_volatility_120d"] = returns.rolling(120).std()
    output["sector_distance_52w_high"] = close / high.rolling(252).max() - 1
    sector_sma50 = close.rolling(50).mean()
    sector_sma200 = close.rolling(200).mean()
    output["sector_above_sma50"] = (close > sector_sma50).astype(float)
    output["sector_above_sma200"] = (close > sector_sma200).astype(float)
    output["sector_rsi_14"] = _rsi(close, 14)
    output["sector_regime_bull"] = (
        (close > sector_sma50) & (sector_sma50 > sector_sma200) & (output["sector_ret_60d"] > 0)
    ).astype(float)
    output["sector_regime_bear"] = (
        (close < sector_sma50) & (sector_sma50 < sector_sma200) & (output["sector_ret_60d"] < 0)
    ).astype(float)
    output["sector_regime_sideways"] = (
        (output["sector_regime_bull"] == 0) & (output["sector_regime_bear"] == 0)
    ).astype(float)
    return output[
        [
            "Date",
            "sector_ret_20d",
            "sector_ret_60d",
            "sector_ret_120d",
            "sector_ret_252d",
            "sector_volatility_20d",
            "sector_volatility_60d",
            "sector_distance_52w_high",
            "sector_above_sma50",
            "sector_above_sma200",
            "sector_rsi_14",
            "sector_volatility_120d",
            "sector_regime_bull",
            "sector_regime_bear",
            "sector_regime_sideways",
        ]
    ]


def _build_sector_fallback_frame(
    data: dict[str, pd.DataFrame],
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build equal-weighted sector close series from stock data as fallback."""

    if metadata is None:
        return pd.DataFrame()
    rows = []
    sectors = metadata.set_index("Ticker")["Sector"].to_dict()
    for symbol, frame in data.items():
        sector = sectors.get(symbol)
        if sector is None or frame.empty:
            continue
        stock = frame[["Date", "Close"]].copy()
        stock["Sector"] = sector
        stock["normalized_close"] = stock["Close"] / stock["Close"].iloc[0]
        rows.append(stock[["Date", "Sector", "normalized_close"]])
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(["Date", "Sector"], as_index=False)["normalized_close"]
        .mean()
        .rename(columns={"normalized_close": "sector_close"})
    )


def _market_breadth(data: dict[str, pd.DataFrame], *, benchmark_symbol: str) -> pd.DataFrame:
    """Calculate same-day market breadth from stock closes."""

    rows = []
    for symbol, frame in data.items():
        if symbol == benchmark_symbol or frame.empty:
            continue
        clean = frame[["Date", "Close"]].copy()
        clean["above_sma50"] = clean["Close"] > clean["Close"].rolling(50).mean()
        clean["above_sma200"] = clean["Close"] > clean["Close"].rolling(200).mean()
        clean["positive_20d_return"] = clean["Close"].pct_change(20) > 0
        clean["advance"] = clean["Close"].pct_change() > 0
        clean["decline"] = clean["Close"].pct_change() < 0
        rows.append(
            clean[
                [
                    "Date",
                    "above_sma50",
                    "above_sma200",
                    "positive_20d_return",
                    "advance",
                    "decline",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Date",
                "market_breadth_above_sma50",
                "market_breadth_above_sma200",
                "market_breadth_positive_20d_return",
                "market_breadth_advance_decline_ratio",
            ]
        )
    combined = pd.concat(rows, ignore_index=True)
    grouped = combined.groupby("Date").agg(
        above_sma50=("above_sma50", "mean"),
        above_sma200=("above_sma200", "mean"),
        positive_20d_return=("positive_20d_return", "mean"),
        advances=("advance", "sum"),
        declines=("decline", "sum"),
    )
    grouped["market_breadth_advance_decline_ratio"] = grouped["advances"] / grouped[
        "declines"
    ].replace(0, pd.NA)
    return grouped.reset_index().rename(
        columns={
            "above_sma50": "market_breadth_above_sma50",
            "above_sma200": "market_breadth_above_sma200",
            "positive_20d_return": "market_breadth_positive_20d_return",
        }
    )[
        [
            "Date",
            "market_breadth_above_sma50",
            "market_breadth_above_sma200",
            "market_breadth_positive_20d_return",
            "market_breadth_advance_decline_ratio",
        ]
    ]


def _sector_for_symbol(symbol: str, metadata: pd.DataFrame | None) -> str | None:
    """Return sector for a ticker."""

    if metadata is None or metadata.empty:
        return None
    row = metadata[metadata["Ticker"] == symbol]
    if row.empty:
        return None
    return str(row.iloc[0]["Sector"])


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Calculate RSI."""

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((loss == 0) & (gain == 0), 50.0)
    return pd.to_numeric(rsi, errors="coerce")


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Calculate MACD and signal line."""

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate Average True Range."""

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate ADX, +DI, and -DI."""

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _atr(high, low, close, window)
    plus_di = 100 * plus_dm.rolling(window).sum() / atr.replace(0, pd.NA)
    minus_di = 100 * minus_dm.rolling(window).sum() / atr.replace(0, pd.NA)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    adx = dx.rolling(window).mean()
    return plus_di, minus_di, adx
