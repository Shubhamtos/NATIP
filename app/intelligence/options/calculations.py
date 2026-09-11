"""Deterministic technical and options calculations."""

from __future__ import annotations

from datetime import date

import pandas as pd

from app.intelligence.options.config import OptionsBuyingConfig
from app.intelligence.options.models import EmaDirection, FuturesOiState, OptionContract


def ema(series: pd.Series, period: int) -> pd.Series:
    """Return EMA values for completed candles."""

    values = pd.to_numeric(series, errors="coerce")
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def ema20_direction(values: pd.Series, *, direction_days: int) -> tuple[EmaDirection, list[float]]:
    """Classify EMA20 direction using consecutive completed EMA values only."""

    clean = pd.to_numeric(values, errors="coerce").dropna()
    needed = direction_days + 1
    if len(clean) < needed:
        return EmaDirection.FLAT_OR_MIXED, clean.tail(needed).astype(float).tolist()
    used = clean.tail(needed).astype(float).tolist()
    if all(later > earlier for earlier, later in zip(used, used[1:])):
        return EmaDirection.RISING, used
    if all(later < earlier for earlier, later in zip(used, used[1:])):
        return EmaDirection.FALLING, used
    return EmaDirection.FLAT_OR_MIXED, used


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return Wilder-style RSI."""

    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    output = 100 - (100 / (1 + rs))
    output = output.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    output = output.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return output


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return average true range."""

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def adx_strength(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return ADX trend-strength values without exposing DI as direction signals."""

    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_values = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_values
    )
    minus_di = (
        100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_values
    )
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def prior_resistance(frame: pd.DataFrame, lookback: int) -> float | None:
    """Return highest high of previous completed candles, excluding current candle."""

    if len(frame) < lookback + 1:
        return None
    highs = pd.to_numeric(frame["high"], errors="coerce").iloc[-(lookback + 1) : -1].dropna()
    return None if highs.empty else float(highs.max())


def prior_support(frame: pd.DataFrame, lookback: int) -> float | None:
    """Return lowest low of previous completed candles, excluding current candle."""

    if len(frame) < lookback + 1:
        return None
    lows = pd.to_numeric(frame["low"], errors="coerce").iloc[-(lookback + 1) : -1].dropna()
    return None if lows.empty else float(lows.min())


def prior_average_volume(frame: pd.DataFrame, lookback: int) -> float | None:
    """Return average volume of previous completed candles, excluding current candle."""

    if len(frame) < lookback + 1:
        return None
    volume = pd.to_numeric(frame["volume"], errors="coerce").iloc[-(lookback + 1) : -1].dropna()
    return None if volume.empty else float(volume.mean())


def candle_close_location(row: pd.Series | dict[str, float]) -> float | None:
    """Return close location in candle range, from 0.0 low to 1.0 high."""

    high = _float(row.get("high"))
    low = _float(row.get("low"))
    close = _float(row.get("close"))
    if high is None or low is None or close is None or high <= low:
        return None
    return (close - low) / (high - low)


def futures_oi_state(price_change: float | None, oi_change: float | None) -> FuturesOiState:
    """Classify futures price and OI confirmation."""

    if price_change is None or oi_change is None:
        return FuturesOiState.UNKNOWN
    if price_change > 0 and oi_change > 0:
        return FuturesOiState.LONG_BUILDUP
    if price_change > 0 and oi_change < 0:
        return FuturesOiState.SHORT_COVERING
    if price_change < 0 and oi_change > 0:
        return FuturesOiState.SHORT_BUILDUP
    if price_change < 0 and oi_change < 0:
        return FuturesOiState.LONG_UNWINDING
    return FuturesOiState.UNKNOWN


def spread_pct(contract: OptionContract) -> float | None:
    """Return bid-ask spread percentage."""

    if contract.bid <= 0 or contract.ask <= 0 or contract.ask < contract.bid:
        return None
    mid = contract.mid_price
    return ((contract.ask - contract.bid) / mid) * 100 if mid > 0 else None


def days_to_expiry(expiry: date, as_of: date) -> int:
    """Return calendar days to expiry."""

    return (expiry - as_of).days


def _float(value: object) -> float | None:
    try:
        output = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return output if pd.notna(output) else None


def latest_indicator_snapshot(
    frame: pd.DataFrame, config: OptionsBuyingConfig
) -> dict[str, object]:
    """Return latest deterministic indicator values for completed daily candles."""

    clean = frame.copy().reset_index(drop=True)
    close = pd.to_numeric(clean["close"], errors="coerce")
    ema10_series = ema(close, config.ema_fast_period)
    ema20_series = ema(close, config.ema_trend_period)
    direction, used = ema20_direction(ema20_series, direction_days=config.ema_direction_days)
    atr_series = atr(clean, config.atr_period)
    rsi_series = rsi(close, config.rsi_period)
    adx_series = adx_strength(clean, config.adx_period)
    return {
        "close": float(close.iloc[-1]),
        "ema10": _last_float(ema10_series),
        "ema20": _last_float(ema20_series),
        "ema20_direction": direction,
        "ema20_values_used": used,
        "rsi14": _last_float(rsi_series),
        "adx14": _last_float(adx_series),
        "atr14": _last_float(atr_series),
        "resistance": prior_resistance(clean, config.breakout_lookback),
        "support": prior_support(clean, config.breakout_lookback),
        "average_volume": prior_average_volume(clean, config.volume_lookback),
        "latest_volume": _float(clean.iloc[-1].get("volume")),
        "close_location": candle_close_location(clean.iloc[-1]),
    }


def _last_float(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.iloc[-1])
