"""Reusable technical-analysis indicator helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DarvasBox:
    """Latest Darvas Box screening state."""

    top: float
    bottom: float
    status: str
    breakout: bool
    breakdown: bool
    volume_confirmed: bool
    height_percent: float
    latest_close: float
    latest_volume: float | None
    average_volume: float | None


@dataclass(frozen=True, slots=True)
class DetectedPattern:
    """Rule-based overall chart pattern detected from OHLCV data."""

    name: str
    status: str
    start_index: int
    end_index: int
    confidence: float
    reason: str
    levels: dict[str, float]


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Return a rolling moving average.

    Args:
        series: Input price series.
        window: Rolling window length.

    Returns:
        Rolling moving-average series.
    """

    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=window).mean()


def recent_support(frame: pd.DataFrame, window: int = 20) -> float | None:
    """Return recent support from low prices.

    Args:
        frame: OHLCV data frame.
        window: Number of recent rows to inspect.

    Returns:
        Recent support level, if available.
    """

    lows = pd.to_numeric(frame["low"], errors="coerce").dropna()
    if lows.empty:
        return None
    return float(lows.tail(window).min())


def recent_resistance(frame: pd.DataFrame, window: int = 20) -> float | None:
    """Return recent resistance from high prices.

    Args:
        frame: OHLCV data frame.
        window: Number of recent rows to inspect.

    Returns:
        Recent resistance level, if available.
    """

    highs = pd.to_numeric(frame["high"], errors="coerce").dropna()
    if highs.empty:
        return None
    return float(highs.tail(window).max())


def latest_atr_stop(
    frame: pd.DataFrame,
    window: int = 14,
    multiplier: float = 2.0,
) -> float | None:
    """Return ATR-based stop from latest close.

    Args:
        frame: OHLCV data frame.
        window: ATR rolling window.
        multiplier: ATR distance from latest close.

    Returns:
        Latest ATR stop level, if available.
    """

    atr = average_true_range(frame, window)
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if atr is None or closes.empty:
        return None
    return float(closes.iloc[-1] - atr * multiplier)


def average_true_range(frame: pd.DataFrame, window: int = 14) -> float | None:
    """Return latest average true range.

    Args:
        frame: OHLCV data frame.
        window: ATR rolling window.

    Returns:
        Latest ATR value, if available.
    """

    highs = pd.to_numeric(frame["high"], errors="coerce")
    lows = pd.to_numeric(frame["low"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    previous_close = closes.shift(1)
    true_range = pd.concat(
        [
            highs - lows,
            (highs - previous_close).abs(),
            (lows - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean().dropna()
    if atr.empty:
        return None
    return float(atr.iloc[-1])


def detect_darvas_box(
    frame: pd.DataFrame,
    *,
    lookback: int = 20,
    volume_multiplier: float = 1.2,
) -> DarvasBox | None:
    """Detect the latest Darvas Box state from OHLCV data.

    The current candle is compared against the prior lookback window. A breakout
    requires the latest close above the box top with above-average volume.

    Args:
        frame: OHLCV data frame.
        lookback: Number of prior candles used to build the box.
        volume_multiplier: Minimum latest-volume multiple versus average volume.

    Returns:
        Latest Darvas Box state, if enough data exists.
    """

    required = ["high", "low", "close"]
    if any(column not in frame.columns for column in required) or len(frame) < lookback + 1:
        return None

    data = frame.copy()
    for column in ["high", "low", "close", "volume"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    clean = data.dropna(subset=["high", "low", "close"])
    if len(clean) < lookback + 1:
        return None

    prior = clean.iloc[-(lookback + 1) : -1]
    latest = clean.iloc[-1]
    top = float(prior["high"].max())
    bottom = float(prior["low"].min())
    latest_close = float(latest["close"])
    latest_volume = _optional_float(latest.get("volume"))
    average_volume = _optional_float(prior["volume"].dropna().mean()) if "volume" in prior else None
    volume_confirmed = (
        latest_volume is not None
        and average_volume is not None
        and average_volume > 0
        and latest_volume >= average_volume * volume_multiplier
    )
    breakout = latest_close > top and volume_confirmed
    breakdown = latest_close < bottom
    if breakout:
        status = "Breakout"
    elif breakdown:
        status = "Breakdown"
    elif latest_close >= top * 0.98:
        status = "Near breakout"
    elif bottom <= latest_close <= top:
        status = "In box"
    else:
        status = "Outside box"

    height_percent = (top - bottom) / latest_close if latest_close else 0.0
    return DarvasBox(
        top=top,
        bottom=bottom,
        status=status,
        breakout=breakout,
        breakdown=breakdown,
        volume_confirmed=volume_confirmed,
        height_percent=height_percent,
        latest_close=latest_close,
        latest_volume=latest_volume,
        average_volume=average_volume,
    )


def detect_darvax_patterns(frame: pd.DataFrame) -> list[DetectedPattern]:
    """Detect broad DarvaX-style chart setups from OHLCV data.

    These are rule-based approximations of broader patterns from the DARVAX
    notes: Darvas box, all-time-high zone, breakout-retest, Fibonacci pullback,
    High-Dry-Fry base, and 1-2-3/double-bottom style reversal.
    """

    clean = _clean_frame(frame)
    if len(clean) < 10:
        return []

    patterns: list[DetectedPattern] = []
    darvas = detect_darvas_box(clean)
    if darvas is not None:
        patterns.append(
            DetectedPattern(
                name="Darvas Box",
                status=darvas.status,
                start_index=max(0, len(clean) - 21),
                end_index=len(clean) - 1,
                confidence=0.78 if darvas.breakout else 0.62,
                reason=(
                    "Price is being compared with the recent Darvas box; breakout "
                    "requires close above box top with volume confirmation."
                ),
                levels={"top": darvas.top, "bottom": darvas.bottom},
            )
        )

    for detector in (
        _detect_life_high_zone,
        _detect_breakout_retest,
        _detect_fibonacci_pullback,
        _detect_high_dry_fry_base,
        _detect_double_bottom_123,
    ):
        detected = detector(clean)
        if detected is not None:
            patterns.append(detected)
    return patterns


def _optional_float(value: object) -> float | None:
    """Return float value unless unavailable."""

    if value is None or pd.isna(value):
        return None
    return float(value)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return numeric OHLCV frame with unusable rows removed."""

    required = ["high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame(columns=["high", "low", "close", "volume"])
    clean = frame.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna(subset=["high", "low", "close"]).reset_index(drop=True)


def _detect_life_high_zone(frame: pd.DataFrame) -> DetectedPattern | None:
    """Detect price near selected-window high / uncharted territory."""

    if len(frame) < 30:
        return None
    highs = frame["high"]
    closes = frame["close"]
    latest_close = float(closes.iloc[-1])
    prior_high = float(highs.iloc[:-1].max())
    if prior_high <= 0 or latest_close < prior_high * 0.98:
        return None
    status = "Breakout" if latest_close > prior_high else "Near life high"
    return DetectedPattern(
        name="Life High / Uncharted Territory",
        status=status,
        start_index=0,
        end_index=len(frame) - 1,
        confidence=0.7 if status == "Breakout" else 0.55,
        reason=(
            "Latest close is within 2% of the selected-window high, matching the "
            "DarvaX preference for life-high or uncharted-territory stocks."
        ),
        levels={"life_high": prior_high},
    )


def _detect_breakout_retest(frame: pd.DataFrame) -> DetectedPattern | None:
    """Detect a prior resistance breakout followed by a retest hold."""

    if len(frame) < 35:
        return None
    prior = frame.iloc[-35:-8]
    recent = frame.iloc[-8:]
    resistance = float(prior["high"].max())
    if resistance <= 0:
        return None
    broke = bool((recent["close"].iloc[:-2] > resistance * 1.01).any())
    retested = bool((recent["low"].tail(5) <= resistance * 1.03).any())
    held = float(frame["close"].iloc[-1]) >= resistance
    if not (broke and retested and held):
        return None
    return DetectedPattern(
        name="Darvas Breakout-Retest",
        status="Retest holding",
        start_index=len(frame) - 35,
        end_index=len(frame) - 1,
        confidence=0.72,
        reason=(
            "Price broke above an older resistance zone, retested near that level, "
            "and is still closing above it."
        ),
        levels={"retest_level": resistance},
    )


def _detect_fibonacci_pullback(frame: pd.DataFrame) -> DetectedPattern | None:
    """Detect a 50-61.8% pullback after a swing move."""

    if len(frame) < 35:
        return None
    window = frame.tail(min(90, len(frame))).reset_index(drop=True)
    low_index = int(window["low"].idxmin())
    after_low = window.iloc[low_index:]
    if len(after_low) < 8:
        return None
    high_index = int(after_low["high"].idxmax())
    swing_low = float(window["low"].iloc[low_index])
    swing_high = float(window["high"].iloc[high_index])
    latest_close = float(window["close"].iloc[-1])
    if swing_high <= swing_low or high_index >= len(window) - 2:
        return None
    retracement = (swing_high - latest_close) / (swing_high - swing_low)
    if not 0.47 <= retracement <= 0.65:
        return None
    fib_50 = swing_high - (swing_high - swing_low) * 0.5
    fib_618 = swing_high - (swing_high - swing_low) * 0.618
    return DetectedPattern(
        name="Fibonacci 50-61.8 Pullback",
        status="In golden pullback zone",
        start_index=max(0, len(frame) - len(window) + low_index),
        end_index=len(frame) - 1,
        confidence=0.66,
        reason=(
            "Latest close is in the 50-61.8% retracement zone after a prior swing, "
            "matching the PDF's preferred deeper pullback area."
        ),
        levels={"fib_50": fib_50, "fib_618": fib_618, "swing_high": swing_high},
    )


def _detect_high_dry_fry_base(frame: pd.DataFrame) -> DetectedPattern | None:
    """Detect High-Dry-Fry: sharp rise, correction, tight base."""

    if len(frame) < 45 or "volume" not in frame.columns:
        return None
    window = frame.tail(min(70, len(frame))).reset_index(drop=True)
    peak_index = int(window["high"].iloc[:-5].idxmax())
    pre_peak = window.iloc[: peak_index + 1]
    if len(pre_peak) < 8:
        return None
    base = window.tail(6)
    swing_low = float(pre_peak["low"].min())
    peak = float(window["high"].iloc[peak_index])
    post_peak_low = float(window["low"].iloc[peak_index:].min())
    rise = (peak - swing_low) / swing_low if swing_low else 0.0
    correction = (peak - post_peak_low) / peak if peak else 0.0
    base_range = (float(base["high"].max()) - float(base["low"].min())) / float(
        base["close"].iloc[-1]
    )
    avg_volume = (
        float(window["volume"].dropna().mean()) if not window["volume"].dropna().empty else 0.0
    )
    peak_volume = float(window["volume"].iloc[max(0, peak_index - 2) : peak_index + 1].max())
    volume_pop = avg_volume > 0 and peak_volume >= avg_volume * 1.4
    latest_close = float(window["close"].iloc[-1])
    base_high = float(base["high"].iloc[:-1].max())
    if not (0.18 <= rise <= 0.65 and 0.08 <= correction <= 0.28 and base_range <= 0.09):
        return None
    status = "Base breakout" if latest_close > base_high else "Tight base after dry fry"
    return DetectedPattern(
        name="DarvaX High-Dry-Fry Base",
        status=status,
        start_index=max(0, len(frame) - len(window) + peak_index),
        end_index=len(frame) - 1,
        confidence=0.74 if volume_pop else 0.6,
        reason=(
            "Stock had a fast impulse move, corrected from the peak, then formed a "
            "tight small-range base. Volume pop strengthens the setup."
        ),
        levels={"peak": peak, "base_high": base_high, "base_low": float(base["low"].min())},
    )


def _detect_double_bottom_123(frame: pd.DataFrame) -> DetectedPattern | None:
    """Detect approximate 1-2-3 / double-bottom reversal structure."""

    if len(frame) < 40:
        return None
    window = frame.tail(min(80, len(frame))).reset_index(drop=True)
    lows = window["low"]
    first_low_index = int(lows.iloc[:-10].idxmin())
    first_low = float(lows.iloc[first_low_index])
    after_first = window.iloc[first_low_index + 5 :]
    if len(after_first) < 12:
        return None
    second_low_index = int(after_first["low"].idxmin())
    second_low = float(window["low"].iloc[second_low_index])
    if first_low <= 0 or abs(second_low - first_low) / first_low > 0.04:
        return None
    neckline = float(window["high"].iloc[first_low_index:second_low_index].max())
    latest_close = float(window["close"].iloc[-1])
    if latest_close < neckline * 0.99:
        return None
    return DetectedPattern(
        name="1-2-3 / Double Bottom",
        status="Neckline breakout" if latest_close > neckline else "At neckline",
        start_index=max(0, len(frame) - len(window) + first_low_index),
        end_index=len(frame) - 1,
        confidence=0.64,
        reason=(
            "Two similar swing lows formed with price now testing or crossing the "
            "neckline, matching the PDF's broader reversal setup."
        ),
        levels={"neckline": neckline, "first_low": first_low, "second_low": second_low},
    )
