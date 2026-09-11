"""Astro chart-reaction screener for shadow research.

The screener answers a narrow research question: after a timestamped astro
event, did the stock chart show a measurable price/volume reaction? It does
not infer causality and must not create trading recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class AstroChartReactionResult:
    """Chart reaction around one or more astro event windows."""

    symbol: str
    company: str
    reaction_score: float
    reaction_label: str
    direction: str
    latest_close: float | None
    latest_timestamp: datetime | None
    event_date: datetime | None
    event_priority: int | None
    event_name: str
    events_checked: int
    post_5d_return_pct: float | None
    post_10d_return_pct: float | None
    post_5d_excess_pct: float | None
    volume_confirmation: str
    range_confirmation: str
    trend_flip: bool
    reasons: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FutureAstroTrendWatchResult:
    """Forward-looking watchlist around upcoming astro windows."""

    symbol: str
    company: str
    watch_score: float
    watch_label: str
    setup_direction: str
    latest_close: float | None
    latest_timestamp: datetime | None
    next_event_date: datetime | None
    days_to_event: int | None
    event_priority: int | None
    event_name: str
    events_next_window: int
    distance_to_resistance_pct: float | None
    distance_to_support_pct: float | None
    volume_state: str
    volatility_state: str
    trend_state: str
    reasons: tuple[str, ...]
    error: str | None = None


def scan_astro_chart_reactions_with_yfinance(
    *,
    symbols: list[tuple[str, str]],
    events: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    reaction_days: int = 5,
    benchmark_symbol: str = "^NSEI",
    batch_size: int = 80,
) -> tuple[list[AstroChartReactionResult], list[AstroChartReactionResult]]:
    """Scan symbols for chart reactions near supplied astro events.

    Args:
        symbols: NSE symbol and company-name pairs.
        events: Astro event marker dictionaries containing at least ``timestamp``.
        start: Download start datetime.
        end: Download end datetime.
        reaction_days: Post-event trading sessions used for the main reaction.
        benchmark_symbol: Benchmark used for excess-return confirmation.
        batch_size: yfinance batch size.

    Returns:
        Matched reaction rows and error rows.
    """

    import yfinance as yf

    clean_events = _normalize_events(events, start=start, end=end)
    if not clean_events:
        return [], [
            AstroChartReactionResult(
                symbol="UNIVERSE",
                company="All",
                reaction_score=0.0,
                reaction_label="No events",
                direction="N/A",
                latest_close=None,
                latest_timestamp=None,
                event_date=None,
                event_priority=None,
                event_name="No eligible past astro/trend-change events in selected window.",
                events_checked=0,
                post_5d_return_pct=None,
                post_10d_return_pct=None,
                post_5d_excess_pct=None,
                volume_confirmation="N/A",
                range_confirmation="N/A",
                trend_flip=False,
                reasons=("Select a longer lookback or run near completed event dates.",),
                error="No eligible completed event windows.",
            )
        ]

    results: list[AstroChartReactionResult] = []
    errors: list[AstroChartReactionResult] = []
    for batch in _chunks(symbols, batch_size):
        yahoo_symbols = [_yahoo_symbol(symbol) for symbol, _ in batch]
        tickers = " ".join(_unique([*yahoo_symbols, benchmark_symbol]))
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        benchmark = _frame_from_bulk_download(data, benchmark_symbol)
        for symbol, company in batch:
            frame = _frame_from_bulk_download(data, _yahoo_symbol(symbol))
            if frame.empty:
                errors.append(
                    _error_result(
                        symbol=symbol,
                        company=company,
                        message="No historical candles returned from Yahoo.",
                    )
                )
                continue
            results.append(
                score_astro_chart_reaction(
                    symbol=symbol,
                    company=company,
                    frame=frame,
                    benchmark_frame=benchmark,
                    events=clean_events,
                    reaction_days=reaction_days,
                )
            )

    matched = [row for row in results if row.reaction_score > 0]
    matched.sort(key=lambda row: (row.reaction_score, row.post_5d_excess_pct or 0.0), reverse=True)
    return matched, errors


def scan_future_astro_trend_watchlist_with_yfinance(
    *,
    symbols: list[tuple[str, str]],
    events: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    as_of: datetime,
    future_days: int = 15,
    benchmark_symbol: str = "^NSEI",
    batch_size: int = 80,
) -> tuple[list[FutureAstroTrendWatchResult], list[FutureAstroTrendWatchResult]]:
    """Scan charts for setups heading into upcoming astro event windows.

    Args:
        symbols: NSE symbol and company-name pairs.
        events: Astro event marker dictionaries.
        start: OHLCV download start.
        end: OHLCV download end.
        as_of: Current scan timestamp.
        future_days: Calendar days ahead to consider for upcoming events.
        benchmark_symbol: Downloaded for provider consistency; not scored yet.
        batch_size: yfinance batch size.

    Returns:
        Matched watchlist rows and errors.
    """

    import yfinance as yf

    future_events = _normalize_future_events(events, as_of=as_of, future_days=future_days)
    if not future_events:
        return [], [
            _future_error_result(
                symbol="UNIVERSE",
                company="All",
                message="No upcoming astro/full-moon/alignment windows in the selected range.",
            )
        ]

    results: list[FutureAstroTrendWatchResult] = []
    errors: list[FutureAstroTrendWatchResult] = []
    for batch in _chunks(symbols, batch_size):
        yahoo_symbols = [_yahoo_symbol(symbol) for symbol, _ in batch]
        tickers = " ".join(_unique([*yahoo_symbols, benchmark_symbol]))
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        for symbol, company in batch:
            frame = _frame_from_bulk_download(data, _yahoo_symbol(symbol))
            if frame.empty:
                errors.append(
                    _future_error_result(
                        symbol=symbol,
                        company=company,
                        message="No historical candles returned from Yahoo.",
                    )
                )
                continue
            results.append(
                score_future_astro_trend_watch(
                    symbol=symbol,
                    company=company,
                    frame=frame,
                    events=future_events,
                    as_of=as_of,
                )
            )
    matched = [row for row in results if row.watch_score > 0]
    matched.sort(key=lambda row: (row.watch_score, -(row.days_to_event or 99)), reverse=True)
    return matched, errors


def score_future_astro_trend_watch(
    *,
    symbol: str,
    company: str,
    frame: pd.DataFrame,
    events: list[dict[str, Any]],
    as_of: datetime,
) -> FutureAstroTrendWatchResult:
    """Score whether a chart is technically sensitive before future astro windows.

    Args:
        symbol: NSE symbol.
        company: Company name.
        frame: Stock OHLCV data indexed by date.
        events: Upcoming astro events.
        as_of: Scan timestamp.

    Returns:
        Watchlist result for the stock.
    """

    clean = _prepare_frame(frame)
    if len(clean) < 80:
        return _future_error_result(
            symbol=symbol,
            company=company,
            message="At least 80 daily candles required for future watch scoring.",
        )
    event = min(events, key=lambda item: pd.Timestamp(item["timestamp"]))
    latest_close = float(clean["close"].iloc[-1])
    latest_timestamp = _timestamp_to_datetime(clean.index[-1])
    close = clean["close"].astype(float)
    high = clean["high"].astype(float)
    low = clean["low"].astype(float)
    volume = clean["volume"].astype(float) if "volume" in clean else pd.Series(index=clean.index)

    resistance = float(high.iloc[-20:].max())
    support = float(low.iloc[-20:].min())
    distance_to_resistance = (resistance / latest_close - 1) * 100 if latest_close > 0 else None
    distance_to_support = (latest_close / support - 1) * 100 if support > 0 else None
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else pd.NA
    range_10d = (high.iloc[-10:].max() / low.iloc[-10:].min() - 1) * 100
    range_40d = (high.iloc[-40:].max() / low.iloc[-40:].min() - 1) * 100
    range_ratio = range_10d / range_40d if range_40d > 0 else None
    volume_5 = volume.iloc[-5:].mean()
    volume_20 = volume.iloc[-20:].mean()
    volume_ratio = volume_5 / volume_20 if volume_20 and volume_20 > 0 else None
    trend_state = _trend_state(
        latest_close=latest_close,
        sma20=_optional_float(sma20),
        sma50=_optional_float(sma50),
        sma200=_optional_float(sma200),
    )
    score, direction, reasons = _future_watch_score(
        distance_to_resistance=distance_to_resistance,
        distance_to_support=distance_to_support,
        range_10d=range_10d,
        range_ratio=range_ratio,
        volume_ratio=volume_ratio,
        trend_state=trend_state,
        event_priority=int(event.get("priority") or 10),
        events_next_window=len(events),
    )
    event_time = pd.Timestamp(event["timestamp"])
    as_of_time = pd.Timestamp(as_of).tz_localize(None)
    return FutureAstroTrendWatchResult(
        symbol=symbol,
        company=company,
        watch_score=round(score, 2),
        watch_label=_future_watch_label(score),
        setup_direction=direction,
        latest_close=latest_close,
        latest_timestamp=latest_timestamp,
        next_event_date=_timestamp_to_datetime(event_time),
        days_to_event=max(0, int((event_time.normalize() - as_of_time.normalize()).days)),
        event_priority=int(event.get("priority") or 10),
        event_name=str(event.get("event") or "Astro trend-change window"),
        events_next_window=len(events),
        distance_to_resistance_pct=round(distance_to_resistance, 2)
        if distance_to_resistance is not None
        else None,
        distance_to_support_pct=round(distance_to_support, 2)
        if distance_to_support is not None
        else None,
        volume_state=_future_volume_state(volume_ratio),
        volatility_state=_future_volatility_state(range_ratio),
        trend_state=trend_state,
        reasons=reasons,
    )


def score_astro_chart_reaction(
    *,
    symbol: str,
    company: str,
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    events: list[dict[str, Any]],
    reaction_days: int = 5,
) -> AstroChartReactionResult:
    """Score one stock chart against completed astro event windows.

    Args:
        symbol: NSE symbol.
        company: Company name.
        frame: Stock OHLCV data indexed by date.
        benchmark_frame: Benchmark OHLCV data indexed by date.
        events: Normalized event markers.
        reaction_days: Main post-event reaction window.

    Returns:
        Best matching chart reaction result for the symbol.
    """

    clean = _prepare_frame(frame)
    benchmark = _prepare_frame(benchmark_frame)
    if len(clean) < 30:
        return _error_result(symbol=symbol, company=company, message="At least 30 candles required.")

    scored: list[AstroChartReactionResult] = []
    for event in events:
        event_time = pd.Timestamp(event["timestamp"])
        scored_event = _score_single_event(
            symbol=symbol,
            company=company,
            frame=clean,
            benchmark_frame=benchmark,
            event=event,
            event_time=event_time,
            reaction_days=reaction_days,
            events_checked=len(events),
        )
        if scored_event is not None:
            scored.append(scored_event)

    if not scored:
        return _error_result(
            symbol=symbol,
            company=company,
            message="No completed event windows overlap this chart.",
        )
    return max(scored, key=lambda row: (row.reaction_score, abs(row.post_5d_excess_pct or 0.0)))


def _score_single_event(
    *,
    symbol: str,
    company: str,
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    event: dict[str, Any],
    event_time: pd.Timestamp,
    reaction_days: int,
    events_checked: int = 1,
) -> AstroChartReactionResult | None:
    """Score a single completed event for one stock frame."""

    index = frame.index
    position = int(index.searchsorted(event_time.normalize()))
    if position >= len(frame):
        return None
    if position + reaction_days >= len(frame):
        return None

    event_position = min(position, len(frame) - 1)
    prior_position = max(event_position - 1, 0)
    pre_position = max(event_position - reaction_days, 0)
    post_position = min(event_position + reaction_days, len(frame) - 1)
    post10_position = min(event_position + 10, len(frame) - 1)

    prior_close = float(frame["close"].iloc[prior_position])
    event_close = float(frame["close"].iloc[event_position])
    pre_close = float(frame["close"].iloc[pre_position])
    post_close = float(frame["close"].iloc[post_position])
    post10_close = float(frame["close"].iloc[post10_position])
    if prior_close <= 0 or pre_close <= 0:
        return None

    pre_return = event_close / pre_close - 1
    post_return = post_close / prior_close - 1
    post10_return = post10_close / prior_close - 1
    benchmark_return = _window_return(
        benchmark_frame,
        start_date=index[prior_position],
        end_date=index[post_position],
    )
    excess = post_return - benchmark_return if benchmark_return is not None else None

    volume = frame["volume"] if "volume" in frame else pd.Series(index=frame.index, dtype=float)
    volume_ratio = _post_window_ratio(volume, event_position, reaction_days, lookback=20)
    range_series = (frame["high"] - frame["low"]) / frame["close"].replace(0, pd.NA)
    range_ratio = _post_window_ratio(range_series, event_position, reaction_days, lookback=20)
    trend_flip = bool(pre_return * post_return < 0 and abs(post_return) >= 0.02)
    score = _reaction_score(
        post_return=post_return,
        post10_return=post10_return,
        excess=excess,
        volume_ratio=volume_ratio,
        range_ratio=range_ratio,
        trend_flip=trend_flip,
        priority=int(event.get("priority") or 10),
    )
    direction = _reaction_direction(post_return=post_return, excess=excess, range_ratio=range_ratio)
    reasons = _reaction_reasons(
        post_return=post_return,
        post10_return=post10_return,
        excess=excess,
        volume_ratio=volume_ratio,
        range_ratio=range_ratio,
        trend_flip=trend_flip,
    )
    return AstroChartReactionResult(
        symbol=symbol,
        company=company,
        reaction_score=round(score, 2),
        reaction_label=_reaction_label(score),
        direction=direction,
        latest_close=_optional_float(frame["close"].iloc[-1]),
        latest_timestamp=_timestamp_to_datetime(frame.index[-1]),
        event_date=_timestamp_to_datetime(event_time),
        event_priority=int(event.get("priority") or 10),
        event_name=str(event.get("event") or "Astro trend-change window"),
        events_checked=events_checked,
        post_5d_return_pct=round(post_return * 100, 2),
        post_10d_return_pct=round(post10_return * 100, 2),
        post_5d_excess_pct=round(excess * 100, 2) if excess is not None else None,
        volume_confirmation=_confirmation_label(volume_ratio, high=1.25, low=0.80),
        range_confirmation=_confirmation_label(range_ratio, high=1.20, low=0.85),
        trend_flip=trend_flip,
        reasons=reasons,
    )


def _reaction_score(
    *,
    post_return: float,
    post10_return: float,
    excess: float | None,
    volume_ratio: float | None,
    range_ratio: float | None,
    trend_flip: bool,
    priority: int,
) -> float:
    """Return a bounded reaction-quality score."""

    score = min(abs(post_return) * 260, 24)
    score += min(abs(post10_return) * 160, 16)
    if excess is not None:
        score += min(abs(excess) * 300, 24)
    if volume_ratio is not None:
        score += max(0.0, min((volume_ratio - 1.0) * 18, 12))
    if range_ratio is not None:
        score += max(0.0, min((range_ratio - 1.0) * 16, 10))
    if trend_flip:
        score += 10
    score += max(0, 10 - priority) * 1.5
    return max(0.0, min(100.0, score))


def _reaction_label(score: float) -> str:
    """Map reaction score to a human label."""

    if score >= 75:
        return "Strong chart reaction"
    if score >= 55:
        return "Moderate chart reaction"
    if score >= 35:
        return "Weak chart reaction"
    return "No clear chart reaction"


def _reaction_direction(
    *,
    post_return: float,
    excess: float | None,
    range_ratio: float | None,
) -> str:
    """Describe the observed post-event chart direction."""

    if abs(post_return) < 0.01 and range_ratio is not None and range_ratio >= 1.2:
        return "Volatility only"
    if post_return > 0 and (excess is None or excess >= 0):
        return "Up confirmation"
    if post_return < 0 and (excess is None or excess <= 0):
        return "Down confirmation"
    return "Mixed reaction"


def _reaction_reasons(
    *,
    post_return: float,
    post10_return: float,
    excess: float | None,
    volume_ratio: float | None,
    range_ratio: float | None,
    trend_flip: bool,
) -> tuple[str, ...]:
    """Build concise scoring reasons."""

    reasons = [
        f"5D move {post_return * 100:+.2f}%",
        f"10D move {post10_return * 100:+.2f}%",
    ]
    if excess is not None:
        reasons.append(f"5D excess vs Nifty {excess * 100:+.2f}%")
    if volume_ratio is not None and volume_ratio >= 1.25:
        reasons.append(f"volume expanded {volume_ratio:.2f}x")
    elif volume_ratio is not None and volume_ratio <= 0.80:
        reasons.append(f"volume stayed quiet {volume_ratio:.2f}x")
    if range_ratio is not None and range_ratio >= 1.20:
        reasons.append(f"range expanded {range_ratio:.2f}x")
    if trend_flip:
        reasons.append("pre/post trend flipped")
    return tuple(reasons)


def _future_watch_score(
    *,
    distance_to_resistance: float | None,
    distance_to_support: float | None,
    range_10d: float,
    range_ratio: float | None,
    volume_ratio: float | None,
    trend_state: str,
    event_priority: int,
    events_next_window: int,
) -> tuple[float, str, tuple[str, ...]]:
    """Score technical sensitivity before a future astro event window."""

    score = 0.0
    reasons: list[str] = []
    if distance_to_resistance is not None and 0 <= distance_to_resistance <= 3:
        score += 24
        reasons.append(f"price is {distance_to_resistance:.2f}% below 20D resistance")
    elif distance_to_resistance is not None and 3 < distance_to_resistance <= 7:
        score += 12
        reasons.append(f"price is within {distance_to_resistance:.2f}% of 20D resistance")
    if distance_to_support is not None and 0 <= distance_to_support <= 3:
        score += 20
        reasons.append(f"price is {distance_to_support:.2f}% above 20D support")
    elif distance_to_support is not None and 3 < distance_to_support <= 7:
        score += 10
        reasons.append(f"price is within {distance_to_support:.2f}% of 20D support")
    if range_10d <= 8:
        score += 16
        reasons.append(f"10D range is tight at {range_10d:.2f}%")
    if range_ratio is not None and range_ratio <= 0.55:
        score += 14
        reasons.append("short-term range is compressed versus 40D range")
    if volume_ratio is not None and volume_ratio <= 0.80:
        score += 12
        reasons.append(f"volume is drying up at {volume_ratio:.2f}x 20D average")
    elif volume_ratio is not None and volume_ratio >= 1.30:
        score += 8
        reasons.append(f"volume is already expanding at {volume_ratio:.2f}x 20D average")
    if trend_state in {"Uptrend", "Downtrend"}:
        score += 10
        reasons.append(f"clear {trend_state.lower()} context")
    elif trend_state == "Sideways compression":
        score += 12
        reasons.append("sideways compression can break either way")
    score += max(0, 10 - event_priority) * 1.4
    if events_next_window >= 3:
        score += 8
        reasons.append(f"{events_next_window} astro windows are clustered ahead")

    near_resistance = distance_to_resistance is not None and distance_to_resistance <= 5
    near_support = distance_to_support is not None and distance_to_support <= 5
    if near_resistance and trend_state in {"Uptrend", "Sideways compression"}:
        direction = "Upside breakout watch"
    elif near_support and trend_state in {"Downtrend", "Sideways compression"}:
        direction = "Downside breakdown watch"
    elif near_resistance and near_support:
        direction = "Two-way volatility watch"
    else:
        direction = "Low-conviction watch"
    return max(0.0, min(100.0, score)), direction, tuple(reasons)


def _future_watch_label(score: float) -> str:
    """Map future watch score to a label."""

    if score >= 75:
        return "High-priority watch"
    if score >= 55:
        return "Medium-priority watch"
    if score >= 35:
        return "Low-priority watch"
    return "No clear setup"


def _trend_state(
    *,
    latest_close: float,
    sma20: float | None,
    sma50: float | None,
    sma200: float | None,
) -> str:
    """Return a compact trend-state label."""

    if sma20 is None or sma50 is None:
        return "Unknown"
    if latest_close > sma20 > sma50 and (sma200 is None or latest_close > sma200):
        return "Uptrend"
    if latest_close < sma20 < sma50 and (sma200 is None or latest_close < sma200):
        return "Downtrend"
    if abs(sma20 / sma50 - 1) <= 0.025:
        return "Sideways compression"
    return "Mixed"


def _future_volume_state(volume_ratio: float | None) -> str:
    """Return future-watch volume state."""

    if volume_ratio is None:
        return "Unavailable"
    if volume_ratio <= 0.80:
        return "Dry-up"
    if volume_ratio >= 1.30:
        return "Expansion"
    return "Normal"


def _future_volatility_state(range_ratio: float | None) -> str:
    """Return future-watch volatility state."""

    if range_ratio is None:
        return "Unavailable"
    if range_ratio <= 0.55:
        return "Compression"
    if range_ratio >= 1.20:
        return "Expansion"
    return "Normal"


def _post_window_ratio(
    series: pd.Series,
    event_position: int,
    reaction_days: int,
    *,
    lookback: int,
) -> float | None:
    """Return post-event average divided by trailing pre-event average."""

    values = pd.to_numeric(series, errors="coerce")
    baseline = values.iloc[max(0, event_position - lookback) : event_position].mean()
    post = values.iloc[event_position : event_position + reaction_days + 1].mean()
    if pd.isna(baseline) or baseline <= 0 or pd.isna(post):
        return None
    return float(post / baseline)


def _window_return(
    frame: pd.DataFrame,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float | None:
    """Return benchmark close return over nearest available dates."""

    if frame.empty:
        return None
    start_pos = int(frame.index.searchsorted(start_date))
    end_pos = int(frame.index.searchsorted(end_date))
    if start_pos >= len(frame) or end_pos >= len(frame):
        return None
    start_close = float(frame["close"].iloc[start_pos])
    end_close = float(frame["close"].iloc[end_pos])
    if start_close <= 0:
        return None
    return end_close / start_close - 1


def _confirmation_label(value: float | None, *, high: float, low: float) -> str:
    """Return simple confirmation text."""

    if value is None:
        return "Unavailable"
    if value >= high:
        return "Strong"
    if value <= low:
        return "Quiet"
    return "Normal"


def _normalize_events(
    events: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Normalize and keep only completed event windows."""

    normalized: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None)
    for event in events:
        raw_time = event.get("timestamp") or event.get("date")
        if raw_time is None:
            continue
        timestamp = pd.Timestamp(raw_time).tz_localize(None)
        if timestamp < start_ts or timestamp > end_ts:
            continue
        row = dict(event)
        row["timestamp"] = timestamp
        normalized.append(row)
    normalized.sort(key=lambda item: item["timestamp"])
    return normalized


def _normalize_future_events(
    events: list[dict[str, Any]],
    *,
    as_of: datetime,
    future_days: int,
) -> list[dict[str, Any]]:
    """Normalize and keep only future events."""

    normalized: list[dict[str, Any]] = []
    as_of_ts = pd.Timestamp(as_of).tz_localize(None)
    end_ts = as_of_ts + pd.Timedelta(days=future_days)
    for event in events:
        raw_time = event.get("timestamp") or event.get("date")
        if raw_time is None:
            continue
        timestamp = pd.Timestamp(raw_time).tz_localize(None)
        if timestamp < as_of_ts.normalize() or timestamp > end_ts:
            continue
        row = dict(event)
        row["timestamp"] = timestamp
        normalized.append(row)
    normalized.sort(key=lambda item: item["timestamp"])
    return normalized


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return sorted numeric OHLCV data indexed by timezone-naive date."""

    clean = frame.copy()
    if "timestamp" in clean.columns:
        clean = clean.set_index("timestamp")
    clean.index = pd.to_datetime(clean.index, errors="coerce").tz_localize(None)
    clean = clean[~clean.index.isna()].sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    for column in ["open", "high", "low", "close", "volume"]:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna(subset=["close"])


def _frame_from_bulk_download(data: pd.DataFrame, yahoo_symbol: str) -> pd.DataFrame:
    """Return normalized OHLCV frame for one yfinance symbol."""

    if data.empty or not yahoo_symbol:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        if yahoo_symbol in data.columns.get_level_values(0):
            frame = data[yahoo_symbol].copy()
        elif yahoo_symbol in data.columns.get_level_values(1):
            frame = data.xs(yahoo_symbol, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = data.copy()
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    columns = [column for column in ["open", "high", "low", "close", "volume"] if column in frame]
    return frame[columns].dropna(how="all") if columns else pd.DataFrame()


def _yahoo_symbol(symbol: str) -> str:
    """Return a Yahoo-compatible NSE symbol."""

    cleaned = symbol.strip().upper()
    if "." in cleaned or cleaned.startswith("^"):
        return cleaned
    return f"{cleaned}.NS"


def _chunks(values: list[tuple[str, str]], size: int) -> list[list[tuple[str, str]]]:
    """Split values into chunks."""

    return [values[index : index + size] for index in range(0, len(values), size)]


def _unique(values: list[str]) -> list[str]:
    """Return order-preserving unique strings."""

    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _timestamp_to_datetime(value: Any) -> datetime | None:
    """Convert timestamp-like value to datetime."""

    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value if isinstance(value, datetime) else None


def _optional_float(value: Any) -> float | None:
    """Return finite float or None."""

    if value is None or pd.isna(value):
        return None
    return float(value)


def _error_result(*, symbol: str, company: str, message: str) -> AstroChartReactionResult:
    """Return an error result row."""

    return AstroChartReactionResult(
        symbol=symbol,
        company=company,
        reaction_score=0.0,
        reaction_label="Unavailable",
        direction="N/A",
        latest_close=None,
        latest_timestamp=None,
        event_date=None,
        event_priority=None,
        event_name="N/A",
        events_checked=0,
        post_5d_return_pct=None,
        post_10d_return_pct=None,
        post_5d_excess_pct=None,
        volume_confirmation="N/A",
        range_confirmation="N/A",
        trend_flip=False,
        reasons=(message,),
        error=message,
    )


def _future_error_result(*, symbol: str, company: str, message: str) -> FutureAstroTrendWatchResult:
    """Return an unavailable future-watch row."""

    return FutureAstroTrendWatchResult(
        symbol=symbol,
        company=company,
        watch_score=0.0,
        watch_label="Unavailable",
        setup_direction="N/A",
        latest_close=None,
        latest_timestamp=None,
        next_event_date=None,
        days_to_event=None,
        event_priority=None,
        event_name="N/A",
        events_next_window=0,
        distance_to_resistance_pct=None,
        distance_to_support_pct=None,
        volume_state="N/A",
        volatility_state="N/A",
        trend_state="N/A",
        reasons=(message,),
        error=message,
    )
