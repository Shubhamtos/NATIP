"""Dashboard analysis helpers.

The helpers produce informational summaries only. They do not generate trading
recommendations or price targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import mean
from typing import Any

import pandas as pd

from app.intelligence.technical.indicators import detect_darvas_box, detect_darvax_patterns
from app.providers.market import HistoricalBar, MarketQuote


@dataclass(frozen=True, slots=True)
class AnalysisItem:
    """Single dashboard analysis item."""

    label: str
    value: str
    note: str


@dataclass(frozen=True, slots=True)
class TechnicalAnalysisSummary:
    """Dashboard technical-analysis summary."""

    bias: str
    interpretation: str
    items: list[AnalysisItem]


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    """Dashboard data-quality summary."""

    status: str
    items: list[AnalysisItem]
    warnings: list[str]


def technical_items(quote: MarketQuote, bars: list[HistoricalBar]) -> list[AnalysisItem]:
    """Build technical-analysis summary items.

    Args:
        quote: Current quote.
        bars: Historical bars.

    Returns:
        Informational technical summary items.
    """

    closes = [bar.close_price for bar in bars if bar.close_price is not None]
    latest_close = closes[-1] if closes else quote.last_price
    first_close = closes[0] if closes else None
    average_volume = _average([bar.volume for bar in bars if bar.volume is not None])
    period_change = _percent_change(first_close, latest_close)

    return [
        AnalysisItem(
            "Period change",
            _format_percent(period_change),
            "Change between first and latest available close in the selected interval.",
        ),
        AnalysisItem(
            "Average volume",
            _format_number(average_volume),
            "Mean traded volume across the selected chart window.",
        ),
        AnalysisItem(
            "Latest close",
            _format_number(latest_close),
            "Most recent close in the selected historical series.",
        ),
    ]


def technical_analysis_summary(
    quote: MarketQuote,
    bars: list[HistoricalBar],
) -> TechnicalAnalysisSummary:
    """Build a detailed technical-analysis summary.

    Args:
        quote: Current quote.
        bars: Historical bars.

    Returns:
        Technical-analysis summary for the dashboard.
    """

    closes = [float(bar.close_price) for bar in bars if bar.close_price is not None]
    highs = [float(bar.high_price) for bar in bars if bar.high_price is not None]
    lows = [float(bar.low_price) for bar in bars if bar.low_price is not None]
    volumes = [float(bar.volume) for bar in bars if bar.volume is not None]
    frame = _bars_to_frame(bars)

    latest_close = closes[-1] if closes else quote.last_price
    sma_20 = _moving_average(closes, 20)
    sma_50 = _moving_average(closes, 50)
    rsi_14 = _rsi(closes, 14)
    macd, macd_signal = _macd(closes)
    adx_14 = _adx(highs, lows, closes, 14)
    atr_14 = _atr(highs, lows, closes, 14)
    high_52w = max(highs[-252:]) if highs else None
    low_52w = min(lows[-252:]) if lows else None
    support = min(lows[-20:]) if lows else None
    resistance = max(highs[-20:]) if highs else None
    average_volume = _average_float(volumes)
    recent_volume = _average_float(volumes[-5:])
    baseline_volume = _average_float(volumes[-20:])
    volume_trend = _volume_trend(recent_volume, baseline_volume)
    darvas_box = detect_darvas_box(frame)
    darvax_patterns = [
        pattern for pattern in detect_darvax_patterns(frame) if pattern.name != "Darvas Box"
    ]
    darvax_pattern_names = [pattern.name for pattern in darvax_patterns]
    darvax_label = ", ".join(darvax_pattern_names[:3]) or "None detected"
    bias = _technical_bias(
        latest_close=latest_close,
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        macd=macd,
        macd_signal=macd_signal,
        adx_14=adx_14,
    )

    items = [
        AnalysisItem("Technical bias", bias, "Directional read from trend and momentum."),
        AnalysisItem("20-period SMA", _format_number(sma_20), "Short-term moving average."),
        AnalysisItem("50-period SMA", _format_number(sma_50), "Medium-term moving average."),
        AnalysisItem(
            "RSI 14", _format_number(rsi_14), "Momentum oscillator; above 70 is extended."
        ),
        AnalysisItem("MACD", _format_number(macd), "12/26 EMA momentum spread."),
        AnalysisItem("MACD signal", _format_number(macd_signal), "9-period EMA of MACD."),
        AnalysisItem("ADX 14", _format_number(adx_14), "Trend-strength gauge."),
        AnalysisItem("ATR 14", _format_number(atr_14), "Volatility range estimate."),
        AnalysisItem("Support", _format_number(support), "Lowest low in the recent 20 candles."),
        AnalysisItem(
            "Resistance",
            _format_number(resistance),
            "Highest high in the recent 20 candles.",
        ),
        AnalysisItem(
            "Avg volume",
            _format_number(average_volume),
            "Average traded volume in the selected window.",
        ),
        AnalysisItem(
            "Volume trend",
            volume_trend,
            "Compares recent 5-candle volume with 20-candle baseline.",
        ),
        AnalysisItem(
            "52W high",
            _format_number(high_52w),
            "Highest high from available daily-like history.",
        ),
        AnalysisItem(
            "52W low",
            _format_number(low_52w),
            "Lowest low from available daily-like history.",
        ),
        AnalysisItem(
            "Darvas Box",
            darvas_box.status if darvas_box else "N/A",
            "Breakout screening from recent price box and volume confirmation.",
        ),
        AnalysisItem(
            "DarvaX patterns",
            darvax_label,
            "Rule-based broader chart patterns extracted from the DARVAX notes.",
        ),
    ]
    return TechnicalAnalysisSummary(
        bias=bias,
        interpretation=_technical_interpretation(
            bias=bias,
            latest_close=latest_close,
            sma_20=sma_20,
            sma_50=sma_50,
            rsi_14=rsi_14,
            macd=macd,
            macd_signal=macd_signal,
            adx_14=adx_14,
            atr_14=atr_14,
            support=support,
            resistance=resistance,
            volume_trend=volume_trend,
            darvas_status=darvas_box.status if darvas_box else None,
            darvax_patterns=darvax_pattern_names,
        ),
        items=items,
    )


def data_quality_summary(
    quote: MarketQuote,
    bars: list[HistoricalBar],
    profile: dict[str, Any],
    *,
    gemini_active: bool,
) -> DataQualitySummary:
    """Build a data-quality summary for the dashboard.

    Args:
        quote: Current quote.
        bars: Historical bars.
        profile: Yahoo Finance company profile.
        gemini_active: Whether Gemini reasoning is enabled.

    Returns:
        Data-quality summary.
    """

    now = datetime.now(UTC)
    latest_bar = max((bar.timestamp for bar in bars), default=None)
    age = now - latest_bar if latest_bar else None
    warnings: list[str] = []
    available_profile_fields = sum(
        1
        for key in [
            "marketCap",
            "totalRevenue",
            "profitMargins",
            "returnOnEquity",
            "debtToEquity",
            "trailingPE",
            "priceToBook",
            "enterpriseToEbitda",
        ]
        if profile.get(key) is not None
    )

    if quote.last_price is None:
        warnings.append("Quote price is missing.")
    if len(bars) < 30:
        warnings.append("Historical candles are limited.")
    if latest_bar is None:
        warnings.append("No historical candle timestamp is available.")
    elif age and age > timedelta(days=10):
        warnings.append("Latest historical candle appears stale.")
    if available_profile_fields < 4:
        warnings.append("Company profile fundamentals are partial.")
    if not gemini_active:
        warnings.append("Gemini reasoning is disabled or unavailable.")

    status = "Fresh"
    if warnings:
        status = "Partial" if quote.last_price is not None and bars else "Weak"

    items = [
        AnalysisItem("Status", status, "Overall readiness for analysis."),
        AnalysisItem("Candles", str(len(bars)), "Historical candles loaded from Yahoo Finance."),
        AnalysisItem(
            "Latest candle",
            latest_bar.isoformat() if latest_bar else "N/A",
            "Most recent candle timestamp.",
        ),
        AnalysisItem(
            "Profile fields",
            f"{available_profile_fields}/8",
            "Core Yahoo profile fields available.",
        ),
        AnalysisItem(
            "Gemini",
            "On" if gemini_active else "Off",
            "AI reasoning enrichment status.",
        ),
    ]
    return DataQualitySummary(status=status, items=items, warnings=warnings)


def fundamental_items(profile: dict[str, Any]) -> list[AnalysisItem]:
    """Build company fundamentals summary items.

    Args:
        profile: Yahoo Finance company profile.

    Returns:
        Informational fundamentals summary items.
    """

    return [
        AnalysisItem("Market cap", _format_number(profile.get("marketCap")), "Company scale."),
        AnalysisItem("Revenue", _format_number(profile.get("totalRevenue")), "Trailing revenue."),
        AnalysisItem("Profit margin", _format_percent(profile.get("profitMargins")), "Net margin."),
        AnalysisItem(
            "Debt to equity",
            _format_number(profile.get("debtToEquity")),
            "Leverage ratio.",
        ),
        AnalysisItem(
            "Return on equity",
            _format_percent(profile.get("returnOnEquity")),
            "ROE.",
        ),
    ]


def valuation_items(profile: dict[str, Any]) -> list[AnalysisItem]:
    """Build valuation summary items.

    Args:
        profile: Yahoo Finance company profile.

    Returns:
        Informational valuation summary items.
    """

    return [
        AnalysisItem(
            "Trailing P/E",
            _format_number(profile.get("trailingPE")),
            "Earnings multiple.",
        ),
        AnalysisItem(
            "Forward P/E",
            _format_number(profile.get("forwardPE")),
            "Forward earnings multiple.",
        ),
        AnalysisItem(
            "Price/book",
            _format_number(profile.get("priceToBook")),
            "Book-value multiple.",
        ),
        AnalysisItem(
            "Enterprise value / EBITDA",
            _format_number(profile.get("enterpriseToEbitda")),
            "EV to EBITDA multiple.",
        ),
    ]


def sentiment_items(profile: dict[str, Any], quote: MarketQuote) -> list[AnalysisItem]:
    """Build market sentiment summary items.

    Args:
        profile: Yahoo Finance company profile.
        quote: Current quote.

    Returns:
        Informational market sentiment summary items.
    """

    return [
        AnalysisItem(
            "Analyst count",
            _format_number(profile.get("numberOfAnalystOpinions")),
            "Yahoo Finance analyst coverage count when available.",
        ),
        AnalysisItem(
            "Beta",
            _format_number(profile.get("beta")),
            "Historical sensitivity to broad market movement.",
        ),
        AnalysisItem(
            "Price vs previous close",
            _format_percent(_percent_change(quote.close_price, quote.last_price)),
            "Intraday move relative to previous close when available.",
        ),
    ]


def risk_items(
    profile: dict[str, Any],
    quote: MarketQuote,
    bars: list[HistoricalBar],
) -> list[AnalysisItem]:
    """Build risk-management summary items.

    Args:
        profile: Yahoo Finance company profile.
        quote: Current quote.
        bars: Historical bars.

    Returns:
        Informational risk summary items.
    """

    highs = [bar.high_price for bar in bars if bar.high_price is not None]
    lows = [bar.low_price for bar in bars if bar.low_price is not None]
    window_high = max(highs) if highs else quote.high_price
    window_low = min(lows) if lows else quote.low_price
    range_percent = _percent_change(window_low, window_high)
    beta = profile.get("beta")

    return [
        AnalysisItem(
            "Window high",
            _format_number(window_high),
            "Highest price in selected window.",
        ),
        AnalysisItem("Window low", _format_number(window_low), "Lowest price in selected window."),
        AnalysisItem(
            "Window range",
            _format_percent(range_percent),
            "Selected-window high/low spread.",
        ),
        AnalysisItem("Beta", _format_number(beta), "Broad-market sensitivity when available."),
    ]


def macro_items() -> list[AnalysisItem]:
    """Build macro summary placeholders.

    Returns:
        Macro context items.
    """

    return [
        AnalysisItem(
            "Macro data source",
            "Not connected",
            "Wire RBI, inflation, rates, currency, and global index feeds here.",
        ),
        AnalysisItem(
            "Interpretation",
            "Neutral",
            "No live macro feed is configured, so the dashboard avoids inference.",
        ),
    ]


def sector_items(sector: str) -> list[AnalysisItem]:
    """Build sector outlook placeholders.

    Args:
        sector: Selected stock sector.

    Returns:
        Sector context items.
    """

    return [
        AnalysisItem("Sector", sector, "Mapped from the local NSE symbol catalog."),
        AnalysisItem(
            "Outlook data source",
            "Not connected",
            "Add sector index, breadth, earnings, and relative-strength feeds here.",
        ),
    ]


def _average(values: list[int]) -> float | None:
    """Return an average value.

    Args:
        values: Numeric values.

    Returns:
        Average or None.
    """

    return mean(values) if values else None


def _bars_to_frame(bars: list[HistoricalBar]) -> pd.DataFrame:
    """Convert bars to a DataFrame for shared technical helpers."""

    return pd.DataFrame(
        [
            {
                "high": bar.high_price,
                "low": bar.low_price,
                "close": bar.close_price,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )


def _average_float(values: list[float]) -> float | None:
    """Return an average float value."""

    return mean(values) if values else None


def _moving_average(values: list[float], window: int) -> float | None:
    """Return moving average for the latest window."""

    if len(values) < window:
        return None
    return mean(values[-window:])


def _rsi(values: list[float], window: int) -> float | None:
    """Return RSI for the latest window."""

    if len(values) <= window:
        return None

    changes = [current - previous for previous, current in zip(values, values[1:], strict=False)]
    recent_changes = changes[-window:]
    gains = [max(change, 0.0) for change in recent_changes]
    losses = [abs(min(change, 0.0)) for change in recent_changes]
    average_gain = mean(gains)
    average_loss = mean(losses)
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _macd(values: list[float]) -> tuple[float | None, float | None]:
    """Return latest MACD and signal values."""

    if len(values) < 35:
        return None, None

    ema_12 = _ema_series(values, 12)
    ema_26 = _ema_series(values, 26)
    macd_series = [short - long for short, long in zip(ema_12, ema_26, strict=True)]
    signal_series = _ema_series(macd_series, 9)
    return macd_series[-1], signal_series[-1]


def _atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    window: int,
) -> float | None:
    """Return average true range."""

    true_ranges = _true_ranges(highs, lows, closes)
    if len(true_ranges) < window:
        return None
    return mean(true_ranges[-window:])


def _adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    window: int,
) -> float | None:
    """Return average directional index."""

    if len(highs) <= window or len(lows) <= window or len(closes) <= window:
        return None

    true_ranges = _true_ranges(highs, lows, closes)
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for index in range(1, min(len(highs), len(lows))):
        up_move = highs[index] - highs[index - 1]
        down_move = lows[index - 1] - lows[index]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)

    dx_values: list[float] = []
    for index in range(window - 1, len(true_ranges)):
        tr_sum = sum(true_ranges[index - window + 1 : index + 1])
        if tr_sum == 0:
            continue
        plus_di = 100 * sum(plus_dm[index - window + 1 : index + 1]) / tr_sum
        minus_di = 100 * sum(minus_dm[index - window + 1 : index + 1]) / tr_sum
        denominator = plus_di + minus_di
        if denominator:
            dx_values.append(100 * abs(plus_di - minus_di) / denominator)

    if len(dx_values) < window:
        return None
    return mean(dx_values[-window:])


def _true_ranges(
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> list[float]:
    """Return true range values."""

    count = min(len(highs), len(lows), len(closes))
    if count < 2:
        return []

    ranges: list[float] = []
    for index in range(1, count):
        ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - closes[index - 1]),
                abs(lows[index] - closes[index - 1]),
            )
        )
    return ranges


def _ema_series(values: list[float], span: int) -> list[float]:
    """Return an EMA series."""

    multiplier = 2.0 / (span + 1)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])
    return ema_values


def _volume_trend(recent_volume: float | None, baseline_volume: float | None) -> str:
    """Return volume trend text."""

    if recent_volume is None or baseline_volume in (None, 0):
        return "N/A"
    change = (recent_volume - baseline_volume) / baseline_volume
    if change > 0.15:
        return "Rising"
    if change < -0.15:
        return "Falling"
    return "Stable"


def _technical_bias(
    *,
    latest_close: float | int | None,
    sma_20: float | None,
    sma_50: float | None,
    rsi_14: float | None,
    macd: float | None,
    macd_signal: float | None,
    adx_14: float | None,
) -> str:
    """Return a technical bias label."""

    score = 0
    if latest_close is not None and sma_20 is not None:
        score += 1 if float(latest_close) >= sma_20 else -1
    if sma_20 is not None and sma_50 is not None:
        score += 1 if sma_20 >= sma_50 else -1
    if rsi_14 is not None:
        if 45 <= rsi_14 <= 70:
            score += 1
        elif rsi_14 < 35 or rsi_14 > 75:
            score -= 1
    if macd is not None and macd_signal is not None:
        score += 1 if macd >= macd_signal else -1
    if adx_14 is not None and adx_14 >= 25:
        score += 1 if score > 0 else -1

    if score >= 2:
        return "Bullish"
    if score <= -2:
        return "Bearish"
    return "Neutral"


def _technical_interpretation(
    *,
    bias: str,
    latest_close: float | int | None,
    sma_20: float | None,
    sma_50: float | None,
    rsi_14: float | None,
    macd: float | None,
    macd_signal: float | None,
    adx_14: float | None,
    atr_14: float | None,
    support: float | None,
    resistance: float | None,
    volume_trend: str,
    darvas_status: str | None,
    darvax_patterns: list[str],
) -> str:
    """Return a concise technical interpretation."""

    details: list[str] = [f"Technical setup is {bias.lower()}."]
    if latest_close is not None and sma_20 is not None:
        relation = "above" if float(latest_close) >= sma_20 else "below"
        details.append(f"Price is {relation} the 20-period average.")
    if sma_20 is not None and sma_50 is not None:
        relation = "above" if sma_20 >= sma_50 else "below"
        details.append(f"20-period average is {relation} the 50-period average.")
    if rsi_14 is not None:
        details.append(f"RSI is {rsi_14:.2f}.")
    if macd is not None and macd_signal is not None:
        relation = "above" if macd >= macd_signal else "below"
        details.append(f"MACD is {relation} signal.")
    if adx_14 is not None:
        details.append(f"ADX is {adx_14:.2f}.")
    if atr_14 is not None:
        details.append(f"ATR is {_format_number(atr_14)}.")
    if support is not None and resistance is not None:
        details.append(
            "Recent support/resistance are "
            f"{_format_number(support)} and {_format_number(resistance)}."
        )
    if volume_trend != "N/A":
        details.append(f"Volume trend is {volume_trend.lower()}.")
    if darvas_status:
        details.append(f"Darvas Box status is {darvas_status.lower()}.")
    if darvax_patterns:
        details.append(f"Detected broader DarvaX setups: {', '.join(darvax_patterns)}.")
    return " ".join(details)


def _percent_change(start: float | int | None, end: float | int | None) -> float | None:
    """Return percent change.

    Args:
        start: Starting value.
        end: Ending value.

    Returns:
        Percent change as a decimal.
    """

    if start in (None, 0) or end is None:
        return None
    return (float(end) - float(start)) / float(start)


def _format_percent(value: float | int | None) -> str:
    """Format a decimal as percent."""

    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _format_number(value: float | int | None) -> str:
    """Format a number."""

    if value is None:
        return "N/A"
    if abs(float(value)) >= 1_000_000_000:
        return f"{float(value) / 1_000_000_000:.2f}B"
    if abs(float(value)) >= 1_000_000:
        return f"{float(value) / 1_000_000:.2f}M"
    return f"{float(value):,.2f}"
