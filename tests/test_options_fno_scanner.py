from __future__ import annotations

import pandas as pd

from app.intelligence.options import (
    OptionRight,
    OptionsBuyingConfig,
    OptionsScanAction,
    scan_underlying_options_setup,
)
from app.intelligence.options.calculations import prior_resistance, prior_support


def _trend_frame(direction: str = "up", *, breakout: bool = True) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for index in range(70):
        close = 100 + index * 0.7 if direction == "up" else 150 - index * 0.7
        rows.append(
            {
                "open": close - 0.4 if direction == "up" else close + 0.4,
                "high": close + 0.8,
                "low": close - 0.8,
                "close": close,
                "volume": 1000,
            }
        )
    if breakout and direction == "up":
        prior_high = max(row["high"] for row in rows[-20:])
        rows[-1].update(
            {
                "open": prior_high + 0.2,
                "high": prior_high + 2.0,
                "low": prior_high - 0.5,
                "close": prior_high + 1.6,
                "volume": 2000,
            }
        )
    if breakout and direction == "down":
        prior_low = min(row["low"] for row in rows[-20:])
        rows[-1].update(
            {
                "open": prior_low - 0.2,
                "high": prior_low + 0.5,
                "low": prior_low - 2.0,
                "close": prior_low - 1.6,
                "volume": 2000,
            }
        )
    return pd.DataFrame(rows)


def test_fno_scanner_returns_call_setup_for_clean_bullish_underlying() -> None:
    frame = _trend_frame("up")
    pivot = prior_resistance(frame, 20)
    assert pivot is not None
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=99,
        maximum_gap_from_pivot_atr=99,
    )

    result = scan_underlying_options_setup(
        symbol="ABC",
        daily_bars=frame,
        market_daily_bars=_trend_frame("up", breakout=False),
        sector_daily_bars=_trend_frame("up", breakout=False),
        confirmation_30m_bar={"close": pivot + 0.2},
        session_vwap=pivot,
        stock_relative_strength_vs_market=0.03,
        stock_relative_strength_vs_sector=0.02,
        config=config,
    )

    assert result.action == OptionsScanAction.CALL_SETUP
    assert result.side == OptionRight.CALL
    assert "call_breakout" in result.passed_rules
    assert "DTE" in result.option_selection_rule


def test_fno_scanner_returns_put_setup_for_clean_bearish_underlying() -> None:
    frame = _trend_frame("down")
    pivot = prior_support(frame, 20)
    assert pivot is not None
    config = OptionsBuyingConfig(
        maximum_ema10_extension_atr=99,
        maximum_gap_from_pivot_atr=99,
    )

    result = scan_underlying_options_setup(
        symbol="ABC",
        daily_bars=frame,
        market_daily_bars=_trend_frame("down", breakout=False),
        sector_daily_bars=_trend_frame("down", breakout=False),
        confirmation_30m_bar={"close": pivot - 0.2},
        session_vwap=pivot,
        stock_relative_strength_vs_market=-0.03,
        stock_relative_strength_vs_sector=-0.02,
        config=config,
    )

    assert result.action == OptionsScanAction.PUT_SETUP
    assert result.side == OptionRight.PUT
    assert "put_breakdown" in result.passed_rules


def test_fno_scanner_fails_closed_when_data_is_insufficient() -> None:
    result = scan_underlying_options_setup(
        symbol="ABC",
        daily_bars=_trend_frame("up").head(10),
        market_daily_bars=None,
        sector_daily_bars=None,
        confirmation_30m_bar=None,
        session_vwap=None,
        stock_relative_strength_vs_market=None,
        stock_relative_strength_vs_sector=None,
    )

    assert result.action == OptionsScanAction.AVOID
    assert "data_available" in result.failed_rules
