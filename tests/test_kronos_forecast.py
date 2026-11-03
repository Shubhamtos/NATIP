"""Tests for NATIP's Kronos stock-direction research integration."""

from __future__ import annotations

import pandas as pd
import pytest

from app.intelligence.kronos_forecast import (
    KronosForecastError,
    forecast_stock_direction,
    normalize_nse_symbol,
    normalize_stock_symbol,
)


def _history(rows: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), index=dates, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=dates,
    )


class FakePredictor:
    """Deterministic predictor that records the supplied historical context."""

    device = "cpu-test"

    def __init__(self, final_multiplier: float) -> None:
        self.final_multiplier = final_multiplier
        self.context_timestamps: pd.Series | None = None

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        **_: object,
    ) -> pd.DataFrame:
        self.context_timestamps = x_timestamp.copy()
        start = float(df["close"].iloc[-1])
        finish = start * self.final_multiplier
        closes = pd.Series(
            [start + (finish - start) * (index + 1) / pred_len for index in range(pred_len)]
        )
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "close": closes,
                "volume": 1_000_000.0,
                "amount": closes * 1_000_000.0,
            }
        )


def test_forecast_uses_only_history_available_by_signal_date() -> None:
    history = _history()
    as_of = history.index[-20]
    predictor = FakePredictor(1.08)

    result = forecast_stock_direction(
        history,
        symbol="RELIANCE",
        predictor=predictor,
        horizon=5,
        lookback=120,
        sample_count=1,
        as_of=as_of,
    )

    assert result.direction == "UP"
    assert result.signal_date == as_of
    assert result.lookback == 120
    assert predictor.context_timestamps is not None
    assert pd.Timestamp(predictor.context_timestamps.max()) == as_of
    assert result.forecast.index.min() > as_of


@pytest.mark.parametrize(
    ("multiplier", "expected"),
    [(1.08, "UP"), (0.92, "DOWN"), (1.001, "SIDEWAYS")],
)
def test_direction_labels_follow_forecast_move(multiplier: float, expected: str) -> None:
    result = forecast_stock_direction(
        _history(),
        symbol="TCS.NS",
        predictor=FakePredictor(multiplier),
        horizon=10,
        lookback=252,
        sample_count=1,
    )

    assert result.direction == expected


def test_invalid_symbol_and_short_history_fail_closed() -> None:
    with pytest.raises(KronosForecastError, match="NSE"):
        normalize_nse_symbol("AAPL.US")
    with pytest.raises(KronosForecastError, match="At least"):
        forecast_stock_direction(
            _history(80),
            symbol="INFY",
            predictor=FakePredictor(1.05),
            horizon=5,
            lookback=120,
            sample_count=1,
        )


def test_us_tickers_preserve_yahoo_format_and_market():
    assert normalize_stock_symbol(" aapl ", "US") == "AAPL"
    assert normalize_stock_symbol("BRK-B", "US") == "BRK-B"
    assert normalize_stock_symbol("RELIANCE", "NSE") == "RELIANCE.NS"
    for ticker in ("AAPL.NS", "AAPL.US", "../AAPL", "^GSPC"):
        with pytest.raises(KronosForecastError):
            normalize_stock_symbol(ticker, "US")
    result = forecast_stock_direction(
        _history(),
        symbol="AAPL",
        market="US",
        predictor=FakePredictor(1.05),
        horizon=10,
        lookback=252,
    )
    assert result.symbol == "AAPL"
    assert result.market == "US"
