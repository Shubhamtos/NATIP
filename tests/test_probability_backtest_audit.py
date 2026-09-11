"""Tests for probability model audit and backtest safeguards."""

from __future__ import annotations

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio


def test_backtest_uses_non_overlapping_20_day_periods() -> None:
    """A 40-date prediction set should produce two non-overlapping periods."""

    rows = []
    dates = pd.bdate_range("2025-01-01", periods=40)
    for date in dates:
        rows.extend(
            [
                {
                    "Date": date,
                    "symbol": "AAA.NS",
                    "p_outperform": 0.80,
                    "future_stock_return": 0.04,
                    "future_nifty_return": 0.01,
                },
                {
                    "Date": date,
                    "symbol": "BBB.NS",
                    "p_outperform": 0.70,
                    "future_stock_return": 0.02,
                    "future_nifty_return": 0.01,
                },
            ]
        )
    result = backtest_top_n_portfolio(pd.DataFrame(rows), top_n=2, transaction_cost_bps=0)

    assert result["periods"] == 2
    assert result["non_overlapping_holding_periods"] is True
    assert result["max_weight_sum"] <= 1.0
    assert result["leverage_enabled"] is False


def test_backtest_removes_duplicate_symbol_predictions() -> None:
    """Duplicate Date/symbol rows should not create duplicate portfolio exposure."""

    predictions = pd.DataFrame(
        [
            {
                "Date": "2025-01-01",
                "symbol": "AAA.NS",
                "p_outperform": 0.90,
                "future_stock_return": 0.10,
                "future_nifty_return": 0.01,
            },
            {
                "Date": "2025-01-01",
                "symbol": "AAA.NS",
                "p_outperform": 0.80,
                "future_stock_return": 0.05,
                "future_nifty_return": 0.01,
            },
        ]
    )
    result = backtest_top_n_portfolio(predictions, top_n=2, transaction_cost_bps=0)

    assert result["duplicate_prediction_rows_removed"] == 1
    assert result["max_weight_sum"] == 1.0
