"""Regression tests for portfolio metric arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from app.backtesting.metrics.performance import (
    calculate_performance_metrics,
    stock_performance_summary,
)


def test_total_return_compounds_instead_of_summing() -> None:
    """Two 10% periods produce 21%, not 20%."""

    periods = pd.DataFrame(
        {
            "exit_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "net_return": [0.10, 0.10],
            "gross_return": [0.10, 0.10],
            "benchmark_return": [0.0, 0.0],
            "turnover": [1.0, 1.0],
            "gross_exposure": [1.0, 1.0],
        }
    )
    trades = pd.DataFrame({"net_return": [0.10, 0.10], "holding_days": [20, 20]})

    metrics = calculate_performance_metrics(periods, trades)

    assert metrics["total_return"] == pytest.approx(0.21)


def test_return_below_minus_one_is_rejected() -> None:
    """A loss beyond capital is evidence of leverage or a calculation defect."""

    periods = pd.DataFrame(
        {
            "exit_date": ["2024-01-31"],
            "net_return": [-1.1],
            "gross_return": [-1.1],
            "benchmark_return": [0.0],
            "turnover": [1.0],
            "gross_exposure": [1.0],
        }
    )

    with pytest.raises(ValueError, match="leverage"):
        calculate_performance_metrics(periods, pd.DataFrame())


def test_stock_performance_summary_ranks_by_weighted_net_contribution() -> None:
    trades = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "AAA.NS", "BBB.NS"],
            "entry_date": ["2025-01-02", "2025-02-03", "2025-01-02"],
            "exit_date": ["2025-01-22", "2025-02-21", "2025-01-22"],
            "net_return": [0.10, -0.02, 0.20],
            "gross_contribution": [0.0105, -0.0015, 0.0045],
            "net_contribution": [0.01, -0.002, 0.004],
            "exit_reason": ["rebalance", "stop_loss", "rebalance"],
            "holding_days": [20, 18, 20],
            "fold": [1, 2, 1],
        }
    )

    summary = stock_performance_summary(trades)

    assert summary.iloc[0]["symbol"] == "AAA.NS"
    assert summary.iloc[0]["total_net_contribution"] == pytest.approx(0.008)
    assert summary.iloc[0]["win_rate"] == pytest.approx(0.5)
    assert summary.iloc[0]["stop_loss_count"] == 1
    assert summary.iloc[0]["data_review"] == "PASS"
    assert summary.iloc[1]["sample_quality"] == "LOW_SAMPLE"
