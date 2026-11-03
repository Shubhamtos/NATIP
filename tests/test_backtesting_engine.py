"""Portfolio execution and exposure safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.engine.portfolio import run_strategy_backtest


def test_signal_at_close_trades_only_at_next_session_open() -> None:
    """No position may fill on the same day that produced its score."""

    panel, universe = _synthetic_panel()
    config = BacktestConfig()
    config.data.min_avg_traded_value_20d = 0
    config.strategy.top_n = 2
    config.strategy.stop_loss_pct = None
    config.costs.stt_buy_bps = 0
    config.costs.stt_sell_bps = 0
    config.costs.exchange_transaction_bps = 0
    config.costs.sebi_bps = 0
    config.costs.stamp_duty_buy_bps = 0
    config.costs.slippage_bps = 0

    def score(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("symbol")["adj_close"].pct_change(20)

    result = run_strategy_backtest(panel, universe, score, config)

    assert not result.trades.empty
    assert (
        pd.to_datetime(result.trades["entry_date"]) > pd.to_datetime(result.trades["signal_date"])
    ).all()
    assert (result.periods["gross_exposure"] <= 1.0 + 1e-9).all()
    assert (result.periods["net_exposure"] <= 1.0 + 1e-9).all()


def _synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2023-01-02", periods=180)
    rows = []
    for index, symbol in enumerate(["AAA.NS", "BBB.NS", "^NSEI"]):
        close = np.linspace(100 + index * 5, 140 + index * 5, len(dates))
        for date, price in zip(dates, close, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "raw_open": price,
                    "raw_high": price * 1.01,
                    "raw_low": price * 0.99,
                    "raw_close": price,
                    "adj_open": price,
                    "adj_high": price * 1.01,
                    "adj_low": price * 0.99,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
    universe = pd.DataFrame(
        {"symbol": ["AAA.NS", "BBB.NS"], "company": ["A", "B"], "sector": ["IT", "Bank"]}
    )
    return pd.DataFrame(rows), universe
