"""Walk-forward chronology safeguards."""

from __future__ import annotations

import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.validation.robustness import build_walk_forward_folds


def test_walk_forward_uses_twenty_trading_session_embargo() -> None:
    """The final training label must end 20 sessions before validation begins."""

    dates = pd.Series(pd.bdate_range("2018-01-01", "2025-12-31"))
    config = BacktestConfig()
    folds = build_walk_forward_folds(dates, config)

    assert folds
    sessions = pd.DatetimeIndex(dates)
    for fold in folds:
        train_position = sessions.get_loc(fold.train_end)
        test_position = sessions.get_loc(fold.test_start)
        assert test_position - train_position - 1 == 20
