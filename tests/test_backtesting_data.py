"""Market-data normalization and session cutoff tests."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from app.backtesting.data.yahoo import _normalize, latest_completed_nse_day


def test_latest_completed_day_excludes_live_session() -> None:
    """A run before the NSE close must not consume a partial same-day bar."""

    now = datetime(2026, 10, 22, 11, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert latest_completed_nse_day(now) == pd.Timestamp("2026-10-21")


def test_latest_completed_day_rolls_weekend_back() -> None:
    """Weekend runs should use Friday as their conservative upper bound."""

    now = datetime(2026, 10, 25, 18, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert latest_completed_nse_day(now) == pd.Timestamp("2026-10-23")


def test_old_cache_is_upgraded_with_adjustment_audit_columns() -> None:
    """Existing Parquet caches must gain new quality fields during normalization."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "raw_open": [100, 50],
            "raw_high": [101, 51],
            "raw_low": [99, 49],
            "raw_close": [100, 50],
            "adj_open": [50, 50],
            "adj_high": [50.5, 51],
            "adj_low": [49.5, 49],
            "adj_close": [50, 50],
            "volume": [1000, 1000],
        }
    )

    normalized = _normalize(frame)

    assert "corporate_action_adjustment" in normalized
    assert "is_abnormal_move" in normalized
    assert normalized.loc[1, "corporate_action_adjustment"]
