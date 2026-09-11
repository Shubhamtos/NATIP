"""Tests for the astro chart-reaction screener."""

from __future__ import annotations

import pandas as pd

from app.dashboard.astro_chart_screener import (
    score_astro_chart_reaction,
    score_future_astro_trend_watch,
)


def test_score_astro_chart_reaction_detects_volume_confirmed_move() -> None:
    """A clear post-event price/volume move should receive a useful score."""

    index = pd.date_range("2026-01-01", periods=45, freq="B")
    close = pd.Series(range(100, 145), index=index, dtype=float)
    stock = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": [100_000] * 20 + [180_000] * 25,
        },
        index=index,
    )
    benchmark = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
        },
        index=index,
    )

    result = score_astro_chart_reaction(
        symbol="TEST",
        company="Test Company",
        frame=stock,
        benchmark_frame=benchmark,
        events=[{"timestamp": index[25], "priority": 3, "event": "Test astro window"}],
    )

    assert result.error is None
    assert result.reaction_score >= 55
    assert result.direction == "Up confirmation"
    assert result.volume_confirmation == "Strong"
    assert result.post_5d_excess_pct is not None
    assert result.post_5d_excess_pct > 0


def test_score_future_astro_trend_watch_flags_sensitive_chart() -> None:
    """A compressed chart near resistance should be flagged before a future window."""

    index = pd.date_range("2026-01-01", periods=120, freq="B")
    close = pd.Series([100 + index_ * 0.2 for index_ in range(120)], index=index)
    close.iloc[-10:] = [122.0, 122.1, 122.3, 122.2, 122.4, 122.5, 122.6, 122.55, 122.7, 122.8]
    stock = pd.DataFrame(
        {
            "open": close - 0.15,
            "high": close + 0.35,
            "low": close - 0.35,
            "close": close,
            "volume": [200_000] * 115 + [100_000] * 5,
        },
        index=index,
    )

    result = score_future_astro_trend_watch(
        symbol="TEST",
        company="Test Company",
        frame=stock,
        events=[{"timestamp": "2026-06-20", "priority": 3, "event": "Future astro window"}],
        as_of=pd.Timestamp("2026-06-15").to_pydatetime(),
    )

    assert result.error is None
    assert result.watch_score >= 35
    assert result.watch_label in {"Low-priority watch", "Medium-priority watch", "High-priority watch"}
    assert result.setup_direction == "Upside breakout watch"
    assert result.volume_state == "Dry-up"
