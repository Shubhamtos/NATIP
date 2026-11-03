"""No-look-ahead and point-in-time data tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtesting.data.yahoo import merge_point_in_time_fundamentals
from app.backtesting.signals.research import research_signal
from app.backtesting.signals.technical import momentum_6_1


def test_future_price_mutation_cannot_change_past_momentum() -> None:
    """A backward-looking signal at T must be invariant to all observations after T."""

    dates = pd.bdate_range("2024-01-01", periods=180)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA.NS",
            "adj_close": np.linspace(100, 180, len(dates)),
        }
    )
    cutoff = frame.index[150]
    original = momentum_6_1(frame).loc[cutoff]
    mutated = frame.copy()
    mutated.loc[cutoff + 1 :, "adj_close"] *= 100

    assert momentum_6_1(mutated).loc[cutoff] == original


def test_fundamental_value_is_unavailable_before_publication() -> None:
    """Quarter-end values cannot leak into dates before the result was published."""

    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-10", "2024-07-19", "2024-07-22"]),
            "symbol": ["AAA.NS"] * 3,
        }
    )
    facts = pd.DataFrame(
        {
            "symbol": ["AAA.NS"],
            "period_end": ["2024-06-30"],
            "publication_date": ["2024-07-19"],
            "sales_yoy": [0.20],
        }
    )
    merged = merge_point_in_time_fundamentals(market, facts)

    assert pd.isna(merged.loc[0, "sales_yoy"])
    assert merged.loc[1, "sales_yoy"] == 0.20
    assert (
        merged.loc[merged["available_from"].notna(), "available_from"]
        <= merged.loc[merged["available_from"].notna(), "date"]
    ).all()


def test_signal_research_forward_return_starts_at_next_open() -> None:
    """Signal diagnostics must not award the untradeable T-close to T+1-open move."""

    dates = pd.bdate_range("2024-01-01", periods=20)
    opens = [100, *[200 + 2 * index for index in range(19)]]
    stock = pd.DataFrame(
        {
            "date": dates,
            "symbol": "AAA.NS",
            "adj_open": opens,
            "adj_close": [100] * 20,
        }
    )
    benchmark = stock.assign(symbol="^NSEI", adj_open=100, adj_close=100)
    panel = pd.concat([stock, benchmark], ignore_index=True)

    def constant_signal(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=frame.index)

    result = research_signal(
        panel,
        constant_signal,
        horizons=[5],
        quantiles=1,
        rebalance_frequency="weekly",
    )
    first = result.scored_panel[result.scored_panel["symbol"] == "AAA.NS"].iloc[0]

    signal_location = dates.get_loc(first["date"])
    assert first["forward_return_5d"] == (
        opens[signal_location + 6] / opens[signal_location + 1]
    ) - 1
