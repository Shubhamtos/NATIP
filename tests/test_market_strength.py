"""Tests for deterministic same-date market-strength research."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from app.database.market_research import MarketResearchStore
from app.intelligence.market_strength import (
    apply_ready_scan,
    breakout_candidates,
    build_market_strength_snapshot,
    market_strength_alerts,
)


def _history(multiplier: float, *, periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    close = pd.Series(100 * np.exp(np.linspace(0, multiplier, periods)))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.linspace(1_000_000, 1_500_000, periods),
        }
    )


def test_market_strength_ranks_same_date_universe() -> None:
    histories = {
        "AAA.NS": _history(1.0),
        "BBB.NS": _history(0.6),
        "CCC.NS": _history(0.2),
        "DDD.NS": _history(-0.2),
        "^NSEI": _history(0.3),
    }
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"],
            "company": ["A", "B", "C", "D"],
            "sector": ["Tech", "Tech", "Bank", "Bank"],
        }
    )

    snapshot = build_market_strength_snapshot(histories, universe)

    assert snapshot.stocks.iloc[0]["Ticker"] == "AAA.NS"
    assert snapshot.stocks["RS Rating"].between(1, 99).all()
    assert snapshot.stocks["Data Date"].nunique() == 1
    assert snapshot.stocks["Trend Checks"].between(0, 8).all()
    assert {
        "SMA20",
        "SMA150",
        "SMA200 Rising",
        "Price Above Long MAs",
    }.issubset(snapshot.stocks.columns)
    assert snapshot.sectors["Stocks"].sum() == 4
    assert {"3M Return", "RS Momentum"}.issubset(snapshot.sectors.columns)
    assert snapshot.breadth["eligible_stocks"] == 4


def test_future_stock_row_after_benchmark_date_cannot_enter_snapshot() -> None:
    histories = {
        "AAA.NS": _history(0.8),
        "BBB.NS": _history(0.4),
        "^NSEI": _history(0.3),
    }
    future = histories["AAA.NS"].tail(1).copy()
    future.loc[:, "Date"] = histories["AAA.NS"]["Date"].iloc[-1].to_pydatetime() + timedelta(
        days=10
    )
    future.loc[:, "Close"] = 10_000.0
    histories["AAA.NS"] = pd.concat([histories["AAA.NS"], future], ignore_index=True)
    universe = pd.DataFrame(
        {
            "symbol": ["AAA.NS", "BBB.NS"],
            "company": ["A", "B"],
            "sector": ["Tech", "Bank"],
        }
    )

    snapshot = build_market_strength_snapshot(histories, universe)

    assert snapshot.as_of_date == histories["^NSEI"]["Date"].max()
    latest = snapshot.stocks.loc[snapshot.stocks["Ticker"] == "AAA.NS", "Close"].iloc[0]
    assert float(latest) < 1_000


def test_ready_scans_and_watchlist_alerts() -> None:
    stocks = pd.DataFrame(
        {
            "Ticker": ["AAA.NS", "BBB.NS"],
            "RS Rating": [95.0, 15.0],
            "RS Composite": [0.8, -0.3],
            "RS Change 5D": [8.0, -20.0],
            "Stage": ["Stage 2", "Stage 4"],
            "Distance 52W High": [-0.01, -0.40],
            "Near Breakout": [True, False],
            "Breakout": [False, False],
            "Pocket Pivot": [True, False],
            "Volume Ratio": [1.2, 0.8],
            "1D Return": [0.01, -0.02],
            "1M Return": [0.08, -0.10],
            "Distance to Pivot": [-0.01, -0.20],
        }
    )

    assert apply_ready_scan(stocks, "RS Leaders")["Ticker"].tolist() == ["AAA.NS"]
    breakouts = breakout_candidates(stocks)
    assert breakouts["Ticker"].tolist() == ["AAA.NS"]
    assert breakouts.iloc[0]["Setup"] == "Near breakout · Pocket pivot"
    alerts = market_strength_alerts(stocks, {"AAA.NS", "BBB.NS"})
    assert {"RS leader", "Near breakout", "RS deterioration"}.issubset(set(alerts["Alert"]))


def test_market_research_store_round_trip(tmp_path) -> None:
    store = MarketResearchStore(tmp_path / "research.sqlite3")
    store.upsert_watchlist("AAA.NS", note="Leader", tags="breakout")
    store.append_journal(
        ticker="AAA.NS",
        thesis="Strong relative strength",
        invalidation="Break below SMA50",
        horizon="5–20 trading days",
        context='{"rs": 95}',
    )

    assert store.list_watchlist()[0]["ticker"] == "AAA.NS"
    assert store.list_journal()[0]["invalidation"] == "Break below SMA50"
    store.remove_watchlist("AAA.NS")
    assert store.list_watchlist() == []
