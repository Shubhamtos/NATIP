"""Synthetic fixtures test date safety, per-metric denominators, and UI behavior."""

import numpy as np
import pandas as pd
import pytest

from app.dashboard.market_breadth_data import load_breadth_histories, update_breadth_data
from app.intelligence.market_breadth import build_breadth_report, normalize_history


def history(direction=1, periods=280):
    close = np.arange(periods) * direction + (500 if direction < 0 else 100)
    return pd.DataFrame(
        {
            "Date": pd.bdate_range("2023-01-02", periods=periods),
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close.astype(float),
            "Volume": 1000.0,
        }
    )


def data():
    stocks = {"UP.NS": history(), "DOWN.NS": history(-1), "^NSEI": history()}
    universe = pd.DataFrame(
        {
            "symbol": ["UP.NS", "DOWN.NS", "MISSING.NS"],
            "company": ["Up", "Down", "Missing"],
            "sector": ["Tech", "Banks", "Tech"],
        }
    )
    return stocks, universe


def report():
    stocks, universe = data()
    return build_breadth_report(stocks, universe, as_of=stocks["^NSEI"].Date.max())


def test_counts_and_metric_denominators_do_not_count_missing_as_declining():
    result = report()
    row = result.timeline.iloc[-1]
    assert row.Requested == 3
    assert row.Eligible == row["AD Eligible"] == 2
    assert row.Advancers == row.Decliners == 1
    assert row.Unchanged == 0
    assert row["Above SMA50 %"] == 50
    assert row["Coverage %"] == pytest.approx(200 / 3)
    assert row["New 52W High Count"] == row["New 52W Low Count"] == 1
    assert row["AD Ratio"] == 1
    assert len(result.stocks) == 3
    assert result.sectors.Requested.sum() == 3


def test_new_listing_can_contribute_without_long_window():
    stocks, universe = data()
    stocks["MISSING.NS"] = history().tail(30)
    result = build_breadth_report(stocks, universe, as_of=stocks["^NSEI"].Date.max())
    row = result.timeline.iloc[-1]
    assert row.Eligible == 3
    assert row["SMA20 Eligible"] == 3
    assert row["SMA50 Eligible"] == row["SMA200 Eligible"] == 2
    assert row["Above SMA20 %"] == pytest.approx(200 / 3)


def test_future_changes_cannot_affect_historical_report():
    stocks, universe = data()
    cutoff = stocks["^NSEI"].Date.iloc[-15]
    before = build_breadth_report(stocks, universe, as_of=cutoff)
    stocks["UP.NS"].loc[stocks["UP.NS"].Date.gt(cutoff), ["Open", "High", "Low", "Close"]] *= 100
    after = build_breadth_report(stocks, universe, as_of=cutoff)
    pd.testing.assert_frame_equal(before.timeline, after.timeline)
    pd.testing.assert_frame_equal(before.stocks, after.stocks)
    assert after.timeline.Date.max() == cutoff


@pytest.mark.parametrize(
    "problem", ["missing", "zero_volume", "invalid_ohlc", "source_stale", "infinite"]
)
def test_invalid_bar_does_not_get_filled_or_bridge_daily_returns(problem):
    stocks, universe = data()
    frame = stocks["UP.NS"].copy()
    if problem == "missing":
        frame = frame.drop(frame.index[-2])
    elif problem == "zero_volume":
        frame.loc[frame.index[-2], "Volume"] = 0
    elif problem == "invalid_ohlc":
        frame.loc[frame.index[-2], "High"] = 1
    elif problem == "infinite":
        frame.loc[frame.index[-2], "Close"] = np.inf
    else:
        frame["is_stale_row"] = False
        frame.loc[frame.index[-2], "is_stale_row"] = True
    stocks["UP.NS"] = frame
    result = build_breadth_report(stocks, universe, as_of=stocks["^NSEI"].Date.max())
    row = result.stocks.set_index("Ticker").loc["UP.NS"]
    assert pd.isna(row["Daily Return"])
    assert pd.isna(row["Above SMA20"])
    assert result.timeline.iloc[-1]["AD Eligible"] == 1


def test_no_decliners_ratio_is_missing_not_infinite():
    stocks, universe = data()
    stocks["DOWN.NS"] = history()
    result = build_breadth_report(stocks, universe, as_of=stocks["^NSEI"].Date.max())
    assert pd.isna(result.timeline.iloc[-1]["AD Ratio"])
    assert result.timeline.iloc[-1]["Up Volume %"] == 100


def test_structural_break_resets_returns_and_rolling_windows():
    stocks, universe = data()
    break_date = stocks["UP.NS"].Date.iloc[-10]
    result = build_breadth_report(
        stocks,
        universe,
        as_of=stocks["^NSEI"].Date.max(),
        structural_breaks={"UP.NS": (str(break_date.date()),)},
    )
    stock = result.stocks.set_index("Ticker").loc["UP.NS"]
    assert pd.isna(stock["Above SMA20"])
    assert pd.isna(stock["New 52W High"])
    assert pd.isna(stock["20D Return"])
    assert result.timeline.loc[result.timeline.Date.eq(break_date), "AD Eligible"].iloc[0] == 1


def test_ad_line_resets_after_low_coverage_gap():
    stocks, universe = data()
    gap = stocks["UP.NS"].Date.iloc[-4]
    stocks["UP.NS"] = stocks["UP.NS"].loc[stocks["UP.NS"].Date.ne(gap)]
    result = build_breadth_report(stocks, universe, as_of=stocks["^NSEI"].Date.max())
    values = result.timeline.set_index("Date")
    assert pd.isna(values.loc[gap, "AD Line"])
    assert not values.loc[gap, "Reliable AD"]
    assert values.iloc[-2]["Reliable AD"]
    assert values.iloc[-2]["AD Line"] == 0


def test_weekend_resolves_to_actual_benchmark_date_and_duplicate_rows_do_not_inflate():
    stocks, universe = data()
    stocks["UP.NS"] = pd.concat([stocks["UP.NS"], stocks["UP.NS"].tail(1)])
    universe = pd.concat([universe, universe.head(1)])
    cutoff = stocks["^NSEI"].Date.max() + pd.Timedelta(days=2)
    result = build_breadth_report(stocks, universe, as_of=cutoff)
    assert result.as_of == stocks["^NSEI"].Date.max()
    assert result.timeline.iloc[-1].Requested == 3
    assert result.timeline.iloc[-1].Eligible == 2


def test_adjusted_ohlc_selected_together_and_missing_adj_field_rejected():
    frame = history()
    for field in ("Open", "High", "Low", "Close"):
        frame[f"adj_{field}"] = frame[field] * 0.5
    assert normalize_history(frame).Close.iloc[-1] == frame.Close.iloc[-1] * 0.5
    with pytest.raises(ValueError, match="Open"):
        normalize_history(frame.drop(columns="adj_Open"))


def test_cache_uses_fresher_complete_source(tmp_path):
    old = tmp_path / "data/backtesting/cache"
    new = tmp_path / "data/kronos/price_cache"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    history().iloc[:-20].to_parquet(old / "UP_NS.parquet")
    history().set_index("Date").to_parquet(new / "UP_NS.parquet")
    stocks, audit = load_breadth_histories(tmp_path, ["UP.NS"], history().Date.max())
    assert normalize_history(stocks["UP.NS"]).Date.max() == history().Date.max()
    assert "kronos" in audit.iloc[0]["Cache Source"]


def test_update_continues_after_one_ticker_fails(tmp_path):
    class Store:
        def __init__(self, *args, **kwargs):
            pass

        def load_symbol(self, symbol, **kwargs):
            if symbol == "FAIL.NS":
                raise ValueError("Fixture failure")
            return pd.DataFrame({"date": [kwargs["end"]]}), "yfinance_incremental"

    progress = []
    audit = update_breadth_data(
        tmp_path,
        ["FAIL.NS", "GOOD.NS"],
        store_factory=Store,
        on_progress=lambda n, total, ticker: progress.append(ticker),
    )
    assert audit.Status.tolist() == ["FAILED", "UPDATED", "UPDATED"]
    assert progress == ["FAIL.NS", "GOOD.NS", "^NSEI"]


def test_ui_filters_downloads_and_chart_controls(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "app.dashboard.market_breadth.cached_breadth",
        lambda *a, **kw: (report(), "TEST FIXTURES ONLY"),
    )
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from app.dashboard.market_breadth import render_market_breadth\n"
        f"render_market_breadth(Path({str(tmp_path)!r}))"
    ).run(timeout=15)
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    app.selectbox(key="breadth-sector").set_value("Tech").run()
    app.selectbox(key="breadth-participation").set_value("Advancing").run()
    assert not app.exception
    assert app.dataframe[1].value.Ticker.tolist() == ["UP.NS"]
    app.text_input(key="breadth-search").set_value("not-a-stock").run()
    assert app.dataframe[1].value.empty
    assert not app.exception


def test_cloud_without_price_caches_still_shows_ema_tabs_and_initializes(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    initialized = []

    def cached(*args, **kwargs):
        if not initialized:
            raise ValueError("No adjusted cache in fresh hosted runtime")
        return report(), "TEST FIXTURE"

    cached.clear = lambda: None

    def update(root, symbols, **kwargs):
        initialized.append(tuple(symbols))
        return pd.DataFrame({"Ticker": symbols, "Status": ["UPDATED"] * len(symbols)})

    monkeypatch.setattr("app.dashboard.market_breadth.cached_breadth", cached)
    monkeypatch.setattr(
        "app.dashboard.market_breadth.load_breadth_universe", lambda *a: (data()[1], "TEST")
    )
    monkeypatch.setattr("app.dashboard.market_breadth.update_breadth_data", update)
    app = AppTest.from_string(
        "from pathlib import Path\n"
        "from app.dashboard.market_breadth import render_market_breadth\n"
        f"render_market_breadth(Path({str(tmp_path)!r}))"
    ).run(timeout=15)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Market overview", "Near EMA200", "EMA Crossings"]
    assert any("EMA-1.2" in item.value for item in app.caption)
    assert not initialized
    app.button(key="breadth-initialize-near").click().run(timeout=15)
    assert not app.exception
    assert len(initialized) == 1
    assert "breadth-request-update" not in app.session_state
    assert len(app.get("plotly_chart")) == 2
    assert app.number_input(key="near-ema200-distance").value == 3.0
