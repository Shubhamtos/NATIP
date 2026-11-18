"""Synthetic EMA fixtures: actual crosses, touches, gaps, and historical cutoffs."""

import numpy as np
import pandas as pd
import pytest

from app.intelligence.ema_screeners import ema_crossings, ema_observations, near_ema200


def observations(prices):
    close = pd.Series(prices, index=pd.bdate_range("2023-01-02", periods=len(prices)))
    return ema_observations(close, pd.Series(0, index=close.index))


def stocks():
    rows = []
    for ticker, price in (("UP.NS", 110.0), ("DOWN.NS", 90.0), ("TOUCH.NS", 100.0)):
        features = observations([100.0] * 259 + [price])
        rows.append(
            features.iloc[-1].to_dict()
            | {
                "Ticker": ticker,
                "Company": ticker,
                "Sector": "Tech",
                "Status": "PASS",
                "Date": features.index[-1],
                "Close": price,
            }
        )
    return pd.DataFrame(rows)


def test_ema_matches_existing_recursive_pandas_convention():
    prices = pd.Series(np.arange(1.0, 281.0), index=pd.bdate_range("2023-01-02", periods=280))
    calculated = ema_observations(prices, pd.Series(0, index=prices.index))
    for window in (20, 50, 200):
        expected = prices.ewm(span=window, adjust=False, min_periods=window).mean()
        pd.testing.assert_series_equal(calculated[f"EMA{window}"], expected, check_names=False)
        assert calculated[f"EMA{window}"].iloc[: window - 1].isna().all()


def test_current_upward_and_downward_crosses_with_all_periods():
    frame = stocks()
    up = ema_crossings(frame, match="All")
    assert up.Ticker.tolist() == ["UP.NS"]
    assert up.iloc[0]["Crossed EMAs"] == "EMA20, EMA50, EMA200"
    assert up.iloc[0]["Sessions Since Latest Cross"] == 0
    assert up.iloc[0]["Latest Cross Date"] == frame.Date.iloc[0]
    down = ema_crossings(frame, direction="Down", match="All")
    assert down.Ticker.tolist() == ["DOWN.NS"]
    assert down.iloc[0].Direction == "Crossed below"
    assert len(ema_crossings(frame, direction="Both", match="All")) == 2


def test_touch_without_cross_and_nontradable_row_never_match():
    frame = stocks()
    frame.loc[frame.Ticker.eq("UP.NS"), "Status"] = "MISSING / NON-TRADING"
    assert ema_crossings(frame).empty
    assert "TOUCH.NS" not in ema_crossings(frame, direction="Both").Ticker.tolist()


def test_any_vs_all_periods_requires_real_cross_for_each():
    frame = stocks().iloc[:1].copy()
    frame["EMA200 Up Age"] = np.nan
    frame["EMA200 Up Date"] = pd.NaT
    assert len(ema_crossings(frame, match="Any")) == 1
    assert ema_crossings(frame, match="All").empty
    assert len(ema_crossings(frame, windows=[20, 50], match="All")) == 1


def test_recent_cross_age_is_measured_in_sessions_and_must_still_hold():
    prices = [100.0] * 250 + [110.0] * 10
    features = observations(prices)
    row = features.iloc[-1].to_dict() | {"Ticker": "A.NS", "Status": "PASS", "Close": 110.0}
    frame = pd.DataFrame([row])
    assert ema_crossings(frame, within_sessions=5).empty
    assert len(ema_crossings(frame, within_sessions=10, match="All")) == 1
    assert ema_crossings(frame, within_sessions=10).iloc[0]["Sessions Since Latest Cross"] == 9
    frame["Close"] = 80.0
    assert ema_crossings(frame, within_sessions=10).empty


def test_near_ema200_distance_and_sides():
    frame = stocks()
    assert near_ema200(frame).Ticker.tolist() == ["TOUCH.NS"]
    above = near_ema200(frame, distance_pct=10, side="Above")
    assert set(above.Ticker) == {"UP.NS", "TOUCH.NS"}
    below = near_ema200(frame, distance_pct=11, side="Below")
    assert set(below.Ticker) == {"DOWN.NS", "TOUCH.NS"}
    assert above.iloc[0]["EMA200 Distance %"] == 0
    assert below.loc[below.Ticker.eq("DOWN.NS"), "EMA200 Distance %"].iloc[0] < 0


def test_missing_session_and_structural_break_restart_warmup_and_events():
    prices = pd.Series([100.0] * 259 + [110.0], index=pd.bdate_range("2023-01-02", periods=260))
    prices.iloc[-5] = np.nan
    frame = ema_observations(prices, pd.Series(0, index=prices.index))
    assert frame.iloc[-1][["EMA20", "EMA50", "EMA200"]].isna().all()
    assert pd.isna(frame.iloc[-1]["EMA20 Up Date"])
    prices = prices.fillna(100)
    segments = pd.Series(0, index=prices.index)
    segments.iloc[-10:] = 1
    frame = ema_observations(prices, segments)
    assert pd.isna(frame.iloc[-1]["EMA200"])


def test_future_values_cannot_change_past_ema_or_crosses():
    prices = pd.Series(
        [100.0] * 259 + [110.0] * 20, index=pd.bdate_range("2023-01-02", periods=279)
    )
    segments = pd.Series(0, index=prices.index)
    before = ema_observations(prices, segments)
    prices.iloc[-10:] = 10000
    after = ema_observations(prices, segments)
    pd.testing.assert_frame_equal(before.iloc[:-10], after.iloc[:-10])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(windows=[]),
        dict(windows=[100]),
        dict(direction="Sideways"),
        dict(within_sessions=0),
        dict(match="Invalid"),
    ],
)
def test_invalid_cross_filters_rejected(kwargs):
    with pytest.raises(ValueError):
        ema_crossings(stocks(), **kwargs)


def test_ema_screener_ui_controls_empty_states_and_downloads(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app.intelligence.market_breadth import BreadthReport

    frame = stocks()
    report = BreadthReport(
        frame.Date.iloc[0], pd.DataFrame(), frame, pd.DataFrame(), pd.DataFrame()
    )
    monkeypatch.setattr("app.dashboard.market_breadth.cached_breadth", lambda *a: (report, "TEST"))
    app = AppTest.from_string(
        "from app.dashboard.market_breadth import cached_breadth\n"
        "from app.dashboard.ema_screeners import render_near_ema200, render_ema_crossings\n"
        "report, _ = cached_breadth('', '', '')\n"
        "render_near_ema200(report)\n"
        "render_ema_crossings(report)\n"
    ).run(timeout=15)
    assert not app.exception
    assert app.dataframe[0].value.Ticker.tolist() == ["TOUCH.NS"]
    assert app.dataframe[1].value.Ticker.tolist() == ["UP.NS"]
    app.selectbox(key="ema-cross-direction").set_value("Below").run()
    assert not app.exception
    assert app.dataframe[1].value.Ticker.tolist() == ["DOWN.NS"]
    app.multiselect(key="ema-cross-windows").set_value([]).run()
    assert not app.exception
    assert any("at least one EMA" in element.value for element in app.info)
