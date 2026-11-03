"""Daily Time Machine math, chronology and conservative execution tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from app.intelligence.daily_time_machine import (
    DailyTradePlan,
    _trade_outcome,
    simulate_daily_plan,
    validate_daily_plan,
)


def history(rows=340):
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, rows)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.01,
            "Low": np.minimum(open_, close) * 0.99,
            "Close": close,
            "Volume": 1e6,
        },
        index=pd.bdate_range("2023-01-01", periods=rows),
    )


def test_reproducible_and_outcomes_sum_to_one():
    h = history()
    plan = DailyTradePlan(100, 95, 110, paths=100)
    a = simulate_daily_plan(h, plan, as_of=h.index[-1])
    b = simulate_daily_plan(h, plan, as_of=h.index[-1])
    pd.testing.assert_frame_equal(a["outcomes"], b["outcomes"])
    assert sum(a["frequencies"].values()) == pytest.approx(1)
    assert a["bands"].shape == (21, 3)


def test_future_bars_cannot_change_result():
    h = history()
    cutoff = h.index[280]
    plan = DailyTradePlan(100, 95, 110, paths=100)
    a = simulate_daily_plan(h, plan, as_of=cutoff)
    h.loc[h.index > cutoff, ["Open", "High", "Low", "Close"]] *= 100
    b = simulate_daily_plan(h, plan, as_of=cutoff)
    pd.testing.assert_frame_equal(a["outcomes"], b["outcomes"])


def test_gap_stop_uses_worse_open_not_stop_price():
    plan = DailyTradePlan(100, 95, 110, cost_bps=0, slippage_bps=0)
    row = _trade_outcome(np.array([[90, 100, 89, 99]]), plan)
    assert row["Outcome"] == "STOP"
    assert row["Exit Price"] == 90
    assert row["Net Return %"] == pytest.approx(-10)


def test_both_touched_is_stop_first():
    row = _trade_outcome(np.array([[100, 120, 90, 110]]), DailyTradePlan(100, 95, 110))
    assert row["Outcome"] == "STOP"


def test_costs_charge_both_sides_and_slippage():
    plan = DailyTradePlan(100, 95, 110, cost_bps=10, slippage_bps=5)
    row = _trade_outcome(np.array([[100, 101, 99, 100]]), plan)
    expected = (100 * 0.9995 * 0.999 / (100 * 1.0005 * 1.001) - 1) * 100
    assert row["Net Return %"] == pytest.approx(expected)
    assert row["Net Return %"] < 0


def test_invalid_plan_fails_but_isolated_zero_volume_is_reported():
    h = history()
    with pytest.raises(ValueError, match="stop"):
        simulate_daily_plan(h, DailyTradePlan(100, 101, 110), as_of=h.index[-1])
    h.iloc[100, h.columns.get_loc("Volume")] = 0
    result = simulate_daily_plan(h, DailyTradePlan(100, 95, 110, paths=100), as_of=h.index[-1])
    assert result["history_rows"] == len(h) - 1
    assert len(result["excluded_bars"]) == 1
    assert "volume" in result["excluded_bars"][0]["Reason"]


def test_two_bad_bars_do_not_create_returns_across_gap():
    h = history()
    h[["Open", "High", "Low", "Close"]] = 100.0
    h.loc[h.index[170] :, ["Open", "High", "Low", "Close"]] = 1000.0
    h.loc[h.index[[169, 170]], "Volume"] = 0
    result = simulate_daily_plan(h, DailyTradePlan(100, 95, 110, paths=100), as_of=h.index[-1])
    assert len(result["excluded_bars"]) == 2
    assert result["frequencies"]["NEITHER"] == 1
    assert result["outcomes"]["Gross Return %"].eq(0).all()


def test_replay_skips_windows_with_bad_candles():
    h = history()
    bad_date = h.index[265]
    h.loc[bad_date, "Volume"] = 0
    result = validate_daily_plan(h, DailyTradePlan(100, 95, 110, paths=100), as_of=h.index[-1])
    assert not ((result.Origin < bad_date) & (result["Outcome End"] >= bad_date)).any()


def test_walkforward_is_nonoverlapping_and_chronological():
    h = history()
    plan = DailyTradePlan(100, 95, 110, paths=100)
    result = validate_daily_plan(h, plan, as_of=h.index[-1])
    assert len(result) >= 3
    assert result.Origin.lt(result["Outcome End"]).all()
    assert (result["Outcome End"].iloc[:-1].to_numpy() <= result.Origin.iloc[1:].to_numpy()).all()
    assert result["Train Rows"].min() >= 253
