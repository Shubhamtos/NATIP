"""Same-date, cancellation and data safety tests for Kronos F&O research."""

from datetime import datetime

import pandas as pd

from app.dashboard import kronos_fno_screener as screen


class Predictor:
    device = "fixture-only"

    def __init__(self):
        self.latest = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.latest.append(x_timestamp.max())
        close = float(df.close.iloc[-1])
        return pd.DataFrame(
            {
                "open": [close * 1.02] * pred_len,
                "high": [close * 1.05] * pred_len,
                "low": [close * 0.99] * pred_len,
                "close": [close * 1.04] * pred_len,
                "volume": [1_000_000] * pred_len,
            }
        )


def history(last="2026-11-02", volume=10_000_000):
    index = pd.bdate_range(end=last, periods=260)
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": volume}, index=index
    )


def parameters():
    return {"horizon": 5, "lookback": 120, "samples": 1, "min_turnover": 1.0, "update": False}


def test_completed_nse_bar_bound():
    assert screen.completed_nse_day(
        datetime.fromisoformat("2026-11-03T11:00:00+05:30")
    ) == pd.Timestamp("2026-11-02")
    assert screen.completed_nse_day(
        datetime.fromisoformat("2026-11-03T16:00:00+05:30")
    ) == pd.Timestamp("2026-11-03")


def test_scan_excludes_future_and_unmatched_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "completed_nse_day", lambda: pd.Timestamp("2026-11-02"))
    future = history("2026-11-03")
    old = history("2026-10-30")
    job = screen.ScanJob(("AAA.NS", "BBB.NS"), parameters())
    model = Predictor()
    screen.screen_job(
        tmp_path,
        job,
        lambda: model,
        history_loader=lambda symbol, **kwargs: future if symbol == "AAA.NS" else old,
    )
    assert job.status == "complete"
    assert len(job.forecasts) == 1
    assert model.latest == [pd.Timestamp("2026-11-02")]
    assert any("differs" in row["Reason"] for row in job.rows)
    assert (tmp_path / "outputs/kronos_fno/latest.csv").exists()


def test_one_failed_stock_does_not_stop_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "completed_nse_day", lambda: pd.Timestamp("2026-11-02"))

    def loader(symbol, **kwargs):
        if symbol == "BAD.NS":
            raise ValueError("Unavailable")
        return history()

    job = screen.ScanJob(("BAD.NS", "GOOD.NS"), parameters())
    screen.screen_job(tmp_path, job, Predictor, history_loader=loader)
    assert job.done == 2
    assert "GOOD.NS" in job.forecasts


def test_cancel_does_not_start_inference(tmp_path):
    job = screen.ScanJob(("AAA.NS",), parameters())
    job.cancel.set()

    def forbidden():
        raise AssertionError("Model must not load after cancellation")

    screen.screen_job(tmp_path, job, forbidden)
    assert job.status == "cancelled"
    assert not job.forecasts


def test_zero_volume_and_low_liquidity_are_not_ranked(tmp_path, monkeypatch):
    monkeypatch.setattr(screen, "completed_nse_day", lambda: pd.Timestamp("2026-11-02"))
    job = screen.ScanJob(("ZERO.NS", "THIN.NS"), parameters())
    screen.screen_job(
        tmp_path,
        job,
        Predictor,
        history_loader=lambda symbol, **kwargs: history(volume=0 if symbol == "ZERO.NS" else 1),
    )
    assert not job.forecasts
    assert all(row["Status"] == "SKIPPED" for row in job.rows)
