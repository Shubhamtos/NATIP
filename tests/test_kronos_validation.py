"""Verify historical forecasting cutoffs, trading-session outcomes, and metrics."""

import numpy as np
import pandas as pd
import pytest

from app.intelligence.kronos_forecast import KronosForecastError
from app.intelligence.kronos_validation import (
    choose_evaluation_dates,
    direction_label,
    evaluate_forecast_case,
    summarize_results,
)


class RecordingPredictor:
    """Predict unchanged prices while recording the history actually supplied."""

    device = "test"

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        self.timestamps = x_timestamp
        self.rows = len(df)
        price = df.close.iloc[-1]
        return pd.DataFrame(
            {
                "open": [price] * pred_len,
                "high": [price] * pred_len,
                "low": [price] * pred_len,
                "close": [price] * pred_len,
                "volume": [1_000_000] * pred_len,
            }
        )


def history():
    dates = pd.bdate_range("2022-01-01", periods=650).delete(600)
    close = np.linspace(100, 150, len(dates))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )


def test_exact_512_candles_and_actual_session_outcome():
    frame = history()
    predictor = RecordingPredictor()
    signal = frame.index[595]
    result = evaluate_forecast_case(
        frame,
        ticker="RELIANCE.NS",
        signal_date=signal,
        predictor=predictor,
        sample_count=2,
        seed=42,
    )
    assert predictor.rows == 512
    assert predictor.timestamps.max() == signal
    assert result["outcome_date"] == frame.index[605].date().isoformat()
    assert result["no_future_input"]
    assert result["actual_return_pct"] == pytest.approx(
        (frame.Close.iloc[605] / frame.Close.iloc[595] - 1) * 100
    )
    changed = frame.copy()
    changed.loc[changed.index > signal, "Close"] *= 2
    second = evaluate_forecast_case(
        changed,
        ticker="RELIANCE.NS",
        signal_date=signal,
        predictor=predictor,
        sample_count=2,
        seed=42,
    )
    assert second["predicted_price"] == result["predicted_price"]
    assert second["actual_price"] != result["actual_price"]


def test_incomplete_context_is_rejected():
    frame = history()
    with pytest.raises(KronosForecastError, match="Incomplete"):
        evaluate_forecast_case(
            frame,
            ticker="TCS.NS",
            signal_date=frame.index[200],
            predictor=RecordingPredictor(),
            sample_count=1,
            seed=42,
        )


def test_nonoverlapping_shared_dates():
    frame = history()
    dates = choose_evaluation_dates(
        {"TCS": frame, "INFY": frame}, periods=8, horizon=10, lookback=512
    )
    positions = frame.index.get_indexer(dates)
    assert (np.diff(positions) >= 10).all()
    assert positions.min() >= 511
    assert positions.max() + 10 < len(frame)


def test_labels_use_same_neutral_band():
    assert direction_label(1, 1) == "SIDEWAYS"
    assert direction_label(-1, 1) == "SIDEWAYS"
    assert direction_label(1.1, 1) == "UP"
    assert direction_label(-1.1, 1) == "DOWN"


def test_summary_does_not_count_repeated_seeds_as_independent_cases():
    rows = []
    for seed, correct in [(42, True), (123, False)]:
        rows.append(
            {
                "sample_count": 1,
                "ticker": "TCS.NS",
                "signal_date": "2024-01-01",
                "seed": seed,
                "actual_direction": "UP",
                "direction_correct": correct,
                "predicted_direction": "UP" if correct else "DOWN",
                "sign_correct": correct,
                "return_error_pp": 1,
                "actual_return_pct": 2,
                "absolute_price_error": 1,
                "persistence_absolute_price_error": 2,
                "momentum_direction": "UP",
                "runtime_seconds": 1,
            }
        )
    result = summarize_results(pd.DataFrame(rows)).iloc[0]
    assert result.unique_stock_dates == 1
    assert result.forecasts == 2
    assert result.direction_accuracy_pct == 50
    assert result.same_direction_across_seeds_pct == 0
