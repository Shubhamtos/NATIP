"""Historical inference audit for a fixed pretrained Kronos model."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from app.intelligence.kronos_forecast import (
    KronosForecastError,
    KronosPredictorProtocol,
    _validate_history,
    forecast_stock_direction,
)


def direction_label(return_pct: float, band_pct: float) -> str:
    """Classify an observed return using the forecast-date neutral band."""
    return "UP" if return_pct > band_pct else "DOWN" if return_pct < -band_pct else "SIDEWAYS"


def evaluate_forecast_case(
    history: pd.DataFrame,
    *,
    ticker: str,
    signal_date: pd.Timestamp,
    predictor: KronosPredictorProtocol,
    sample_count: int,
    seed: int,
    horizon: int = 10,
    lookback: int = 512,
) -> dict[str, Any]:
    """Run a cutoff-only forecast and compare with later observed sessions.

    Args:
        history: Adjusted daily OHLCV including outcomes for scoring only.
        ticker: NSE ticker.
        signal_date: Last session allowed into model inputs.
        predictor: Loaded fixed pretrained model.
        sample_count: Number of sampled forecast paths.
        seed: Recorded random seed for reproducibility.
        horizon: Actual trading-session outcome horizon.
        lookback: Required number of input candles.

    Returns:
        Forecast, realized outcome, error, and leakage-audit fields.
    """
    clean = _validate_history(history)
    cutoff = pd.Timestamp(signal_date).normalize()
    past = clean.loc[clean.index <= cutoff].tail(lookback)
    future = clean.loc[clean.index > cutoff].head(horizon)
    if len(past) != lookback or len(future) != horizon or past.index[-1] != cutoff:
        raise KronosForecastError("Incomplete context/outcome window for historical evaluation.")
    assert past.index.max() < future.index.min()
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    started = time.perf_counter()
    result = forecast_stock_direction(
        past,
        symbol=ticker,
        predictor=predictor,
        horizon=horizon,
        lookback=lookback,
        sample_count=sample_count,
        as_of=cutoff,
    )
    actual_price = float(future["Close"].iloc[-1])
    actual_return = (actual_price / result.latest_price - 1) * 100
    predicted_price = float(result.forecast["close"].iloc[-1])
    actual_direction = direction_label(actual_return, result.neutral_band_pct)
    momentum_return = (past["Close"].iloc[-1] / past["Close"].iloc[-horizon - 1] - 1) * 100
    return {
        "ticker": ticker,
        "signal_date": cutoff.date().isoformat(),
        "outcome_date": future.index[-1].date().isoformat(),
        "sample_count": sample_count,
        "seed": seed,
        "lookback": len(past),
        "horizon": horizon,
        "latest_price": result.latest_price,
        "predicted_price": predicted_price,
        "actual_price": actual_price,
        "predicted_return_pct": result.expected_return_pct,
        "actual_return_pct": actual_return,
        "neutral_band_pct": result.neutral_band_pct,
        "predicted_direction": result.direction,
        "actual_direction": actual_direction,
        "direction_correct": result.direction == actual_direction,
        "sign_correct": np.sign(result.expected_return_pct) == np.sign(actual_return),
        "return_error_pp": result.expected_return_pct - actual_return,
        "absolute_price_error": abs(predicted_price - actual_price),
        "persistence_absolute_price_error": abs(result.latest_price - actual_price),
        "momentum_direction": direction_label(float(momentum_return), result.neutral_band_pct),
        "context_start": past.index[0].date().isoformat(),
        "context_end": past.index[-1].date().isoformat(),
        "first_outcome_date": future.index[0].date().isoformat(),
        "no_future_input": bool(past.index.max() == cutoff < future.index.min()),
        "calendar_mismatch": not result.forecast.index.equals(future.index),
        "warnings": " | ".join(result.warnings),
        "runtime_seconds": time.perf_counter() - started,
    }


def choose_evaluation_dates(
    histories: dict[str, pd.DataFrame], *, periods: int, horizon: int, lookback: int
) -> pd.DatetimeIndex:
    """Choose identical recent non-overlapping outcome windows for all stocks."""
    eligible = None
    for history in histories.values():
        clean = _validate_history(history)
        dates = clean.index[lookback - 1 : -horizon]
        eligible = dates if eligible is None else eligible.intersection(dates)
    if eligible is None or len(eligible) < periods * horizon:
        raise ValueError("Insufficient shared historical dates for the requested evaluation.")
    return eligible[::-horizon][:periods].sort_values()


def summarize_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare sampling settings on matched observations without treating seeds as new cases."""
    rows = []
    for sample_count, group in predictions.groupby("sample_count"):
        recalls = [
            part["direction_correct"].mean() for _, part in group.groupby("actual_direction")
        ]
        directional = group.loc[group["predicted_direction"].ne("SIDEWAYS")]
        case_agreement = group.groupby(["ticker", "signal_date"])["predicted_direction"].nunique()
        rows.append(
            {
                "sample_count": sample_count,
                "forecasts": len(group),
                "unique_stock_dates": len(case_agreement),
                "unique_dates": group["signal_date"].nunique(),
                "stocks": group["ticker"].nunique(),
                "seeds": group["seed"].nunique(),
                "direction_accuracy_pct": 100 * group["direction_correct"].mean(),
                "sign_accuracy_pct": 100 * group["sign_correct"].mean(),
                "balanced_direction_accuracy_pct": 100 * np.mean(recalls),
                "up_down_calls": len(directional),
                "up_down_label_accuracy_pct": 100 * directional["direction_correct"].mean(),
                "return_mae_pp": group["return_error_pp"].abs().mean(),
            "unchanged_return_mae_pp": group["actual_return_pct"].abs().mean(),
                "return_rmse_pp": np.sqrt(group["return_error_pp"].pow(2).mean()),
                "price_mae": group["absolute_price_error"].mean(),
                "unchanged_price_mae": group["persistence_absolute_price_error"].mean(),
                "always_up_accuracy_pct": 100 * group["actual_direction"].eq("UP").mean(),
                "always_down_accuracy_pct": 100 * group["actual_direction"].eq("DOWN").mean(),
                "always_sideways_accuracy_pct": 100
                * group["actual_direction"].eq("SIDEWAYS").mean(),
                "momentum_direction_accuracy_pct": 100
                * group["momentum_direction"].eq(group["actual_direction"]).mean(),
                "same_direction_across_seeds_pct": 100 * case_agreement.eq(1).mean(),
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


def date_bootstrap(predictions: pd.DataFrame, iterations: int = 2000) -> pd.DataFrame:
    """Resample whole signal dates, keeping stocks and repeated seeds clustered.

    Args:
        predictions: Completed matched audit records.
        iterations: Number of deterministic bootstrap replications.

    Returns:
        Accuracy intervals and paired differences against one sample path.
    """
    correct = predictions.pivot_table(
        index="signal_date", columns="sample_count", values="direction_correct", aggfunc="sum"
    )
    counts = predictions.pivot_table(
        index="signal_date", columns="sample_count", values="direction_correct", aggfunc="size"
    )
    rng = np.random.default_rng(20261101)
    draws = rng.integers(0, len(correct), size=(iterations, len(correct)))
    rates = correct.to_numpy()[draws].sum(axis=1) / counts.to_numpy()[draws].sum(axis=1)
    rows = []
    for column, samples in enumerate(correct.columns):
        ci = np.percentile(rates[:, column] * 100, [2.5, 97.5])
        delta = (rates[:, column] - rates[:, 0]) * 100
        delta_ci = np.percentile(delta, [2.5, 97.5])
        rows.append({
            "sample_count": samples, "date_clusters": len(correct), "iterations": iterations,
            "direction_accuracy_ci_low_pct": ci[0], "direction_accuracy_ci_high_pct": ci[1],
            "difference_vs_1_path_ci_low_pp": delta_ci[0],
            "difference_vs_1_path_ci_high_pp": delta_ci[1],
        })
    return pd.DataFrame(rows)


def run_validation(
    histories: dict[str, pd.DataFrame],
    predictor: KronosPredictorProtocol,
    dates: pd.DatetimeIndex,
    *,
    seeds: tuple[int, ...] = (42, 123),
    sample_counts: tuple[int, ...] = (1, 2, 3),
    horizon: int = 10,
    lookback: int = 512,
    existing: pd.DataFrame | None = None,
    on_progress: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Evaluate matched settings and checkpoint every completed inference for resuming."""
    rows = existing.to_dict("records") if existing is not None else []
    completed = {(r["ticker"], str(r["signal_date"]), r["sample_count"], r["seed"]) for r in rows}
    for date in dates:
        for ticker, history in histories.items():
            for seed in seeds:
                for samples in sample_counts:
                    key = (ticker, date.date().isoformat(), samples, seed)
                    if key in completed:
                        continue
                    rows.append(
                        evaluate_forecast_case(
                            history,
                            ticker=ticker,
                            signal_date=date,
                            predictor=predictor,
                            sample_count=samples,
                            seed=seed,
                            horizon=horizon,
                            lookback=lookback,
                        )
                    )
                    if on_progress is not None:
                        on_progress(pd.DataFrame(rows))
    predictions = pd.DataFrame(rows)
    assert predictions["no_future_input"].all()
    counts = predictions.groupby("sample_count").size()
    assert counts.nunique() == 1, "Sampling settings must use exactly matched observations."
    return predictions
