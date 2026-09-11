"""Leak-safe target generation for stock probability modeling."""

from __future__ import annotations

import pandas as pd

from app.probability.config import HORIZON_DAYS, OUTPERFORM_THRESHOLD, UNDERPERFORM_THRESHOLD


def add_forward_excess_return_labels(
    features: pd.DataFrame,
    nifty_frame: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Add 20-trading-day excess-return labels.

    Args:
        features: Backward-looking feature rows.
        nifty_frame: Nifty benchmark OHLCV data.
        horizon_days: Forward return horizon.

    Returns:
        Dataset with future returns, excess return and class label. Rows without a full
        future horizon are dropped to prevent accidental leakage.
    """

    output = features.copy().sort_values(["symbol", "Date"])
    output["future_stock_return"] = output.groupby("symbol")["Close"].transform(
        lambda close: close.shift(-horizon_days) / close - 1
    )
    nifty = nifty_frame[["Date", "Close"]].copy().sort_values("Date")
    nifty["future_nifty_return"] = nifty["Close"].shift(-horizon_days) / nifty["Close"] - 1
    output = output.merge(
        nifty[["Date", "future_nifty_return"]],
        on="Date",
        how="left",
    )
    output["excess_return"] = output["future_stock_return"] - output["future_nifty_return"]
    output["label"] = output["excess_return"].map(_classify_excess_return)
    return output.dropna(
        subset=["future_stock_return", "future_nifty_return", "excess_return", "label"]
    ).reset_index(drop=True)


def _classify_excess_return(excess_return: float) -> int:
    """Map excess return to outperform, neutral, or underperform."""

    if excess_return > OUTPERFORM_THRESHOLD:
        return 1
    if excess_return < UNDERPERFORM_THRESHOLD:
        return -1
    return 0
