from __future__ import annotations

import pandas as pd

from pathlib import Path

from app.probability.data_download import load_stock_universe, load_universe_metadata
from app.probability.features import (
    ADVANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_features_for_symbol,
)
from app.probability.targets import add_forward_excess_return_labels


def _sample_ohlcv(rows: int = 320, *, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = pd.Series([start_price + index * 0.5 for index in range(rows)], dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": [100_000 + index * 100 for index in range(rows)],
        }
    )


def test_probability_feature_engineering_uses_expected_columns() -> None:
    stock = _sample_ohlcv()
    nifty = _sample_ohlcv(start_price=90.0)

    features = build_features_for_symbol("TEST.NS", stock, nifty)

    assert not features.empty
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["symbol"].unique().tolist() == ["TEST.NS"]
    assert features["Date"].min() > stock["Date"].min()


def test_advanced_breakout_features_use_prior_highs() -> None:
    stock = _sample_ohlcv(rows=320)
    nifty = _sample_ohlcv(rows=320, start_price=90.0)
    breakout_index = 260
    stock.loc[breakout_index, "Close"] = stock.loc[: breakout_index - 1, "High"].max() * 1.05
    stock.loc[breakout_index, "High"] = stock.loc[breakout_index, "Close"] * 1.01

    features = build_features_for_symbol("TEST.NS", stock, nifty)
    breakout_row = features[features["Date"] == stock.loc[breakout_index, "Date"]].iloc[0]
    prior_20d_high = stock.loc[: breakout_index - 1, "High"].tail(20).max()

    assert set(ADVANCED_FEATURE_COLUMNS).issubset(features.columns)
    assert breakout_row["breakout_prev_20d_high"] == 1.0
    assert breakout_row["distance_prev_20d_high"] == stock.loc[breakout_index, "Close"] / prior_20d_high - 1


def test_probability_targets_drop_rows_without_future_horizon() -> None:
    stock = _sample_ohlcv()
    nifty = _sample_ohlcv(start_price=90.0)
    features = build_features_for_symbol("TEST.NS", stock, nifty)

    labeled = add_forward_excess_return_labels(features, nifty, horizon_days=20)

    assert not labeled.empty
    assert labeled["label"].isin([-1, 0, 1]).all()
    assert labeled["Date"].max() <= features["Date"].max() - pd.tseries.offsets.BDay(20)
    assert "future_stock_return" in labeled.columns
    assert "future_nifty_return" in labeled.columns


def test_probability_universe_is_editable_csv() -> None:
    symbols = load_stock_universe(Path("stocks.csv"))

    assert symbols == [
        "RELIANCE.NS",
        "TCS.NS",
        "ICICIBANK.NS",
        "M&M.NS",
        "SUNPHARMA.NS",
        "ITC.NS",
        "BEL.NS",
        "TATASTEEL.NS",
        "BHARTIARTL.NS",
        "LT.NS",
    ]


def test_probability_default_universe_has_150_metadata_rows() -> None:
    metadata = load_universe_metadata()

    assert len(metadata) == 150
    assert metadata["Ticker"].nunique() == 150
    assert set(metadata.columns) == {"Ticker", "Company", "Sector", "MarketCapCategory", "Group"}
    assert metadata["Group"].value_counts().to_dict() == {
        "DiversifiedMidcap": 50,
        "NiftyNext50": 50,
        "Nifty50Additional": 40,
        "Existing10": 10,
    }
