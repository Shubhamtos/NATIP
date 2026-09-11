"""Latest stock probability screener."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    FEATURES_PATH,
    FULL_RANKING_OUTPUT,
    MODEL_PATH,
    SCREENER_OUTPUT,
    TOP20_RANKING_OUTPUT,
    ensure_probability_dirs,
)
from app.probability.data_download import (
    download_universe,
    load_stock_universe,
    load_universe_metadata,
)
from app.probability.features import build_feature_dataset
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.train import LABEL_TO_CLASS


def run_latest_screener(
    *,
    stocks_csv: Path | None = None,
    output_path: Path = SCREENER_OUTPUT,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download latest data, calculate features, score stocks and save ranked CSV."""

    try:
        from joblib import load
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("Install joblib before running the probability screener.") from exc

    ensure_probability_dirs()
    symbols = load_stock_universe(stocks_csv) if stocks_csv else load_stock_universe()
    metadata = load_universe_metadata(stocks_csv) if stocks_csv else load_universe_metadata()
    sector_symbols = sector_benchmark_symbols(metadata)
    data = download_universe(
        [*symbols, *sector_symbols],
        include_benchmark=True,
        force=force_download,
        update_cached=force_download,
    )
    stock_data = {
        symbol: frame for symbol, frame in data.items() if symbol in [*symbols, BENCHMARK_SYMBOL]
    }
    sector_frames = {
        sector: data[benchmark]
        for sector, benchmark in SECTOR_YAHOO_SYMBOLS.items()
        if benchmark in data and not data[benchmark].empty
    }
    features = build_feature_dataset(
        stock_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=metadata,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    if features.empty:
        raise ValueError("No feature rows were generated.")
    latest = features.sort_values("Date").groupby("symbol", as_index=False).tail(1)

    model = load(MODEL_PATH)
    feature_columns = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    probabilities = model.predict_proba(latest[feature_columns])
    predicted_class_index = probabilities.argmax(axis=1)
    class_to_label = {0: -1, 1: 0, 2: 1}
    output = pd.DataFrame(
        {
            "Rank": range(1, len(latest) + 1),
            "Ticker": latest["symbol"].to_list(),
            "P_Underperform": probabilities[:, LABEL_TO_CLASS[-1]],
            "P_Neutral": probabilities[:, LABEL_TO_CLASS[0]],
            "P_Outperform": probabilities[:, LABEL_TO_CLASS[1]],
            "Predicted_Class": [class_to_label[int(value)] for value in predicted_class_index],
            "Data Date": latest["Date"].dt.date.astype(str).to_list(),
        }
    ).merge(metadata, on="Ticker", how="left")
    output = output.sort_values("P_Outperform", ascending=False)
    output["Rank"] = range(1, len(output) + 1)
    output = output[
        [
            "Rank",
            "Ticker",
            "Sector",
            "MarketCapCategory",
            "P_Outperform",
            "P_Neutral",
            "P_Underperform",
            "Predicted_Class",
            "Company",
            "Group",
            "Data Date",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    output.to_csv(FULL_RANKING_OUTPUT, index=False)
    output.head(20).to_csv(TOP20_RANKING_OUTPUT, index=False)
    return output.reset_index(drop=True)
