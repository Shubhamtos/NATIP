"""Train NATIP's 20-day Nifty outperformance probability model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import (
    BACKTEST_OUTPUT,
    BENCHMARK_SYMBOL,
    DATA_DIR,
    DATA_QUALITY_REPORT,
    FEATURES_PATH,
    FEATURE_IMPORTANCE_OUTPUT,
    MODEL_PATH,
    MODEL_COMPARISON_OUTPUT,
    MODEL_METRICS_OUTPUT,
    PROBABILITY_BUCKET_OUTPUT,
    REGIME_COMPARISON_OUTPUT,
    SECTOR_DIAGNOSTICS_OUTPUT,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.data_download import (
    download_universe,
    load_stock_universe,
    load_universe_metadata,
)
from app.probability.data_quality import build_data_quality_report
from app.probability.features import build_feature_dataset, feature_columns_for_variant
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.targets import add_forward_excess_return_labels
from app.probability.train import LABEL_TO_CLASS, train_xgb_classifier
from app.probability.validate import evaluate_classifier, walk_forward_validation


def main() -> None:
    """Run the 150-stock probability pipeline end to end."""

    ensure_probability_dirs()
    previous_metrics = _load_previous_metrics()
    _snapshot_previous_10_artifacts(previous_metrics)

    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    symbols = load_stock_universe(STOCKS_UNIVERSE_2026_08_CSV)
    sector_symbols = sector_benchmark_symbols(universe)
    data = download_universe(symbols=[*symbols, *sector_symbols], include_benchmark=True)
    stock_data = {
        symbol: frame for symbol, frame in data.items() if symbol in [*symbols, BENCHMARK_SYMBOL]
    }
    sector_frames = _sector_frames_from_data(data, universe)
    features = build_feature_dataset(
        stock_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=universe,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    dataset = add_forward_excess_return_labels(features, data[BENCHMARK_SYMBOL])
    dataset = dataset.merge(
        universe[["Ticker", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    dataset_path = DATA_DIR / "probability_training_dataset.csv"
    dataset.to_csv(dataset_path, index=False)

    quality_report = build_data_quality_report(universe=universe, data=stock_data, dataset=dataset)
    DATA_QUALITY_REPORT.write_text(
        json.dumps(_json_safe(quality_report), indent=2), encoding="utf-8"
    )

    try:
        from joblib import load
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("Install joblib before loading the trained model.") from exc

    variant_results = {}
    for variant in ("baseline", "sector", "sector_regime"):
        variant_results[variant] = _train_and_evaluate_variant(
            variant=variant,
            dataset=dataset,
            xgb_classifier=XGBClassifier,
            load_model=load,
        )

    final_result = variant_results["sector_regime"]
    metrics = final_result["metrics"]
    summary = final_result["summary"]
    shutil.copy2(summary["model_path"], MODEL_PATH)
    shutil.copy2(summary["features_path"], FEATURES_PATH)
    _save_feature_importance(metrics)
    DATA_DIR.joinpath("probability_validation_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2), encoding="utf-8"
    )
    MODEL_METRICS_OUTPUT.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    BACKTEST_OUTPUT.write_text(
        json.dumps(_json_safe(metrics.get("backtests", {})), indent=2), encoding="utf-8"
    )

    comparison = _compare_with_previous_metrics(previous_metrics, metrics)
    MODEL_COMPARISON_OUTPUT.write_text(
        json.dumps(_json_safe(comparison), indent=2), encoding="utf-8"
    )
    regime_comparison = _variant_comparison_report(variant_results)
    REGIME_COMPARISON_OUTPUT.write_text(
        json.dumps(_json_safe(regime_comparison), indent=2), encoding="utf-8"
    )
    bucket_frames = []
    sector_frames_diag = []
    for variant, result in variant_results.items():
        predictions = result["predictions"]
        bucket_frames.append(_probability_bucket_analysis(predictions, variant=variant))
        sector_frames_diag.append(_sector_diagnostics(predictions, variant=variant))
    pd.concat(bucket_frames, ignore_index=True).to_csv(PROBABILITY_BUCKET_OUTPUT, index=False)
    pd.concat(sector_frames_diag, ignore_index=True).to_csv(SECTOR_DIAGNOSTICS_OUTPUT, index=False)
    print(
        json.dumps(
            {
                **summary,
                "dataset_path": str(dataset_path),
                "data_quality_report": str(DATA_QUALITY_REPORT),
                "metrics_path": str(MODEL_METRICS_OUTPUT),
                "comparison_path": str(MODEL_COMPARISON_OUTPUT),
                "sector_regime_comparison_path": str(REGIME_COMPARISON_OUTPUT),
                "probability_bucket_path": str(PROBABILITY_BUCKET_OUTPUT),
                "sector_diagnostics_path": str(SECTOR_DIAGNOSTICS_OUTPUT),
            },
            indent=2,
        )
    )


def _json_safe(value):
    """Convert pandas/numpy scalar values into JSON-safe Python values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _sector_frames_from_data(
    data: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Map universe sectors to downloaded sector benchmark frames."""

    frames: dict[str, pd.DataFrame] = {}
    for sector in universe["Sector"].dropna().astype(str).unique():
        benchmark = SECTOR_YAHOO_SYMBOLS.get(sector)
        if benchmark and benchmark in data and not data[benchmark].empty:
            frames[sector] = data[benchmark]
    return frames


def _train_and_evaluate_variant(
    *,
    variant: str,
    dataset: pd.DataFrame,
    xgb_classifier,
    load_model,
) -> dict:
    """Train one feature variant and return metrics plus test predictions."""

    feature_columns = feature_columns_for_variant(variant)
    model_path = Path(f"models/probability/xgb_outperformance_model_{variant}.joblib")
    features_path = Path(f"models/probability/feature_columns_{variant}.json")
    summary = train_xgb_classifier(
        dataset,
        model_path=model_path,
        features_path=features_path,
        feature_columns=feature_columns,
    )
    model = load_model(summary["model_path"])
    model.natip_feature_columns = feature_columns
    test = dataset[dataset["Date"] >= "2025-01-01"].dropna(subset=feature_columns + ["label"])
    metrics = evaluate_classifier(model, test) if not test.empty else {}
    metrics["walk_forward_validation"] = walk_forward_validation(
        lambda: xgb_classifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        ),
        dataset,
        feature_columns=feature_columns,
    )
    predictions = _test_predictions(model, test, feature_columns)
    backtests = {
        f"top_{top_n}": backtest_top_n_portfolio(predictions, top_n=top_n) for top_n in (5, 10, 20)
    }
    metrics["backtests"] = backtests
    predictions.to_csv(DATA_DIR / f"probability_test_predictions_{variant}.csv", index=False)
    Path(f"reports/probability/stock_probability_150_model_metrics_{variant}.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    return {"summary": summary, "metrics": metrics, "predictions": predictions}


def _test_predictions(model, test: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build test-set probability prediction rows."""

    probabilities = model.predict_proba(test[feature_columns])
    predictions = test[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "label",
        ]
    ].copy()
    predictions["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
    predictions["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
    predictions["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
    predictions["predicted_class"] = pd.Series(
        probabilities.argmax(axis=1), index=predictions.index
    ).map({0: -1, 1: 0, 2: 1})
    return predictions


def _variant_comparison_report(results: dict[str, dict]) -> dict:
    """Compare baseline, sector, and sector+regime variants."""

    report = {}
    for variant, result in results.items():
        metrics = result["metrics"]
        top5 = (metrics.get("backtests") or {}).get("top_5", {})
        report[variant] = {
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "weighted_f1": (metrics.get("classification_report") or {})
            .get("weighted avg", {})
            .get("f1-score"),
            "log_loss": metrics.get("log_loss"),
            "calibration": metrics.get("calibration"),
            "top_5_cagr": top5.get("cagr"),
            "top_5_sharpe": top5.get("sharpe"),
            "top_5_max_drawdown": top5.get("max_drawdown"),
            "top_5_turnover": top5.get("turnover"),
            "top_5_excess_cagr_vs_nifty": (
                top5.get("cagr") - top5.get("benchmark_cagr")
                if top5.get("cagr") is not None and top5.get("benchmark_cagr") is not None
                else None
            ),
            "top_5_win_rate": top5.get("win_rate"),
        }
    return report


def _probability_bucket_analysis(predictions: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    """Analyze realized outcomes by predicted outperform probability bucket."""

    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    labels = ["0-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"]
    frame = predictions.copy()
    frame["bucket"] = pd.cut(
        frame["p_outperform"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    rows = []
    for bucket in labels:
        group = frame[frame["bucket"] == bucket]
        rows.append(
            {
                "Variant": variant,
                "P_Outperform_Bucket": bucket,
                "ObservationCount": int(len(group)),
                "ActualOutperformRate": float((group["label"] == 1).mean()) if len(group) else None,
                "AverageFutureStockReturn": (
                    float(group["future_stock_return"].mean()) if len(group) else None
                ),
                "AverageFutureExcessReturn": (
                    float(group["excess_return"].mean()) if len(group) else None
                ),
                "MedianExcessReturn": (
                    float(group["excess_return"].median()) if len(group) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _sector_diagnostics(predictions: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    """Return sector-level prediction diagnostics."""

    rows = []
    for sector, group in predictions.groupby("Sector"):
        rows.append(
            {
                "Variant": variant,
                "Sector": sector,
                "PredictionCount": int(len(group)),
                "SuccessRate": float((group["label"] == 1).mean()),
                "AverageFutureExcessReturn": float(group["excess_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["Variant", "AverageFutureExcessReturn"], ascending=False)


def _load_previous_metrics() -> dict | None:
    """Load previous 10-stock metrics before retraining overwrites them."""

    metrics_path = DATA_DIR / "probability_validation_metrics.json"
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _snapshot_previous_10_artifacts(previous_metrics: dict | None) -> None:
    """Preserve existing 10-stock metrics/model artifacts when present."""

    if previous_metrics is None:
        return
    snapshot_path = DATA_DIR / "probability_validation_metrics_10_stock_snapshot.json"
    if not snapshot_path.exists():
        snapshot_path.write_text(
            json.dumps(_json_safe(previous_metrics), indent=2), encoding="utf-8"
        )
    for source, destination in [
        (
            Path("models/probability/xgb_outperformance_model.joblib"),
            Path("models/probability/xgb_outperformance_model_10_stock_snapshot.joblib"),
        ),
        (
            Path("models/probability/feature_columns.json"),
            Path("models/probability/feature_columns_10_stock_snapshot.json"),
        ),
    ]:
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)


def _save_feature_importance(metrics: dict) -> None:
    """Save model feature importance to CSV."""

    importance = metrics.get("feature_importance") or {}
    frame = pd.DataFrame(
        [{"Feature": feature, "Importance": value} for feature, value in importance.items()]
    ).sort_values("Importance", ascending=False)
    frame.to_csv(FEATURE_IMPORTANCE_OUTPUT, index=False)


def _compare_with_previous_metrics(previous: dict | None, current: dict) -> dict:
    """Compare previous 10-stock model metrics with the new 150-stock model."""

    return {
        "previous_model": "10-stock model" if previous else "Unavailable",
        "current_model": "150-stock model",
        "metrics": {
            metric: {
                "previous": _extract_metric(previous, metric) if previous else None,
                "current": _extract_metric(current, metric),
            }
            for metric in [
                "balanced_accuracy",
                "weighted_f1",
                "log_loss",
                "top_5_cagr",
                "top_5_sharpe",
                "top_5_max_drawdown",
                "top_5_win_rate",
                "top_5_turnover",
                "top_5_benchmark_cagr",
            ]
        },
    }


def _extract_metric(metrics: dict | None, metric: str):
    """Extract comparable metrics from a metrics dictionary."""

    if not metrics:
        return None
    if metric == "weighted_f1":
        return (metrics.get("classification_report") or {}).get("weighted avg", {}).get("f1-score")
    if metric.startswith("top_5_"):
        key = metric.replace("top_5_", "")
        return (metrics.get("backtests") or {}).get("top_5", {}).get(key)
    return metrics.get(metric)


if __name__ == "__main__":
    main()
