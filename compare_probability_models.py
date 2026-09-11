"""Compare ML models for NATIP's 150-stock probability screener."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import (
    BENCHMARK_SYMBOL,
    DATA_DIR,
    FEATURES_PATH,
    FULL_RANKING_OUTPUT,
    MODEL_DIR,
    MODEL_PATH,
    REPORT_DIR,
    STOCKS_UNIVERSE_2026_08_CSV,
    TOP20_RANKING_OUTPUT,
    ensure_probability_dirs,
)
from app.probability.data_download import (
    download_universe,
    load_stock_universe,
    load_universe_metadata,
)
from app.probability.features import build_feature_dataset, feature_columns_for_variant
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.targets import add_forward_excess_return_labels
from app.probability.train import LABEL_TO_CLASS

COMPARISON_DIR = REPORT_DIR / "model_comparison"
PREDICTION_DIR = DATA_DIR / "model_comparison_predictions"
MODEL_NAMES = ("xgboost", "lightgbm", "random_forest")
ENSEMBLE_NAMES = ("ensemble_xgb_lgbm", "ensemble_all_equal", "ensemble_validation_weighted")


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """Container for a trained estimator and its metadata."""

    name: str
    model: Any
    feature_importance: pd.DataFrame


def main() -> None:
    """Run the model comparison end-to-end."""

    ensure_probability_dirs()
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    dataset = _load_or_build_dataset()
    feature_columns = feature_columns_for_variant("sector_regime")
    shared = dataset.dropna(subset=feature_columns + ["label"]).copy()
    train, validation, test = _time_splits(shared)

    trained = [_train_model(name, train, validation, feature_columns) for name in MODEL_NAMES]
    validation_predictions = {
        item.name: _predict_frame(item.model, validation, feature_columns, item.name)
        for item in trained
    }
    test_predictions = {
        item.name: _predict_frame(item.model, test, feature_columns, item.name) for item in trained
    }
    latest_features = _build_latest_feature_rows(feature_columns)
    latest_predictions = {
        item.name: _predict_latest(item.model, latest_features, feature_columns, item.name)
        for item in trained
    }

    ensemble_weights = _validation_weights(validation_predictions)
    ensembles = _build_ensembles(
        test_predictions=test_predictions,
        latest_predictions=latest_predictions,
        validation_weights=ensemble_weights,
    )
    all_predictions = {**test_predictions, **ensembles["test"]}
    all_latest = {**latest_predictions, **ensembles["latest"]}

    metrics = {
        name: _evaluate_predictions(predictions) for name, predictions in all_predictions.items()
    }
    walk_forward = {
        name: _walk_forward_predictions(name, shared, feature_columns) for name in MODEL_NAMES
    }
    walk_forward_ensembles = {
        "ensemble_xgb_lgbm": _average_predictions(
            [walk_forward["xgboost"], walk_forward["lightgbm"]],
            [0.5, 0.5],
            "ensemble_xgb_lgbm",
        ),
        "ensemble_all_equal": _average_predictions(
            [walk_forward[name] for name in MODEL_NAMES],
            [1 / 3, 1 / 3, 1 / 3],
            "ensemble_all_equal",
        ),
        "ensemble_validation_weighted": _average_predictions(
            [walk_forward[name] for name in MODEL_NAMES],
            [ensemble_weights[name] for name in MODEL_NAMES],
            "ensemble_validation_weighted",
        ),
    }
    walk_forward = {**walk_forward, **walk_forward_ensembles}
    for name, predictions in walk_forward.items():
        metrics[name]["walk_forward"] = _evaluate_predictions(predictions)
    metrics["ensemble_validation_weights"] = ensemble_weights

    bucket = pd.concat(
        [
            _probability_bucket_analysis(predictions, model=name)
            for name, predictions in all_predictions.items()
        ],
        ignore_index=True,
    )
    walk_forward_bucket = pd.concat(
        [
            _probability_bucket_analysis(predictions, model=name, scope="walk_forward")
            for name, predictions in walk_forward.items()
        ],
        ignore_index=True,
    )
    sector = pd.concat(
        [
            _sector_diagnostics(predictions, model=name)
            for name, predictions in all_predictions.items()
        ],
        ignore_index=True,
    )
    model_audit = _model_audit_report(all_predictions, metrics)
    model_selected_returns = _model_selected_returns(all_predictions)
    metrics_by_year = _metrics_by_group(all_predictions, group_column="year", scope="test_year")
    metrics_by_fold = _metrics_by_group(
        walk_forward,
        group_column="fold",
        scope="walk_forward_fold",
    )
    comparison = _comparison_table(metrics)
    agreement = _model_agreement(latest_predictions, latest_features)
    best_model = _best_out_of_sample_model(metrics)
    best_ensemble = _best_ensemble(metrics)

    for name, predictions in all_predictions.items():
        predictions.to_csv(PREDICTION_DIR / f"{name}_test_predictions.csv", index=False)
    for name, latest in all_latest.items():
        latest.to_csv(COMPARISON_DIR / f"{name}_latest_ranking.csv", index=False)
        latest.head(20).to_csv(COMPARISON_DIR / f"{name}_top20.csv", index=False)
    agreement.to_csv(COMPARISON_DIR / "latest_model_agreement.csv", index=False)
    bucket.to_csv(COMPARISON_DIR / "probability_bucket_analysis_by_model.csv", index=False)
    pd.concat([bucket, walk_forward_bucket], ignore_index=True).to_csv(
        COMPARISON_DIR / "probability_bucket_report.csv",
        index=False,
    )
    sector.to_csv(COMPARISON_DIR / "sector_diagnostics_by_model.csv", index=False)
    model_audit.to_csv(COMPARISON_DIR / "model_audit_report.csv", index=False)
    model_selected_returns.to_csv(COMPARISON_DIR / "model_selected_returns.csv", index=False)
    pd.concat([metrics_by_year, metrics_by_fold], ignore_index=True).to_csv(
        COMPARISON_DIR / "model_metrics_by_year_and_fold.csv",
        index=False,
    )
    comparison.to_csv(COMPARISON_DIR / "model_comparison_table.csv", index=False)
    _write_backtest_audit_report(metrics, all_predictions)
    _save_feature_importance_comparison(trained)
    (COMPARISON_DIR / "model_comparison_metrics.json").write_text(
        json.dumps(_json_safe(metrics), indent=2),
        encoding="utf-8",
    )
    (COMPARISON_DIR / "model_comparison_summary.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "best_out_of_sample_model": best_model,
                    "best_ensemble": best_ensemble,
                    "ensemble_validation_weights": ensemble_weights,
                    "shared_observations": {
                        "train": len(train),
                        "validation": len(validation),
                        "test": len(test),
                        "features": len(feature_columns),
                    },
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    # Deploy best ensemble/latest ranking only if it improves walk-forward over the best single model.
    selected_latest = all_latest[best_ensemble if best_ensemble else best_model]
    selected_latest.to_csv(FULL_RANKING_OUTPUT, index=False)
    selected_latest.head(20).to_csv(TOP20_RANKING_OUTPUT, index=False)
    selected_latest.to_csv(REPORT_DIR / "stock_probability_latest.csv", index=False)
    _deploy_selected_model(trained, best_model, feature_columns)
    print(
        json.dumps(
            {
                "comparison_table": str(COMPARISON_DIR / "model_comparison_table.csv"),
                "bucket_analysis": str(COMPARISON_DIR / "probability_bucket_analysis_by_model.csv"),
                "audit_report": str(COMPARISON_DIR / "model_audit_report.csv"),
                "selected_returns": str(COMPARISON_DIR / "model_selected_returns.csv"),
                "backtest_audit": str(COMPARISON_DIR / "backtest_audit_report.txt"),
                "sector_diagnostics": str(COMPARISON_DIR / "sector_diagnostics_by_model.csv"),
                "agreement": str(COMPARISON_DIR / "latest_model_agreement.csv"),
                "best_out_of_sample_model": best_model,
                "best_ensemble": best_ensemble,
                "deployed_ranking": str(FULL_RANKING_OUTPUT),
            },
            indent=2,
        )
    )


def _load_or_build_dataset() -> pd.DataFrame:
    """Load existing feature dataset or rebuild with unchanged 150-stock universe."""

    path = DATA_DIR / "probability_training_dataset.csv"
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["Date"])
        return _ensure_dataset_metadata(cached)

    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    symbols = load_stock_universe(STOCKS_UNIVERSE_2026_08_CSV)
    sector_symbols = sector_benchmark_symbols(universe)
    data = download_universe([*symbols, *sector_symbols], include_benchmark=True)
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
        metadata=universe,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    dataset = add_forward_excess_return_labels(features, data[BENCHMARK_SYMBOL])
    dataset = dataset.merge(
        universe[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    dataset.to_csv(path, index=False)
    return _ensure_dataset_metadata(dataset)


def _ensure_dataset_metadata(dataset: pd.DataFrame) -> pd.DataFrame:
    """Ensure cached datasets have current universe metadata columns."""

    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    metadata = universe[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
        columns={"Ticker": "symbol"}
    )
    output = dataset.copy()
    for column in ["Company", "Sector", "MarketCapCategory", "Group"]:
        if column in output.columns:
            output = output.drop(columns=[column])
    return output.merge(metadata, on="symbol", how="left")


def _time_splits(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return shared time-based train, validation, and final test observations."""

    dates = pd.to_datetime(dataset["Date"])
    return (
        dataset[dates <= "2022-12-31"].copy(),
        dataset[(dates >= "2023-01-01") & (dates <= "2024-12-31")].copy(),
        dataset[dates >= "2025-01-01"].copy(),
    )


def _train_model(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    *,
    walk_forward: bool = False,
) -> TrainedModel:
    """Train one model with the common training/validation observations."""

    if name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200 if walk_forward else 500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            early_stopping_rounds=30,
        )
        model.fit(
            train[feature_columns],
            train["label"].map(LABEL_TO_CLASS),
            eval_set=[(validation[feature_columns], validation["label"].map(LABEL_TO_CLASS))],
            verbose=False,
        )
    elif name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=200 if walk_forward else 500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multiclass",
            random_state=42,
            verbose=-1,
        )
        model.fit(train[feature_columns], train["label"].map(LABEL_TO_CLASS))
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=80 if walk_forward else 300,
            max_depth=7 if walk_forward else 9,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train[feature_columns], train["label"].map(LABEL_TO_CLASS))
    else:
        raise ValueError(f"Unknown model: {name}")

    model.natip_feature_columns = feature_columns
    return TrainedModel(
        name=name, model=model, feature_importance=_feature_importance(name, model, feature_columns)
    )


def _predict_frame(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Predict probabilities for a labeled frame."""

    probabilities = model.predict_proba(frame[feature_columns])
    output = frame[
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
    output["Model"] = model_name
    output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
    output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
    output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
    output["predicted_class"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map(
        {0: -1, 1: 0, 2: 1}
    )
    return output


def _predict_latest(
    model: Any,
    latest_features: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Predict latest ranking rows."""

    probabilities = model.predict_proba(latest_features[feature_columns])
    output = latest_features[
        ["symbol", "Sector", "MarketCapCategory", "Company", "Group", "Date"]
    ].copy()
    output["Model"] = model_name
    output["P_Underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
    output["P_Neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
    output["P_Outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
    output["Predicted_Class"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map(
        {0: -1, 1: 0, 2: 1}
    )
    output = output.rename(columns={"symbol": "Ticker", "Date": "Data Date"})
    output = output.sort_values("P_Outperform", ascending=False).reset_index(drop=True)
    output.insert(0, "Rank", range(1, len(output) + 1))
    return output[
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
            "Model",
        ]
    ]


def _build_latest_feature_rows(feature_columns: list[str]) -> pd.DataFrame:
    """Build latest unlabeled feature rows for current ranking output."""

    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    symbols = load_stock_universe(STOCKS_UNIVERSE_2026_08_CSV)
    sector_symbols = sector_benchmark_symbols(universe)
    data = download_universe([*symbols, *sector_symbols], include_benchmark=True)
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
        metadata=universe,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    features = features.merge(
        universe[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    return (
        features.dropna(subset=feature_columns)
        .sort_values("Date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate classifier and portfolio metrics from prediction rows."""

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        log_loss,
    )

    if predictions.empty:
        return {}

    y_true = predictions["label"].map(LABEL_TO_CLASS)
    y_pred = predictions["predicted_class"].map(LABEL_TO_CLASS)
    probabilities = predictions[["p_underperform", "p_neutral", "p_outperform"]].to_numpy()
    report = classification_report(
        y_true,
        y_pred,
        labels=[LABEL_TO_CLASS[-1], LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]],
        output_dict=True,
        zero_division=0,
    )
    calibration_true, calibration_pred = calibration_curve(
        (predictions["label"] == 1).astype(int),
        predictions["p_outperform"],
        n_bins=10,
        strategy="quantile",
    )
    backtest = backtest_top_n_portfolio(predictions, top_n=5)
    selected = _selected_outperform_stats(predictions)
    actual_outperform = (predictions["label"] == 1).astype(float)
    brier_outperform = float(((predictions["p_outperform"] - actual_outperform) ** 2).mean())
    outperform_key = str(LABEL_TO_CLASS[1])
    return {
        "number_predictions": int(len(predictions)),
        "predicted_outperform": int((predictions["predicted_class"] == 1).sum()),
        "predicted_neutral": int((predictions["predicted_class"] == 0).sum()),
        "predicted_underperform": int((predictions["predicted_class"] == -1).sum()),
        "actual_outperform_rate": float((predictions["label"] == 1).mean()),
        "precision_outperform": report[outperform_key]["precision"],
        "recall_outperform": report[outperform_key]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1, 2]),
        "brier_outperform": brier_outperform,
        "calibration_mae": float(abs(calibration_true - calibration_pred).mean()),
        "calibration": {
            "predicted_probability": calibration_pred.tolist(),
            "observed_frequency": calibration_true.tolist(),
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "cagr": backtest.get("cagr"),
        "sharpe": backtest.get("sharpe"),
        "sortino": backtest.get("sortino"),
        "max_drawdown": backtest.get("max_drawdown"),
        "win_rate": backtest.get("win_rate"),
        "turnover": backtest.get("turnover"),
        "avg_future_excess_return": selected["selected_avg_future_excess_return"],
        "overall_avg_future_excess_return": float(predictions["excess_return"].mean()),
        "selected_outperform_stats": selected,
        "backtest_audit": {
            "top_n": backtest.get("top_n"),
            "rebalance_every_days": backtest.get("rebalance_every_days"),
            "leverage_enabled": backtest.get("leverage_enabled"),
            "max_weight_sum": backtest.get("max_weight_sum"),
            "duplicate_prediction_rows_removed": backtest.get("duplicate_prediction_rows_removed"),
            "skipped_rebalance_dates": backtest.get("skipped_rebalance_dates"),
            "non_overlapping_holding_periods": backtest.get("non_overlapping_holding_periods"),
            "periods": backtest.get("periods"),
        },
        "excess_cagr_vs_nifty": (
            backtest.get("cagr") - backtest.get("benchmark_cagr")
            if backtest.get("cagr") is not None and backtest.get("benchmark_cagr") is not None
            else None
        ),
        "benchmark_cagr": backtest.get("benchmark_cagr"),
    }


def _selected_outperform_stats(predictions: pd.DataFrame) -> dict[str, Any]:
    """Summarize realized returns for rows predicted as Outperform by a model."""

    selected = predictions[predictions["predicted_class"] == 1]
    if selected.empty:
        return {
            "selected_observations": 0,
            "selected_avg_future_stock_return": None,
            "selected_median_future_stock_return": None,
            "selected_avg_future_excess_return": None,
            "selected_median_future_excess_return": None,
            "selected_win_rate_vs_nifty": None,
        }
    return {
        "selected_observations": int(len(selected)),
        "selected_avg_future_stock_return": float(selected["future_stock_return"].mean()),
        "selected_median_future_stock_return": float(selected["future_stock_return"].median()),
        "selected_avg_future_excess_return": float(selected["excess_return"].mean()),
        "selected_median_future_excess_return": float(selected["excess_return"].median()),
        "selected_win_rate_vs_nifty": float((selected["excess_return"] > 0).mean()),
    }


def _walk_forward_predictions(
    model_name: str,
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate walk-forward predictions using only past data for each fold."""

    dates = pd.to_datetime(dataset["Date"])
    current = dates.min() + pd.DateOffset(years=3)
    end = dates.max()
    frames = []
    fold = 1
    while current < end:
        test_end = current + pd.DateOffset(months=6)
        train = dataset[dates < current].dropna(subset=feature_columns + ["label"])
        test = dataset[(dates >= current) & (dates < test_end)].dropna(
            subset=feature_columns + ["label"]
        )
        if not train.empty and not test.empty:
            model = _train_model(
                model_name,
                train,
                test,
                feature_columns,
                walk_forward=True,
            ).model
            fold_predictions = _predict_frame(model, test, feature_columns, model_name)
            fold_predictions["fold"] = fold
            fold_predictions["fold_train_end"] = current
            fold_predictions["fold_test_start"] = current
            fold_predictions["fold_test_end"] = test_end
            frames.append(fold_predictions)
            fold += 1
        current = test_end
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validation_weights(validation_predictions: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Derive ensemble weights from validation-period log loss only."""

    scores = {}
    for name, predictions in validation_predictions.items():
        metrics = _evaluate_predictions(predictions)
        scores[name] = 1 / max(metrics["log_loss"], 1e-9)
    total = sum(scores.values())
    return {name: score / total for name, score in scores.items()}


def _build_ensembles(
    *,
    test_predictions: dict[str, pd.DataFrame],
    latest_predictions: dict[str, pd.DataFrame],
    validation_weights: dict[str, float],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build equal and validation-weighted ensemble predictions."""

    return {
        "test": {
            "ensemble_xgb_lgbm": _average_predictions(
                [test_predictions["xgboost"], test_predictions["lightgbm"]],
                [0.5, 0.5],
                "ensemble_xgb_lgbm",
            ),
            "ensemble_all_equal": _average_predictions(
                [test_predictions[name] for name in MODEL_NAMES],
                [1 / 3, 1 / 3, 1 / 3],
                "ensemble_all_equal",
            ),
            "ensemble_validation_weighted": _average_predictions(
                [test_predictions[name] for name in MODEL_NAMES],
                [validation_weights[name] for name in MODEL_NAMES],
                "ensemble_validation_weighted",
            ),
        },
        "latest": {
            "ensemble_xgb_lgbm": _average_latest(
                [latest_predictions["xgboost"], latest_predictions["lightgbm"]],
                [0.5, 0.5],
                "ensemble_xgb_lgbm",
            ),
            "ensemble_all_equal": _average_latest(
                [latest_predictions[name] for name in MODEL_NAMES],
                [1 / 3, 1 / 3, 1 / 3],
                "ensemble_all_equal",
            ),
            "ensemble_validation_weighted": _average_latest(
                [latest_predictions[name] for name in MODEL_NAMES],
                [validation_weights[name] for name in MODEL_NAMES],
                "ensemble_validation_weighted",
            ),
        },
    }


def _average_predictions(
    frames: list[pd.DataFrame], weights: list[float], name: str
) -> pd.DataFrame:
    """Average prediction probabilities for labeled frames."""

    base = (
        frames[0]
        .drop(columns=["Model", "p_underperform", "p_neutral", "p_outperform", "predicted_class"])
        .copy()
    )
    for column in ["p_underperform", "p_neutral", "p_outperform"]:
        base[column] = sum(
            frame[column].to_numpy() * weight for frame, weight in zip(frames, weights, strict=True)
        )
    _normalize_probability_columns(base, ["p_underperform", "p_neutral", "p_outperform"])
    base["predicted_class"] = (
        base[["p_underperform", "p_neutral", "p_outperform"]].to_numpy().argmax(axis=1)
    )
    base["predicted_class"] = base["predicted_class"].map({0: -1, 1: 0, 2: 1})
    base["Model"] = name
    return base


def _average_latest(frames: list[pd.DataFrame], weights: list[float], name: str) -> pd.DataFrame:
    """Average latest ranking probabilities."""

    base = (
        frames[0]
        .drop(
            columns=[
                "Rank",
                "Model",
                "P_Underperform",
                "P_Neutral",
                "P_Outperform",
                "Predicted_Class",
            ]
        )
        .copy()
    )
    for column in ["P_Underperform", "P_Neutral", "P_Outperform"]:
        base[column] = sum(
            frame[column].to_numpy() * weight for frame, weight in zip(frames, weights, strict=True)
        )
    _normalize_probability_columns(base, ["P_Underperform", "P_Neutral", "P_Outperform"])
    base["Predicted_Class"] = (
        base[["P_Underperform", "P_Neutral", "P_Outperform"]].to_numpy().argmax(axis=1)
    )
    base["Predicted_Class"] = base["Predicted_Class"].map({0: -1, 1: 0, 2: 1})
    base["Model"] = name
    base = base.sort_values("P_Outperform", ascending=False).reset_index(drop=True)
    base.insert(0, "Rank", range(1, len(base) + 1))
    return base


def _probability_bucket_analysis(
    predictions: pd.DataFrame,
    *,
    model: str,
    scope: str = "test",
) -> pd.DataFrame:
    """Analyze realized performance by probability bucket."""

    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.000001]
    labels = ["0-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-100%"]
    frame = predictions.copy()
    frame["bucket"] = pd.cut(
        frame["p_outperform"], bins=bins, labels=labels, include_lowest=True, right=False
    )
    rows = []
    for bucket in labels:
        group = frame[frame["bucket"] == bucket]
        rows.append(
            {
                "Model": model,
                "Scope": scope,
                "P_Outperform_Bucket": bucket,
                "Count": int(len(group)),
                "ObservationCount": int(len(group)),
                "PredictedAvgProbability": (
                    float(group["p_outperform"].mean()) if len(group) else None
                ),
                "ActualOutperformRate": float((group["label"] == 1).mean()) if len(group) else None,
                "AverageFutureReturn": (
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


def _normalize_probability_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Normalize probability columns in place."""

    totals = frame[columns].sum(axis=1).replace(0, 1)
    for column in columns:
        frame[column] = frame[column] / totals


def _sector_diagnostics(predictions: pd.DataFrame, *, model: str) -> pd.DataFrame:
    """Generate sector-level diagnostics."""

    rows = []
    for sector, group in predictions.groupby("Sector"):
        rows.append(
            {
                "Model": model,
                "Sector": sector,
                "PredictionCount": int(len(group)),
                "SuccessRate": float((group["label"] == 1).mean()),
                "AverageFutureExcessReturn": float(group["excess_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _model_audit_report(
    predictions_by_model: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Create the independent model audit report requested for test predictions."""

    rows = []
    for name, predictions in predictions_by_model.items():
        metric = metrics[name]
        selected = metric.get("selected_outperform_stats", {})
        rows.append(
            {
                "Model": name,
                "Scope": "test",
                "NumberPredictions": metric.get("number_predictions"),
                "PredictedOutperform": metric.get("predicted_outperform"),
                "PredictedNeutral": metric.get("predicted_neutral"),
                "PredictedUnderperform": metric.get("predicted_underperform"),
                "ActualOutperformRate": metric.get("actual_outperform_rate"),
                "PrecisionOutperform": metric.get("precision_outperform"),
                "RecallOutperform": metric.get("recall_outperform"),
                "MacroF1": metric.get("f1_macro"),
                "WeightedF1": metric.get("f1_weighted"),
                "BalancedAccuracy": metric.get("balanced_accuracy"),
                "LogLoss": metric.get("log_loss"),
                "BrierOutperform": metric.get("brier_outperform"),
                "CalibrationMAE": metric.get("calibration_mae"),
                "SelectedObservations": selected.get("selected_observations"),
                "SelectedAvgFuture20DStockReturn": selected.get(
                    "selected_avg_future_stock_return"
                ),
                "SelectedMedianFuture20DStockReturn": selected.get(
                    "selected_median_future_stock_return"
                ),
                "SelectedAvgFuture20DExcessReturn": selected.get(
                    "selected_avg_future_excess_return"
                ),
                "SelectedMedianFuture20DExcessReturn": selected.get(
                    "selected_median_future_excess_return"
                ),
                "SelectedWinRateVsNifty": selected.get("selected_win_rate_vs_nifty"),
                "CAGR": metric.get("cagr"),
                "Sharpe": metric.get("sharpe"),
                "Sortino": metric.get("sortino"),
                "MaxDrawdown": metric.get("max_drawdown"),
                "BacktestWinRate": metric.get("win_rate"),
                "Turnover": metric.get("turnover"),
                "BenchmarkCAGR": metric.get("benchmark_cagr"),
                "ExcessCAGRVsNifty": metric.get("excess_cagr_vs_nifty"),
            }
        )
    return pd.DataFrame(rows)


def _model_selected_returns(predictions_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Save model-selected Outperform observations and their realized returns."""

    frames = []
    columns = [
        "Model",
        "Date",
        "symbol",
        "Sector",
        "MarketCapCategory",
        "p_outperform",
        "p_neutral",
        "p_underperform",
        "predicted_class",
        "label",
        "future_stock_return",
        "future_nifty_return",
        "excess_return",
    ]
    for name, predictions in predictions_by_model.items():
        selected = predictions[predictions["predicted_class"] == 1].copy()
        if selected.empty:
            continue
        selected["Model"] = name
        frames.append(selected[columns])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _metrics_by_group(
    predictions_by_model: dict[str, pd.DataFrame],
    *,
    group_column: str,
    scope: str,
) -> pd.DataFrame:
    """Calculate audit metrics by year or walk-forward fold."""

    rows = []
    for name, predictions in predictions_by_model.items():
        if predictions.empty:
            continue
        frame = predictions.copy()
        frame["year"] = pd.to_datetime(frame["Date"]).dt.year
        if group_column not in frame.columns:
            continue
        for group_value, group in frame.groupby(group_column):
            if group.empty:
                continue
            metric = _evaluate_predictions(group)
            selected = metric.get("selected_outperform_stats", {})
            rows.append(
                {
                    "Model": name,
                    "Scope": scope,
                    "Group": group_value,
                    "NumberPredictions": metric.get("number_predictions"),
                    "PredictedOutperform": metric.get("predicted_outperform"),
                    "ActualOutperformRate": metric.get("actual_outperform_rate"),
                    "PrecisionOutperform": metric.get("precision_outperform"),
                    "RecallOutperform": metric.get("recall_outperform"),
                    "MacroF1": metric.get("f1_macro"),
                    "WeightedF1": metric.get("f1_weighted"),
                    "BalancedAccuracy": metric.get("balanced_accuracy"),
                    "LogLoss": metric.get("log_loss"),
                    "BrierOutperform": metric.get("brier_outperform"),
                    "SelectedObservations": selected.get("selected_observations"),
                    "SelectedAvgFuture20DExcessReturn": selected.get(
                        "selected_avg_future_excess_return"
                    ),
                    "SelectedWinRateVsNifty": selected.get("selected_win_rate_vs_nifty"),
                    "CAGR": metric.get("cagr"),
                    "Sharpe": metric.get("sharpe"),
                    "MaxDrawdown": metric.get("max_drawdown"),
                }
            )
    return pd.DataFrame(rows)


def _write_backtest_audit_report(
    metrics: dict[str, dict[str, Any]],
    predictions_by_model: dict[str, pd.DataFrame],
) -> None:
    """Write a plain-English audit of the corrected backtest assumptions."""

    lines = [
        "NATIP probability model backtest audit",
        "",
        "Findings fixed:",
        "- Avg Excess Return was previously calculated from the complete shared test set, so it was identical for every model.",
        "- The portfolio backtest previously rebalanced on every date while using 20-day forward returns, which compounded overlapping 20-day outcomes.",
        "",
        "Current backtest assumptions:",
        "- Leverage is disabled.",
        "- Positions are equal-weighted.",
        "- Portfolio weights are asserted to be <= 100%.",
        "- Rebalancing uses non-overlapping 20-trading-day steps.",
        "- Portfolio period return is the weighted average of selected stock forward returns less transaction costs.",
        "- Duplicate Date/symbol prediction rows are removed before backtesting.",
        "- NaN and infinite returns are rejected before performance metrics are calculated.",
        "",
        "Model audit summary:",
    ]
    for name in predictions_by_model:
        metric = metrics[name]
        selected = metric.get("selected_outperform_stats", {})
        audit = metric.get("backtest_audit", {})
        lines.extend(
            [
                f"- {name}: selected={selected.get('selected_observations')}, "
                f"selected_avg_excess={selected.get('selected_avg_future_excess_return')}, "
                f"CAGR={metric.get('cagr')}, MaxDD={metric.get('max_drawdown')}, "
                f"periods={audit.get('periods')}, max_weight_sum={audit.get('max_weight_sum')}, "
                f"duplicates_removed={audit.get('duplicate_prediction_rows_removed')}",
            ]
        )
    (COMPARISON_DIR / "backtest_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


def _comparison_table(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create requested comparison table."""

    rows = []
    for name, metric in metrics.items():
        if name == "ensemble_validation_weights":
            continue
        rows.append(
            {
                "Model": name,
                "F1": metric.get("f1_weighted"),
                "LogLoss": metric.get("log_loss"),
                "Calibration": metric.get("calibration_mae"),
                "CAGR": metric.get("cagr"),
                "Sharpe": metric.get("sharpe"),
                "MaxDD": metric.get("max_drawdown"),
                "Avg Excess Return": metric.get("avg_future_excess_return"),
            }
        )
    return pd.DataFrame(rows).sort_values(["F1", "LogLoss"], ascending=[False, True])


def _model_agreement(
    latest_predictions: dict[str, pd.DataFrame],
    latest_features: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate latest model agreement score across base models."""

    output = latest_features[
        ["symbol", "Sector", "MarketCapCategory", "Company", "Group", "Date"]
    ].copy()
    output = output.rename(columns={"symbol": "Ticker", "Date": "Data Date"})
    for name, frame in latest_predictions.items():
        output = output.merge(
            frame[["Ticker", "Predicted_Class", "P_Outperform"]].rename(
                columns={
                    "Predicted_Class": f"{name}_class",
                    "P_Outperform": f"{name}_p_outperform",
                }
            ),
            on="Ticker",
            how="left",
        )
    class_columns = [f"{name}_class" for name in MODEL_NAMES]
    output["model_agreement_score"] = output[class_columns].eq(1).sum(axis=1)
    return output.sort_values("model_agreement_score", ascending=False)


def _best_out_of_sample_model(metrics: dict[str, dict[str, Any]]) -> str:
    """Pick best base model by walk-forward F1, then log loss."""

    base = {
        name: metric["walk_forward"]
        for name, metric in metrics.items()
        if name in MODEL_NAMES and "walk_forward" in metric
    }
    return sorted(
        base, key=lambda name: (base[name]["f1_weighted"], -base[name]["log_loss"]), reverse=True
    )[0]


def _best_ensemble(metrics: dict[str, dict[str, Any]]) -> str | None:
    """Keep ensemble only if walk-forward out-of-sample metrics improve."""

    best_base = _best_out_of_sample_model(metrics)
    base_score = metrics[best_base]["walk_forward"]["f1_weighted"]
    ensembles = {name: metrics[name] for name in ENSEMBLE_NAMES if name in metrics}
    if not ensembles:
        return None
    best = sorted(
        ensembles,
        key=lambda name: (
            ensembles[name]["walk_forward"]["f1_weighted"],
            -ensembles[name]["walk_forward"]["log_loss"],
        ),
        reverse=True,
    )[0]
    return best if ensembles[best]["walk_forward"]["f1_weighted"] > base_score else None


def _feature_importance(name: str, model: Any, feature_columns: list[str]) -> pd.DataFrame:
    """Extract feature importance from a trained model."""

    values = getattr(model, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["Model", "Feature", "Importance"])
    return pd.DataFrame({"Model": name, "Feature": feature_columns, "Importance": values})


def _save_feature_importance_comparison(trained: list[TrainedModel]) -> None:
    """Save feature importance across base models."""

    pd.concat([item.feature_importance for item in trained], ignore_index=True).to_csv(
        COMPARISON_DIR / "feature_importance_by_model.csv",
        index=False,
    )


def _deploy_selected_model(
    trained: list[TrainedModel],
    selected_model: str,
    feature_columns: list[str],
) -> None:
    """Deploy the best selected base model for dashboard-compatible scoring."""

    try:
        from joblib import dump
    except ImportError:
        return
    selected = next(item for item in trained if item.name == selected_model)
    dump(selected.model, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def _json_safe(value):
    """Convert pandas/numpy scalar values into JSON-safe values."""

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


if __name__ == "__main__":
    main()
