"""Audit advanced price and market features for the NSE probability screener."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    DATA_DIR,
    REPORT_DIR,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.data_download import (
    download_universe,
    load_stock_universe,
    load_universe_metadata,
)
from app.probability.features import (
    ADVANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MARKET_REGIME_FEATURE_COLUMNS,
    SECTOR_RELATIVE_FEATURE_COLUMNS,
    build_feature_dataset,
    feature_columns_for_variant,
)
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.targets import add_forward_excess_return_labels
from app.probability.train import LABEL_TO_CLASS
from compare_probability_models import (
    _evaluate_predictions,
    _feature_importance,
    _predict_frame,
    _probability_bucket_analysis,
    _time_splits,
    _train_model,
    _walk_forward_predictions,
)

ADVANCED_DATASET_PATH = DATA_DIR / "probability_training_dataset_advanced.csv"
ADVANCED_AUDIT_OUTPUT = REPORT_DIR / "advanced_feature_audit.csv"
FEATURE_IMPORTANCE_COMPARISON_OUTPUT = REPORT_DIR / "feature_importance_comparison.csv"
ADVANCED_MODEL_COMPARISON_OUTPUT = REPORT_DIR / "advanced_model_comparison.csv"
PROBABILITY_BUCKET_COMPARISON_OUTPUT = REPORT_DIR / "probability_bucket_comparison.csv"


def main() -> None:
    """Run the advanced feature audit without changing deployed model artifacts."""

    ensure_probability_dirs()
    dataset = _build_advanced_dataset()
    baseline_columns = feature_columns_for_variant("sector_regime")
    advanced_columns = feature_columns_for_variant("advanced")
    train, validation, test = _time_splits(dataset.dropna(subset=["label"]).copy())

    audit = _advanced_feature_audit(dataset, baseline_columns, advanced_columns)
    audit.to_csv(ADVANCED_AUDIT_OUTPUT, index=False)

    results = {}
    for variant, columns in {
        "baseline_sector_regime": baseline_columns,
        "advanced": advanced_columns,
    }.items():
        clean_train = train.dropna(subset=columns + ["label"])
        clean_validation = validation.dropna(subset=columns + ["label"])
        clean_test = test.dropna(subset=columns + ["label"])
        trained = _train_model("xgboost", clean_train, clean_validation, columns)
        predictions = _predict_frame(trained.model, clean_test, columns, variant)
        test_metrics = _evaluate_predictions(predictions)
        walk_forward_predictions = _walk_forward_predictions("xgboost", dataset, columns)
        walk_forward_metrics = _evaluate_predictions(walk_forward_predictions)
        results[variant] = {
            "columns": columns,
            "model": trained.model,
            "feature_importance": trained.feature_importance,
            "predictions": predictions,
            "test_metrics": test_metrics,
            "walk_forward_predictions": walk_forward_predictions,
            "walk_forward_metrics": walk_forward_metrics,
        }

    comparison = _comparison_rows(results)
    comparison.to_csv(ADVANCED_MODEL_COMPARISON_OUTPUT, index=False)

    buckets = pd.concat(
        [
            _probability_bucket_analysis(
                result["predictions"],
                model=variant,
                scope="test",
            )
            for variant, result in results.items()
        ],
        ignore_index=True,
    )
    buckets.to_csv(PROBABILITY_BUCKET_COMPARISON_OUTPUT, index=False)

    importance = _feature_importance_comparison(results, validation, audit)
    importance.to_csv(FEATURE_IMPORTANCE_COMPARISON_OUTPUT, index=False)

    print(
        json.dumps(
            {
                "advanced_feature_audit": str(ADVANCED_AUDIT_OUTPUT),
                "feature_importance_comparison": str(FEATURE_IMPORTANCE_COMPARISON_OUTPUT),
                "advanced_model_comparison": str(ADVANCED_MODEL_COMPARISON_OUTPUT),
                "probability_bucket_comparison": str(PROBABILITY_BUCKET_COMPARISON_OUTPUT),
                "advanced_dataset": str(ADVANCED_DATASET_PATH),
            },
            indent=2,
        )
    )


def _build_advanced_dataset() -> pd.DataFrame:
    """Build a fresh dataset so newly added feature columns are present."""

    if ADVANCED_DATASET_PATH.exists():
        cached = pd.read_csv(ADVANCED_DATASET_PATH, parse_dates=["Date"])
        required = set(feature_columns_for_variant("advanced")) | {"label", "future_stock_return"}
        if required.issubset(cached.columns):
            return cached

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
    dataset.to_csv(ADVANCED_DATASET_PATH, index=False)
    return dataset


def _advanced_feature_audit(
    dataset: pd.DataFrame,
    baseline_columns: list[str],
    advanced_columns: list[str],
) -> pd.DataFrame:
    """Record existing coverage, newly added columns, and redundancy flags."""

    requested_groups = {
        "nifty_relative_momentum": [
            "rs_vs_nifty_20d",
            "rs_vs_nifty_60d",
            "rs_vs_nifty_120d",
        ],
        "sector_relative_momentum": [
            "stock_vs_sector_ret_20d",
            "stock_vs_sector_ret_60d",
            "stock_vs_sector_ret_120d",
            "sector_momentum_20d",
            "sector_momentum_60d",
            "sector_momentum_120d",
        ],
        "nifty_market_regime": [
            "nifty_above_sma20",
            "nifty_above_sma50",
            "nifty_above_sma200",
            "market_regime_bull_trend",
            "market_regime_bear_trend",
            "market_regime_sideways",
        ],
        "market_breadth": [
            "market_breadth_above_sma50",
            "market_breadth_above_sma200",
            "market_breadth_positive_20d_return",
            "market_breadth_advance_decline_ratio",
        ],
    }
    train_validation = dataset[pd.to_datetime(dataset["Date"]) <= "2024-12-31"]
    rows = []
    for group, columns in requested_groups.items():
        for column in columns:
            rows.append(_feature_audit_row(column, group, dataset, baseline_columns, advanced_columns))
    for column in ADVANCED_FEATURE_COLUMNS:
        row = _feature_audit_row(column, "advanced_added_or_verified", dataset, baseline_columns, advanced_columns)
        row.update(_redundancy_check(column, train_validation, baseline_columns))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_audit_row(
    column: str,
    group: str,
    dataset: pd.DataFrame,
    baseline_columns: list[str],
    advanced_columns: list[str],
) -> dict[str, Any]:
    """Build one feature audit row."""

    present = column in dataset.columns
    non_null = int(dataset[column].notna().sum()) if present else 0
    return {
        "Feature": column,
        "Group": group,
        "AlreadyInBaseline": column in baseline_columns,
        "InAdvancedFeatureSet": column in advanced_columns,
        "PresentInDataset": present,
        "NonNullRows": non_null,
        "MissingPct": float(dataset[column].isna().mean()) if present else None,
        "LeakageControl": _leakage_note(column),
    }


def _redundancy_check(
    column: str,
    dataset: pd.DataFrame,
    baseline_columns: list[str],
) -> dict[str, Any]:
    """Flag advanced features that are highly correlated with baseline features."""

    if column not in dataset.columns:
        return {"MaxAbsCorrelationWithBaseline": None, "MostCorrelatedBaselineFeature": None, "RedundancyFlag": "missing"}
    candidates = [item for item in baseline_columns if item in dataset.columns]
    frame = dataset[[column, *candidates]].dropna().tail(20_000)
    if frame.empty:
        return {"MaxAbsCorrelationWithBaseline": None, "MostCorrelatedBaselineFeature": None, "RedundancyFlag": "insufficient_data"}
    correlations = frame.corr(numeric_only=True)[column].drop(labels=[column]).abs().dropna()
    if correlations.empty:
        return {"MaxAbsCorrelationWithBaseline": None, "MostCorrelatedBaselineFeature": None, "RedundancyFlag": "insufficient_data"}
    best_feature = str(correlations.idxmax())
    best_value = float(correlations.loc[best_feature])
    return {
        "MaxAbsCorrelationWithBaseline": best_value,
        "MostCorrelatedBaselineFeature": best_feature,
        "RedundancyFlag": "high_redundancy" if best_value >= 0.95 else "ok",
    }


def _comparison_rows(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare baseline and advanced models on test and walk-forward metrics."""

    rows = []
    baseline_walk_forward = results["baseline_sector_regime"]["walk_forward_metrics"]
    for variant, result in results.items():
        for scope, metric in {
            "test": result["test_metrics"],
            "walk_forward": result["walk_forward_metrics"],
        }.items():
            predictions = (
                result["predictions"] if scope == "test" else result["walk_forward_predictions"]
            )
            rows.append(
                {
                    "Variant": variant,
                    "Scope": scope,
                    "FeatureCount": len(result["columns"]),
                    "Rows": len(predictions),
                    "LogLoss": metric.get("log_loss"),
                    "CalibrationMAE": metric.get("calibration_mae"),
                    "PRAUCOutperform": _pr_auc(predictions),
                    "OutperformPrecision": metric.get("precision_outperform"),
                    "OutperformRecall": metric.get("recall_outperform"),
                    "MacroF1": metric.get("f1_macro"),
                    "WeightedF1": metric.get("f1_weighted"),
                    "BalancedAccuracy": metric.get("balanced_accuracy"),
                    "SelectedAvgExcessReturn": metric.get("avg_future_excess_return"),
                    "CAGR": metric.get("cagr"),
                    "Sharpe": metric.get("sharpe"),
                    "MaxDrawdown": metric.get("max_drawdown"),
                    "DeploymentFlag": _deployment_flag(
                        variant,
                        scope,
                        metric,
                        baseline_walk_forward,
                    ),
                }
            )
    comparison = pd.DataFrame(rows)
    return comparison


def _feature_importance_comparison(
    results: dict[str, dict[str, Any]],
    validation: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    """Compare model, permutation, and optional SHAP importance."""

    rows = []
    for variant, result in results.items():
        model = result["model"]
        columns = result["columns"]
        model_importance = _feature_importance(variant, model, columns)
        permutation = _permutation_importance(model, validation, columns)
        shap_values = _optional_shap_importance(model, validation, columns)
        for _, row in model_importance.iterrows():
            feature = str(row["Feature"])
            rows.append(
                {
                    "Variant": variant,
                    "Feature": feature,
                    "ModelImportance": float(row["Importance"]),
                    "PermutationImportanceValidation": permutation.get(feature),
                    "ShapMeanAbsValidation": shap_values.get(feature),
                    "IsAdvancedFeature": feature in ADVANCED_FEATURE_COLUMNS,
                    "RedundancyFlag": _audit_value(audit, feature, "RedundancyFlag"),
                    "MaxAbsCorrelationWithBaseline": _audit_value(
                        audit, feature, "MaxAbsCorrelationWithBaseline"
                    ),
                    "ContributionFlag": _contribution_flag(
                        feature,
                        permutation.get(feature),
                        _audit_value(audit, feature, "RedundancyFlag"),
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Variant", "PermutationImportanceValidation", "ModelImportance"],
        ascending=[True, False, False],
    )


def _permutation_importance(
    model: Any,
    validation: pd.DataFrame,
    columns: list[str],
) -> dict[str, float]:
    """Calculate validation-only permutation importance."""

    try:
        from sklearn.inspection import permutation_importance
    except ImportError:
        return {}
    clean = validation.dropna(subset=columns + ["label"]).tail(5_000)
    if clean.empty:
        return {}
    result = permutation_importance(
        model,
        clean[columns],
        clean["label"].map(LABEL_TO_CLASS),
        n_repeats=3,
        random_state=42,
        scoring="neg_log_loss",
    )
    return dict(zip(columns, result.importances_mean.tolist(), strict=True))


def _optional_shap_importance(
    model: Any,
    validation: pd.DataFrame,
    columns: list[str],
) -> dict[str, float]:
    """Return SHAP importance if the optional package is installed."""

    try:
        import shap
    except ImportError:
        return {}
    clean = validation.dropna(subset=columns + ["label"]).tail(500)
    if clean.empty:
        return {}
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(clean[columns])
    if isinstance(values, list):
        mean_abs = sum(abs(value).mean(axis=0) for value in values) / len(values)
    else:
        mean_abs = abs(values).mean(axis=0)
        if getattr(mean_abs, "ndim", 1) > 1:
            mean_abs = mean_abs.mean(axis=1)
    return dict(zip(columns, [float(value) for value in mean_abs], strict=True))


def _pr_auc(predictions: pd.DataFrame) -> float | None:
    """Calculate binary PR-AUC for the Outperform class."""

    if predictions.empty:
        return None
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return None
    return float(
        average_precision_score(
            (predictions["label"] == 1).astype(int),
            predictions["p_outperform"],
        )
    )


def _deployment_flag(
    variant: str,
    scope: str,
    metric: dict[str, Any],
    baseline_walk_forward: dict[str, Any],
) -> str:
    """Flag whether a variant should be promoted based on walk-forward only."""

    if variant != "advanced" or scope != "walk_forward":
        return "reference"
    baseline_log_loss = baseline_walk_forward.get("log_loss")
    baseline_calibration = baseline_walk_forward.get("calibration_mae")
    if baseline_log_loss is None or baseline_calibration is None:
        return "insufficient_baseline"
    if (
        metric.get("log_loss") is not None
        and metric.get("calibration_mae") is not None
        and metric["log_loss"] <= baseline_log_loss
        and metric["calibration_mae"] <= baseline_calibration
    ):
        return "candidate_for_further_validation"
    return "do_not_deploy_walk_forward_not_improved"


def _contribution_flag(
    feature: str,
    permutation_importance: float | None,
    redundancy_flag: Any,
) -> str:
    """Flag advanced features using validation-only importance and redundancy."""

    if feature not in ADVANCED_FEATURE_COLUMNS:
        return "baseline_feature"
    if redundancy_flag == "high_redundancy":
        return "drop_candidate_high_redundancy"
    if permutation_importance is None:
        return "insufficient_validation_importance"
    if permutation_importance <= 0:
        return "drop_candidate_no_validation_gain"
    return "contributes_on_validation"


def _audit_value(audit: pd.DataFrame, feature: str, column: str) -> Any:
    """Fetch a value from the audit table."""

    match = audit[audit["Feature"] == feature]
    if match.empty or column not in match.columns:
        return None
    return match.iloc[-1][column]


def _leakage_note(column: str) -> str:
    """Describe feature-specific leakage control."""

    if "prev_" in column or "breakout_" in column:
        return "Prior high uses high.shift(1).rolling(...).max()."
    if column.startswith("market_breadth"):
        return "Breadth uses same-date universe data and backward-looking moving averages/returns."
    if column.startswith("beta_") or column.startswith("corr_"):
        return "Rolling covariance/correlation uses historical daily returns only."
    return "Backward-looking rolling or same-date feature."


if __name__ == "__main__":
    main()
