"""Compare baseline, pruned advanced, and full advanced feature sets."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.features import (
    ADVANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MARKET_REGIME_FEATURE_COLUMNS,
    SECTOR_RELATIVE_FEATURE_COLUMNS,
    feature_columns_for_variant,
)
from app.probability.train import LABEL_TO_CLASS
from audit_advanced_probability_features import _build_advanced_dataset
from compare_probability_models import (
    _evaluate_predictions,
    _feature_importance,
    _predict_frame,
    _probability_bucket_analysis,
    _time_splits,
    _train_model,
)

PRUNED_FEATURES = [
    "beta_252d",
    "beta_120d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "sector_rsi_14",
    "corr_nifty_20d",
    "relative_momentum_20d_change",
]

OUTPUT_MODEL_COMPARISON = REPORT_DIR / "pruned_model_comparison.csv"
OUTPUT_COMMON_SAMPLE = REPORT_DIR / "common_sample_comparison.csv"
OUTPUT_DISTRIBUTION = REPORT_DIR / "probability_distribution_comparison.csv"
OUTPUT_IMPORTANCE = REPORT_DIR / "pruned_feature_importance.csv"


def main() -> None:
    """Run common-sample model comparison without changing deployed artifacts."""

    ensure_probability_dirs()
    dataset = _build_advanced_dataset().dropna(subset=["label"]).copy()
    variants = _variant_columns()
    common_columns = _common_required_columns(variants)
    common = dataset.dropna(subset=common_columns + ["label"]).copy()
    train, validation, test = _time_splits(common)
    _assert_common_counts(variants, train, validation, test)

    results = {}
    for variant, columns in variants.items():
        trained = _train_model("xgboost", train, validation, columns)
        predictions = _predict_frame(trained.model, test, columns, variant)
        walk_forward_predictions = _walk_forward_predictions_common(variant, common, columns)
        results[variant] = {
            "columns": columns,
            "model": trained.model,
            "feature_importance": trained.feature_importance,
            "test_predictions": predictions,
            "test_metrics": _evaluate_predictions(predictions),
            "walk_forward_predictions": walk_forward_predictions,
            "walk_forward_metrics": _evaluate_predictions(walk_forward_predictions),
        }

    comparison = _comparison_report(results)
    comparison.to_csv(OUTPUT_MODEL_COMPARISON, index=False)
    common_sample = _common_sample_report(common, train, validation, test, variants, results)
    common_sample.to_csv(OUTPUT_COMMON_SAMPLE, index=False)
    distribution = _probability_distribution_report(results)
    distribution.to_csv(OUTPUT_DISTRIBUTION, index=False)
    importance = _importance_report(results, validation)
    importance.to_csv(OUTPUT_IMPORTANCE, index=False)

    print(
        json.dumps(
            {
                "pruned_model_comparison": str(OUTPUT_MODEL_COMPARISON),
                "common_sample_comparison": str(OUTPUT_COMMON_SAMPLE),
                "probability_distribution_comparison": str(OUTPUT_DISTRIBUTION),
                "pruned_feature_importance": str(OUTPUT_IMPORTANCE),
                "recommendation": _recommendation(comparison),
            },
            indent=2,
        )
    )


def _variant_columns() -> dict[str, list[str]]:
    """Return the three requested feature variants."""

    baseline = feature_columns_for_variant("sector_regime")
    pruned = [*baseline, *PRUNED_FEATURES]
    full_advanced = [
        *FEATURE_COLUMNS,
        *SECTOR_RELATIVE_FEATURE_COLUMNS,
        *MARKET_REGIME_FEATURE_COLUMNS,
        *[feature for feature in ADVANCED_FEATURE_COLUMNS if feature != "volume_to_avg20_confirm"],
    ]
    return {
        "A_baseline_70": _dedupe(baseline),
        "B_pruned_advanced_77": _dedupe(pruned),
        "C_full_advanced": _dedupe(full_advanced),
    }


def _common_required_columns(variants: dict[str, list[str]]) -> list[str]:
    """Return all columns required to force identical stock/date observations."""

    return _dedupe([feature for columns in variants.values() for feature in columns])


def _walk_forward_predictions_common(
    variant: str,
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate expanding walk-forward predictions on the already-common sample."""

    dates = pd.to_datetime(dataset["Date"])
    current = dates.min() + pd.DateOffset(years=3)
    end = dates.max()
    frames = []
    fold = 1
    while current < end:
        test_end = current + pd.DateOffset(months=6)
        train = dataset[dates < current]
        test = dataset[(dates >= current) & (dates < test_end)]
        if not train.empty and not test.empty:
            model = _train_model(
                "xgboost",
                train,
                test,
                feature_columns,
                walk_forward=True,
            ).model
            fold_predictions = _predict_frame(model, test, feature_columns, variant)
            fold_predictions["fold"] = fold
            fold_predictions["fold_train_end"] = current
            fold_predictions["fold_test_start"] = current
            fold_predictions["fold_test_end"] = test_end
            frames.append(fold_predictions)
            fold += 1
        current = test_end
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _comparison_report(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build metric comparison for final test and walk-forward."""

    rows = []
    for variant, result in results.items():
        for scope, predictions_key, metrics_key in [
            ("final_test", "test_predictions", "test_metrics"),
            ("walk_forward", "walk_forward_predictions", "walk_forward_metrics"),
        ]:
            predictions = result[predictions_key]
            metrics = result[metrics_key]
            rows.append(
                {
                    "Variant": variant,
                    "Scope": scope,
                    "FeatureCount": len(result["columns"]),
                    "Rows": len(predictions),
                    "PredictedOutperform": metrics.get("predicted_outperform"),
                    "LogLoss": metrics.get("log_loss"),
                    "CalibrationMAE": metrics.get("calibration_mae"),
                    "PRAUCOutperform": _pr_auc(predictions),
                    "OutperformPrecision": metrics.get("precision_outperform"),
                    "OutperformRecall": metrics.get("recall_outperform"),
                    "MacroF1": metrics.get("f1_macro"),
                    "WeightedF1": metrics.get("f1_weighted"),
                    "BalancedAccuracy": metrics.get("balanced_accuracy"),
                    "SelectedAvgExcessReturn": metrics.get("avg_future_excess_return"),
                    "CAGR": metrics.get("cagr"),
                    "Sharpe": metrics.get("sharpe"),
                    "MaxDrawdown": metrics.get("max_drawdown"),
                    "RecommendationFlag": _variant_recommendation_flag(
                        variant,
                        scope,
                        metrics,
                        results["A_baseline_70"]["walk_forward_metrics"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _common_sample_report(
    common: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    variants: dict[str, list[str]],
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Report common-sample assertions and probability buckets."""

    rows = []
    expected_counts = {
        "train": len(train),
        "validation": len(validation),
        "final_test": len(test),
    }
    for variant, columns in variants.items():
        rows.extend(
            [
                {
                    "ReportType": "common_sample_assertion",
                    "Variant": variant,
                    "Scope": scope,
                    "Rows": count,
                    "FeatureCount": len(columns),
                    "CommonRowsIdentical": True,
                    "Detail": "All variants use rows non-null across the union of full advanced features.",
                }
                for scope, count in expected_counts.items()
            ]
        )
    for scope, frame in [
        ("all_common", common),
        ("train", train),
        ("validation", validation),
        ("final_test", test),
    ]:
        rows.append(
            {
                "ReportType": "common_sample_total",
                "Variant": "all",
                "Scope": scope,
                "Rows": len(frame),
                "FeatureCount": None,
                "CommonRowsIdentical": True,
                "Detail": "Common stock/date sample used for every model.",
            }
        )
    bucket_rows = []
    for variant, result in results.items():
        bucket_rows.append(
            _probability_bucket_analysis(
                result["test_predictions"],
                model=variant,
                scope="final_test",
            )
        )
        bucket_rows.append(
            _probability_bucket_analysis(
                result["walk_forward_predictions"],
                model=variant,
                scope="walk_forward",
            )
        )
    buckets = pd.concat(bucket_rows, ignore_index=True)
    bucket_report = buckets.rename(
        columns={
            "Model": "Variant",
            "Count": "Rows",
            "P_Outperform_Bucket": "Detail",
        }
    )
    bucket_report["ReportType"] = "probability_bucket"
    bucket_report["FeatureCount"] = bucket_report["Variant"].map(
        {variant: len(columns) for variant, columns in variants.items()}
    )
    bucket_report["CommonRowsIdentical"] = True
    return pd.concat([pd.DataFrame(rows), bucket_report], ignore_index=True, sort=False)


def _probability_distribution_report(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Report P_Outperform distribution by model, final test, and fold."""

    rows = []
    for variant, result in results.items():
        rows.append(_distribution_row(variant, "final_test", "all", result["test_predictions"]))
        walk_forward = result["walk_forward_predictions"]
        rows.append(_distribution_row(variant, "walk_forward", "all", walk_forward))
        if "fold" in walk_forward.columns:
            for fold, group in walk_forward.groupby("fold"):
                rows.append(_distribution_row(variant, "walk_forward", fold, group))
    return pd.DataFrame(rows)


def _importance_report(
    results: dict[str, dict[str, Any]],
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """Save feature importance for all variants using validation-only permutation checks."""

    rows = []
    for variant, result in results.items():
        permutation = _permutation_importance(result["model"], validation, result["columns"])
        for _, row in result["feature_importance"].iterrows():
            feature = str(row["Feature"])
            rows.append(
                {
                    "Variant": variant,
                    "Feature": feature,
                    "ModelImportance": float(row["Importance"]),
                    "PermutationImportanceValidation": permutation.get(feature),
                    "IsPrunedFeature": feature in PRUNED_FEATURES,
                    "IsFullAdvancedFeature": feature in ADVANCED_FEATURE_COLUMNS,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["Variant", "PermutationImportanceValidation", "ModelImportance"],
        ascending=[True, False, False],
    )


def _distribution_row(
    variant: str,
    scope: str,
    fold: Any,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize probability distribution."""

    probabilities = predictions["p_outperform"] if "p_outperform" in predictions else pd.Series(dtype=float)
    if probabilities.empty:
        return {
            "Variant": variant,
            "Scope": scope,
            "Fold": fold,
            "Rows": 0,
            "PredictedOutperform": 0,
            "MinPOutperform": None,
            "MedianPOutperform": None,
            "MeanPOutperform": None,
            "P90Outperform": None,
            "P95Outperform": None,
            "P99Outperform": None,
            "MaxPOutperform": None,
            "RowsAbove40Pct": 0,
        }
    return {
        "Variant": variant,
        "Scope": scope,
        "Fold": fold,
        "Rows": int(len(probabilities)),
        "PredictedOutperform": int((predictions["predicted_class"] == 1).sum()),
        "MinPOutperform": float(probabilities.min()),
        "MedianPOutperform": float(probabilities.median()),
        "MeanPOutperform": float(probabilities.mean()),
        "P90Outperform": float(probabilities.quantile(0.90)),
        "P95Outperform": float(probabilities.quantile(0.95)),
        "P99Outperform": float(probabilities.quantile(0.99)),
        "MaxPOutperform": float(probabilities.max()),
        "RowsAbove40Pct": int((probabilities > 0.40).sum()),
    }


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
    clean = validation.tail(5_000)
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


def _pr_auc(predictions: pd.DataFrame) -> float | None:
    """Calculate binary PR-AUC for Outperform."""

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


def _variant_recommendation_flag(
    variant: str,
    scope: str,
    metrics: dict[str, Any],
    baseline_walk_forward: dict[str, Any],
) -> str:
    """Recommend pruned only when walk-forward improves over baseline."""

    if scope != "walk_forward":
        return "final_test_not_used_for_selection"
    if variant == "A_baseline_70":
        return "baseline_reference"
    if variant != "B_pruned_advanced_77":
        return "compare_only"
    checks = [
        metrics.get("log_loss", float("inf")) < baseline_walk_forward.get("log_loss", float("inf")),
        metrics.get("calibration_mae", float("inf"))
        <= baseline_walk_forward.get("calibration_mae", float("inf")),
        metrics.get("f1_weighted", 0) >= baseline_walk_forward.get("f1_weighted", 0),
        metrics.get("sharpe", float("-inf")) >= baseline_walk_forward.get("sharpe", float("-inf")),
    ]
    return "recommend_pruned" if all(checks) else "do_not_recommend_pruned"


def _recommendation(comparison: pd.DataFrame) -> str:
    """Return final walk-forward-only recommendation string."""

    match = comparison[
        (comparison["Variant"] == "B_pruned_advanced_77")
        & (comparison["Scope"] == "walk_forward")
    ]
    if match.empty:
        return "do_not_recommend_pruned"
    return str(match.iloc[0]["RecommendationFlag"])


def _assert_common_counts(
    variants: dict[str, list[str]],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Assert every variant has identical split row counts on the common sample."""

    expected = (len(train), len(validation), len(test))
    for variant, columns in variants.items():
        observed = (
            len(train.dropna(subset=columns + ["label"])),
            len(validation.dropna(subset=columns + ["label"])),
            len(test.dropna(subset=columns + ["label"])),
        )
        if observed != expected:
            raise AssertionError(
                f"{variant} row counts are not identical: observed={observed}, expected={expected}"
            )


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order while removing duplicate feature names."""

    return list(dict.fromkeys(items))


if __name__ == "__main__":
    main()
