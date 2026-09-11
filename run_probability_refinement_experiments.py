"""Run feature pruning, ablation, normalization, and binary calibration experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.features import (
    ADVANCED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    MARKET_REGIME_FEATURE_COLUMNS,
    SECTOR_RELATIVE_FEATURE_COLUMNS,
    feature_columns_for_variant,
)
from app.probability.train import LABEL_TO_CLASS
from audit_advanced_probability_features import _build_advanced_dataset

OUTPUT_ABLATION = REPORT_DIR / "feature_ablation_report.csv"
OUTPUT_PRUNED_COMPARISON = REPORT_DIR / "pruned_model_comparison.csv"
OUTPUT_BINARY_COMPARISON = REPORT_DIR / "binary_model_comparison.csv"
OUTPUT_CALIBRATION = REPORT_DIR / "calibration_comparison.csv"
OUTPUT_PERCENTILE = REPORT_DIR / "probability_percentile_analysis.csv"
OUTPUT_RECOMMENDED = REPORT_DIR / "recommended_feature_set.json"
AUDIT_ESTIMATORS = 60

PRUNED_FEATURES = [
    "beta_252d",
    "beta_120d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "sector_rsi_14",
    "corr_nifty_20d",
    "relative_momentum_20d_change",
]

DROP_CANDIDATES = [
    "distance_prev_20d_high",
    "distance_prev_50d_high",
    "distance_prev_100d_high",
    "breakout_prev_20d_high",
    "breakout_prev_50d_high",
    "breakout_prev_100d_high",
    "bb_width_to_avg_120d",
    "momentum_20d_change",
    "momentum_20d_vs_60d",
    "relative_momentum_20d_vs_60d",
    "up_down_volume_ratio_20d",
    "price_return_x_volume_ratio",
    "volume_trend_20d",
    "positive_price_volume_confirmation",
    "volatility_20d_to_60d",
    "volatility_20d_change",
    "sector_above_sma50",
    "sector_above_sma200",
    "sector_regime_bull",
    "sector_regime_bear",
    "sector_regime_sideways",
    "sector_volatility_120d",
    "beta_60d",
    "beta_120d",
    "beta_252d",
    "corr_nifty_60d",
    "corr_nifty_120d",
]

ABLATION_GROUPS = {
    "breakout_features": [
        "distance_prev_20d_high",
        "distance_prev_50d_high",
        "distance_prev_100d_high",
        "breakout_prev_20d_high",
        "breakout_prev_50d_high",
        "breakout_prev_100d_high",
    ],
    "volume_price_features": [
        "up_down_volume_ratio_20d",
        "price_return_x_volume_ratio",
        "volume_trend_20d",
        "positive_price_volume_confirmation",
    ],
    "volatility_features": [
        "bb_width_to_avg_120d",
        "volatility_20d_to_60d",
        "volatility_20d_change",
    ],
    "momentum_acceleration_features": [
        "momentum_20d_change",
        "momentum_20d_vs_60d",
        "relative_momentum_20d_vs_60d",
    ],
        "sector_regime_features": [
        "sector_above_sma50",
        "sector_above_sma200",
        "sector_regime_bull",
        "sector_regime_bear",
        "sector_regime_sideways",
        "sector_volatility_120d",
    ],
    "beta_correlation_features": ["beta_60d", "corr_nifty_60d", "corr_nifty_120d"],
    "long_window_beta_features": ["beta_120d", "beta_252d"],
}

RAW_PRICE_SCALE_FEATURES = [
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_20",
    "ema_50",
    "atr_14",
    "macd",
    "macd_signal",
]

NORMALIZED_FEATURES = [
    "sma20_to_sma50",
    "sma50_to_sma200",
    "atr14_to_close",
    "macd_to_close",
    "macd_signal_to_close",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Model specification for experiment loops."""

    name: str
    kind: str
    target: str


def main() -> None:
    """Run all requested refinement experiments."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(_build_advanced_dataset())
    train_validation = dataset[pd.to_datetime(dataset["Date"]) <= "2024-12-31"].copy()

    variants = _feature_variants()
    pruned_results = _evaluate_multiclass_variants(train_validation, variants)
    pruned_results.to_csv(OUTPUT_PRUNED_COMPARISON, index=False)

    ablation = _run_ablation(train_validation, variants["C_full_advanced"])
    ablation.to_csv(OUTPUT_ABLATION, index=False)

    binary, calibration, percentile = _run_binary_experiments(train_validation, variants)
    binary.to_csv(OUTPUT_BINARY_COMPARISON, index=False)
    calibration.to_csv(OUTPUT_CALIBRATION, index=False)
    percentile.to_csv(OUTPUT_PERCENTILE, index=False)

    recommended = _recommended_feature_set(variants, pruned_results, ablation)
    OUTPUT_RECOMMENDED.write_text(json.dumps(_json_safe(recommended), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "feature_ablation_report": str(OUTPUT_ABLATION),
                "pruned_model_comparison": str(OUTPUT_PRUNED_COMPARISON),
                "binary_model_comparison": str(OUTPUT_BINARY_COMPARISON),
                "calibration_comparison": str(OUTPUT_CALIBRATION),
                "probability_percentile_analysis": str(OUTPUT_PERCENTILE),
                "recommended_feature_set": str(OUTPUT_RECOMMENDED),
                "selection_period": "walk-forward folds ending before 2025-01-01; final test untouched",
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add normalized price-scale replacements and binary targets."""

    output = dataset.copy()
    output["sma20_to_sma50"] = output["sma_20"] / output["sma_50"] - 1
    output["sma50_to_sma200"] = output["sma_50"] / output["sma_200"] - 1
    output["atr14_to_close"] = output["atr_14"] / output["Close"]
    output["macd_to_close"] = output["macd"] / output["Close"]
    output["macd_signal_to_close"] = output["macd_signal"] / output["Close"]
    output["buy_target"] = (output["excess_return"] > 0.05).astype(int)
    output["sell_target"] = (output["excess_return"] < -0.05).astype(int)
    return output.replace([float("inf"), float("-inf")], pd.NA)


def _feature_variants() -> dict[str, list[str]]:
    """Return requested feature variants."""

    baseline = feature_columns_for_variant("sector_regime")
    full_advanced = [
        *FEATURE_COLUMNS,
        *SECTOR_RELATIVE_FEATURE_COLUMNS,
        *MARKET_REGIME_FEATURE_COLUMNS,
        *[feature for feature in ADVANCED_FEATURE_COLUMNS if feature != "volume_to_avg20_confirm"],
    ]
    pruned = [*baseline, *PRUNED_FEATURES]
    normalized_pruned = [
        *[feature for feature in pruned if feature not in RAW_PRICE_SCALE_FEATURES],
        *NORMALIZED_FEATURES,
    ]
    return {
        "A_baseline": _dedupe(baseline),
        "B_pruned_advanced": _dedupe(pruned),
        "C_full_advanced": _dedupe(full_advanced),
        "D_normalized_pruned": _dedupe(normalized_pruned),
    }


def _evaluate_multiclass_variants(
    dataset: pd.DataFrame,
    variants: dict[str, list[str]],
) -> pd.DataFrame:
    """Evaluate multiclass variants with median imputation and identical rows."""

    rows = []
    for variant, features in variants.items():
        print(f"[multiclass] {variant}", flush=True)
        predictions = _walk_forward_predictions(
            dataset,
            features,
            target="label",
            model_kind="xgboost",
            binary=False,
            variant=variant,
        )
        rows.append(
            {
                **_multiclass_metrics(predictions),
                "Experiment": "multiclass_feature_variant",
                "Variant": variant,
                "FeatureCount": len(features),
                "Rows": len(predictions),
                "Fold": "all",
            }
        )
        for fold, group in predictions.groupby("fold"):
            rows.append(
                {
                    **_multiclass_metrics(group),
                    "Experiment": "multiclass_feature_variant",
                    "Variant": variant,
                    "FeatureCount": len(features),
                    "Rows": len(group),
                    "Fold": fold,
                }
            )
    return pd.DataFrame(rows)


def _run_ablation(dataset: pd.DataFrame, full_features: list[str]) -> pd.DataFrame:
    """Run logical-group and individual feature ablations."""

    rows = []
    baseline_predictions = _walk_forward_predictions(
        dataset,
        full_features,
        target="label",
        model_kind="xgboost",
        binary=False,
        variant="full_advanced_reference",
    )
    baseline_by_fold = {
        fold: _multiclass_metrics(group) for fold, group in baseline_predictions.groupby("fold")
    }
    baseline_all = _multiclass_metrics(baseline_predictions)
    rows.append(
        {
            **baseline_all,
            "Ablation": "full_advanced_reference",
            "AblationType": "reference",
            "RemovedFeatures": "",
            "RemovedCount": 0,
            "FoldsImproved": 0,
            "FoldsWorsened": 0,
            "Decision": "reference",
            "Evidence": "Full advanced reference with redundant volume_to_avg20_confirm excluded.",
        }
    )
    ablations = {**ABLATION_GROUPS, **{f"remove_{feature}": [feature] for feature in DROP_CANDIDATES}}
    for name, removed in ablations.items():
        print(f"[ablation] {name}", flush=True)
        features = [feature for feature in full_features if feature not in removed]
        predictions = _walk_forward_predictions(
            dataset,
            features,
            target="label",
            model_kind="xgboost",
            binary=False,
            variant=name,
        )
        metrics = _multiclass_metrics(predictions)
        fold_metrics = {
            fold: _multiclass_metrics(group) for fold, group in predictions.groupby("fold")
        }
        folds_improved, folds_worsened = _fold_improvement_counts(fold_metrics, baseline_by_fold)
        decision = (
            "remove_candidate"
            if folds_improved > folds_worsened and folds_improved >= 2
            else "keep_insufficient_fold_evidence"
        )
        rows.append(
            {
                **metrics,
                "Ablation": name,
                "AblationType": "group" if name in ABLATION_GROUPS else "individual",
                "RemovedFeatures": ",".join(removed),
                "RemovedCount": len(removed),
                "FoldsImproved": folds_improved,
                "FoldsWorsened": folds_worsened,
                "Decision": decision,
                "Evidence": (
                    f"Improved/equal on {folds_improved} folds and worsened on "
                    f"{folds_worsened} folds versus full advanced reference."
                ),
            }
        )
    return pd.DataFrame(rows)


def _run_binary_experiments(
    dataset: pd.DataFrame,
    variants: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run BUY/SELL binary model and calibration experiments."""

    binary_rows = []
    calibration_rows = []
    percentile_rows = []
    model_specs = [
        ModelSpec("xgboost", "xgboost", "buy_target"),
        ModelSpec("random_forest", "random_forest", "buy_target"),
        ModelSpec("lightgbm", "lightgbm", "buy_target"),
        ModelSpec("xgboost", "xgboost", "sell_target"),
        ModelSpec("random_forest", "random_forest", "sell_target"),
        ModelSpec("lightgbm", "lightgbm", "sell_target"),
    ]
    for variant, features in variants.items():
        if variant == "C_full_advanced":
            feature_set = features
        elif variant == "D_normalized_pruned":
            feature_set = features
        elif variant == "B_pruned_advanced":
            feature_set = features
        else:
            feature_set = features
        for spec in model_specs:
            print(f"[binary] {variant} {spec.name} {spec.target}", flush=True)
            predictions = _walk_forward_predictions(
                dataset,
                feature_set,
                target=spec.target,
                model_kind=spec.kind,
                binary=True,
                variant=f"{variant}_{spec.name}_{spec.target}",
            )
            for calibration in ("uncalibrated", "platt", "isotonic"):
                calibrated = predictions.copy()
                probability_column = f"p_{calibration}"
                if calibration == "uncalibrated":
                    calibrated["probability"] = calibrated["p_positive"]
                else:
                    calibrated["probability"] = calibrated[probability_column]
                metric = _binary_metrics(calibrated, spec.target)
                binary_rows.append(
                    {
                        **metric,
                        "Variant": variant,
                        "Model": spec.name,
                        "Target": spec.target,
                        "Calibration": calibration,
                        "FeatureCount": len(feature_set),
                        "Rows": len(calibrated),
                    }
                )
                calibration_rows.append(
                    {
                        "Variant": variant,
                        "Model": spec.name,
                        "Target": spec.target,
                        "Calibration": calibration,
                        "Rows": len(calibrated),
                        "LogLoss": metric["LogLoss"],
                        "Brier": metric["Brier"],
                        "CalibrationMAE": metric["CalibrationMAE"],
                        "PRAUC": metric["PRAUC"],
                    }
                )
                percentile_rows.extend(
                    _percentile_rows(
                        calibrated,
                        variant=variant,
                        model=spec.name,
                        target=spec.target,
                        calibration=calibration,
                    )
                )
    return pd.DataFrame(binary_rows), pd.DataFrame(calibration_rows), pd.DataFrame(percentile_rows)


def _walk_forward_predictions(
    dataset: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    model_kind: str,
    binary: bool,
    variant: str,
) -> pd.DataFrame:
    """Create expanding walk-forward predictions using pre-2025 data only."""

    dates = pd.to_datetime(dataset["Date"])
    current = dates.min() + pd.DateOffset(years=3)
    end = min(pd.Timestamp("2025-01-01"), dates.max())
    frames = []
    fold = 1
    while current < end:
        test_end = min(current + pd.DateOffset(months=6), end)
        train = dataset[dates < current].dropna(subset=[target])
        test = dataset[(dates >= current) & (dates < test_end)].dropna(subset=[target])
        if not train.empty and not test.empty:
            if binary:
                frame = _binary_fold_predictions(
                    train,
                    test,
                    features,
                    target=target,
                    model_kind=model_kind,
                    variant=variant,
                    fold=fold,
                )
            else:
                model = _fit_model(train, features, target=target, model_kind=model_kind, binary=False)
                frame = _multiclass_prediction_frame(model, test, features, variant, fold)
            frames.append(frame)
            fold += 1
        current = test_end
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _binary_fold_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    model_kind: str,
    variant: str,
    fold: int,
) -> pd.DataFrame:
    """Train/calibrate one binary fold using only in-fold past data."""

    dates = pd.to_datetime(train["Date"])
    calibration_start = dates.quantile(0.8)
    fit_frame = train[dates < calibration_start]
    calibration_frame = train[dates >= calibration_start]
    if fit_frame.empty or calibration_frame.empty:
        fit_frame = train
        calibration_frame = train
    model = _fit_model(fit_frame, features, target=target, model_kind=model_kind, binary=True)
    calibration_probability = _positive_probability(model, calibration_frame, features)
    test_probability = _positive_probability(model, test, features)
    platt = _fit_platt(calibration_probability, calibration_frame[target])
    isotonic = _fit_isotonic(calibration_probability, calibration_frame[target])
    output = _base_prediction_columns(test).copy()
    output["Variant"] = variant
    output["fold"] = fold
    output["target"] = target
    output["actual"] = test[target].astype(int).to_numpy()
    output["p_positive"] = test_probability
    output["p_platt"] = platt.predict_proba(test_probability.reshape(-1, 1))[:, 1]
    output["p_isotonic"] = isotonic.predict(test_probability)
    output["predicted"] = (output["p_positive"] >= 0.5).astype(int)
    return output


def _fit_model(
    train: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    model_kind: str,
    binary: bool,
):
    """Fit a median-imputed model with missing indicators."""

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    model = _estimator(model_kind, binary=binary)
    y = train[target].map(LABEL_TO_CLASS) if not binary else train[target].astype(int)
    pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    pipeline.fit(train[features], y)
    return pipeline


def _estimator(model_kind: str, *, binary: bool):
    """Create estimator without changing tuned hyperparameters intentionally."""

    if model_kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=AUDIT_ESTIMATORS,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic" if binary else "multi:softprob",
            eval_metric="logloss" if binary else "mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )
    if model_kind == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=80,
            max_depth=9,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    if model_kind == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=AUDIT_ESTIMATORS,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary" if binary else "multiclass",
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unknown model kind: {model_kind}")


def _multiclass_prediction_frame(model, test: pd.DataFrame, features: list[str], variant: str, fold: int) -> pd.DataFrame:
    """Build multiclass prediction rows."""

    probabilities = model.predict_proba(test[features])
    output = _base_prediction_columns(test)
    output["Model"] = variant
    output["fold"] = fold
    output["label"] = test["label"].to_numpy()
    output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
    output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
    output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
    output["predicted_class"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map(
        {0: -1, 1: 0, 2: 1}
    )
    return output


def _base_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return common prediction identity/result columns."""

    return frame[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
        ]
    ].copy()


def _positive_probability(model, frame: pd.DataFrame, features: list[str]):
    """Return positive class probability even if a fold has a missing class."""

    probabilities = model.predict_proba(frame[features])
    estimator = model.steps[-1][1]
    classes = list(getattr(estimator, "classes_", []))
    if 1 not in classes:
        return pd.Series(0.0, index=frame.index).to_numpy()
    return probabilities[:, classes.index(1)]


def _multiclass_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate multiclass predictions."""

    if predictions.empty:
        return _empty_metrics()
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        log_loss,
    )

    y_true = predictions["label"].map(LABEL_TO_CLASS)
    y_pred = predictions["predicted_class"].map(LABEL_TO_CLASS)
    probs = predictions[["p_underperform", "p_neutral", "p_outperform"]].to_numpy()
    report = classification_report(
        y_true,
        y_pred,
        labels=[LABEL_TO_CLASS[-1], LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]],
        output_dict=True,
        zero_division=0,
    )
    actual_outperform = (predictions["label"] == 1).astype(int)
    calibration_true, calibration_pred = calibration_curve(
        actual_outperform,
        predictions["p_outperform"],
        n_bins=10,
        strategy="quantile",
    )
    selected = predictions[predictions["predicted_class"] == 1]
    backtest = backtest_top_n_portfolio(predictions, top_n=5)
    return {
        "LogLoss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
        "Brier": float(((predictions["p_outperform"] - actual_outperform) ** 2).mean()),
        "CalibrationMAE": float(abs(calibration_true - calibration_pred).mean()),
        "PRAUC": float(average_precision_score(actual_outperform, predictions["p_outperform"])),
        "OutperformPrecision": report[str(LABEL_TO_CLASS[1])]["precision"],
        "OutperformRecall": report[str(LABEL_TO_CLASS[1])]["recall"],
        "MacroF1": report["macro avg"]["f1-score"],
        "WeightedF1": report["weighted avg"]["f1-score"],
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "AvgSelectedExcessReturn": (
            float(selected["excess_return"].mean()) if not selected.empty else None
        ),
        "CAGR": backtest.get("cagr"),
        "Sharpe": backtest.get("sharpe"),
        "MaxDD": backtest.get("max_drawdown"),
        "PredictedOutperform": int((predictions["predicted_class"] == 1).sum()),
    }


def _binary_metrics(predictions: pd.DataFrame, target: str) -> dict[str, Any]:
    """Evaluate binary BUY/SELL probabilities."""

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score

    y_true = predictions["actual"].astype(int)
    probability = predictions["probability"].clip(0, 1)
    calibration_true, calibration_pred = calibration_curve(
        y_true, probability, n_bins=10, strategy="quantile"
    )
    top10 = _top_fraction(predictions, 0.10)
    top5 = _top_fraction(predictions, 0.05)
    return {
        "LogLoss": float(log_loss(y_true, probability, labels=[0, 1])),
        "Brier": float(brier_score_loss(y_true, probability)),
        "CalibrationMAE": float(abs(calibration_true - calibration_pred).mean()),
        "PRAUC": float(average_precision_score(y_true, probability)),
        "PrecisionAtTop10Pct": float(top10["actual"].mean()) if not top10.empty else None,
        "PrecisionAtTop5Pct": float(top5["actual"].mean()) if not top5.empty else None,
        "ClassifierPrecisionAt50Pct": float(
            precision_score(y_true, (probability >= 0.5).astype(int), zero_division=0)
        ),
        "AvgExcessReturnTop10Pct": _avg_excess_for_target(top10, target),
        "AvgExcessReturnTop5Pct": _avg_excess_for_target(top5, target),
    }


def _avg_excess_for_target(frame: pd.DataFrame, target: str) -> float | None:
    """Return directional excess-return quality for top binary selections."""

    if frame.empty:
        return None
    value = frame["excess_return"].mean()
    return float(-value if target == "sell_target" else value)


def _top_fraction(predictions: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return top probability rows."""

    count = max(1, int(len(predictions) * fraction))
    return predictions.sort_values("probability", ascending=False).head(count)


def _percentile_rows(
    predictions: pd.DataFrame,
    *,
    variant: str,
    model: str,
    target: str,
    calibration: str,
) -> list[dict[str, Any]]:
    """Analyze probability percentiles by fold and decile."""

    rows = []
    frame = predictions.copy()
    frame["probability_percentile"] = frame.groupby("fold")["probability"].rank(pct=True)
    frame["percentile_bucket"] = pd.cut(
        frame["probability_percentile"],
        bins=[0, 0.5, 0.75, 0.9, 0.95, 1.0],
        labels=["0-50", "50-75", "75-90", "90-95", "95-100"],
        include_lowest=True,
    )
    for (fold, bucket), group in frame.groupby(["fold", "percentile_bucket"], observed=False):
        rows.append(
            {
                "Variant": variant,
                "Model": model,
                "Target": target,
                "Calibration": calibration,
                "Fold": fold,
                "PercentileBucket": bucket,
                "Rows": int(len(group)),
                "AvgProbability": float(group["probability"].mean()) if len(group) else None,
                "ActualPositiveRate": float(group["actual"].mean()) if len(group) else None,
                "AvgDirectionalExcessReturn": _avg_excess_for_target(group, target),
            }
        )
    return rows


def _fit_platt(probabilities, actual):
    """Fit Platt/sigmoid calibration from validation probabilities."""

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(probabilities.reshape(-1, 1), actual.astype(int))
    return model


def _fit_isotonic(probabilities, actual):
    """Fit isotonic calibration from validation probabilities."""

    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probabilities, actual.astype(int))
    return model


def _fold_improvement_counts(
    candidate: dict[Any, dict[str, Any]],
    baseline: dict[Any, dict[str, Any]],
) -> tuple[int, int]:
    """Count folds where candidate removal is equal/better or worse."""

    improved = 0
    worsened = 0
    for fold, candidate_metric in candidate.items():
        baseline_metric = baseline.get(fold)
        if baseline_metric is None:
            continue
        is_better = (
            candidate_metric["LogLoss"] <= baseline_metric["LogLoss"]
            and candidate_metric["Brier"] <= baseline_metric["Brier"]
            and candidate_metric["CalibrationMAE"] <= baseline_metric["CalibrationMAE"]
        )
        if is_better:
            improved += 1
        else:
            worsened += 1
    return improved, worsened


def _recommended_feature_set(
    variants: dict[str, list[str]],
    comparison: pd.DataFrame,
    ablation: pd.DataFrame,
) -> dict[str, Any]:
    """Build recommended feature-set artifact from validation evidence only."""

    baseline = comparison[
        (comparison["Variant"] == "A_baseline") & (comparison["Fold"].astype(str) == "all")
    ].iloc[0]
    pruned = comparison[
        (comparison["Variant"] == "B_pruned_advanced") & (comparison["Fold"].astype(str) == "all")
    ].iloc[0]
    recommend_pruned = (
        pruned["LogLoss"] <= baseline["LogLoss"]
        and pruned["Brier"] <= baseline["Brier"]
        and pruned["CalibrationMAE"] <= baseline["CalibrationMAE"]
        and pruned["Sharpe"] >= baseline["Sharpe"]
    )
    removed = []
    for _, row in ablation.iterrows():
        if row["Decision"] == "remove_candidate":
            removed.append(
                {
                    "feature_or_group": row["Ablation"],
                    "features": str(row["RemovedFeatures"]).split(",") if row["RemovedFeatures"] else [],
                    "reason": "Removal was equal/better across multiple walk-forward folds.",
                    "validation_evidence": {
                        "folds_improved": int(row["FoldsImproved"]),
                        "folds_worsened": int(row["FoldsWorsened"]),
                        "log_loss": row["LogLoss"],
                        "brier": row["Brier"],
                        "calibration_mae": row["CalibrationMAE"],
                    },
                }
            )
    kept = variants["B_pruned_advanced"] if recommend_pruned else variants["A_baseline"]
    return {
        "selection_rule": "Final test untouched; recommendation based on walk-forward pre-2025 only.",
        "recommended_variant": "B_pruned_advanced" if recommend_pruned else "A_baseline",
        "features_kept": kept,
        "features_removed": removed,
        "pruned_recommendation_evidence": {
            "baseline": baseline.to_dict(),
            "pruned": pruned.to_dict(),
            "decision": "recommend_pruned" if recommend_pruned else "do_not_recommend_pruned",
        },
        "notes": [
            "volume_to_avg20_confirm removed because it duplicates volume_to_avg20.",
            "Median imputation and missing indicators are fitted inside training folds.",
            "Binary BUY/SELL calibration uses in-fold past calibration data only.",
        ],
    }


def _empty_metrics() -> dict[str, Any]:
    """Return empty metric placeholders."""

    return {
        "LogLoss": None,
        "Brier": None,
        "CalibrationMAE": None,
        "PRAUC": None,
        "OutperformPrecision": None,
        "OutperformRecall": None,
        "MacroF1": None,
        "WeightedF1": None,
        "BalancedAccuracy": None,
        "AvgSelectedExcessReturn": None,
        "CAGR": None,
        "Sharpe": None,
        "MaxDD": None,
        "PredictedOutperform": 0,
    }


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    return list(dict.fromkeys(items))


def _json_safe(value):
    """Convert pandas/numpy scalars to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
