"""Validation-only V3 model-improvement research for NATIP.

This script does not modify V1 production artifacts, the frozen V2 BUY
candidate, or any final-test-selected thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from rebuild_clean_probability_dataset import CLEAN_CACHE_DIR, _safe_symbol
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_platt,
    _positive_probability,
    _transform_with_medians,
)
from run_ranker_binary_signal_research import _fit_classifier as _fit_default_classifier
from run_stock_specific_feature_variant import (
    _common_evaluation_rows,
    _frozen_features,
    _ic_summary,
    _prepare_dataset,
    _safe_float,
)
from run_v2_backward_forward_selection import BINARY_TOP, CLASSIFIER_TOP, CLEAN_DEMERGER_DATASET
from run_v2_binary_stability_diagnosis import (
    CURRENT_BINARY_ID,
    TUNED_BINARY_ID,
    _binary_oof_with_audit,
    _fold_frames,
    _primary_baseline_oof,
)
from run_v2_model_family_comparison import _selected_26_features
from run_v2_xgb_hyperparameter_optimization import TrialSpec, _binary_specs

OUTPUT_REGRESSION = REPORT_DIR / "v3_excess_regression_results.csv"
OUTPUT_BUY_REGRESSION = REPORT_DIR / "v3_buy_regression_stratification.csv"
OUTPUT_META = REPORT_DIR / "v3_oof_meta_model_results.csv"
OUTPUT_META_CONFIDENCE = REPORT_DIR / "v3_meta_buy_confidence.csv"
OUTPUT_FEATURE_AUDIT = REPORT_DIR / "v3_new_feature_audit.csv"
OUTPUT_FEATURE_ABLATION = REPORT_DIR / "v3_new_feature_ablation.csv"
OUTPUT_REPORT = REPORT_DIR / "v3_model_improvement_report.txt"

RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 2000
TAIL_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)
NEW_FEATURES = [
    "traded_value",
    "median_traded_value_20d",
    "median_traded_value_60d",
    "traded_value_to_avg20",
    "intraday_range_pct",
    "close_location_value",
    "gap_pct",
    "abnormal_volume",
    "volume_volatility_20d",
    "residual_return_20d",
    "residual_return_60d",
    "residual_volatility_60d",
]


@dataclass(frozen=True, slots=True)
class FeatureVariant:
    """Feature-ablation variant."""

    name: str
    description: str
    features: list[str]
    added_features: list[str]


def main() -> None:
    """Run V3 validation-only model-improvement research."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into V3 research.")

    enriched = _add_v3_candidate_features(pre_final)
    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(enriched, features_77 + NEW_FEATURES)
    folds = _embargo_folds(common)

    print("[phase 0] first-stage OOF scores", flush=True)
    primary = _primary_full_oof(common, folds, features_26)
    specs = {spec.trial_id: spec for spec in _binary_specs(common)}
    current_binary, _ = _binary_oof_with_audit(common, folds, features_77, specs[CURRENT_BINARY_ID])
    tuned_binary, _ = _binary_oof_with_audit(common, folds, features_77, specs[TUNED_BINARY_ID])
    first_stage = _first_stage_frame(primary, current_binary, tuned_binary)

    print("[phase 1] excess-return regression", flush=True)
    regression = _regression_oof(common, folds, features_26)
    regression_results = _regression_results(regression)
    regression_results.to_csv(OUTPUT_REGRESSION, index=False)
    buy_regression = _buy_regression_stratification(first_stage, regression)
    buy_regression.to_csv(OUTPUT_BUY_REGRESSION, index=False)

    print("[phase 2] meta-models", flush=True)
    meta_input = _meta_input(first_stage, regression)
    meta_predictions = _meta_oof(meta_input, folds)
    meta_results = _meta_results(meta_predictions, first_stage)
    meta_results.to_csv(OUTPUT_META, index=False)
    meta_confidence = _meta_buy_confidence(meta_predictions, first_stage)
    meta_confidence.to_csv(OUTPUT_META_CONFIDENCE, index=False)

    print("[phase 5] new feature audit and grouped ablation", flush=True)
    feature_audit = _new_feature_audit(enriched)
    feature_audit.to_csv(OUTPUT_FEATURE_AUDIT, index=False)
    if OUTPUT_FEATURE_ABLATION.exists():
        ablation = pd.read_csv(OUTPUT_FEATURE_ABLATION)
    else:
        ablation = _new_feature_ablation(common, folds, features_26, current_binary)
    ablation.to_csv(OUTPUT_FEATURE_ABLATION, index=False)

    decision = _decision(
        regression_results=regression_results,
        buy_regression=buy_regression,
        meta_results=meta_results,
        meta_confidence=meta_confidence,
        feature_ablation=ablation,
    )
    OUTPUT_REPORT.write_text(
        _report(
            regression_results=regression_results,
            buy_regression=buy_regression,
            meta_results=meta_results,
            meta_confidence=meta_confidence,
            feature_audit=feature_audit,
            feature_ablation=ablation,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v3_excess_regression_results": str(OUTPUT_REGRESSION),
                "v3_buy_regression_stratification": str(OUTPUT_BUY_REGRESSION),
                "v3_oof_meta_model_results": str(OUTPUT_META),
                "v3_meta_buy_confidence": str(OUTPUT_META_CONFIDENCE),
                "v3_new_feature_audit": str(OUTPUT_FEATURE_AUDIT),
                "v3_new_feature_ablation": str(OUTPUT_FEATURE_ABLATION),
                "v3_model_improvement_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "frozen_v2_buy_candidate_modified": False,
                "buy_rules_modified": False,
                "decision": decision,
            },
            indent=2,
        )
    )


def _add_v3_candidate_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add stock-specific candidate features using backward-looking transforms only."""

    frame = dataset.sort_values(["symbol", "Date"]).copy()
    frame = _merge_clean_ohlcv(frame)
    group = frame.groupby("symbol", group_keys=False)
    previous_close = group["Close"].shift(1)
    frame["traded_value"] = frame["Close"] * frame["Volume"]
    frame["median_traded_value_20d"] = group["traded_value"].transform(
        lambda item: item.rolling(20, min_periods=10).median()
    )
    frame["median_traded_value_60d"] = group["traded_value"].transform(
        lambda item: item.rolling(60, min_periods=30).median()
    )
    avg20 = group["traded_value"].transform(lambda item: item.rolling(20, min_periods=10).mean())
    frame["traded_value_to_avg20"] = frame["traded_value"] / avg20.replace(0, np.nan)
    frame["intraday_range_pct"] = (frame["High"] - frame["Low"]) / frame["Close"].replace(
        0, np.nan
    )
    price_range = (frame["High"] - frame["Low"]).replace(0, np.nan)
    frame["close_location_value"] = ((frame["Close"] - frame["Low"]) / price_range).clip(0, 1)
    frame["gap_pct"] = (frame["Open"] / previous_close.replace(0, np.nan)) - 1
    volume_avg20 = group["Volume"].transform(lambda item: item.rolling(20, min_periods=10).mean())
    frame["abnormal_volume"] = frame["Volume"] / volume_avg20.replace(0, np.nan)
    frame["volume_volatility_20d"] = group["Volume"].transform(
        lambda item: item.pct_change(fill_method=None).rolling(20, min_periods=10).std()
    )
    for horizon in (20, 60):
        market_col = f"nifty_ret_{horizon}d"
        sector_col = f"sector_momentum_{horizon}d"
        stock_col = f"ret_{horizon}d"
        if {market_col, sector_col, stock_col}.issubset(frame.columns):
            frame[f"residual_return_{horizon}d"] = (
                frame[stock_col] - 0.50 * frame[market_col] - 0.50 * frame[sector_col]
            )
        else:
            frame[f"residual_return_{horizon}d"] = np.nan
    if {"volatility_60d", "nifty_volatility_60d"}.issubset(frame.columns):
        sector_vol = frame.get("sector_volatility_120d", pd.Series(np.nan, index=frame.index))
        frame["residual_volatility_60d"] = (
            frame["volatility_60d"] - 0.50 * frame["nifty_volatility_60d"] - 0.50 * sector_vol
        )
    else:
        frame["residual_volatility_60d"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


def _merge_clean_ohlcv(dataset: pd.DataFrame) -> pd.DataFrame:
    """Merge adjusted OHLCV from clean cache for stock-specific V3 features."""

    if {"Open", "High", "Low", "Volume"}.issubset(dataset.columns):
        return dataset
    frames = []
    for symbol in sorted(dataset["symbol"].dropna().unique()):
        path = CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["Date"])
        if "is_tradable_row" in raw.columns:
            raw = raw[raw["is_tradable_row"].astype(bool)].copy()
        columns = {
            "adj_Open": "Open",
            "adj_High": "High",
            "adj_Low": "Low",
            "adj_Close": "cache_Close",
            "Volume": "Volume",
        }
        available = ["Date", *[column for column in columns if column in raw.columns]]
        item = raw[available].rename(columns=columns)
        item["symbol"] = symbol
        frames.append(item)
    if not frames:
        for column in ("Open", "High", "Low", "Volume"):
            dataset[column] = np.nan
        return dataset
    ohlcv = pd.concat(frames, ignore_index=True)
    merged = dataset.merge(ohlcv, on=["Date", "symbol"], how="left")
    for column in ("Open", "High", "Low"):
        if column not in merged.columns:
            merged[column] = merged["Close"]
        merged[column] = merged[column].fillna(merged["Close"])
    if "Volume" not in merged.columns:
        merged["Volume"] = np.nan
    return merged


def _first_stage_frame(
    primary: pd.DataFrame, current_binary: pd.DataFrame, tuned_binary: pd.DataFrame
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    frame = primary.rename(
        columns={
            "p_outperform": "xgb26_p_outperform",
            "p_neutral": "xgb26_p_neutral",
            "p_underperform": "xgb26_p_underperform",
        }
    )[
        [
            "Date",
            "symbol",
            "fold",
            "Sector",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
            "xgb26_p_outperform",
            "xgb26_p_neutral",
            "xgb26_p_underperform",
            "classifier_percentile",
        ]
    ].copy()
    current = current_binary[key + ["sigmoid_probability", "sigmoid_percentile"]].rename(
        columns={
            "sigmoid_probability": "current_binary_probability",
            "sigmoid_percentile": "current_binary_percentile",
        }
    )
    tuned = tuned_binary[key + ["sigmoid_probability", "sigmoid_percentile"]].rename(
        columns={
            "sigmoid_probability": "tuned_binary_probability",
            "sigmoid_percentile": "tuned_binary_percentile",
        }
    )
    frame = frame.merge(current, on=key, how="inner").merge(tuned, on=key, how="inner")
    frame["probability_margin"] = frame["xgb26_p_outperform"] - frame["xgb26_p_underperform"]
    frame["current_buy"] = (
        (frame["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (frame["current_binary_percentile"] >= 1 - BINARY_TOP)
    )
    frame["high_confidence_buy"] = frame["current_buy"] & (
        frame["tuned_binary_percentile"] >= 1 - BINARY_TOP
    )
    return frame


def _primary_full_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]
) -> pd.DataFrame:
    """Generate full 3-class XGB OOF probabilities for V3 meta-model inputs."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_default_classifier(
            train, features, target="label", model_name="xgboost", binary=False
        )
        probabilities = model.predict_proba(_transform_with_medians(validation, features, medians))
        output = validation[
            [
                "Date",
                "symbol",
                "Sector",
                "future_stock_return",
                "future_nifty_return",
                "excess_return",
                "buy_target",
                "market_regime_label",
            ]
        ].copy()
        output["fold"] = fold["fold"]
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _regression_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_xgb_regressor(train, features)
        prediction = model.predict(_transform_with_medians(validation, features, medians))
        output = validation[
            [
                "Date",
                "symbol",
                "Sector",
                "future_stock_return",
                "future_nifty_return",
                "excess_return",
                "buy_target",
                "market_regime_label",
            ]
        ].copy()
        output["fold"] = fold["fold"]
        output["predicted_20d_excess_return"] = prediction
        output["predicted_excess_percentile"] = output.groupby("Date")[
            "predicted_20d_excess_return"
        ].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fit_xgb_regressor(train: pd.DataFrame, features: list[str]):
    from xgboost import XGBRegressor

    medians = train[features].median(numeric_only=True).fillna(0)
    x_train = _transform_with_medians(train, features, medians)
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1.0,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(x_train, train["excess_return"].astype(float))
    return model, medians


def _regression_results(regression: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = regression["excess_return"].astype(float)
    pred = regression["predicted_20d_excess_return"].astype(float)
    rows.append(
        {
            "Section": "overall",
            "Bucket": "all",
            "Count": int(len(regression)),
            "PearsonIC": _safe_float(pred.corr(y, method="pearson")),
            "SpearmanIC": _safe_float(pred.corr(y, method="spearman")),
            "MAE": _safe_float((pred - y).abs().mean()),
            "RMSE": _safe_float(np.sqrt(((pred - y) ** 2).mean())),
        }
        | _ic_summary(regression, "predicted_20d_excess_return")
    )
    for label, bucket in _decile_groups(regression, "predicted_20d_excess_return"):
        rows.append(_performance_row(bucket, "predicted_return_decile", label))
    for fraction in TAIL_FRACTIONS:
        rows.append(
            _performance_row(
                _top_same_date_fraction(regression, "predicted_20d_excess_return", fraction),
                "top_predicted_return",
                _tail_label(fraction),
            )
        )
    return pd.DataFrame(rows)


def _buy_regression_stratification(
    first_stage: pd.DataFrame, regression: pd.DataFrame
) -> pd.DataFrame:
    merged = first_stage.merge(
        regression[["Date", "symbol", "fold", "predicted_20d_excess_return"]],
        on=["Date", "symbol", "fold"],
        how="inner",
    )
    current_buy = merged[merged["current_buy"]].copy()
    rows = [_performance_row(current_buy, "current_buy", "all")]
    for label, bucket in _quantile_groups(
        current_buy, "predicted_20d_excess_return", labels=("low", "medium", "high")
    ):
        rows.append(_performance_row(bucket, "current_buy_predicted_excess_tercile", label))
    for fraction in TAIL_FRACTIONS:
        top = _top_same_date_fraction(current_buy, "predicted_20d_excess_return", fraction)
        rows.append(
            _performance_row(top, "current_buy_top_predicted_excess", _tail_label(fraction))
        )
    for label, bucket in _quantile_groups(
        current_buy, "predicted_20d_excess_return", labels=("low", "medium", "high")
    ):
        rows.append(
            _bootstrap_difference_row(
                current_buy,
                bucket,
                "current_buy_predicted_excess_tercile_bootstrap",
                f"{label}_minus_all",
            )
        )
    return _add_stability(pd.DataFrame(rows), current_buy)


def _meta_input(first_stage: pd.DataFrame, regression: pd.DataFrame) -> pd.DataFrame:
    return first_stage.merge(
        regression[["Date", "symbol", "fold", "predicted_20d_excess_return"]],
        on=["Date", "symbol", "fold"],
        how="inner",
    )


def _meta_oof(meta_input: pd.DataFrame, folds: list[dict[str, Any]]) -> pd.DataFrame:
    meta_features = [
        "xgb26_p_outperform",
        "xgb26_p_neutral",
        "xgb26_p_underperform",
        "classifier_percentile",
        "probability_margin",
        "current_binary_probability",
        "current_binary_percentile",
        "tuned_binary_probability",
        "tuned_binary_percentile",
        "predicted_20d_excess_return",
    ]
    frames = []
    for fold in folds:
        validation = meta_input[meta_input["Date"].isin(fold["validation_dates"])].copy()
        train = meta_input[meta_input["Date"] < fold["validation_start"]].copy()
        train = train[~train["Date"].isin(fold.get("embargo_dates", []))].copy()
        if validation.empty or train["Date"].nunique() < 40:
            continue
        fit, calibration = _fit_calibration_split(train)
        logistic, medians = _fit_logistic_meta(fit, meta_features)
        raw_probability = _positive_probability(
            logistic, _transform_with_medians(validation, meta_features, medians)
        )
        calibration_probability = _positive_probability(
            logistic, _transform_with_medians(calibration, meta_features, medians)
        )
        sigmoid = _fit_platt(calibration_probability, calibration["buy_target"])
        output = validation.copy()
        output["MetaModel"] = "logistic"
        output["meta_raw_probability"] = raw_probability
        output["meta_probability"] = sigmoid.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        output["meta_percentile"] = output.groupby("Date")["meta_probability"].rank(pct=True)
        frames.append(output)
        xgb_meta, xgb_medians = _fit_shallow_xgb_meta(fit, calibration, meta_features)
        xgb_raw = _positive_probability(
            xgb_meta, _transform_with_medians(validation, meta_features, xgb_medians)
        )
        xgb_cal = _positive_probability(
            xgb_meta, _transform_with_medians(calibration, meta_features, xgb_medians)
        )
        xgb_sigmoid = _fit_platt(xgb_cal, calibration["buy_target"])
        xgb_output = validation.copy()
        xgb_output["MetaModel"] = "shallow_xgboost"
        xgb_output["meta_raw_probability"] = xgb_raw
        xgb_output["meta_probability"] = xgb_sigmoid.predict_proba(xgb_raw.reshape(-1, 1))[:, 1]
        xgb_output["meta_percentile"] = xgb_output.groupby("Date")["meta_probability"].rank(
            pct=True
        )
        frames.append(xgb_output)
    return pd.concat(frames, ignore_index=True)


def _fit_logistic_meta(train: pd.DataFrame, features: list[str]):
    from sklearn.linear_model import LogisticRegression

    medians = train[features].median(numeric_only=True).fillna(0)
    x_train = _transform_with_medians(train, features, medians)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(x_train, train["buy_target"].astype(int))
    return model, medians


def _fit_shallow_xgb_meta(fit: pd.DataFrame, calibration: pd.DataFrame, features: list[str]):
    spec = TrialSpec(
        "v3_shallow_meta",
        "meta",
        {
            "n_estimators": 120,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_child_weight": 10.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 1.0,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
    )
    from run_v2_xgb_hyperparameter_optimization import _fit_xgb

    return _fit_xgb(fit, calibration, features, target="buy_target", spec=spec, binary=True)


def _meta_results(meta_predictions: pd.DataFrame, first_stage: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    rows = []
    current_buy = first_stage[first_stage["current_buy"]].copy()
    high_confidence = first_stage[first_stage["high_confidence_buy"]].copy()
    rows.append(_performance_row(current_buy, "benchmark", "current_buy"))
    rows.append(_performance_row(high_confidence, "benchmark", "current_plus_tuned_high"))
    for model_name, frame in meta_predictions.groupby("MetaModel"):
        y = frame["buy_target"].astype(int)
        p = frame["meta_probability"].clip(1e-6, 1 - 1e-6)
        overall = {
            "Section": "meta_overall",
            "Bucket": model_name,
            "Count": int(len(frame)),
            "PRAUC": _safe_float(average_precision_score(y, p)),
            "LogLoss": _safe_float(log_loss(y, p, labels=[0, 1])),
            "Brier": _safe_float(brier_score_loss(y, p)),
        }
        rows.append(overall)
        for fraction in TAIL_FRACTIONS:
            top = _top_same_date_fraction(frame, "meta_probability", fraction)
            rows.append(
                _performance_row(top, f"meta_tail_{model_name}", _tail_label(fraction))
                | {
                    "PRAUC": overall["PRAUC"],
                    "LogLoss": overall["LogLoss"],
                    "Brier": overall["Brier"],
                }
            )
    return pd.DataFrame(rows)


def _meta_buy_confidence(
    meta_predictions: pd.DataFrame, first_stage: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    buy_keys = first_stage[first_stage["current_buy"]][["Date", "symbol", "fold"]]
    for model_name, frame in meta_predictions.groupby("MetaModel"):
        buy_frame = frame.merge(buy_keys, on=["Date", "symbol", "fold"], how="inner")
        rows.append(_performance_row(buy_frame, f"{model_name}_existing_buy", "all"))
        for label, bucket in _quantile_groups(
            buy_frame, "meta_probability", labels=("low_meta", "medium_meta", "high_meta")
        ):
            rows.append(_performance_row(bucket, f"{model_name}_existing_buy_meta_tercile", label))
            rows.append(
                _bootstrap_difference_row(
                    buy_frame,
                    bucket,
                    f"{model_name}_existing_buy_meta_tercile_bootstrap",
                    f"{label}_minus_all",
                )
            )
        rows.extend(_fold_year_rows(buy_frame, f"{model_name}_existing_buy"))
    return pd.DataFrame(rows)


def _new_feature_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in NEW_FEATURES:
        series = dataset[feature]
        rows.append(
            {
                "Feature": feature,
                "MissingPct": _safe_float(series.isna().mean()),
                "InfiniteCount": int(np.isinf(series.replace([np.inf, -np.inf], np.nan)).sum()),
                "Mean": _safe_float(series.mean()),
                "Median": _safe_float(series.median()),
                "Std": _safe_float(series.std(ddof=0)),
                "P01": _safe_float(series.quantile(0.01)),
                "P05": _safe_float(series.quantile(0.05)),
                "P95": _safe_float(series.quantile(0.95)),
                "P99": _safe_float(series.quantile(0.99)),
                "WithinDateVarianceMean": _safe_float(
                    dataset.groupby("Date")[feature].var().mean()
                ),
                "BackwardLookingOnly": True,
            }
        )
    return pd.DataFrame(rows)


def _new_feature_ablation(
    common: pd.DataFrame,
    folds: list[dict[str, Any]],
    features_26: list[str],
    current_binary: pd.DataFrame,
) -> pd.DataFrame:
    groups = {
        "liquidity_value": [
            "traded_value",
            "median_traded_value_20d",
            "median_traded_value_60d",
            "traded_value_to_avg20",
        ],
        "price_volume_microstructure": [
            "intraday_range_pct",
            "close_location_value",
            "gap_pct",
            "abnormal_volume",
            "volume_volatility_20d",
        ],
        "residual_features": [
            "residual_return_20d",
            "residual_return_60d",
            "residual_volatility_60d",
        ],
    }
    variants = [
        FeatureVariant("xgb26_baseline", "Current 26-feature XGB", features_26, []),
        *[
            FeatureVariant(
                f"xgb26_plus_{name}",
                f"26-feature XGB + {name}",
                [*features_26, *items],
                items,
            )
            for name, items in groups.items()
        ],
        FeatureVariant(
            "xgb26_plus_all_v3",
            "26-feature XGB + all V3 candidates",
            [*features_26, *NEW_FEATURES],
            NEW_FEATURES,
        ),
    ]
    rows = []
    for variant in variants:
        print(f"[feature ablation] {variant.name}", flush=True)
        primary = _primary_baseline_oof(common, folds, variant.features)
        selected = _selected_current_buy(primary, current_binary)
        rows.append(
            _performance_row(selected, "feature_ablation", variant.name)
            | {
                "FeatureCount": len(variant.features),
                "AddedFeatureCount": len(variant.added_features),
                "AddedFeatures": ",".join(variant.added_features),
            }
            | _ic_summary(primary, "p_outperform")
        )
    return pd.DataFrame(rows)


def _selected_current_buy(primary: pd.DataFrame, current_binary: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = primary.merge(
        current_binary[key + ["sigmoid_percentile"]],
        on=key,
        how="inner",
    )
    return merged[
        (merged["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (merged["sigmoid_percentile"] >= 1 - BINARY_TOP)
    ].copy()


def _performance_row(frame: pd.DataFrame, section: str, bucket: str) -> dict[str, Any]:
    return {
        "Section": section,
        "Bucket": bucket,
        "Count": int(len(frame)),
        "Precision": _safe_float(frame["buy_target"].mean()) if not frame.empty else None,
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": (
            _safe_float((frame["excess_return"] > 0).mean()) if not frame.empty else None
        ),
    }


def _bootstrap_difference_row(
    benchmark: pd.DataFrame,
    candidate: pd.DataFrame,
    section: str,
    bucket: str,
) -> dict[str, Any]:
    """Date-block bootstrap difference: candidate minus benchmark."""

    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(benchmark["Date"]).union(set(candidate["Date"]))))
    benchmark_groups = {date: group for date, group in benchmark.groupby("Date")}
    candidate_groups = {date: group for date, group in candidate.groupby("Date")}
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        bench = _concat_date_groups(benchmark_groups, sample_dates)
        cand = _concat_date_groups(candidate_groups, sample_dates)
        bench_metrics = _performance_row(bench, "benchmark", "sample")
        cand_metrics = _performance_row(cand, "candidate", "sample")
        diffs.append(
            {
                "PrecisionDiff": _none_safe(cand_metrics["Precision"])
                - _none_safe(bench_metrics["Precision"]),
                "AverageExcessDiff": _none_safe(cand_metrics["AverageExcessReturn"])
                - _none_safe(bench_metrics["AverageExcessReturn"]),
                "MedianExcessDiff": _none_safe(cand_metrics["MedianExcessReturn"])
                - _none_safe(bench_metrics["MedianExcessReturn"]),
                "NiftyWinRateDiff": _none_safe(cand_metrics["NiftyWinRate"])
                - _none_safe(bench_metrics["NiftyWinRate"]),
            }
        )
    diff_frame = pd.DataFrame(diffs)
    row = {
        "Section": section,
        "Bucket": bucket,
        "Count": int(len(candidate)),
        "BootstrapSamples": BOOTSTRAP_SAMPLES,
    }
    for column in diff_frame.columns:
        row[f"{column}Mean"] = _safe_float(diff_frame[column].mean())
        row[f"{column}CI025"] = _safe_float(diff_frame[column].quantile(0.025))
        row[f"{column}CI975"] = _safe_float(diff_frame[column].quantile(0.975))
        row[f"{column}StatisticallyAboveZero"] = bool(diff_frame[column].quantile(0.025) > 0)
    return row


def _concat_date_groups(
    groups: dict[pd.Timestamp, pd.DataFrame], dates: np.ndarray
) -> pd.DataFrame:
    pieces = [groups[date] for date in dates if date in groups]
    if pieces:
        return pd.concat(pieces, ignore_index=True)
    return pd.DataFrame(
        columns=["Date", "buy_target", "future_stock_return", "excess_return"]
    )


def _none_safe(value: float | None) -> float:
    return 0.0 if value is None or pd.isna(value) else float(value)


def _fold_year_rows(frame: pd.DataFrame, section: str) -> list[dict[str, Any]]:
    rows = []
    for fold, group in frame.groupby("fold"):
        rows.append(_performance_row(group, f"{section}_fold", str(fold)))
    for year, group in frame.groupby(frame["Date"].dt.year):
        rows.append(_performance_row(group, f"{section}_year", str(int(year))))
    return rows


def _add_stability(summary: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    rows = summary.to_dict(orient="records")
    rows.extend(_fold_year_rows(source, "current_buy_source"))
    return pd.DataFrame(rows)


def _decile_groups(frame: pd.DataFrame, column: str) -> list[tuple[str, pd.DataFrame]]:
    valid = frame.dropna(subset=[column]).copy()
    if valid.empty:
        return []
    valid["bucket"] = pd.qcut(valid[column], q=10, labels=False, duplicates="drop")
    return [(f"decile_{int(label) + 1}", group) for label, group in valid.groupby("bucket")]


def _quantile_groups(
    frame: pd.DataFrame, column: str, *, labels: tuple[str, ...]
) -> list[tuple[str, pd.DataFrame]]:
    valid = frame.dropna(subset=[column]).copy()
    if valid.empty:
        return []
    valid["bucket"] = pd.qcut(valid[column], q=len(labels), labels=labels, duplicates="drop")
    return [(str(label), group) for label, group in valid.groupby("bucket")]


def _top_same_date_fraction(frame: pd.DataFrame, column: str, fraction: float) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("Date"):
        if group.empty:
            continue
        count = max(1, int(np.ceil(len(group) * fraction)))
        pieces.append(group.sort_values(column, ascending=False).head(count))
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def _tail_label(fraction: float) -> str:
    return f"Top{fraction * 100:g}%"


def _decision(
    *,
    regression_results: pd.DataFrame,
    buy_regression: pd.DataFrame,
    meta_results: pd.DataFrame,
    meta_confidence: pd.DataFrame,
    feature_ablation: pd.DataFrame,
) -> str:
    current = _row_lookup(meta_results, "benchmark", "current_buy")
    high = _row_lookup(meta_results, "benchmark", "current_plus_tuned_high")
    best_meta = (
        meta_results[meta_results["Section"].str.startswith("meta_tail_", na=False)]
        .sort_values(["Precision", "MedianExcessReturn", "Count"], ascending=False)
        .head(1)
    )
    regression_terciles = buy_regression[
        buy_regression["Section"].eq("current_buy_predicted_excess_tercile")
    ].copy()
    best_feature = (
        feature_ablation[~feature_ablation["Bucket"].eq("xgb26_baseline")]
        .sort_values(["Precision", "MedianExcessReturn", "Count"], ascending=False)
        .head(1)
    )
    if not best_meta.empty and current and _materially_better(best_meta.iloc[0], current):
        return "META_MODEL_ADDS_VALUE"
    if _monotonic_tercile_improvement(regression_terciles) and current:
        return "REGRESSION_ADDS_VALUE"
    if not best_feature.empty and current and _materially_better(best_feature.iloc[0], current):
        return "NEW_FEATURES_ADD_VALUE"
    if high and current and high["Precision"] and current["Precision"]:
        return "NO_MATERIAL_IMPROVEMENT"
    return "NO_MATERIAL_IMPROVEMENT"


def _monotonic_tercile_improvement(frame: pd.DataFrame) -> bool:
    """Require broad monotonic separation before calling regression useful."""

    if set(frame["Bucket"]) != {"low", "medium", "high"}:
        return False
    indexed = frame.set_index("Bucket")
    precision = indexed["Precision"]
    average = indexed["AverageExcessReturn"]
    median = indexed["MedianExcessReturn"]
    return bool(
        precision["high"] >= precision["medium"] >= precision["low"]
        and average["high"] >= average["medium"] >= average["low"]
        and median["high"] >= median["medium"] >= median["low"]
    )


def _materially_better(candidate: pd.Series, benchmark: dict[str, Any]) -> bool:
    return bool(
        candidate.get("Count", 0) >= 100
        and candidate.get("Precision", 0) > (benchmark.get("Precision") or 0) + 0.03
        and candidate.get("MedianExcessReturn", -1) >= (benchmark.get("MedianExcessReturn") or -1)
        and candidate.get("AverageExcessReturn", -1) >= (benchmark.get("AverageExcessReturn") or -1)
    )


def _row_lookup(frame: pd.DataFrame, section: str, bucket: str) -> dict[str, Any] | None:
    rows = frame[frame["Section"].eq(section) & frame["Bucket"].eq(bucket)]
    return rows.iloc[0].to_dict() if not rows.empty else None


def _report(**items: Any) -> str:
    regression = items["regression_results"]
    buy_regression = items["buy_regression"]
    meta = items["meta_results"]
    confidence = items["meta_confidence"]
    feature_ablation = items["feature_ablation"]
    decision = items["decision"]
    return "\n".join(
        [
            "NATIP V3 Model-Improvement Research",
            "====================================",
            "",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Frozen V2 BUY candidate modified: NO.",
            "BUY rules modified: NO.",
            "",
            "Benchmark:",
            "XGB26 Top 1% AND Current Binary77 Top 0.5%; tuned Binary77 is confidence only.",
            "",
            "Regression overall:",
            regression[regression["Section"].eq("overall")].to_string(index=False),
            "",
            "Regression within existing BUY:",
            buy_regression.head(12).to_string(index=False),
            "",
            "Meta-model evaluation:",
            meta.head(20).to_string(index=False),
            "",
            "Meta confidence within existing BUY:",
            confidence.head(20).to_string(index=False),
            "",
            "New-feature grouped ablation:",
            feature_ablation.to_string(index=False),
            "",
            f"Decision: {decision}",
            "",
            "Decision policy: do not promote any V3 candidate unless improvement survives "
            "multiple folds, years, date-block bootstrap review, and reasonable sample count.",
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
