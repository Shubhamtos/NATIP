"""Validation-only V2 XGBoost hyperparameter optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_isotonic,
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
from run_v2_backward_forward_selection import (
    BINARY_TOP,
    CLASSIFIER_TOP,
    CLEAN_DEMERGER_DATASET,
)
from run_v2_model_family_comparison import _selected_26_features

PRIMARY_TRIALS = 14
BINARY_TRIALS = 10
TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 2000

OUTPUT_PRIMARY_TRIALS = REPORT_DIR / "v2_xgb26_tuning_trials.csv"
OUTPUT_PRIMARY_TOP = REPORT_DIR / "v2_xgb26_top_configs.csv"
OUTPUT_BINARY_TRIALS = REPORT_DIR / "v2_binary77_tuning_trials.csv"
OUTPUT_BINARY_TOP = REPORT_DIR / "v2_binary77_top_configs.csv"
OUTPUT_CALIBRATION = REPORT_DIR / "v2_binary77_calibration.csv"
OUTPUT_COMPARISON = REPORT_DIR / "v2_tuned_hybrid_comparison.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "v2_tuned_hybrid_bootstrap.csv"
OUTPUT_CONFIG = REPORT_DIR / "v2_tuned_xgb_config.json"
OUTPUT_REPORT = REPORT_DIR / "v2_xgb_tuning_report.txt"


@dataclass(frozen=True, slots=True)
class TrialSpec:
    """XGBoost trial configuration."""

    trial_id: str
    phase: str
    params: dict[str, Any]
    scale_pos_weight: float | None = None
    is_baseline: bool = False


def main() -> None:
    """Run focused XGBoost tuning with OOF validation only."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into tuning research.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    if len(features_26) != 26:
        raise AssertionError(f"Expected 26 features, got {len(features_26)}")
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    print("[phase1] primary XGB26 trials", flush=True)
    primary_specs = _primary_specs()
    primary_predictions = {
        spec.trial_id: _classifier_oof(common, folds, features_26, spec) for spec in primary_specs
    }
    primary_trials = pd.DataFrame(
        [_primary_trial_row(spec, primary_predictions[spec.trial_id]) for spec in primary_specs]
    )
    primary_trials.to_csv(OUTPUT_PRIMARY_TRIALS, index=False)
    top_primary = _top_primary_configs(primary_trials)
    top_primary.to_csv(OUTPUT_PRIMARY_TOP, index=False)
    selected_primary_id = str(top_primary.iloc[0]["TrialID"])
    selected_primary = next(spec for spec in primary_specs if spec.trial_id == selected_primary_id)
    selected_primary_predictions = primary_predictions[selected_primary_id]
    baseline_primary_predictions = primary_predictions["primary_baseline"]

    print("[fixed] current Binary77 baseline", flush=True)
    baseline_binary = _binary_oof(common, folds, features_77, _binary_baseline_spec())
    primary_hybrids = pd.DataFrame(
        [
            _hybrid_row(
                primary_predictions[spec.trial_id],
                baseline_binary,
                spec.trial_id,
                "primary_tuning_current_binary77",
                common,
            )
            for spec in primary_specs
        ]
    )

    print("[phase3] binary XGB77 trials", flush=True)
    binary_specs = _binary_specs(common)
    binary_predictions = {
        spec.trial_id: _binary_oof(common, folds, features_77, spec) for spec in binary_specs
    }
    binary_trials = pd.DataFrame(
        [
            _binary_trial_row(spec, binary_predictions[spec.trial_id], calibration="sigmoid")
            for spec in binary_specs
        ]
    )
    binary_trials.to_csv(OUTPUT_BINARY_TRIALS, index=False)
    binary_hybrids = pd.DataFrame(
        [
            _hybrid_row(
                selected_primary_predictions,
                binary_predictions[spec.trial_id],
                spec.trial_id,
                "binary_tuning_selected_primary",
                common,
            )
            for spec in binary_specs
        ]
    )
    binary_top = _top_binary_configs(binary_trials, binary_hybrids)
    binary_top.to_csv(OUTPUT_BINARY_TOP, index=False)
    selected_binary_id = str(binary_top.iloc[0]["TrialID"])
    selected_binary = next(spec for spec in binary_specs if spec.trial_id == selected_binary_id)
    selected_binary_predictions = binary_predictions[selected_binary_id]

    print("[phase4] calibration", flush=True)
    calibration = _calibration_report(selected_binary_predictions)
    calibration.to_csv(OUTPUT_CALIBRATION, index=False)

    comparison = pd.DataFrame(
        [
            _hybrid_row(
                baseline_primary_predictions,
                baseline_binary,
                "A_current_xgb26_current_binary77",
                "robustness",
                common,
            ),
            _hybrid_row(
                selected_primary_predictions,
                baseline_binary,
                "B_tuned_xgb26_current_binary77",
                "robustness",
                common,
            ),
            _hybrid_row(
                selected_primary_predictions,
                selected_binary_predictions,
                "C_tuned_xgb26_tuned_binary77",
                "robustness",
                common,
            ),
        ]
    )
    comparison = pd.concat([comparison, primary_hybrids, binary_hybrids], ignore_index=True)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    bootstrap = _bootstrap_report(
        baseline_primary_predictions,
        baseline_binary,
        selected_primary_predictions,
        selected_binary_predictions,
    )
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    config = _config_payload(
        selected_primary,
        selected_binary,
        primary_trials,
        binary_trials,
        comparison,
        bootstrap,
        features_26,
        features_77,
    )
    OUTPUT_CONFIG.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(
        _summary(
            primary_trials,
            top_primary,
            binary_trials,
            binary_top,
            calibration,
            comparison,
            bootstrap,
            config,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_xgb26_tuning_trials": str(OUTPUT_PRIMARY_TRIALS),
                "v2_xgb26_top_configs": str(OUTPUT_PRIMARY_TOP),
                "v2_binary77_tuning_trials": str(OUTPUT_BINARY_TRIALS),
                "v2_binary77_top_configs": str(OUTPUT_BINARY_TOP),
                "v2_binary77_calibration": str(OUTPUT_CALIBRATION),
                "v2_tuned_hybrid_comparison": str(OUTPUT_COMPARISON),
                "v2_tuned_hybrid_bootstrap": str(OUTPUT_BOOTSTRAP),
                "v2_tuned_xgb_config": str(OUTPUT_CONFIG),
                "v2_xgb_tuning_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "features_modified": False,
                "buy_percentile_structure_modified": False,
            },
            indent=2,
        )
    )


def _primary_specs() -> list[TrialSpec]:
    return [
        TrialSpec("primary_baseline", "primary", _baseline_params(binary=False), is_baseline=True),
        *[
            TrialSpec(f"primary_trial_{index:02d}", "primary", params)
            for index, params in enumerate(
                _sample_params(PRIMARY_TRIALS, seed=RANDOM_SEED), start=1
            )
        ],
    ]


def _binary_specs(common: pd.DataFrame) -> list[TrialSpec]:
    ratio = common["buy_target"].eq(0).sum() / max(1, common["buy_target"].eq(1).sum())
    moderate_weight = float(min(max(ratio * 0.5, 1.0), 3.0))
    specs = [
        TrialSpec("binary_baseline", "binary", _baseline_params(binary=True), is_baseline=True),
        TrialSpec(
            "binary_baseline_weighted",
            "binary",
            _baseline_params(binary=True),
            scale_pos_weight=moderate_weight,
        ),
    ]
    for index, params in enumerate(_sample_params(BINARY_TRIALS, seed=RANDOM_SEED + 100), start=1):
        weight = moderate_weight if index % 3 == 0 else None
        specs.append(
            TrialSpec(
                f"binary_trial_{index:02d}",
                "binary",
                params,
                scale_pos_weight=weight,
            )
        )
    return specs


def _binary_baseline_spec() -> TrialSpec:
    return TrialSpec("binary_baseline", "binary", _baseline_params(binary=True), is_baseline=True)


def _sample_params(count: int, *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    params = []
    for _ in range(count):
        params.append(
            {
                "n_estimators": int(rng.integers(300, 2001)),
                "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.10)))),
                "max_depth": int(rng.integers(2, 9)),
                "min_child_weight": float(np.exp(rng.uniform(np.log(1), np.log(20)))),
                "subsample": float(rng.uniform(0.60, 1.00)),
                "colsample_bytree": float(rng.uniform(0.50, 1.00)),
                "gamma": float(rng.uniform(0, 5)),
                "reg_alpha": float(np.exp(rng.uniform(np.log(0.001), np.log(10)))),
                "reg_lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(30)))),
            }
        )
    return params


def _baseline_params(*, binary: bool) -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic" if binary else "multi:softprob",
        "eval_metric": "logloss" if binary else "mlogloss",
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "n_jobs": -1,
    }


def _classifier_oof(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    spec: TrialSpec,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        if spec.is_baseline:
            model, medians = _fit_default_classifier(
                train,
                features,
                target="label",
                model_name="xgboost",
                binary=False,
            )
        else:
            fit, inner_validation = _fit_calibration_split(train)
            model, medians = _fit_xgb(
                fit,
                inner_validation,
                features,
                target="label",
                spec=spec,
                binary=False,
            )
        probabilities = model.predict_proba(_transform_with_medians(validation, features, medians))
        output = _base_frame(validation, fold["fold"])
        output["TrialID"] = spec.trial_id
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_oof(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    spec: TrialSpec,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit, calibration = _fit_calibration_split(train)
        if spec.is_baseline:
            model, medians = _fit_default_classifier(
                fit,
                features,
                target="buy_target",
                model_name="xgboost",
                binary=True,
            )
        else:
            model, medians = _fit_xgb(
                fit,
                calibration,
                features,
                target="buy_target",
                spec=spec,
                binary=True,
            )
        calibration_probability = _positive_probability(
            model, _transform_with_medians(calibration, features, medians)
        )
        raw_probability = _positive_probability(
            model, _transform_with_medians(validation, features, medians)
        )
        sigmoid = _fit_platt(calibration_probability, calibration["buy_target"])
        isotonic = _fit_isotonic(calibration_probability, calibration["buy_target"])
        output = _base_frame(validation, fold["fold"])[
            [
                "Date",
                "symbol",
                "fold",
                "future_stock_return",
                "future_nifty_return",
                "excess_return",
                "buy_target",
                "market_regime_label",
            ]
        ].copy()
        output["TrialID"] = spec.trial_id
        output["raw_probability"] = raw_probability
        output["sigmoid_probability"] = sigmoid.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        output["isotonic_probability"] = isotonic.predict(raw_probability)
        for calibration_name in ("raw", "sigmoid", "isotonic"):
            output[f"{calibration_name}_percentile"] = output.groupby("Date")[
                f"{calibration_name}_probability"
            ].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fit_xgb(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    spec: TrialSpec,
    binary: bool,
):
    from xgboost import XGBClassifier

    params = {**spec.params}
    params["objective"] = "binary:logistic" if binary else "multi:softprob"
    params["eval_metric"] = "logloss" if binary else "mlogloss"
    params["random_state"] = RANDOM_SEED
    params["tree_method"] = "hist"
    params["n_jobs"] = -1
    params["early_stopping_rounds"] = 50
    if binary and spec.scale_pos_weight is not None:
        params["scale_pos_weight"] = spec.scale_pos_weight
    medians = fit[features].median(numeric_only=True).fillna(0)
    x_fit = _transform_with_medians(fit, features, medians)
    x_validation = _transform_with_medians(validation, features, medians)
    y_fit = fit[target].astype(int) if binary else fit[target].map(LABEL_TO_CLASS)
    y_validation = (
        validation[target].astype(int) if binary else validation[target].map(LABEL_TO_CLASS)
    )
    model = XGBClassifier(**params)
    model.fit(x_fit, y_fit, eval_set=[(x_validation, y_validation)], verbose=False)
    return model, medians


def _primary_trial_row(spec: TrialSpec, predictions: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    y_true = predictions["label"].map(LABEL_TO_CLASS)
    probabilities = _probability_matrix(
        predictions[["p_underperform", "p_neutral", "p_outperform"]]
    )
    actual = predictions["buy_target"].astype(int)
    row = {
        "TrialID": spec.trial_id,
        "Phase": spec.phase,
        "IsBaseline": spec.is_baseline,
        **_params_for_row(spec),
        "Rows": int(len(predictions)),
        "PRAUC": float(average_precision_score(actual, predictions["p_outperform"])),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(brier_score_loss(actual, predictions["p_outperform"])),
        "MeanIC": _ic_summary(predictions, "p_outperform")["MeanIC"],
    }
    for fraction in TOP_FRACTIONS:
        selected = _top_by_date(predictions, "p_outperform", fraction)
        row[f"Top{_pct(fraction)}Precision"] = _safe_float(selected["buy_target"].mean())
    return row


def _binary_trial_row(
    spec: TrialSpec, predictions: pd.DataFrame, *, calibration: str
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    probability = predictions[f"{calibration}_probability"].clip(0, 1)
    actual = predictions["buy_target"].astype(int)
    row = {
        "TrialID": spec.trial_id,
        "Phase": spec.phase,
        "IsBaseline": spec.is_baseline,
        "Calibration": calibration,
        **_params_for_row(spec),
        "Rows": int(len(predictions)),
        "PRAUC": float(average_precision_score(actual, probability)),
        "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
        "Brier": float(brier_score_loss(actual, probability)),
    }
    for fraction in TOP_FRACTIONS:
        selected = _top_by_date(predictions, f"{calibration}_probability", fraction)
        row[f"Top{_pct(fraction)}Precision"] = _safe_float(selected["buy_target"].mean())
    return row


def _hybrid_row(
    primary: pd.DataFrame,
    binary: pd.DataFrame,
    trial_id: str,
    phase: str,
    common: pd.DataFrame,
) -> dict[str, Any]:
    key = ["Date", "symbol", "fold"]
    merged = primary.merge(
        binary[key + ["sigmoid_percentile"]],
        on=key,
        how="inner",
    )
    selected = merged[
        (merged["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (merged["sigmoid_percentile"] >= 1 - BINARY_TOP)
    ].copy()
    base_rate = float(common["buy_target"].mean())
    row = {
        "TrialID": trial_id,
        "Phase": phase,
        "SignalCount": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()),
        "PrecisionLift": _safe_float(selected["buy_target"].mean() / base_rate),
        "AverageFutureStockReturn": _safe_float(selected["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(selected["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
    }
    row.update(_stability(selected, "fold"))
    row.update(_stability(selected.assign(year=selected["Date"].dt.year), "year"))
    return row


def _calibration_report(predictions: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import brier_score_loss, log_loss

    rows = []
    raw_top = set(_top_key_frame(predictions, "raw_probability", BINARY_TOP))
    for calibration in ("raw", "sigmoid", "isotonic"):
        probability = predictions[f"{calibration}_probability"].clip(0, 1)
        top = set(_top_key_frame(predictions, f"{calibration}_probability", BINARY_TOP))
        overlap = len(raw_top & top)
        union = len(raw_top | top)
        rows.append(
            {
                "Calibration": calibration,
                "LogLoss": float(log_loss(predictions["buy_target"], probability, labels=[0, 1])),
                "Brier": float(brier_score_loss(predictions["buy_target"], probability)),
                "ReliabilityMAE": _reliability_mae(predictions["buy_target"], probability),
                "Top0.5MembershipCount": len(top),
                "Top0.5JaccardVsRaw": _safe_float(overlap / union) if union else None,
            }
        )
    return pd.DataFrame(rows)


def _top_primary_configs(primary_trials: pd.DataFrame) -> pd.DataFrame:
    return (
        primary_trials.sort_values(
            ["Top1%Precision", "Top0.5%Precision", "PRAUC", "MeanIC"],
            ascending=False,
        )
        .head(3)
        .copy()
    )


def _top_binary_configs(binary_trials: pd.DataFrame, binary_hybrids: pd.DataFrame) -> pd.DataFrame:
    joined = binary_trials.merge(
        binary_hybrids[
            [
                "TrialID",
                "SignalCount",
                "Precision",
                "AverageExcessReturn",
                "MedianExcessReturn",
                "NiftyWinRate",
            ]
        ],
        on="TrialID",
        how="left",
        suffixes=("_Binary", "_Hybrid"),
    )
    return (
        joined.sort_values(
            ["Precision", "MedianExcessReturn", "AverageExcessReturn", "SignalCount"],
            ascending=False,
        )
        .head(3)
        .copy()
    )


def _bootstrap_report(
    baseline_primary: pd.DataFrame,
    baseline_binary: pd.DataFrame,
    tuned_primary: pd.DataFrame,
    tuned_binary: pd.DataFrame,
) -> pd.DataFrame:
    baseline = _selected_hybrid(baseline_primary, baseline_binary)
    tuned = _selected_hybrid(tuned_primary, tuned_binary)
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(baseline["Date"]).union(set(tuned["Date"]))))
    baseline_groups = _date_metric_groups(baseline)
    tuned_groups = _date_metric_groups(tuned)
    rows = []
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        base_sample = _metrics_from_date_groups(baseline_groups, sample_dates)
        tuned_sample = _metrics_from_date_groups(tuned_groups, sample_dates)
        diffs.append(
            {
                "PrecisionDiff": _safe_float(tuned_sample["Precision"] - base_sample["Precision"]),
                "AverageExcessDiff": _safe_float(
                    tuned_sample["AverageExcessReturn"] - base_sample["AverageExcessReturn"]
                ),
                "MedianExcessDiff": _safe_float(
                    tuned_sample["MedianExcessReturn"] - base_sample["MedianExcessReturn"]
                ),
                "NiftyWinRateDiff": _safe_float(
                    tuned_sample["NiftyWinRate"] - base_sample["NiftyWinRate"]
                ),
            }
        )
    diff_frame = pd.DataFrame(diffs)
    row = {"Comparison": "C_minus_A", "BootstrapSamples": BOOTSTRAP_SAMPLES}
    for column in diff_frame.columns:
        row[f"{column}Mean"] = _safe_float(diff_frame[column].mean())
        row[f"{column}CI025"] = _safe_float(diff_frame[column].quantile(0.025))
        row[f"{column}CI975"] = _safe_float(diff_frame[column].quantile(0.975))
        row[f"{column}StatisticallyAboveZero"] = bool(diff_frame[column].quantile(0.025) > 0)
    rows.append(row)
    return pd.DataFrame(rows)


def _config_payload(
    selected_primary: TrialSpec,
    selected_binary: TrialSpec,
    primary_trials: pd.DataFrame,
    binary_trials: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    features_26: list[str],
    features_77: list[str],
) -> dict[str, Any]:
    best_comparison = comparison[comparison["TrialID"].eq("C_tuned_xgb26_tuned_binary77")]
    current = comparison[comparison["TrialID"].eq("A_current_xgb26_current_binary77")]
    tuned = best_comparison.iloc[0].to_dict()
    baseline = current.iloc[0].to_dict()
    successful = bool(
        tuned["Precision"] > baseline["Precision"]
        and tuned["MedianExcessReturn"] >= baseline["MedianExcessReturn"]
        and bootstrap.iloc[0].get("PrecisionDiffCI025", -1) > 0
        and tuned["SignalCount"] >= max(350, baseline["SignalCount"] * 0.70)
        and tuned["WorstFoldPrecision"] >= max(0.25, baseline["WorstFoldPrecision"] - 0.05)
        and tuned["FoldPositiveAvgExcessPct"] >= baseline["FoldPositiveAvgExcessPct"]
    )
    return {
        "selection_basis": "Validation/OOF only; final test not used.",
        "v1_production_modified": False,
        "features_modified": False,
        "buy_percentile_structure_modified": False,
        "primary_feature_count": len(features_26),
        "primary_features": features_26,
        "binary_feature_count": len(features_77),
        "selected_primary_trial": selected_primary.trial_id,
        "selected_primary_params": selected_primary.params,
        "selected_binary_trial": selected_binary.trial_id,
        "selected_binary_params": selected_binary.params,
        "selected_binary_scale_pos_weight": selected_binary.scale_pos_weight,
        "current_hybrid": baseline,
        "tuned_hybrid": tuned,
        "bootstrap": bootstrap.to_dict(orient="records"),
        "recommendation": (
            "TUNED_REPLACE_CURRENT_V2_RESEARCH_CANDIDATE"
            if successful
            else "KEEP_CURRENT_V2_RESEARCH_CANDIDATE_UNTIL_STABILITY_IMPROVES"
        ),
        "top_primary_trials": primary_trials.head(5).to_dict(orient="records"),
        "top_binary_trials": binary_trials.head(5).to_dict(orient="records"),
    }


def _selected_hybrid(primary: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = primary.merge(binary[key + ["sigmoid_percentile"]], on=key, how="inner")
    return merged[
        (merged["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (merged["sigmoid_percentile"] >= 1 - BINARY_TOP)
    ].copy()


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _base_frame(frame: pd.DataFrame, fold: Any) -> pd.DataFrame:
    output = frame[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
            "label",
        ]
    ].copy()
    output["fold"] = fold
    return output


def _probability_matrix(probabilities: pd.DataFrame) -> pd.DataFrame:
    total = probabilities.sum(axis=1).replace(0, 1)
    return probabilities.div(total, axis=0)


def _top_by_date(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    percentile = frame.groupby("Date")[score].rank(pct=True)
    return frame[percentile >= 1 - fraction].copy()


def _top_key_frame(
    frame: pd.DataFrame, score: str, fraction: float
) -> list[tuple[pd.Timestamp, str]]:
    selected = _top_by_date(frame, score, fraction)
    return list(zip(selected["Date"], selected["symbol"], strict=False))


def _date_metric_groups(frame: pd.DataFrame) -> dict[Any, dict[str, Any]]:
    groups = {}
    for date, group in frame.groupby("Date"):
        groups[date] = {
            "count": int(len(group)),
            "buy_sum": float(group["buy_target"].sum()),
            "excess_sum": float(group["excess_return"].sum()),
            "win_sum": float((group["excess_return"] > 0).sum()),
            "excess_values": group["excess_return"].to_numpy(),
        }
    return groups


def _metrics_from_date_groups(
    groups: dict[Any, dict[str, Any]], sample_dates: np.ndarray
) -> dict[str, float]:
    count = 0
    buy_sum = 0.0
    excess_sum = 0.0
    win_sum = 0.0
    excess_values = []
    for date in sample_dates:
        stats = groups.get(date)
        if not stats:
            continue
        count += stats["count"]
        buy_sum += stats["buy_sum"]
        excess_sum += stats["excess_sum"]
        win_sum += stats["win_sum"]
        excess_values.append(stats["excess_values"])
    if count == 0:
        return {
            "Precision": np.nan,
            "AverageExcessReturn": np.nan,
            "MedianExcessReturn": np.nan,
            "NiftyWinRate": np.nan,
        }
    values = np.concatenate(excess_values) if excess_values else np.array([np.nan])
    return {
        "Precision": buy_sum / count,
        "AverageExcessReturn": excess_sum / count,
        "MedianExcessReturn": float(np.nanmedian(values)),
        "NiftyWinRate": win_sum / count,
    }


def _stability(selected: pd.DataFrame, column: str) -> dict[str, Any]:
    if selected.empty:
        return {
            f"{column.capitalize()}Count": 0,
            f"{column.capitalize()}PositiveAvgExcessPct": None,
            f"{column.capitalize()}PositiveMedianExcessPct": None,
            f"Worst{column.capitalize()}Precision": None,
            f"Worst{column.capitalize()}": None,
        }
    rows = []
    for value, group in selected.groupby(column):
        rows.append(
            {
                column: value,
                "Precision": _safe_float(group["buy_target"].mean()),
                "AverageExcessReturn": _safe_float(group["excess_return"].mean()),
                "MedianExcessReturn": _safe_float(group["excess_return"].median()),
            }
        )
    frame = pd.DataFrame(rows)
    worst = frame.sort_values(["Precision", "AverageExcessReturn"], ascending=True).iloc[0]
    return {
        f"{column.capitalize()}Count": int(len(frame)),
        f"{column.capitalize()}PositiveAvgExcessPct": _safe_float(
            (frame["AverageExcessReturn"] > 0).mean()
        ),
        f"{column.capitalize()}PositiveMedianExcessPct": _safe_float(
            (frame["MedianExcessReturn"] > 0).mean()
        ),
        f"Worst{column.capitalize()}Precision": worst["Precision"],
        f"Worst{column.capitalize()}": worst[column],
    }


def _reliability_mae(actual: pd.Series, probability: pd.Series) -> float:
    bins = pd.qcut(probability.rank(method="first"), q=10, duplicates="drop")
    grouped = pd.DataFrame({"actual": actual, "probability": probability}).groupby(
        bins, observed=False
    )
    return float((grouped["actual"].mean() - grouped["probability"].mean()).abs().mean())


def _params_for_row(spec: TrialSpec) -> dict[str, Any]:
    row = {
        f"param_{key}": value
        for key, value in spec.params.items()
        if key not in {"objective", "eval_metric", "random_state", "tree_method", "n_jobs"}
    }
    row["scale_pos_weight"] = spec.scale_pos_weight
    return row


def _pct(value: float) -> str:
    return f"{value * 100:g}%"


def _summary(
    primary_trials: pd.DataFrame,
    top_primary: pd.DataFrame,
    binary_trials: pd.DataFrame,
    binary_top: pd.DataFrame,
    calibration: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "NATIP V2 XGBoost Hyperparameter Optimization",
            "=============================================",
            "",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Features/targets/percentile rules modified: NO.",
            "",
            "Top primary XGB26 configs:",
            top_primary.to_string(index=False),
            "",
            "Top binary XGB77 configs:",
            binary_top.to_string(index=False),
            "",
            "Calibration:",
            calibration.to_string(index=False),
            "",
            "Hybrid comparison:",
            comparison.to_string(index=False),
            "",
            "Bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "Report answers:",
            f"1. Did tuning improve XGB26? {config['selected_primary_trial'] != 'primary_baseline'}",
            "2. Which XGB26 hyperparameters changed most? See selected_primary_params in JSON.",
            f"3. Did tuning improve Binary77? {config['selected_binary_trial'] != 'binary_baseline'}",
            f"4. Did class weighting help? selected scale_pos_weight={config['selected_binary_scale_pos_weight']}",
            "5. Did calibration help? See v2_binary77_calibration.csv; ranking impact is Top0.5 Jaccard vs raw.",
            f"6. Did fully tuned hybrid beat benchmark precision? {config['tuned_hybrid']['Precision'] > config['current_hybrid']['Precision']}",
            f"7. Did median/average excess improve? avg={config['tuned_hybrid']['AverageExcessReturn'] > config['current_hybrid']['AverageExcessReturn']}, median={config['tuned_hybrid']['MedianExcessReturn'] > config['current_hybrid']['MedianExcessReturn']}",
            f"8. Statistically credible? {bootstrap.iloc[0].get('PrecisionDiffStatisticallyAboveZero')}",
            "9. Fold/year stability is in v2_tuned_hybrid_comparison.csv.",
            f"10. Recommendation: {config['recommendation']}",
            "",
            "Selected config:",
            json.dumps(_json_safe(config), indent=2),
            "",
            "All primary trial rows:",
            primary_trials.to_string(index=False),
            "",
            "All binary trial rows:",
            binary_trials.to_string(index=False),
        ]
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
