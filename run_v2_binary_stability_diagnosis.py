"""Diagnose Binary77 stability after V2 XGBoost tuning."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
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
    _prepare_dataset,
    _safe_float,
)
from run_v2_backward_forward_selection import (
    BINARY_TOP,
    CLASSIFIER_TOP,
    CLEAN_DEMERGER_DATASET,
)
from run_v2_model_family_comparison import _selected_26_features
from run_v2_xgb_hyperparameter_optimization import (
    BOOTSTRAP_SAMPLES,
    RANDOM_SEED,
    TrialSpec,
    _binary_specs,
    _fit_xgb,
)

CURRENT_BINARY_ID = "binary_baseline"
TUNED_BINARY_ID = "binary_trial_03"
MIN_FOLD_SIGNALS = 20

OUTPUT_ZERO_FOLD = REPORT_DIR / "v2_binary_zero_fold_diagnosis.csv"
OUTPUT_PARAMS = REPORT_DIR / "v2_binary_parameter_comparison.csv"
OUTPUT_MEMBERSHIP = REPORT_DIR / "v2_binary_membership_analysis.csv"
OUTPUT_PARETO = REPORT_DIR / "v2_binary_stability_pareto.csv"
OUTPUT_TOP3 = REPORT_DIR / "v2_binary_stable_top3.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "v2_binary_stable_bootstrap.csv"
OUTPUT_RECOMMENDATION = REPORT_DIR / "v2_binary_stability_recommendation.json"
OUTPUT_REPORT = REPORT_DIR / "v2_binary_stability_report.txt"


def main() -> None:
    """Run validation-only stability diagnostics for existing Binary77 trials."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into binary stability diagnostics.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    print("[primary] current XGB26", flush=True)
    primary = _primary_baseline_oof(common, folds, features_26)

    specs = _binary_specs(common)
    predictions: dict[str, pd.DataFrame] = {}
    audits: dict[str, pd.DataFrame] = {}
    for spec in specs:
        print(f"[binary] {spec.trial_id}", flush=True)
        predictions[spec.trial_id], audits[spec.trial_id] = _binary_oof_with_audit(
            common, folds, features_77, spec
        )

    current_binary = predictions[CURRENT_BINARY_ID]
    tuned_binary = predictions[TUNED_BINARY_ID]
    current_hybrid = _selected_hybrid(primary, current_binary)
    tuned_hybrid = _selected_hybrid(primary, tuned_binary)

    zero_fold = _zero_fold_diagnosis(common, folds, primary, tuned_binary, audits[TUNED_BINARY_ID])
    zero_fold.to_csv(OUTPUT_ZERO_FOLD, index=False)
    params = _parameter_comparison(
        next(spec for spec in specs if spec.trial_id == CURRENT_BINARY_ID),
        next(spec for spec in specs if spec.trial_id == TUNED_BINARY_ID),
        audits[CURRENT_BINARY_ID],
        audits[TUNED_BINARY_ID],
    )
    params.to_csv(OUTPUT_PARAMS, index=False)
    membership = _membership_analysis(primary, current_binary, tuned_binary)
    membership.to_csv(OUTPUT_MEMBERSHIP, index=False)
    pareto = _pareto_report(common, primary, predictions)
    pareto.to_csv(OUTPUT_PARETO, index=False)
    top3 = _top3_candidates(pareto)
    top3.to_csv(OUTPUT_TOP3, index=False)
    bootstrap = _bootstrap_top3(primary, predictions, top3)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    recommendation = _recommendation_payload(
        zero_fold=zero_fold,
        params=params,
        membership=membership,
        pareto=pareto,
        top3=top3,
        bootstrap=bootstrap,
        current_hybrid=current_hybrid,
        tuned_hybrid=tuned_hybrid,
    )
    OUTPUT_RECOMMENDATION.write_text(
        json.dumps(_json_safe(recommendation), indent=2), encoding="utf-8"
    )
    OUTPUT_REPORT.write_text(
        _summary(zero_fold, params, membership, pareto, top3, bootstrap, recommendation),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_binary_zero_fold_diagnosis": str(OUTPUT_ZERO_FOLD),
                "v2_binary_parameter_comparison": str(OUTPUT_PARAMS),
                "v2_binary_membership_analysis": str(OUTPUT_MEMBERSHIP),
                "v2_binary_stability_pareto": str(OUTPUT_PARETO),
                "v2_binary_stable_top3": str(OUTPUT_TOP3),
                "v2_binary_stable_bootstrap": str(OUTPUT_BOOTSTRAP),
                "v2_binary_stability_recommendation": str(OUTPUT_RECOMMENDATION),
                "v2_binary_stability_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "features_modified": False,
                "buy_percentile_thresholds_modified": False,
                "broad_search_performed": False,
            },
            indent=2,
        )
    )


def _primary_baseline_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_default_classifier(
            train, features, target="label", model_name="xgboost", binary=False
        )
        probabilities = model.predict_proba(_transform_with_medians(validation, features, medians))
        output = _base_frame(validation, fold["fold"])
        output["p_outperform"] = probabilities[:, 2]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_oof_with_audit(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    spec: TrialSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    audits = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit, calibration = _fit_calibration_split(train)
        if spec.is_baseline:
            model, medians = _fit_default_classifier(
                fit, features, target="buy_target", model_name="xgboost", binary=True
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
                "Sector",
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
        output["sigmoid_percentile"] = output.groupby("Date")["sigmoid_probability"].rank(pct=True)
        frames.append(output)
        audits.append(
            {
                "TrialID": spec.trial_id,
                "fold": fold["fold"],
                "TrainStart": fold["train_start"],
                "TrainEnd": fold["train_end"],
                "ValidationStart": fold["validation_start"],
                "ValidationEnd": fold["validation_end"],
                "BestIteration": getattr(model, "best_iteration", None),
                "BestScore": getattr(model, "best_score", None),
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audits)


def _zero_fold_diagnosis(
    common: pd.DataFrame,
    folds: list[dict[str, Any]],
    primary: pd.DataFrame,
    tuned_binary: pd.DataFrame,
    tuned_audit: pd.DataFrame,
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = primary.merge(
        tuned_binary[key + ["sigmoid_percentile"]],
        on=key,
        how="inner",
    )
    rows = []
    for fold in folds:
        fold_id = fold["fold"]
        validation = common[common["Date"].isin(fold["validation_dates"])].copy()
        fold_frame = merged[merged["fold"].eq(fold_id)].copy()
        xgb_top = fold_frame[fold_frame["classifier_percentile"] >= 1 - CLASSIFIER_TOP]
        binary_top = fold_frame[fold_frame["sigmoid_percentile"] >= 1 - BINARY_TOP]
        selected = _selected_hybrid_from_merged(fold_frame)
        sector_share = _largest_share(selected, "Sector")
        regime_share = _largest_share(selected, "market_regime_label")
        audit_row = tuned_audit[tuned_audit["fold"].eq(fold_id)].iloc[0].to_dict()
        row = {
            "fold": fold_id,
            "TrainStart": fold["train_start"],
            "TrainEnd": fold["train_end"],
            "ValidationStart": fold["validation_start"],
            "ValidationEnd": fold["validation_end"],
            "ValidationObservations": int(len(validation)),
            "BasePositiveRate": _safe_float(validation["buy_target"].mean()),
            "XGBTop1Observations": int(len(xgb_top)),
            "BinaryTop0.5Observations": int(len(binary_top)),
            "IntersectionObservations": int(len(selected)),
            "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
            "PrecisionLift": (
                _safe_float(selected["buy_target"].mean() / validation["buy_target"].mean())
                if not selected.empty and validation["buy_target"].mean()
                else None
            ),
            "AverageFutureStockReturn": _safe_float(selected["future_stock_return"].mean()),
            "MedianFutureStockReturn": _safe_float(selected["future_stock_return"].median()),
            "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
            "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
            "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
            "LargestSectorShare": sector_share,
            "LargestRegimeShare": regime_share,
            "BestIteration": audit_row.get("BestIteration"),
            "BestScore": audit_row.get("BestScore"),
        }
        row["DiagnosisCause"] = _diagnosis_cause(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _parameter_comparison(
    current: TrialSpec,
    tuned: TrialSpec,
    current_audit: pd.DataFrame,
    tuned_audit: pd.DataFrame,
) -> pd.DataFrame:
    fields = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]
    rows = []
    for field in fields:
        rows.append(
            {
                "Parameter": field,
                "CurrentBinary77": current.params.get(field),
                "TunedBinary77": tuned.params.get(field),
                "AbsoluteChange": _diff(tuned.params.get(field), current.params.get(field)),
                "LikelyTailRankingImpact": _parameter_impact(
                    field, current.params.get(field), tuned.params.get(field)
                ),
            }
        )
    rows.append(
        {
            "Parameter": "scale_pos_weight",
            "CurrentBinary77": current.scale_pos_weight,
            "TunedBinary77": tuned.scale_pos_weight,
            "AbsoluteChange": _diff(tuned.scale_pos_weight, current.scale_pos_weight),
            "LikelyTailRankingImpact": "Moderate positive-class weighting can move borderline BUY positives into the top tail, but may destabilize fold-specific rankings.",
        }
    )
    rows.append(
        {
            "Parameter": "early_stopping_iteration_mean",
            "CurrentBinary77": _safe_float(current_audit["BestIteration"].mean()),
            "TunedBinary77": _safe_float(tuned_audit["BestIteration"].mean()),
            "AbsoluteChange": _diff(
                _safe_float(tuned_audit["BestIteration"].mean()),
                _safe_float(current_audit["BestIteration"].mean()),
            ),
            "LikelyTailRankingImpact": "Large effective-tree-count changes alter tail sharpness and can create fold instability if validation regimes differ.",
        }
    )
    return pd.DataFrame(rows)


def _membership_analysis(
    primary: pd.DataFrame, current_binary: pd.DataFrame, tuned_binary: pd.DataFrame
) -> pd.DataFrame:
    current = _selected_hybrid(primary, current_binary)
    tuned = _selected_hybrid(primary, tuned_binary)
    current_keys = set(_keys(current))
    tuned_keys = set(_keys(tuned))
    merged = primary.copy()
    merged["CurrentMember"] = _key_series(merged).isin(current_keys)
    merged["TunedMember"] = _key_series(merged).isin(tuned_keys)
    labels = {
        "BOTH_CURRENT_AND_TUNED": merged["CurrentMember"] & merged["TunedMember"],
        "CURRENT_ONLY": merged["CurrentMember"] & ~merged["TunedMember"],
        "TUNED_ONLY": ~merged["CurrentMember"] & merged["TunedMember"],
        "NEITHER": ~merged["CurrentMember"] & ~merged["TunedMember"],
    }
    rows = []
    for label, mask in labels.items():
        frame = merged[mask].copy()
        rows.append(_membership_row(frame, label, "overall", "all"))
        if label != "NEITHER":
            for fold, group in frame.groupby("fold"):
                rows.append(_membership_row(group, label, "fold", fold))
            for year, group in frame.groupby(frame["Date"].dt.year):
                rows.append(_membership_row(group, label, "year", int(year)))
    return pd.DataFrame(rows)


def _pareto_report(
    common: pd.DataFrame, primary: pd.DataFrame, predictions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    for trial_id, binary in predictions.items():
        selected = _selected_hybrid(primary, binary)
        folds = [
            _metric_by_group(group) | {"fold": fold} for fold, group in selected.groupby("fold")
        ]
        years = [
            _metric_by_group(group) | {"year": year}
            for year, group in selected.groupby(selected["Date"].dt.year)
        ]
        fold_frame = pd.DataFrame(folds)
        year_frame = pd.DataFrame(years)
        row = {
            "TrialID": trial_id,
            "SignalCount": int(len(selected)),
            "Precision": _safe_float(selected["buy_target"].mean()),
            "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
            "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
            "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
            "MinFoldPrecision": (
                _safe_float(fold_frame["Precision"].min()) if not fold_frame.empty else None
            ),
            "MinFoldAverageExcessReturn": (
                _safe_float(fold_frame["AverageExcessReturn"].min())
                if not fold_frame.empty
                else None
            ),
            "MinFoldMedianExcessReturn": (
                _safe_float(fold_frame["MedianExcessReturn"].min())
                if not fold_frame.empty
                else None
            ),
            "MinSignalsAnyFold": int(fold_frame["Count"].min()) if not fold_frame.empty else 0,
            "FoldPrecisionStd": (
                _safe_float(fold_frame["Precision"].std(ddof=0)) if not fold_frame.empty else None
            ),
            "YearPositiveAvgExcessPct": (
                _safe_float((year_frame["AverageExcessReturn"] > 0).mean())
                if not year_frame.empty
                else None
            ),
            "LargestFoldSignalShare": _largest_group_share(selected, "fold"),
            "LargestYearSignalShare": _largest_group_share(
                selected.assign(year=selected["Date"].dt.year), "year"
            ),
        }
        for minimum in (300, 325, 350):
            row[f"EligibleMinSignals{minimum}"] = _eligible(row, minimum)
        row["ParetoRank"] = None
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["ParetoRank"] = _pareto_rank(frame)
    return frame.sort_values(
        ["ParetoRank", "Precision", "SignalCount"], ascending=[True, False, False]
    )


def _top3_candidates(pareto: pd.DataFrame) -> pd.DataFrame:
    current = pareto[pareto["TrialID"].eq(CURRENT_BINARY_ID)].copy()
    eligible = pareto[pareto["EligibleMinSignals325"].eq(True)].copy()
    precision = (
        eligible[~eligible["TrialID"].eq(CURRENT_BINARY_ID)]
        .sort_values(["Precision", "MedianExcessReturn", "SignalCount"], ascending=False)
        .head(1)
    )
    balanced = eligible[
        ~eligible["TrialID"].isin([CURRENT_BINARY_ID, *precision["TrialID"].tolist()])
    ].copy()
    if not balanced.empty:
        balanced["BalancedScore"] = (
            balanced["Precision"].rank(pct=True)
            + balanced["SignalCount"].rank(pct=True)
            + balanced["MedianExcessReturn"].rank(pct=True)
            + balanced["MinFoldPrecision"].rank(pct=True)
        ) / 4
        balanced = balanced.sort_values("BalancedScore", ascending=False).head(1)
    output = []
    if not current.empty:
        current = current.copy()
        current["Role"] = "A_CURRENT_BINARY77_BENCHMARK"
        output.append(current)
    if not precision.empty:
        precision = precision.copy()
        precision["Role"] = "B_BEST_PRECISION_STABLE"
        output.append(precision)
    if not balanced.empty:
        balanced = balanced.copy()
        balanced["Role"] = "C_BEST_BALANCED_STABLE"
        output.append(balanced)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _bootstrap_top3(
    primary: pd.DataFrame, predictions: dict[str, pd.DataFrame], top3: pd.DataFrame
) -> pd.DataFrame:
    current = _selected_hybrid(primary, predictions[CURRENT_BINARY_ID])
    rows = []
    for item in top3.itertuples(index=False):
        candidate = _selected_hybrid(primary, predictions[item.TrialID])
        rows.append(_bootstrap_candidate(current, candidate, item.TrialID, item.Role))
    return pd.DataFrame(rows)


def _bootstrap_candidate(
    current: pd.DataFrame, candidate: pd.DataFrame, trial_id: str, role: str
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(current["Date"]).union(set(candidate["Date"]))))
    current_groups = _date_metric_groups(current)
    candidate_groups = _date_metric_groups(candidate)
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        current_metrics = _metrics_from_date_groups(current_groups, sample_dates)
        candidate_metrics = _metrics_from_date_groups(candidate_groups, sample_dates)
        diffs.append(
            {
                "PrecisionDiff": candidate_metrics["Precision"] - current_metrics["Precision"],
                "AverageExcessDiff": candidate_metrics["AverageExcessReturn"]
                - current_metrics["AverageExcessReturn"],
                "MedianExcessDiff": candidate_metrics["MedianExcessReturn"]
                - current_metrics["MedianExcessReturn"],
                "NiftyWinRateDiff": candidate_metrics["NiftyWinRate"]
                - current_metrics["NiftyWinRate"],
            }
        )
    frame = pd.DataFrame(diffs)
    row = {"TrialID": trial_id, "Role": role, "BootstrapSamples": BOOTSTRAP_SAMPLES}
    for column in frame.columns:
        row[f"{column}Mean"] = _safe_float(frame[column].mean())
        row[f"{column}CI025"] = _safe_float(frame[column].quantile(0.025))
        row[f"{column}CI975"] = _safe_float(frame[column].quantile(0.975))
        row[f"{column}StatisticallyAboveZero"] = bool(frame[column].quantile(0.025) > 0)
    return row


def _recommendation_payload(**kwargs: Any) -> dict[str, Any]:
    pareto = kwargs["pareto"]
    top3 = kwargs["top3"]
    stable = (
        top3[top3["Role"].ne("A_CURRENT_BINARY77_BENCHMARK")] if not top3.empty else pd.DataFrame()
    )
    current = pareto[pareto["TrialID"].eq(CURRENT_BINARY_ID)].iloc[0].to_dict()
    promoted = pd.DataFrame()
    if not stable.empty:
        promoted = stable[
            (stable["Precision"] > current["Precision"])
            & (stable["MedianExcessReturn"] >= current["MedianExcessReturn"])
            & (stable["MinFoldPrecision"] >= current["MinFoldPrecision"])
        ].copy()
    if not promoted.empty:
        decision = "PROMOTE_STABLE_TUNED_BINARY77"
        candidate = (
            promoted.sort_values(
                ["Precision", "MedianExcessReturn", "SignalCount"], ascending=False
            )
            .iloc[0]
            .to_dict()
        )
    elif stable.empty:
        decision = "CONTINUE_LOCAL_BINARY77_SEARCH"
        candidate = None
    else:
        decision = "KEEP_CURRENT_BINARY77"
        candidate = current
    return {
        "decision": decision,
        "selection_basis": "Validation/OOF only; final test not used.",
        "v1_production_modified": False,
        "features_modified": False,
        "buy_percentile_thresholds_modified": False,
        "broad_search_performed": False,
        "current_binary77": current,
        "preferred_candidate": candidate,
        "zero_fold_summary": kwargs["zero_fold"].to_dict(orient="records"),
        "top3": top3.to_dict(orient="records"),
        "bootstrap": kwargs["bootstrap"].to_dict(orient="records"),
        "next_step": (
            "Run a small local interpolation search between current and high-precision tuned Binary77."
            if decision == "CONTINUE_LOCAL_BINARY77_SEARCH"
            else "Do not run local search unless further stability evidence is required."
        ),
    }


def _selected_hybrid(primary: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = primary.merge(binary[key + ["sigmoid_percentile"]], on=key, how="inner")
    return _selected_hybrid_from_merged(merged)


def _selected_hybrid_from_merged(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (frame["sigmoid_percentile"] >= 1 - BINARY_TOP)
    ].copy()


def _metric_by_group(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "Count": int(len(frame)),
        "Precision": _safe_float(frame["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _membership_row(frame: pd.DataFrame, label: str, scope: str, value: Any) -> dict[str, Any]:
    return {"Membership": label, "Scope": scope, "ScopeValue": value, **_metric_by_group(frame)}


def _eligible(row: dict[str, Any], minimum_signals: int) -> bool:
    return bool(
        row["SignalCount"] >= minimum_signals
        and row["MinSignalsAnyFold"] >= MIN_FOLD_SIGNALS
        and row["MinFoldAverageExcessReturn"] is not None
        and row["MinFoldAverageExcessReturn"] > 0
        and (row["MinFoldPrecision"] is not None and row["MinFoldPrecision"] > 0)
        and row["LargestFoldSignalShare"] <= 0.35
        and row["LargestYearSignalShare"] <= 0.70
    )


def _pareto_rank(frame: pd.DataFrame) -> list[int]:
    ranks = []
    metrics = [
        "Precision",
        "SignalCount",
        "MinFoldPrecision",
        "AverageExcessReturn",
        "MedianExcessReturn",
    ]
    for _, row in frame.iterrows():
        rank = 1
        for _, other in frame.iterrows():
            if other["TrialID"] == row["TrialID"]:
                continue
            if all(other[m] >= row[m] for m in metrics) and any(other[m] > row[m] for m in metrics):
                rank += 1
        ranks.append(rank)
    return ranks


def _diagnosis_cause(row: dict[str, Any]) -> str:
    if row["IntersectionObservations"] < 10:
        return "tiny_sample_size"
    if row["Precision"] == 0:
        if row["LargestSectorShare"] and row["LargestSectorShare"] > 0.50:
            return "sector_concentration_with_zero_precision"
        if row["LargestRegimeShare"] and row["LargestRegimeShare"] > 0.80:
            return "regime_concentration_with_zero_precision"
        return "general_model_failure_in_fold"
    if row["AverageExcessReturn"] is not None and row["AverageExcessReturn"] <= 0:
        return "regime_or_ranking_failure"
    return "no_zero_precision_issue"


def _parameter_impact(field: str, current: Any, tuned: Any) -> str:
    if field == "learning_rate":
        return "Lower learning rate with many more trees can sharpen ranking but may overfit fold-specific validation tails."
    if field == "max_depth":
        return "Higher depth captures interactions but can increase tail instability."
    if field == "gamma":
        return "Higher gamma prunes weak splits, changing which observations survive into the extreme tail."
    if field == "reg_lambda":
        return "Higher L2 regularization smooths leaf weights while still allowing a different tail order."
    if field == "n_estimators":
        return "Many more trees plus early stopping changes effective ensemble size and probability separation."
    if field == "reg_alpha":
        return "Alpha near zero keeps most split structure active; impact likely smaller than depth/gamma/lambda."
    return "Parameter shift may alter tail ranking; inspect with fold-level stability rather than overall precision only."


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
    groups: dict[Any, dict[str, Any]], dates: np.ndarray
) -> dict[str, float]:
    count = 0
    buy_sum = excess_sum = win_sum = 0.0
    values = []
    for date in dates:
        stats = groups.get(date)
        if stats is None:
            continue
        count += stats["count"]
        buy_sum += stats["buy_sum"]
        excess_sum += stats["excess_sum"]
        win_sum += stats["win_sum"]
        values.append(stats["excess_values"])
    if count == 0:
        return {
            "Precision": np.nan,
            "AverageExcessReturn": np.nan,
            "MedianExcessReturn": np.nan,
            "NiftyWinRate": np.nan,
        }
    all_values = np.concatenate(values) if values else np.array([np.nan])
    return {
        "Precision": buy_sum / count,
        "AverageExcessReturn": excess_sum / count,
        "MedianExcessReturn": float(np.nanmedian(all_values)),
        "NiftyWinRate": win_sum / count,
    }


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
        ]
    ].copy()
    output["fold"] = fold
    return output


def _keys(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, str, int]]:
    return list(zip(frame["Date"], frame["symbol"], frame["fold"], strict=False))


def _key_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(_keys(frame), index=frame.index)


def _largest_share(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return _safe_float(frame[column].value_counts(normalize=True).max())


def _largest_group_share(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return _safe_float(frame[column].value_counts(normalize=True).max())


def _diff(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return _safe_float(float(left) - float(right))


def _summary(
    zero_fold: pd.DataFrame,
    params: pd.DataFrame,
    membership: pd.DataFrame,
    pareto: pd.DataFrame,
    top3: pd.DataFrame,
    bootstrap: pd.DataFrame,
    recommendation: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "NATIP V2 Binary77 Stability Diagnosis",
            "=====================================",
            "",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Features and BUY percentile thresholds modified: NO.",
            "Broad hyperparameter search performed: NO.",
            "",
            "Zero-fold diagnosis:",
            zero_fold.to_string(index=False),
            "",
            "Parameter comparison:",
            params.to_string(index=False),
            "",
            "Membership analysis:",
            membership.to_string(index=False),
            "",
            "Stability Pareto:",
            pareto.to_string(index=False),
            "",
            "Stable Top 3:",
            top3.to_string(index=False),
            "",
            "Bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "Recommendation:",
            json.dumps(_json_safe(recommendation), indent=2),
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
