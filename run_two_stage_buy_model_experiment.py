"""Validation-only two-stage BUY model research.

The experiment uses only the original 150-stock clean technical/market dataset
and strict OOF XGB26 + Current Binary77 predictions. It does not modify V1/V2
production artifacts and does not use final-test rows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import _fit_classifier, _positive_probability, _transform_with_medians
from run_stock_specific_feature_variant import _common_evaluation_rows, _frozen_features, _prepare_dataset
from run_v2_backward_forward_selection import CLEAN_DEMERGER_DATASET
from run_v2_model_family_comparison import _selected_26_features

CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"

OUTPUT_BASELINE = REPORT_DIR / "baseline_precision_recall_audit.csv"
OUTPUT_STAGE1 = REPORT_DIR / "stage1_candidate_analysis.csv"
OUTPUT_STAGE2_OOF = REPORT_DIR / "stage2_oof_predictions.parquet"
OUTPUT_EQUAL_SIGNAL = REPORT_DIR / "stage2_equal_signal_comparison.csv"
OUTPUT_EQUAL_PRECISION = REPORT_DIR / "stage2_equal_precision_comparison.csv"
OUTPUT_FRONTIER = REPORT_DIR / "stage2_precision_recall_frontier.csv"
OUTPUT_FOLD = REPORT_DIR / "stage2_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "stage2_year_stability.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "stage2_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "stage2_report.txt"

RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 2000
SELECTED_STAGE1_RULE = "xgb_top5_or_binary_top5"
MIN_PRECISION_LEVELS = (0.50, 0.55, 0.60, 0.65)

STAGE1_RULES: tuple[tuple[str, str], ...] = (
    ("xgb_top3", "xgb_top3"),
    ("xgb_top5", "xgb_top5"),
    ("xgb_top10", "xgb_top10"),
    ("binary_top3", "binary_top3"),
    ("binary_top5", "binary_top5"),
    ("binary_top10", "binary_top10"),
    ("xgb_top5_or_binary_top5", "xgb_top5_or_binary_top5"),
    ("xgb_top5_and_binary_top10", "xgb_top5_and_binary_top10"),
    ("mean_percentile_top5", "mean_percentile_top5"),
)


def main() -> None:
    """Run the validation-only two-stage BUY experiment."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into two-stage BUY experiment.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    current = _load_current_oof()
    merged = _merge_features_and_oof(common, current, features_77)
    _assert_oof_chronology(merged, folds)

    baseline_full = _baseline_rows(merged, scope_name="full_validation")
    stage1 = _stage1_analysis(merged)
    selected_stage1 = stage1[
        (stage1["Rule"].eq(SELECTED_STAGE1_RULE)) & (stage1["Scope"].eq("full_validation"))
    ].iloc[0].to_dict()

    print(f"[stage2] selected stage-1 rule: {SELECTED_STAGE1_RULE}", flush=True)
    stage2 = _stage2_oof(merged, folds, features_77)
    stage2.to_parquet(OUTPUT_STAGE2_OOF, index=False)

    eval_frame = stage2[stage2["stage2_eval_available"]].copy()
    baseline_eval = _baseline_rows(eval_frame, scope_name="stage2_eval_scope")
    baseline = pd.concat([baseline_full, baseline_eval], ignore_index=True)
    baseline.to_csv(OUTPUT_BASELINE, index=False)
    stage1.to_csv(OUTPUT_STAGE1, index=False)

    equal_signal = _equal_signal_comparison(eval_frame)
    equal_precision = _equal_precision_comparison(eval_frame)
    frontier = _precision_recall_frontier(eval_frame)
    fold_stability = _stability(eval_frame, frontier, "fold")
    year_stability = _stability(eval_frame, frontier, "year")
    bootstrap = _bootstrap(eval_frame, equal_signal)
    decision = _decision(equal_signal, equal_precision, bootstrap)

    equal_signal.to_csv(OUTPUT_EQUAL_SIGNAL, index=False)
    equal_precision.to_csv(OUTPUT_EQUAL_PRECISION, index=False)
    frontier.to_csv(OUTPUT_FRONTIER, index=False)
    fold_stability.to_csv(OUTPUT_FOLD, index=False)
    year_stability.to_csv(OUTPUT_YEAR, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            baseline=baseline,
            selected_stage1=selected_stage1,
            stage1=stage1,
            equal_signal=equal_signal,
            equal_precision=equal_precision,
            frontier=frontier,
            fold_stability=fold_stability,
            year_stability=year_stability,
            bootstrap=bootstrap,
            decision=decision,
            features_26=features_26,
            features_77=features_77,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "baseline_precision_recall_audit": str(OUTPUT_BASELINE),
                "stage1_candidate_analysis": str(OUTPUT_STAGE1),
                "stage2_oof_predictions": str(OUTPUT_STAGE2_OOF),
                "stage2_equal_signal_comparison": str(OUTPUT_EQUAL_SIGNAL),
                "stage2_equal_precision_comparison": str(OUTPUT_EQUAL_PRECISION),
                "stage2_precision_recall_frontier": str(OUTPUT_FRONTIER),
                "stage2_fold_stability": str(OUTPUT_FOLD),
                "stage2_year_stability": str(OUTPUT_YEAR),
                "stage2_bootstrap": str(OUTPUT_BOOTSTRAP),
                "stage2_report": str(OUTPUT_REPORT),
                "decision": decision,
                "final_test_used": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )


def _load_current_oof() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    frame = frame[frame["Scheme"].eq("EXPANDING_BASELINE")].copy()
    frame = frame[frame["Date"] < FINAL_TEST_START].copy()
    required = [
        "Date",
        "symbol",
        "fold",
        "p_underperform",
        "p_neutral",
        "p_outperform",
        "classifier_percentile",
        "binary_buy_raw_probability",
        "binary_buy_sigmoid_probability",
        "binary_buy_percentile",
        "buy_signal",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssertionError(f"Current OOF missing required columns: {missing}")
    return frame[required].copy()


def _merge_features_and_oof(
    common: pd.DataFrame, current: pd.DataFrame, features_77: list[str]
) -> pd.DataFrame:
    base_cols = [
        "Date",
        "symbol",
        "Sector",
        "MarketCapCategory",
        "future_stock_return",
        "future_nifty_return",
        "excess_return",
        "buy_target",
        "market_regime_label",
        *features_77,
    ]
    merged = common[base_cols].merge(current, on=["Date", "symbol"], how="inner")
    if len(merged) != len(current):
        raise AssertionError(f"OOF merge changed row count: {len(current)} -> {len(merged)}")
    merged["year"] = merged["Date"].dt.year
    merged["mean_percentile_score"] = (
        merged["classifier_percentile"] + merged["binary_buy_percentile"]
    ) / 2
    merged["current_buy_signal"] = (
        (merged["classifier_percentile"] >= 0.99)
        & (merged["binary_buy_percentile"] >= 0.995)
    )
    for name, _ in STAGE1_RULES:
        merged[f"candidate_{name}"] = _candidate_mask(merged, name)
    return merged.sort_values(["Date", "symbol"]).reset_index(drop=True)


def _candidate_mask(frame: pd.DataFrame, rule: str) -> pd.Series:
    if rule == "xgb_top3":
        return frame["classifier_percentile"] >= 0.97
    if rule == "xgb_top5":
        return frame["classifier_percentile"] >= 0.95
    if rule == "xgb_top10":
        return frame["classifier_percentile"] >= 0.90
    if rule == "binary_top3":
        return frame["binary_buy_percentile"] >= 0.97
    if rule == "binary_top5":
        return frame["binary_buy_percentile"] >= 0.95
    if rule == "binary_top10":
        return frame["binary_buy_percentile"] >= 0.90
    if rule == "xgb_top5_or_binary_top5":
        return (frame["classifier_percentile"] >= 0.95) | (frame["binary_buy_percentile"] >= 0.95)
    if rule == "xgb_top5_and_binary_top10":
        return (frame["classifier_percentile"] >= 0.95) & (frame["binary_buy_percentile"] >= 0.90)
    if rule == "mean_percentile_top5":
        return frame.groupby("Date")["mean_percentile_score"].rank(pct=True) >= 0.95
    raise ValueError(rule)


def _assert_oof_chronology(frame: pd.DataFrame, folds: list[dict[str, Any]]) -> None:
    if not frame["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows present in OOF frame.")
    by_fold = frame.groupby("fold")["Date"].agg(["min", "max"])
    for fold in folds:
        fold_id = fold["fold"]
        if fold_id not in by_fold.index:
            continue
        row = by_fold.loc[fold_id]
        if row["min"] < fold["validation_start"] or row["max"] > fold["validation_end"]:
            raise AssertionError(f"Fold {fold_id} OOF dates do not match validation dates.")


def _baseline_rows(frame: pd.DataFrame, *, scope_name: str) -> pd.DataFrame:
    rows = []
    rows.append(
        _classification_row(
            frame,
            frame["current_buy_signal"],
            score=frame["p_outperform"],
            model="current_xgb26_binary77",
            rule="XGB26 Top1 AND Binary77 Top0.5",
            scope=scope_name,
            scope_value="all",
        )
    )
    for fold, group in frame.groupby("fold"):
        rows.append(
            _classification_row(
                group,
                group["current_buy_signal"],
                score=group["p_outperform"],
                model="current_xgb26_binary77",
                rule="XGB26 Top1 AND Binary77 Top0.5",
                scope=f"{scope_name}_fold",
                scope_value=fold,
            )
        )
    return pd.DataFrame(rows)


def _stage1_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_positives = int(frame["buy_target"].sum())
    for name, _ in STAGE1_RULES:
        mask = frame[f"candidate_{name}"]
        selected = frame[mask]
        rows.append(
            {
                "Rule": name,
                "Scope": "full_validation",
                "CandidateCount": int(mask.sum()),
                "CandidateShare": _safe_float(mask.mean()),
                "PositiveRate": _safe_float(selected["buy_target"].mean()),
                "RecallOfAllWinners": _safe_float(selected["buy_target"].sum() / total_positives)
                if total_positives
                else None,
                "Precision": _safe_float(selected["buy_target"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _stage2_oof(
    frame: pd.DataFrame, folds: list[dict[str, Any]], features_77: list[str]
) -> pd.DataFrame:
    stage_features = [
        "p_underperform",
        "p_neutral",
        "p_outperform",
        "classifier_percentile",
        "binary_buy_raw_probability",
        "binary_buy_sigmoid_probability",
        "binary_buy_percentile",
        "mean_percentile_score",
        *features_77,
    ]
    outputs = []
    for fold in folds:
        fold_id = int(fold["fold"])
        validation = frame[frame["fold"].eq(fold_id)].copy()
        validation["stage2_eval_available"] = False
        validation["stage2_candidate"] = validation[f"candidate_{SELECTED_STAGE1_RULE}"]
        validation["stage2_probability"] = 0.0
        if fold_id == min(item["fold"] for item in folds):
            outputs.append(validation)
            continue
        train = frame[
            (frame["fold"] < fold_id) & frame[f"candidate_{SELECTED_STAGE1_RULE}"]
        ].copy()
        validation_candidates = validation[validation["stage2_candidate"]].copy()
        if train.empty or validation_candidates.empty:
            outputs.append(validation)
            continue
        model, medians = _fit_classifier(
            train,
            stage_features,
            target="buy_target",
            model_name="xgboost",
            binary=True,
        )
        probabilities = _positive_probability(
            model, _transform_with_medians(validation_candidates, stage_features, medians)
        )
        validation.loc[validation_candidates.index, "stage2_probability"] = probabilities
        validation["stage2_eval_available"] = True
        outputs.append(validation)
    result = pd.concat(outputs, ignore_index=True)
    result["stage2_percentile"] = result.groupby("Date")["stage2_probability"].rank(pct=True)
    return result


def _classification_row(
    frame: pd.DataFrame,
    buy_mask: pd.Series,
    *,
    score: pd.Series,
    model: str,
    rule: str,
    scope: str,
    scope_value: Any,
) -> dict[str, Any]:
    y_true = frame["buy_target"].astype(int)
    y_pred = buy_mask.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "Model": model,
        "Rule": rule,
        "Scope": scope,
        "ScopeValue": scope_value,
        "Rows": int(len(frame)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": _safe_float(precision),
        "Recall": _safe_float(recall),
        "F1": _safe_float(_fbeta(precision, recall, beta=1.0)),
        "F0_5": _safe_float(_fbeta(precision, recall, beta=0.5)),
        "PRAUC": _safe_float(average_precision_score(y_true, score)) if len(frame) else None,
        "BuySignals": int(buy_mask.sum()),
        "AverageExcessReturn": _safe_float(frame.loc[buy_mask, "excess_return"].mean())
        if buy_mask.any()
        else None,
        "MedianExcessReturn": _safe_float(frame.loc[buy_mask, "excess_return"].median())
        if buy_mask.any()
        else None,
        "NiftyWinRate": _safe_float((frame.loc[buy_mask, "excess_return"] > 0).mean())
        if buy_mask.any()
        else None,
    }


def _equal_signal_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    baseline_mask = frame["current_buy_signal"]
    n = int(baseline_mask.sum())
    stage2_mask = _top_n_mask(frame["stage2_probability"], n)
    return pd.DataFrame(
        [
            _classification_row(
                frame,
                baseline_mask,
                score=frame["p_outperform"],
                model="baseline",
                rule="equal_signal_baseline",
                scope="stage2_eval_scope",
                scope_value="all",
            ),
            _classification_row(
                frame,
                stage2_mask,
                score=frame["stage2_probability"],
                model="two_stage",
                rule=f"{SELECTED_STAGE1_RULE} + Stage2 top {n}",
                scope="stage2_eval_scope",
                scope_value="all",
            ),
        ]
    )


def _equal_precision_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = _classification_row(
        frame,
        frame["current_buy_signal"],
        score=frame["p_outperform"],
        model="baseline",
        rule="equal_precision_baseline",
        scope="stage2_eval_scope",
        scope_value="all",
    )
    target_precision = float(baseline["Precision"] or 0.0)
    stage2_mask = _largest_prefix_at_precision(frame, "stage2_probability", target_precision)
    stage2 = _classification_row(
        frame,
        stage2_mask,
        score=frame["stage2_probability"],
        model="two_stage",
        rule=f"{SELECTED_STAGE1_RULE} + max signals at baseline precision",
        scope="stage2_eval_scope",
        scope_value="all",
    )
    stage2["TargetPrecision"] = target_precision
    return pd.DataFrame([baseline | {"TargetPrecision": target_precision}, stage2])


def _precision_recall_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for minimum_precision in MIN_PRECISION_LEVELS:
        mask = _largest_prefix_at_precision(frame, "stage2_probability", minimum_precision)
        row = _classification_row(
            frame,
            mask,
            score=frame["stage2_probability"],
            model="two_stage",
            rule=f"minimum_precision_{minimum_precision:.2f}",
            scope="stage2_eval_scope",
            scope_value="all",
        )
        row["MinimumPrecision"] = minimum_precision
        row["LargestSectorShare"] = _largest_share(frame[mask], "Sector")
        rows.append(row)
    return pd.DataFrame(rows)


def _stability(frame: pd.DataFrame, frontier: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for _, frontier_row in frontier.iterrows():
        count = int(frontier_row["BuySignals"])
        selected_keys = set(_keys(frame[_top_n_mask(frame["stage2_probability"], count)]))
        for value, group in frame.groupby(group_column):
            mask = pd.Series(_keys(group)).isin(selected_keys).to_numpy()
            row = _classification_row(
                group,
                pd.Series(mask, index=group.index),
                score=group["stage2_probability"],
                model="two_stage",
                rule=str(frontier_row["Rule"]),
                scope=group_column,
                scope_value=value,
            )
            row["MinimumPrecision"] = frontier_row["MinimumPrecision"]
            rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(frame: pd.DataFrame, equal_signal: pd.DataFrame) -> pd.DataFrame:
    n = int(equal_signal[equal_signal["Model"].eq("baseline")]["BuySignals"].iloc[0])
    baseline = frame[frame["current_buy_signal"]].copy()
    stage2 = frame[_top_n_mask(frame["stage2_probability"], n)].copy()
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(baseline["Date"]) | set(stage2["Date"])))
    base_by_date = {date: group for date, group in baseline.groupby("Date")}
    stage_by_date = {date: group for date, group in stage2.groupby("Date")}
    diffs = {"Precision": [], "Recall": [], "AverageExcessReturn": [], "MedianExcessReturn": [], "NiftyWinRate": []}
    total_positives = int(frame["buy_target"].sum())
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        base_sample = pd.concat(
            [base_by_date[d] for d in sample_dates if d in base_by_date], ignore_index=True
        )
        stage_sample = pd.concat(
            [stage_by_date[d] for d in sample_dates if d in stage_by_date], ignore_index=True
        )
        base_metrics = _selected_metrics(base_sample, total_positives)
        stage_metrics = _selected_metrics(stage_sample, total_positives)
        for metric in diffs:
            diffs[metric].append(stage_metrics[metric] - base_metrics[metric])
    rows = []
    for metric, values in diffs.items():
        arr = np.array(values, dtype=float)
        rows.append(
            {
                "Comparison": "two_stage_equal_signal_minus_baseline",
                "Metric": metric,
                "MeanDiff": _safe_float(np.nanmean(arr)),
                "CI95Low": _safe_float(np.nanquantile(arr, 0.025)),
                "CI95High": _safe_float(np.nanquantile(arr, 0.975)),
                "Samples": BOOTSTRAP_SAMPLES,
            }
        )
    return pd.DataFrame(rows)


def _selected_metrics(selected: pd.DataFrame, total_positives: int) -> dict[str, float]:
    if selected.empty:
        return {
            "Precision": 0.0,
            "Recall": 0.0,
            "AverageExcessReturn": np.nan,
            "MedianExcessReturn": np.nan,
            "NiftyWinRate": np.nan,
        }
    return {
        "Precision": float(selected["buy_target"].mean()),
        "Recall": float(selected["buy_target"].sum() / total_positives) if total_positives else 0.0,
        "AverageExcessReturn": float(selected["excess_return"].mean()),
        "MedianExcessReturn": float(selected["excess_return"].median()),
        "NiftyWinRate": float((selected["excess_return"] > 0).mean()),
    }


def _top_n_mask(score: pd.Series, n: int) -> pd.Series:
    mask = pd.Series(False, index=score.index)
    if n <= 0:
        return mask
    top_index = score.sort_values(ascending=False).head(n).index
    mask.loc[top_index] = True
    return mask


def _largest_prefix_at_precision(
    frame: pd.DataFrame, score_column: str, minimum_precision: float
) -> pd.Series:
    ranked = frame.sort_values(score_column, ascending=False).copy()
    positives = ranked["buy_target"].astype(int).cumsum()
    counts = np.arange(1, len(ranked) + 1)
    precision = positives / counts
    valid = np.flatnonzero(precision >= minimum_precision)
    if len(valid) == 0:
        return pd.Series(False, index=frame.index)
    n = int(valid[-1] + 1)
    return _top_n_mask(frame[score_column], n)


def _keys(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    return list(zip(frame["Date"], frame["symbol"], strict=False))


def _largest_share(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return _safe_float(frame[column].value_counts(normalize=True).iloc[0])


def _fbeta(precision: float, recall: float, *, beta: float) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def _decision(
    equal_signal: pd.DataFrame, equal_precision: pd.DataFrame, bootstrap: pd.DataFrame
) -> str:
    base_equal = equal_signal[equal_signal["Model"].eq("baseline")].iloc[0]
    stage_equal = equal_signal[equal_signal["Model"].eq("two_stage")].iloc[0]
    base_precision = float(base_equal["Precision"])
    base_recall = float(base_equal["Recall"])
    stage_precision = float(stage_equal["Precision"])
    stage_recall = float(stage_equal["Recall"])
    eq_precision_stage = equal_precision[equal_precision["Model"].eq("two_stage")].iloc[0]
    robust_precision = _bootstrap_low(bootstrap, "Precision") > 0
    robust_recall = _bootstrap_low(bootstrap, "Recall") > 0
    robust_excess = _bootstrap_low(bootstrap, "AverageExcessReturn") >= 0
    precision_improved = stage_precision > base_precision and robust_precision and robust_excess
    recall_improved = (
        stage_recall > base_recall and robust_recall and robust_excess
    ) or (
        float(eq_precision_stage["Precision"]) >= base_precision
        and float(eq_precision_stage["Recall"]) > base_recall
    )
    if precision_improved and recall_improved:
        return "TWO_STAGE_MODEL_IMPROVES_PRECISION_AND_RECALL"
    if precision_improved:
        return "TWO_STAGE_MODEL_IMPROVES_PRECISION_ONLY"
    if recall_improved:
        return "TWO_STAGE_MODEL_IMPROVES_RECALL_ONLY"
    return "NO_ROBUST_IMPROVEMENT"


def _bootstrap_low(bootstrap: pd.DataFrame, metric: str) -> float:
    row = bootstrap[bootstrap["Metric"].eq(metric)]
    if row.empty:
        return float("-inf")
    return float(row["CI95Low"].iloc[0])


def _report(
    *,
    baseline: pd.DataFrame,
    selected_stage1: dict[str, Any],
    stage1: pd.DataFrame,
    equal_signal: pd.DataFrame,
    equal_precision: pd.DataFrame,
    frontier: pd.DataFrame,
    fold_stability: pd.DataFrame,
    year_stability: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: str,
    features_26: list[str],
    features_77: list[str],
) -> str:
    return "\n".join(
        [
            "Validation-only Two-stage BUY Model Research",
            "",
            f"Decision: {decision}",
            "Final test used: False",
            "V1/V2 production artifacts modified: False",
            "Fundamentals used: False",
            "Stage-1 scores for Stage-2 training: strict prior OOF folds only",
            "Stage-2 fold 1 omitted from Stage-2 comparison because no prior OOF fold exists.",
            "",
            f"Reference XGB26 feature count: {len(features_26)}",
            f"Stage-2 technical/market feature count: {len(features_77)} + OOF score features",
            f"Selected Stage-1 rule: {SELECTED_STAGE1_RULE}",
            json.dumps(_json_safe(selected_stage1), indent=2),
            "",
            "Baseline:",
            baseline.to_string(index=False),
            "",
            "Stage-1 candidate analysis:",
            stage1.to_string(index=False),
            "",
            "Equal signal comparison:",
            equal_signal.to_string(index=False),
            "",
            "Equal precision comparison:",
            equal_precision.to_string(index=False),
            "",
            "Precision-recall frontier:",
            frontier.to_string(index=False),
            "",
            "Bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "Fold stability:",
            fold_stability.to_string(index=False),
            "",
            "Year stability:",
            year_stability.to_string(index=False),
            "",
            "Leakage checks:",
            "- Dataset restricted to Date < 2025-01-01.",
            "- Existing XGB26/Binary77 inputs are strict walk-forward OOF validation scores.",
            "- Stage-2 training uses only candidate rows from previous OOF folds.",
            "- Same clean adjusted-price dataset and original 150-stock universe are used.",
            "- No fundamentals, no random split, no final-test rows.",
        ]
    )


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    main()
