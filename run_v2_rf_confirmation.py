"""Focused RF confirmation test for the V2 26-feature BUY architecture."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_stock_specific_feature_variant import (
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _prepare_dataset,
    _safe_float,
)
from run_v2_55_binary_buy_research import _binary_buy_77_oof
from run_v2_backward_forward_selection import (
    BINARY_TOP,
    CLASSIFIER_TOP,
    CLEAN_DEMERGER_DATASET,
)
from run_v2_model_family_comparison import (
    PrimaryModel,
    _primary_oof,
    _selected_26_features,
    _top_keys,
)

RF_TOPS = (0.20, 0.10, 0.05, 0.02)
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 42
BENCHMARK_PRECISION = 0.5149
BENCHMARK_AVG_EXCESS = 0.0514
BENCHMARK_MEDIAN_EXCESS = 0.0537
BENCHMARK_NIFTY_WIN = 0.6966

OUTPUT_CONFIRMATION = REPORT_DIR / "v2_rf_confirmation.csv"
OUTPUT_DISAGREEMENT = REPORT_DIR / "v2_rf_disagreement_analysis.csv"
OUTPUT_DIVERSITY = REPORT_DIR / "v2_xgb_rf_diversity.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "v2_rf_confirmation_bootstrap.csv"
OUTPUT_DECISION = REPORT_DIR / "v2_rf_decision.json"
OUTPUT_REPORT = REPORT_DIR / "v2_rf_confirmation_report.txt"


def main() -> None:
    """Run RF confirmation analysis without final-test data."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into RF confirmation research.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    if len(features_26) != 26:
        raise AssertionError(f"Expected 26 compact features, got {len(features_26)}")

    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    binary = _binary_buy_77_oof(
        common,
        folds,
        FeatureSet("binary_77", "Unchanged 77-feature Binary XGBoost BUY", features_77),
    )
    base_rate = float(common["buy_target"].mean())

    print("[primary-oof] 26-feature XGBoost", flush=True)
    xgb = _primary_oof(
        common,
        folds,
        PrimaryModel("xgboost", "26-feature XGBoost", features_26),
        model_name="xgboost",
    )
    print("[primary-oof] 26-feature Random Forest", flush=True)
    rf = _primary_oof(
        common,
        folds,
        PrimaryModel("random_forest", "26-feature Random Forest", features_26),
        model_name="random_forest",
    )

    merged = _merge_predictions(xgb, rf, binary)
    baseline = _baseline_buy(merged)
    confirmation = _confirmation_report(merged, baseline, base_rate)
    confirmation.to_csv(OUTPUT_CONFIRMATION, index=False)
    disagreement = _disagreement_report(merged, baseline, base_rate)
    disagreement.to_csv(OUTPUT_DISAGREEMENT, index=False)
    diversity = _diversity_report(xgb, rf)
    diversity.to_csv(OUTPUT_DIVERSITY, index=False)
    bootstrap = _bootstrap_report(merged, confirmation)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    decision = _decision_payload(confirmation, disagreement, diversity, bootstrap)
    OUTPUT_DECISION.write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(
        _summary(confirmation, disagreement, diversity, bootstrap, decision),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_rf_confirmation": str(OUTPUT_CONFIRMATION),
                "v2_rf_disagreement_analysis": str(OUTPUT_DISAGREEMENT),
                "v2_xgb_rf_diversity": str(OUTPUT_DIVERSITY),
                "v2_rf_confirmation_bootstrap": str(OUTPUT_BOOTSTRAP),
                "v2_rf_decision": str(OUTPUT_DECISION),
                "v2_rf_confirmation_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "frozen_buy_thresholds_modified": False,
            },
            indent=2,
        )
    )


def _merge_predictions(xgb: pd.DataFrame, rf: pd.DataFrame, binary: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    base = xgb.rename(
        columns={
            "p_outperform": "xgb_p_outperform",
            "p_outperform_percentile": "xgb_percentile",
        }
    )
    rf_part = rf[key + ["p_outperform", "p_outperform_percentile"]].rename(
        columns={
            "p_outperform": "rf_p_outperform",
            "p_outperform_percentile": "rf_percentile",
        }
    )
    merged = base.merge(rf_part, on=key, how="inner").merge(
        binary[key + ["binary_buy_percentile"]],
        on=key,
        how="inner",
    )
    if len(merged) != len(xgb):
        raise AssertionError("OOF merge changed row count.")
    return merged


def _baseline_buy(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["xgb_percentile"] >= 1 - CLASSIFIER_TOP)
        & (frame["binary_buy_percentile"] >= 1 - BINARY_TOP)
    ].copy()


def _rf_confirmed(frame: pd.DataFrame, rf_top: float) -> pd.DataFrame:
    return frame[frame["rf_percentile"] >= 1 - rf_top].copy()


def _confirmation_report(
    merged: pd.DataFrame, baseline: pd.DataFrame, base_rate: float
) -> pd.DataFrame:
    rows = [
        _metric_row(
            baseline,
            "baseline_xgb_top1_binary_top0.5",
            "overall",
            "all",
            base_rate,
            rf_top=None,
        )
    ]
    for rf_top in RF_TOPS:
        selected = _rf_confirmed(baseline, rf_top)
        overall = _metric_row(
            selected,
            f"xgb_top1_binary_top0.5_rf_top{_pct(rf_top)}",
            "overall",
            "all",
            base_rate,
            rf_top=rf_top,
        )
        fold_rows = [
            _metric_row(
                _rf_confirmed(group, rf_top),
                overall["Rule"],
                "fold",
                fold,
                base_rate,
                rf_top=rf_top,
            )
            for fold, group in baseline.groupby("fold")
        ]
        year_rows = [
            _metric_row(
                _rf_confirmed(group, rf_top),
                overall["Rule"],
                "year",
                int(year),
                base_rate,
                rf_top=rf_top,
            )
            for year, group in baseline.groupby(baseline["Date"].dt.year)
        ]
        fold_frame = pd.DataFrame(fold_rows)
        year_frame = pd.DataFrame(year_rows)
        overall.update(_worst_stability(fold_frame, "Fold"))
        overall.update(_worst_stability(year_frame, "Year"))
        rows.append(overall)
        rows.extend(fold_rows)
        rows.extend(year_rows)
    return pd.DataFrame(rows)


def _disagreement_report(
    merged: pd.DataFrame, baseline: pd.DataFrame, base_rate: float
) -> pd.DataFrame:
    rows = []
    for rf_top in RF_TOPS:
        confirmed = _rf_confirmed(baseline, rf_top)
        confirmed_keys = set(zip(confirmed["Date"], confirmed["symbol"], strict=False))
        not_confirmed = baseline[
            ~baseline.apply(lambda row: (row["Date"], row["symbol"]) in confirmed_keys, axis=1)
        ].copy()
        rows.append(
            _metric_row(
                confirmed,
                f"rf_top{_pct(rf_top)}",
                "group",
                "RF confirms",
                base_rate,
                rf_top=rf_top,
            )
        )
        rows.append(
            _metric_row(
                not_confirmed,
                f"rf_top{_pct(rf_top)}",
                "group",
                "RF does not confirm",
                base_rate,
                rf_top=rf_top,
            )
        )
    return pd.DataFrame(rows)


def _diversity_report(xgb: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = xgb[key + ["p_outperform", "p_outperform_percentile"]].merge(
        rf[key + ["p_outperform", "p_outperform_percentile"]],
        on=key,
        suffixes=("_xgb", "_rf"),
        how="inner",
    )
    rows = []
    for fraction in (0.01, 0.05):
        xgb_top = _top_keys(xgb, "p_outperform", fraction)
        rf_top = _top_keys(rf, "p_outperform", fraction)
        intersection = xgb_top & rf_top
        union = xgb_top | rf_top
        rows.append(
            {
                "TopFraction": fraction,
                "TopLabel": f"Top{_pct(fraction)}",
                "Rows": int(len(merged)),
                "PearsonProbabilityCorrelation": _safe_float(
                    merged["p_outperform_xgb"].corr(merged["p_outperform_rf"], method="pearson")
                ),
                "SpearmanProbabilityCorrelation": _safe_float(
                    merged["p_outperform_xgb"].corr(merged["p_outperform_rf"], method="spearman")
                ),
                "PercentileCorrelation": _safe_float(
                    merged["p_outperform_percentile_xgb"].corr(
                        merged["p_outperform_percentile_rf"], method="spearman"
                    )
                ),
                "XGBTopCount": len(xgb_top),
                "RFTopCount": len(rf_top),
                "Overlap": len(intersection),
                "JaccardSimilarity": _safe_float(len(intersection) / len(union)) if union else None,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_report(merged: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    improved = confirmation[
        (confirmation["Scope"].eq("overall"))
        & confirmation["RFTop"].notna()
        & (confirmation["Precision"] > BENCHMARK_PRECISION)
    ].copy()
    if improved.empty:
        return pd.DataFrame(
            [
                {
                    "Rule": "NO_RF_RULE_EXCEEDED_BENCHMARK_POINT_ESTIMATE",
                    "BootstrapSamples": BOOTSTRAP_SAMPLES,
                    "Note": "No RF-confirmed rule had higher precision than 51.49%; bootstrap not required.",
                }
            ]
        )
    baseline = _baseline_buy(merged)
    rows = []
    for item in improved.itertuples(index=False):
        selected = _rf_confirmed(baseline, float(item.RFTop))
        rows.append(_bootstrap_rule(baseline, selected, item.Rule))
    return pd.DataFrame(rows)


def _bootstrap_rule(baseline: pd.DataFrame, selected: pd.DataFrame, rule: str) -> dict[str, Any]:
    rng = (
        pd.Series(range(BOOTSTRAP_SAMPLES))
        .sample(frac=1, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    dates = pd.Series(sorted(baseline["Date"].drop_duplicates()))
    diffs = []
    for sample_index in rng:
        sampled_dates = dates.sample(
            n=len(dates), replace=True, random_state=RANDOM_SEED + int(sample_index)
        )
        base_sample = _sample_by_dates(baseline, sampled_dates)
        selected_sample = _sample_by_dates(selected, sampled_dates)
        diffs.append(_metric_diffs(base_sample, selected_sample))
    diff_frame = pd.DataFrame(diffs)
    row = {"Rule": rule, "BootstrapSamples": BOOTSTRAP_SAMPLES}
    for metric in diff_frame.columns:
        row[f"{metric}Mean"] = _safe_float(diff_frame[metric].mean())
        row[f"{metric}CI025"] = _safe_float(diff_frame[metric].quantile(0.025))
        row[f"{metric}CI975"] = _safe_float(diff_frame[metric].quantile(0.975))
        row[f"{metric}StatisticallyAboveZero"] = bool(diff_frame[metric].quantile(0.025) > 0)
    return row


def _sample_by_dates(frame: pd.DataFrame, sampled_dates: pd.Series) -> pd.DataFrame:
    pieces = [frame[frame["Date"].eq(date)] for date in sampled_dates]
    return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()


def _metric_diffs(baseline: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | None]:
    return {
        "PrecisionDiff": _safe_float(selected["buy_target"].mean() - baseline["buy_target"].mean()),
        "AverageExcessDiff": _safe_float(
            selected["excess_return"].mean() - baseline["excess_return"].mean()
        ),
        "MedianExcessDiff": _safe_float(
            selected["excess_return"].median() - baseline["excess_return"].median()
        ),
        "NiftyWinRateDiff": _safe_float(
            (selected["excess_return"] > 0).mean() - (baseline["excess_return"] > 0).mean()
        ),
    }


def _metric_row(
    frame: pd.DataFrame,
    rule: str,
    scope: str,
    scope_value: Any,
    base_rate: float,
    *,
    rf_top: float | None,
) -> dict[str, Any]:
    precision = _safe_float(frame["buy_target"].mean()) if not frame.empty else None
    return {
        "Rule": rule,
        "RFTop": rf_top,
        "Scope": scope,
        "ScopeValue": scope_value,
        "SignalCount": int(len(frame)),
        "Precision": precision,
        "DeltaPrecisionVsBenchmark": (
            _safe_float(precision - BENCHMARK_PRECISION) if precision is not None else None
        ),
        "BaseRate": base_rate,
        "PrecisionLift": _safe_float(precision / base_rate) if precision is not None else None,
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _worst_stability(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {f"Worst{label}": None, f"Worst{label}Precision": None}
    worst = frame.sort_values(["Precision", "AverageExcessReturn"], ascending=True).iloc[0]
    return {
        f"Worst{label}": worst["ScopeValue"],
        f"Worst{label}Precision": worst["Precision"],
        f"Worst{label}AverageExcessReturn": worst["AverageExcessReturn"],
        f"Worst{label}MedianExcessReturn": worst["MedianExcessReturn"],
    }


def _decision_payload(
    confirmation: pd.DataFrame,
    disagreement: pd.DataFrame,
    diversity: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> dict[str, Any]:
    overall = confirmation[confirmation["Scope"].eq("overall")].copy()
    rf_rules = overall[overall["RFTop"].notna()].sort_values(
        ["Precision", "SignalCount", "MedianExcessReturn"], ascending=False
    )
    best = rf_rules.head(1).to_dict(orient="records")
    best_rule = rf_rules.iloc[0] if not rf_rules.empty else None
    materially_better = (
        best_rule is not None
        and best_rule["Precision"] > BENCHMARK_PRECISION + 0.02
        and best_rule["SignalCount"] >= 250
        and best_rule["AverageExcessReturn"] >= BENCHMARK_AVG_EXCESS - 0.002
        and best_rule["MedianExcessReturn"] >= BENCHMARK_MEDIAN_EXCESS - 0.002
        and best_rule["NiftyWinRate"] >= BENCHMARK_NIFTY_WIN - 0.02
    )
    if materially_better:
        decision = "RF_ADDS_VALUE"
    elif best_rule is not None and best_rule["Precision"] > 0.45:
        decision = "RF_DIAGNOSTIC_ONLY"
    else:
        decision = "RF_REMOVE_FROM_BUY_ARCHITECTURE"
    return {
        "decision": decision,
        "selection_basis": "Validation/out-of-fold only; final test not used.",
        "v1_production_modified": False,
        "frozen_buy_thresholds_modified": False,
        "benchmark": {
            "signals": 435,
            "precision": BENCHMARK_PRECISION,
            "average_excess_return": BENCHMARK_AVG_EXCESS,
            "median_excess_return": BENCHMARK_MEDIAN_EXCESS,
            "nifty_win_rate": BENCHMARK_NIFTY_WIN,
        },
        "best_rf_rule": best,
        "confirmation_overall": overall.to_dict(orient="records"),
        "disagreement": disagreement.to_dict(orient="records"),
        "diversity": diversity.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "recommendation": (
            "Do not add RF to the BUY architecture unless bootstrap/stability show a material improvement."
        ),
    }


def _summary(
    confirmation: pd.DataFrame,
    disagreement: pd.DataFrame,
    diversity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "NATIP V2 RF Confirmation Test",
            "==============================",
            "",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Frozen BUY thresholds modified: NO.",
            "",
            "Confirmation overall:",
            confirmation[confirmation["Scope"].eq("overall")].to_string(index=False),
            "",
            "RF disagreement analysis:",
            disagreement.to_string(index=False),
            "",
            "XGB/RF diversity:",
            diversity.to_string(index=False),
            "",
            "Bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "Decision:",
            json.dumps(_json_safe(decision), indent=2),
        ]
    )


def _pct(value: float) -> str:
    return f"{value * 100:g}%"


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
