"""Audit current+tuned Binary77 agreement as a confidence upgrader."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_stock_specific_feature_variant import (
    _common_evaluation_rows,
    _frozen_features,
    _prepare_dataset,
    _safe_float,
)
from run_v2_backward_forward_selection import BINARY_TOP, CLASSIFIER_TOP, CLEAN_DEMERGER_DATASET
from run_v2_binary_stability_diagnosis import (
    BOOTSTRAP_SAMPLES,
    CURRENT_BINARY_ID,
    RANDOM_SEED,
    TUNED_BINARY_ID,
    _binary_oof_with_audit,
    _date_metric_groups,
    _json_safe,
    _metrics_from_date_groups,
    _primary_baseline_oof,
)
from run_v2_model_family_comparison import _selected_26_features
from run_v2_xgb_hyperparameter_optimization import _binary_specs

OUTPUT_FOLD = REPORT_DIR / "v2_dual_binary_fold_audit.csv"
OUTPUT_YEAR = REPORT_DIR / "v2_dual_binary_year_audit.csv"
OUTPUT_ZERO = REPORT_DIR / "v2_zero_fold_membership.csv"
OUTPUT_SECTOR_REGIME = REPORT_DIR / "v2_dual_binary_sector_regime.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "v2_dual_binary_bootstrap.csv"
OUTPUT_TUNED_ONLY = REPORT_DIR / "v2_tuned_only_analysis.csv"
OUTPUT_DECISION = REPORT_DIR / "v2_dual_binary_decision.json"
OUTPUT_REPORT = REPORT_DIR / "v2_dual_binary_report.txt"


def main() -> None:
    """Run validation-only dual Binary77 agreement audit."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into dual Binary77 audit.")
    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    print("[primary] current XGB26", flush=True)
    primary = _primary_baseline_oof(common, folds, features_26)
    specs = {spec.trial_id: spec for spec in _binary_specs(common)}
    print("[binary] current Binary77", flush=True)
    current_binary, _ = _binary_oof_with_audit(common, folds, features_77, specs[CURRENT_BINARY_ID])
    print("[binary] tuned Binary77 research", flush=True)
    tuned_binary, _ = _binary_oof_with_audit(common, folds, features_77, specs[TUNED_BINARY_ID])

    merged = _merged_scores(primary, current_binary, tuned_binary)
    groups = _membership_groups(merged)
    both = groups["BOTH"]
    current_full = groups["CURRENT_BUY"]
    tuned_only = groups["TUNED_ONLY"]

    fold_audit = _scoped_audit(both, common, "fold")
    fold_audit.to_csv(OUTPUT_FOLD, index=False)
    year_audit = _scoped_audit(both.assign(year=both["Date"].dt.year), common, "year")
    year_audit.to_csv(OUTPUT_YEAR, index=False)
    zero = _zero_fold_membership(merged)
    zero.to_csv(OUTPUT_ZERO, index=False)
    sector_regime = _sector_regime_audit(both)
    sector_regime.to_csv(OUTPUT_SECTOR_REGIME, index=False)
    bootstrap = _bootstrap_both_vs_current(both, current_full)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    tuned_only_analysis = _tuned_only_analysis(tuned_only)
    tuned_only_analysis.to_csv(OUTPUT_TUNED_ONLY, index=False)
    decision = _decision_payload(
        both=both,
        current_full=current_full,
        tuned_only=tuned_only,
        fold_audit=fold_audit,
        year_audit=year_audit,
        sector_regime=sector_regime,
        bootstrap=bootstrap,
        tuned_only_analysis=tuned_only_analysis,
        zero=zero,
    )
    OUTPUT_DECISION.write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(
        _summary(
            fold_audit, year_audit, zero, sector_regime, bootstrap, tuned_only_analysis, decision
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_dual_binary_fold_audit": str(OUTPUT_FOLD),
                "v2_dual_binary_year_audit": str(OUTPUT_YEAR),
                "v2_zero_fold_membership": str(OUTPUT_ZERO),
                "v2_dual_binary_sector_regime": str(OUTPUT_SECTOR_REGIME),
                "v2_dual_binary_bootstrap": str(OUTPUT_BOOTSTRAP),
                "v2_tuned_only_analysis": str(OUTPUT_TUNED_ONLY),
                "v2_dual_binary_decision": str(OUTPUT_DECISION),
                "v2_dual_binary_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "features_or_thresholds_modified": False,
            },
            indent=2,
        )
    )


def _merged_scores(
    primary: pd.DataFrame, current_binary: pd.DataFrame, tuned_binary: pd.DataFrame
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    frame = primary.rename(columns={"p_outperform": "xgb_p_outperform"})[
        [
            "Date",
            "symbol",
            "fold",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
            "xgb_p_outperform",
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
    merged = frame.merge(current, on=key, how="inner").merge(tuned, on=key, how="inner")
    merged["xgb_top1"] = merged["classifier_percentile"] >= 1 - CLASSIFIER_TOP
    merged["current_top0_5"] = merged["current_binary_percentile"] >= 1 - BINARY_TOP
    merged["tuned_top0_5"] = merged["tuned_binary_percentile"] >= 1 - BINARY_TOP
    merged["current_buy"] = merged["xgb_top1"] & merged["current_top0_5"]
    merged["tuned_buy"] = merged["xgb_top1"] & merged["tuned_top0_5"]
    return merged


def _membership_groups(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "CURRENT_BUY": merged[merged["current_buy"]].copy(),
        "BOTH": merged[merged["current_buy"] & merged["tuned_buy"]].copy(),
        "CURRENT_ONLY": merged[merged["current_buy"] & ~merged["tuned_buy"]].copy(),
        "TUNED_ONLY": merged[~merged["current_buy"] & merged["tuned_buy"]].copy(),
    }


def _scoped_audit(frame: pd.DataFrame, common: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(scope):
        if scope == "fold":
            validation = common[common["Date"].isin(group["Date"].unique())]
        else:
            validation = common[common["Date"].dt.year.eq(value)]
        row = _metric_row(group)
        row.update(
            {
                "Scope": scope,
                "ScopeValue": value,
                "BaseRate": _safe_float(validation["buy_target"].mean()),
            }
        )
        row["PrecisionLift"] = (
            _safe_float(row["Precision"] / row["BaseRate"]) if row["BaseRate"] else None
        )
        row["FlagNegativeAvgExcess"] = bool(
            row["AverageExcessReturn"] is not None and row["AverageExcessReturn"] < 0
        )
        row["FlagNegativeMedianExcess"] = bool(
            row["MedianExcessReturn"] is not None and row["MedianExcessReturn"] < 0
        )
        row["FlagPrecisionBelowBaseRate"] = bool(
            row["Precision"] is not None
            and row["BaseRate"] is not None
            and row["Precision"] < row["BaseRate"]
        )
        row["FlagFewerThan10Observations"] = bool(row["Count"] < 10)
        rows.append(row)
    return pd.DataFrame(rows)


def _zero_fold_membership(merged: pd.DataFrame) -> pd.DataFrame:
    fold2_tuned = merged[(merged["fold"].eq(2)) & merged["tuned_buy"]].copy()
    rows = []
    for _, row in fold2_tuned.iterrows():
        if row["current_buy"] and row["tuned_buy"]:
            membership = "BOTH"
        elif row["tuned_buy"]:
            membership = "TUNED_ONLY"
        elif row["current_buy"]:
            membership = "CURRENT_ONLY"
        else:
            membership = "NEITHER"
        rows.append(
            {
                "Ticker": row["symbol"],
                "Date": row["Date"],
                "Sector": row["Sector"],
                "MarketRegime": row["market_regime_label"],
                "Membership": membership,
                "XGBPOutperform": row["xgb_p_outperform"],
                "XGBPercentile": row["classifier_percentile"],
                "CurrentBinaryProbability": row["current_binary_probability"],
                "CurrentBinaryPercentile": row["current_binary_percentile"],
                "TunedBinaryProbability": row["tuned_binary_probability"],
                "TunedBinaryPercentile": row["tuned_binary_percentile"],
                "FutureStockReturn": row["future_stock_return"],
                "FutureExcessReturn": row["excess_return"],
                "BuyTarget": row["buy_target"],
            }
        )
    return pd.DataFrame(rows)


def _sector_regime_audit(both: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, column in (("sector", "Sector"), ("market_regime", "market_regime_label")):
        for value, group in both.groupby(column):
            row = _metric_row(group)
            row.update(
                {
                    "Scope": scope,
                    "ScopeValue": value,
                    "ShareOfBothSignals": len(group) / max(1, len(both)),
                }
            )
            rows.append(row)
    sector_metrics = [row for row in rows if row["Scope"] == "sector"]
    if sector_metrics:
        strongest = max(sector_metrics, key=lambda item: item["AverageExcessReturn"] or -999)
        excluded = both[both["Sector"] != strongest["ScopeValue"]].copy()
        row = _metric_row(excluded)
        row.update(
            {
                "Scope": "sector_exclusion",
                "ScopeValue": f"exclude_{strongest['ScopeValue']}",
                "ShareOfBothSignals": len(excluded) / max(1, len(both)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_both_vs_current(both: pd.DataFrame, current_full: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(both["Date"]).union(set(current_full["Date"]))))
    both_groups = _date_metric_groups(both)
    current_groups = _date_metric_groups(current_full)
    diffs = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        b = _metrics_from_date_groups(both_groups, sampled)
        c = _metrics_from_date_groups(current_groups, sampled)
        diffs.append(
            {
                "PrecisionDiff": b["Precision"] - c["Precision"],
                "AverageExcessDiff": b["AverageExcessReturn"] - c["AverageExcessReturn"],
                "MedianExcessDiff": b["MedianExcessReturn"] - c["MedianExcessReturn"],
                "NiftyWinRateDiff": b["NiftyWinRate"] - c["NiftyWinRate"],
            }
        )
    frame = pd.DataFrame(diffs)
    row = {"Comparison": "BOTH_minus_CURRENT_FULL", "BootstrapSamples": BOOTSTRAP_SAMPLES}
    for column in frame.columns:
        row[f"{column}Mean"] = _safe_float(frame[column].mean())
        row[f"{column}CI025"] = _safe_float(frame[column].quantile(0.025))
        row[f"{column}CI975"] = _safe_float(frame[column].quantile(0.975))
        row[f"{column}StatisticallyAboveZero"] = bool(frame[column].quantile(0.025) > 0)
    return pd.DataFrame([row])


def _tuned_only_analysis(tuned_only: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "Scope": "overall",
            "ScopeValue": "all",
            **_metric_row(tuned_only),
            **_distribution(tuned_only),
        }
    )
    for fold, group in tuned_only.groupby("fold"):
        rows.append(
            {"Scope": "fold", "ScopeValue": fold, **_metric_row(group), **_distribution(group)}
        )
    for year, group in tuned_only.groupby(tuned_only["Date"].dt.year):
        rows.append(
            {"Scope": "year", "ScopeValue": int(year), **_metric_row(group), **_distribution(group)}
        )
    for sector, group in tuned_only.groupby("Sector"):
        rows.append(
            {"Scope": "sector", "ScopeValue": sector, **_metric_row(group), **_distribution(group)}
        )
    return pd.DataFrame(rows)


def _decision_payload(**kwargs: Any) -> dict[str, Any]:
    both = kwargs["both"]
    current = kwargs["current_full"]
    fold = kwargs["fold_audit"]
    bootstrap = kwargs["bootstrap"].iloc[0].to_dict()
    both_metrics = _metric_row(both)
    current_metrics = _metric_row(current)
    enough = bool(len(both) >= 100)
    fold_ok = bool(
        not fold.empty
        and not fold["FlagNegativeAvgExcess"].any()
        and not fold["FlagNegativeMedianExcess"].any()
        and not fold["FlagPrecisionBelowBaseRate"].any()
        and not fold["FlagFewerThan10Observations"].any()
    )
    bootstrap_ok = bool(
        bootstrap.get("PrecisionDiffCI025", -1) > 0
        and bootstrap.get("AverageExcessDiffCI025", -1) > 0
        and bootstrap.get("MedianExcessDiffCI025", -1) > 0
    )
    if enough and fold_ok and bootstrap_ok:
        decision = "TUNED_BINARY_USE_AS_CONFIDENCE_UPGRADER"
    elif enough and both_metrics["Precision"] > current_metrics["Precision"]:
        decision = "TUNED_BINARY_DIAGNOSTIC_ONLY"
    else:
        decision = "TUNED_BINARY_REJECT"
    return {
        "decision": decision,
        "selection_basis": "Validation/OOF only; tuned Binary77 remains research-only.",
        "v1_production_modified": False,
        "features_or_thresholds_modified": False,
        "current_buy_metrics": current_metrics,
        "both_high_confidence_metrics": both_metrics,
        "confidence_tier_supported": decision == "TUNED_BINARY_USE_AS_CONFIDENCE_UPGRADER",
        "interpretation": {
            "current_plus_tuned_confirms": "BUY / HIGH CONFIDENCE",
            "current_only": "BUY / MEDIUM CONFIDENCE",
            "note": "This is a confidence interpretation only; official BUY rule is unchanged.",
        },
        "fold_flags": fold.to_dict(orient="records"),
        "bootstrap": kwargs["bootstrap"].to_dict(orient="records"),
        "largest_sector_share": _largest_share(both, "Sector"),
        "largest_regime_share": _largest_share(both, "market_regime_label"),
    }


def _metric_row(frame: pd.DataFrame) -> dict[str, Any]:
    winners = frame[frame["excess_return"] > 0.05]
    losers = frame[frame["excess_return"] <= 0.05]
    return {
        "Count": int(len(frame)),
        "Precision": _safe_float(frame["buy_target"].mean()),
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
        "WinnerCount": int(len(winners)),
        "LoserCount": int(len(losers)),
        "AverageWinnerExcess": _safe_float(winners["excess_return"].mean()),
        "AverageLoserExcess": _safe_float(losers["excess_return"].mean()),
    }


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "ExcessP10": _safe_float(frame["excess_return"].quantile(0.10)),
        "ExcessP25": _safe_float(frame["excess_return"].quantile(0.25)),
        "ExcessP75": _safe_float(frame["excess_return"].quantile(0.75)),
        "ExcessP90": _safe_float(frame["excess_return"].quantile(0.90)),
        "LargestSectorShare": _largest_share(frame, "Sector"),
    }


def _largest_share(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return _safe_float(frame[column].value_counts(normalize=True).max())


def _summary(*items: Any) -> str:
    fold, year, zero, sector, bootstrap, tuned_only, decision = items
    return "\n".join(
        [
            "NATIP V2 Dual Binary77 Agreement Audit",
            "======================================",
            "",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Features/model parameters/thresholds modified: NO.",
            "",
            "BOTH fold audit:",
            fold.to_string(index=False),
            "",
            "BOTH year audit:",
            year.to_string(index=False),
            "",
            "Zero-fold membership rows:",
            zero.to_string(index=False),
            "",
            "Sector/regime robustness:",
            sector.to_string(index=False),
            "",
            "BOTH vs current bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "TUNED_ONLY analysis:",
            tuned_only.to_string(index=False),
            "",
            "Decision:",
            json.dumps(_json_safe(decision), indent=2),
        ]
    )


if __name__ == "__main__":
    main()
