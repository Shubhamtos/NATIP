"""V5 validation-only percentile recalculation on the Smallcap-expanded universe."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_stock_specific_feature_variant import (
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _prepare_dataset,
    _safe_float,
)
from run_v2_55_binary_buy_research import _binary_buy_77_oof, _classifier_oof

V5_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_v5_smallcap250_vedl_excluded.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"
OUTPUT_OOF = REPORT_DIR / "v5_smallcap250_validation_percentiles.csv"
OUTPUT_BUY_SUMMARY = REPORT_DIR / "v5_smallcap250_buy_rule_recalculation.csv"
OUTPUT_SUMMARY = REPORT_DIR / "v5_smallcap250_percentile_recalculation_summary.txt"

CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005


def main() -> None:
    """Recalculate same-date validation percentiles using the expanded V5 universe."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(V5_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_26 = _selected_26_features()
    features_77 = _frozen_features()
    _assert_features(features_26, features_77, pre_final)

    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    classifier_set = FeatureSet("v5_xgb26", "V5 expanded-universe XGB26 classifier", features_26)
    binary_set = FeatureSet("v5_binary77", "V5 expanded-universe Binary77 BUY", features_77)

    print("[v5-percentiles] classifier OOF", flush=True)
    classifier = _classifier_oof(common, folds, classifier_set)
    print("[v5-percentiles] binary OOF", flush=True)
    binary = _binary_buy_77_oof(common, folds, binary_set)

    key = ["Date", "symbol", "fold"]
    merged = classifier.merge(binary, on=key, how="inner")
    if len(merged) != len(classifier):
        raise AssertionError("V5 classifier/binary OOF merge changed row count.")
    merged["v5_buy_rule_signal"] = (
        (merged["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
        & (merged["binary_buy_percentile"] >= 1 - BINARY_TOP)
    )
    merged.to_csv(OUTPUT_OOF, index=False)

    summary = _buy_summary(merged)
    summary.to_csv(OUTPUT_BUY_SUMMARY, index=False)
    OUTPUT_SUMMARY.write_text(_summary_text(merged, summary, folds), encoding="utf-8")
    print(
        json.dumps(
            {
                "v5_validation_percentiles": str(OUTPUT_OOF),
                "v5_buy_rule_recalculation": str(OUTPUT_BUY_SUMMARY),
                "v5_percentile_summary": str(OUTPUT_SUMMARY),
                "final_test_used": False,
                "v1_v2_modified": False,
            },
            indent=2,
        )
    )


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 selected features, got {len(features)}.")
    return features


def _assert_features(features_26: list[str], features_77: list[str], dataset: pd.DataFrame) -> None:
    missing = [feature for feature in [*features_26, *features_77] if feature not in dataset.columns]
    if missing:
        raise AssertionError(f"V5 dataset missing required model features: {missing}")
    if len(features_77) != 77:
        raise AssertionError(f"Expected 77 Binary BUY features, got {len(features_77)}.")


def _buy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selected = frame[frame["v5_buy_rule_signal"]].copy()
    base_rate = float(frame["buy_target"].mean())
    rows.append(_row("overall", "all", selected, base_rate))
    for fold, group in frame.groupby("fold"):
        rows.append(_row("fold", fold, group[group["v5_buy_rule_signal"]], base_rate))
    for year, group in frame.groupby(frame["Date"].dt.year):
        rows.append(_row("year", int(year), group[group["v5_buy_rule_signal"]], base_rate))
    for regime, group in frame.groupby("market_regime_label"):
        rows.append(_row("market_regime", regime, group[group["v5_buy_rule_signal"]], base_rate))
    return pd.DataFrame(rows)


def _row(scope: str, scope_value: Any, frame: pd.DataFrame, base_rate: float) -> dict[str, Any]:
    precision = _safe_float(frame["buy_target"].mean()) if not frame.empty else None
    return {
        "Scope": scope,
        "ScopeValue": scope_value,
        "ClassifierTop": CLASSIFIER_TOP,
        "BinaryTop": BINARY_TOP,
        "SignalCount": int(len(frame)),
        "Precision": precision,
        "BaseRate": base_rate,
        "PrecisionLift": _safe_float(precision / base_rate) if precision is not None else None,
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean())
        if not frame.empty
        else None,
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median())
        if not frame.empty
        else None,
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()) if not frame.empty else None,
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()) if not frame.empty else None,
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()) if not frame.empty else None,
    }


def _summary_text(frame: pd.DataFrame, summary: pd.DataFrame, folds: list[dict[str, Any]]) -> str:
    overall = summary[(summary["Scope"] == "overall") & (summary["ScopeValue"] == "all")].iloc[0]
    return "\n".join(
        [
            "NATIP V5 Smallcap 250 Percentile Recalculation",
            "",
            "Scope: validation/out-of-fold only",
            f"Validation rows scored: {len(frame)}",
            f"Unique validation symbols: {frame['symbol'].nunique()}",
            f"Walk-forward folds: {len(folds)}",
            f"BUY rule unchanged: XGB26 Top {CLASSIFIER_TOP:.1%} AND Binary77 Top {BINARY_TOP:.1%}",
            "",
            "Expanded-universe BUY rule result:",
            f"Signals: {int(overall.SignalCount)}",
            f"Precision: {_fmt_pct(overall.Precision)}",
            f"Average excess return: {_fmt_pct(overall.AverageExcessReturn)}",
            f"Median excess return: {_fmt_pct(overall.MedianExcessReturn)}",
            f"Nifty win rate: {_fmt_pct(overall.NiftyWinRate)}",
            "",
            "Final-test data was not used. V1/V2 production artifacts were not modified.",
        ]
    )


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.2f}%"


if __name__ == "__main__":
    main()
