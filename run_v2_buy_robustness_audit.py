"""Validation-only robustness audit for frozen V2 high-precision BUY candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
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
from run_v2_55_binary_buy_research import (
    CLEAN_DEMERGER_DATASET,
    Architecture,
    _assert_feature_counts,
    _binary_buy_77_oof,
    _classifier_oof,
    _merge_all_oof,
    _ranker_77_oof,
    _remove_market_and_sector,
    _remove_market_wide,
    _rule_name,
    _select_agreement,
)

BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260811
SMALL_SAMPLE_COUNT = 30

OUTPUT_BOOTSTRAP = REPORT_DIR / "v2_buy_block_bootstrap.csv"
OUTPUT_FOLD = REPORT_DIR / "v2_buy_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "v2_buy_year_stability.csv"
OUTPUT_REGIME = REPORT_DIR / "v2_buy_regime_stability.csv"
OUTPUT_SECTOR = REPORT_DIR / "v2_buy_sector_stability.csv"
OUTPUT_SENSITIVITY = REPORT_DIR / "v2_buy_threshold_sensitivity.csv"
OUTPUT_REPORT = REPORT_DIR / "v2_vs_v1_robustness_report.txt"


@dataclass(frozen=True, slots=True)
class CandidateRule:
    """Frozen candidate BUY rule for robustness testing."""

    name: str
    display_name: str
    classifier_top: float
    binary_top: float


FROZEN_RULES = (
    CandidateRule("extreme_buy", "Rule 1 - Extreme BUY", 0.005, 0.005),
    CandidateRule("practical_buy", "Rule 2 - Practical BUY", 0.010, 0.005),
)

SENSITIVITY_CLASSIFIER_TOPS = (0.0075, 0.0100, 0.0125)
SENSITIVITY_BINARY_TOPS = (0.0040, 0.0050, 0.0060)


def main() -> None:
    """Run robustness audit using validation/out-of-fold predictions only."""

    ensure_probability_dirs()
    merged, architectures = _build_oof_predictions()
    base_rate = float(merged.drop_duplicates(["Date", "symbol", "fold"])["buy_target"].mean())

    bootstrap = _paired_block_bootstrap(merged, architectures, base_rate)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)

    fold = _stability_by_scope(merged, architectures, base_rate, scope="fold")
    fold.to_csv(OUTPUT_FOLD, index=False)

    year = _year_stability(merged, architectures, base_rate)
    year.to_csv(OUTPUT_YEAR, index=False)

    regime = _stability_by_scope(merged, architectures, base_rate, scope="market_regime_label")
    regime.to_csv(OUTPUT_REGIME, index=False)

    sector = _sector_stability(merged, architectures, base_rate)
    sector.to_csv(OUTPUT_SECTOR, index=False)

    sensitivity = _threshold_sensitivity(merged, architectures, base_rate)
    sensitivity.to_csv(OUTPUT_SENSITIVITY, index=False)

    OUTPUT_REPORT.write_text(
        _report(bootstrap, fold, year, regime, sector, sensitivity, base_rate),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "v2_buy_block_bootstrap": str(OUTPUT_BOOTSTRAP),
                "v2_buy_fold_stability": str(OUTPUT_FOLD),
                "v2_buy_year_stability": str(OUTPUT_YEAR),
                "v2_buy_regime_stability": str(OUTPUT_REGIME),
                "v2_buy_sector_stability": str(OUTPUT_SECTOR),
                "v2_buy_threshold_sensitivity": str(OUTPUT_SENSITIVITY),
                "v2_vs_v1_robustness_report": str(OUTPUT_REPORT),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "final_test_status": "not_used",
                "production_v1_modified": False,
            },
            indent=2,
        )
    )


def _build_oof_predictions() -> tuple[pd.DataFrame, list[Architecture]]:
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_77 = _frozen_features()
    features_55 = _remove_market_wide(features_77)
    features_49 = _remove_market_and_sector(features_77)
    _assert_feature_counts(features_77, features_55, features_49)
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    feature_sets = {
        "v1_77": FeatureSet("v1_77", "V1 77-feature classifier", features_77),
        "v2_b_55": FeatureSet("v2_b_55", "V2-B 55-feature classifier", features_55),
    }
    architectures = [
        Architecture(
            "v1_77_plus_binary77",
            "V1: 77 classifier + 77 Binary BUY",
            feature_sets["v1_77"],
        ),
        Architecture(
            "v2_b_55_plus_binary77",
            "V2-B: 55 classifier + 77 Binary BUY",
            feature_sets["v2_b_55"],
        ),
    ]
    classifier_frames = []
    for feature_set in feature_sets.values():
        print(f"[classifier-oof] {feature_set.name}", flush=True)
        classifier_frames.append(_classifier_oof(common, folds, feature_set))
    classifiers = pd.concat(classifier_frames, ignore_index=True)

    print("[binary-buy-oof] 77-feature Binary BUY", flush=True)
    binary = _binary_buy_77_oof(common, folds, feature_sets["v1_77"])
    print("[ranker-oof] 77-feature XGBRanker diagnostic", flush=True)
    ranker = _ranker_77_oof(common, folds, feature_sets["v1_77"])
    return _merge_all_oof(classifiers, binary, ranker), architectures


def _paired_block_bootstrap(
    merged: pd.DataFrame, architectures: list[Architecture], base_rate: float
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for rule in FROZEN_RULES:
        v1 = _select_for_architecture(merged, architectures[0], rule)
        v2 = _select_for_architecture(merged, architectures[1], rule)
        dates = np.array(sorted(set(v1["Date"].unique()).union(set(v2["Date"].unique()))))
        if len(dates) == 0:
            continue
        v1_by_date = {date: group for date, group in v1.groupby("Date")}
        v2_by_date = {date: group for date, group in v2.groupby("Date")}
        samples = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled_dates = rng.choice(dates, size=len(dates), replace=True)
            sample_v1 = _concat_sampled_dates(v1_by_date, sampled_dates)
            sample_v2 = _concat_sampled_dates(v2_by_date, sampled_dates)
            row = {"Rule": rule.name, "RuleName": rule.display_name}
            row.update(
                {f"V1_{key}": value for key, value in _metrics(sample_v1, base_rate).items()}
            )
            row.update(
                {f"V2_{key}": value for key, value in _metrics(sample_v2, base_rate).items()}
            )
            samples.append(row)
        sample_frame = pd.DataFrame(samples)
        point_v1 = _metrics(v1, base_rate)
        point_v2 = _metrics(v2, base_rate)
        for metric in [
            "Precision",
            "AverageExcessReturn",
            "MedianExcessReturn",
            "NiftyWinRate",
        ]:
            diff = sample_frame[f"V2_{metric}"] - sample_frame[f"V1_{metric}"]
            rows.append(
                {
                    "Rule": rule.name,
                    "RuleName": rule.display_name,
                    "Metric": metric,
                    "V1PointEstimate": point_v1.get(metric),
                    "V1CiLow": sample_frame[f"V1_{metric}"].quantile(0.025),
                    "V1CiHigh": sample_frame[f"V1_{metric}"].quantile(0.975),
                    "V2PointEstimate": point_v2.get(metric),
                    "V2CiLow": sample_frame[f"V2_{metric}"].quantile(0.025),
                    "V2CiHigh": sample_frame[f"V2_{metric}"].quantile(0.975),
                    "DifferencePointEstimate": point_v2.get(metric) - point_v1.get(metric),
                    "DifferenceCiLow": diff.quantile(0.025),
                    "DifferenceCiHigh": diff.quantile(0.975),
                    "StatisticallyDistinguishableFromZero": bool(
                        diff.quantile(0.025) > 0 or diff.quantile(0.975) < 0
                    ),
                    "BootstrapSamples": BOOTSTRAP_SAMPLES,
                    "BootstrapMethod": "date-block paired bootstrap",
                    "BaseRate": base_rate,
                }
            )
    return pd.DataFrame(rows)


def _concat_sampled_dates(
    groups_by_date: dict[pd.Timestamp, pd.DataFrame], sampled_dates: np.ndarray
) -> pd.DataFrame:
    frames = [groups_by_date[date] for date in sampled_dates if date in groups_by_date]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _stability_by_scope(
    merged: pd.DataFrame,
    architectures: list[Architecture],
    base_rate: float,
    *,
    scope: str,
) -> pd.DataFrame:
    rows = []
    for architecture in architectures:
        for rule in FROZEN_RULES:
            selected = _select_for_architecture(merged, architecture, rule)
            for value, group in selected.groupby(scope):
                metrics = _metrics(group, base_rate)
                rows.append(
                    {
                        "Architecture": architecture.name,
                        "ArchitectureName": architecture.display_name,
                        "Rule": rule.name,
                        "RuleName": rule.display_name,
                        "Scope": scope,
                        "ScopeValue": value,
                        **metrics,
                        **_stability_flags(metrics, base_rate),
                    }
                )
    return pd.DataFrame(rows)


def _year_stability(
    merged: pd.DataFrame, architectures: list[Architecture], base_rate: float
) -> pd.DataFrame:
    rows = []
    for architecture in architectures:
        for rule in FROZEN_RULES:
            selected = _select_for_architecture(merged, architecture, rule).copy()
            selected["Year"] = selected["Date"].dt.year
            yearly_rows = []
            for year, group in selected.groupby("Year"):
                metrics = _metrics(group, base_rate)
                yearly_rows.append(
                    {
                        "Architecture": architecture.name,
                        "ArchitectureName": architecture.display_name,
                        "Rule": rule.name,
                        "RuleName": rule.display_name,
                        "Year": int(year),
                        **metrics,
                        **_stability_flags(metrics, base_rate),
                    }
                )
            rows.extend(_add_year_summary(yearly_rows))
    return pd.DataFrame(rows)


def _add_year_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    data_rows = [row for row in rows if row.get("Count", 0) > 0]
    profitable = sum(row.get("AverageExcessReturn", 0) > 0 for row in data_rows)
    positive_median = sum(row.get("MedianExcessReturn", 0) > 0 for row in data_rows)
    worst = min(data_rows, key=lambda row: row.get("AverageExcessReturn", float("inf")))
    best = max(data_rows, key=lambda row: row.get("AverageExcessReturn", float("-inf")))
    for row in rows:
        row["ProfitableYears"] = profitable
        row["PositiveMedianExcessYears"] = positive_median
        row["WorstYear"] = worst["Year"]
        row["WorstYearAverageExcessReturn"] = worst["AverageExcessReturn"]
        row["BestYear"] = best["Year"]
        row["BestYearAverageExcessReturn"] = best["AverageExcessReturn"]
    return rows


def _sector_stability(
    merged: pd.DataFrame, architectures: list[Architecture], base_rate: float
) -> pd.DataFrame:
    rows = []
    for architecture in architectures:
        for rule in FROZEN_RULES:
            selected = _select_for_architecture(merged, architecture, rule)
            total = len(selected)
            sector_groups = list(selected.groupby("Sector"))
            strongest_sector = _strongest_sector(sector_groups)
            largest_sector = _largest_sector(sector_groups)
            for sector, group in sector_groups:
                metrics = _metrics(group, base_rate)
                rows.append(
                    {
                        "Architecture": architecture.name,
                        "ArchitectureName": architecture.display_name,
                        "Rule": rule.name,
                        "RuleName": rule.display_name,
                        "Scope": "sector",
                        "Sector": sector,
                        "LargestSector": largest_sector,
                        "LargestSectorSignalShare": (
                            _safe_float(selected["Sector"].eq(largest_sector).mean())
                            if total
                            else None
                        ),
                        "StrongestPerformingSector": strongest_sector,
                        **metrics,
                        **_stability_flags(metrics, base_rate),
                    }
                )
            if strongest_sector:
                excluded = selected[~selected["Sector"].eq(strongest_sector)]
                rows.append(
                    {
                        "Architecture": architecture.name,
                        "ArchitectureName": architecture.display_name,
                        "Rule": rule.name,
                        "RuleName": rule.display_name,
                        "Scope": "excluding_strongest_sector",
                        "Sector": f"EX_{strongest_sector}",
                        "LargestSector": largest_sector,
                        "LargestSectorSignalShare": (
                            _safe_float(selected["Sector"].eq(largest_sector).mean())
                            if total
                            else None
                        ),
                        "StrongestPerformingSector": strongest_sector,
                        **_metrics(excluded, base_rate),
                    }
                )
            rows.append(
                {
                    "Architecture": architecture.name,
                    "ArchitectureName": architecture.display_name,
                    "Rule": rule.name,
                    "RuleName": rule.display_name,
                    "Scope": "date_concentration",
                    "Sector": "ALL",
                    **_date_concentration_metrics(selected, base_rate),
                }
            )
    return pd.DataFrame(rows)


def _strongest_sector(sector_groups: list[tuple[str, pd.DataFrame]]) -> str | None:
    eligible = [(sector, group) for sector, group in sector_groups if len(group) >= 10]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[1]["excess_return"].mean())[0]


def _largest_sector(sector_groups: list[tuple[str, pd.DataFrame]]) -> str | None:
    if not sector_groups:
        return None
    return max(sector_groups, key=lambda item: len(item[1]))[0]


def _date_concentration_metrics(frame: pd.DataFrame, base_rate: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "UniqueSignalDates": 0,
            "AverageSignalsPerSignalDate": None,
            "MaxSignalsOneDate": None,
            "DateLevelPrecision": None,
            "DateLevelAverageExcessReturn": None,
            "DateLevelMedianExcessReturn": None,
            "DateLevelNiftyWinRate": None,
        }
    per_date = (
        frame.groupby("Date")
        .agg(
            Signals=("symbol", "size"),
            Precision=("buy_target", "mean"),
            AverageExcessReturn=("excess_return", "mean"),
            MedianExcessReturn=("excess_return", "median"),
            NiftyWinRate=("excess_return", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    return {
        "BaseRate": base_rate,
        "UniqueSignalDates": int(len(per_date)),
        "AverageSignalsPerSignalDate": _safe_float(per_date["Signals"].mean()),
        "MaxSignalsOneDate": int(per_date["Signals"].max()),
        "DateLevelPrecision": _safe_float(per_date["Precision"].mean()),
        "DateLevelAverageExcessReturn": _safe_float(per_date["AverageExcessReturn"].mean()),
        "DateLevelMedianExcessReturn": _safe_float(per_date["MedianExcessReturn"].mean()),
        "DateLevelNiftyWinRate": _safe_float(per_date["NiftyWinRate"].mean()),
    }


def _threshold_sensitivity(
    merged: pd.DataFrame, architectures: list[Architecture], base_rate: float
) -> pd.DataFrame:
    rows = []
    for architecture in architectures:
        frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
        for classifier_top in SENSITIVITY_CLASSIFIER_TOPS:
            for binary_top in SENSITIVITY_BINARY_TOPS:
                selected = _select_agreement(frame, classifier_top, binary_top)
                rows.append(
                    {
                        "Architecture": architecture.name,
                        "ArchitectureName": architecture.display_name,
                        "ClassifierTop": classifier_top,
                        "BinaryTop": binary_top,
                        "Rule": _rule_name(classifier_top, binary_top),
                        **_metrics(selected, base_rate),
                    }
                )
    return pd.DataFrame(rows)


def _select_for_architecture(
    merged: pd.DataFrame, architecture: Architecture, rule: CandidateRule
) -> pd.DataFrame:
    frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
    return _select_agreement(frame, rule.classifier_top, rule.binary_top)


def _metrics(frame: pd.DataFrame, base_rate: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "Count": 0,
            "Precision": None,
            "BaseRate": base_rate,
            "PrecisionLift": None,
            "AverageFutureStockReturn": None,
            "MedianFutureStockReturn": None,
            "AverageExcessReturn": None,
            "MedianExcessReturn": None,
            "NiftyWinRate": None,
        }
    precision = float(frame["buy_target"].mean())
    return {
        "Count": int(len(frame)),
        "Precision": _safe_float(precision),
        "BaseRate": base_rate,
        "PrecisionLift": _safe_float(precision / base_rate) if base_rate else None,
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _stability_flags(metrics: dict[str, Any], base_rate: float) -> dict[str, Any]:
    flags = []
    if metrics.get("Count", 0) < SMALL_SAMPLE_COUNT:
        flags.append("SMALL_SAMPLE")
    precision = metrics.get("Precision")
    if precision is not None and precision <= base_rate:
        flags.append("PRECISION_NOT_ABOVE_BASE_RATE")
    if (metrics.get("AverageExcessReturn") or 0) <= 0:
        flags.append("AVG_EXCESS_NON_POSITIVE")
    if (metrics.get("MedianExcessReturn") or 0) <= 0:
        flags.append("MEDIAN_EXCESS_NON_POSITIVE")
    return {"Flags": ";".join(flags) if flags else "OK"}


def _report(
    bootstrap: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    regime: pd.DataFrame,
    sector: pd.DataFrame,
    sensitivity: pd.DataFrame,
    base_rate: float,
) -> str:
    v2_b_boot = bootstrap[
        bootstrap["Metric"].isin(["Precision", "AverageExcessReturn", "MedianExcessReturn"])
    ].copy()
    recommendations = _recommendations(bootstrap, fold, year, regime, sector)
    concentration = sector[sector["Scope"].eq("date_concentration")]
    sector_perf = sector[sector["Scope"].eq("sector")]
    strongest_excluded = sector[sector["Scope"].eq("excluding_strongest_sector")]
    return "\n".join(
        [
            "NATIP V2 BUY Robustness Audit",
            "==============================",
            "",
            "Production impact: NONE. Frozen Version 1 production model/rules were not modified.",
            "Final test usage: NOT USED. This audit uses validation/out-of-fold predictions only.",
            f"Date-block bootstrap samples: {BOOTSTRAP_SAMPLES}, seed: {BOOTSTRAP_SEED}",
            f"Overall validation BUY base rate: {base_rate:.4%}",
            "",
            "Bootstrap headline:",
            v2_b_boot.to_string(index=False),
            "",
            "Rule classification:",
            pd.DataFrame(recommendations).to_string(index=False),
            "",
            "Fold stability:",
            fold.to_string(index=False),
            "",
            "Year stability:",
            year.to_string(index=False),
            "",
            "Regime stability:",
            regime.to_string(index=False),
            "",
            "Sector concentration and performance:",
            sector_perf.to_string(index=False),
            "",
            "Performance excluding strongest-performing sector:",
            strongest_excluded.to_string(index=False),
            "",
            "Date clustering / one-observation-per-date aggregation:",
            concentration.to_string(index=False),
            "",
            "Threshold sensitivity around frozen candidate:",
            sensitivity.to_string(index=False),
            "",
            "Interpretation:",
            _interpretation(recommendations),
        ]
    )


def _recommendations(
    bootstrap: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    regime: pd.DataFrame,
    sector: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for rule in FROZEN_RULES:
        boot_rule = bootstrap[bootstrap["Rule"].eq(rule.name)]
        precision_diff = boot_rule[boot_rule["Metric"].eq("Precision")].iloc[0]
        avg_excess_diff = boot_rule[boot_rule["Metric"].eq("AverageExcessReturn")].iloc[0]
        median_excess_diff = boot_rule[boot_rule["Metric"].eq("MedianExcessReturn")].iloc[0]
        v2_fold = fold[
            (fold["Architecture"].eq("v2_b_55_plus_binary77")) & fold["Rule"].eq(rule.name)
        ]
        v2_year = year[
            (year["Architecture"].eq("v2_b_55_plus_binary77")) & year["Rule"].eq(rule.name)
        ]
        v2_regime = regime[
            (regime["Architecture"].eq("v2_b_55_plus_binary77")) & regime["Rule"].eq(rule.name)
        ]
        v2_sector = sector[
            (sector["Architecture"].eq("v2_b_55_plus_binary77"))
            & sector["Rule"].eq(rule.name)
            & sector["Scope"].eq("sector")
        ]
        bad_folds = int((v2_fold["Flags"] != "OK").sum()) if not v2_fold.empty else 999
        bad_years = int((v2_year["Flags"] != "OK").sum()) if not v2_year.empty else 999
        bad_regimes = int((v2_regime["Flags"] != "OK").sum()) if not v2_regime.empty else 999
        largest_sector_share = (
            float(v2_sector["LargestSectorSignalShare"].max()) if not v2_sector.empty else 1.0
        )
        statistically_better = bool(
            precision_diff["DifferenceCiLow"] > 0
            and avg_excess_diff["DifferenceCiLow"] > 0
            and median_excess_diff["DifferenceCiLow"] > 0
        )
        economically_positive = bool(
            (v2_fold["AverageExcessReturn"] > 0).all()
            and (v2_year["AverageExcessReturn"] > 0).all()
            and (v2_regime["AverageExcessReturn"] > 0).all()
        )
        if statistically_better and economically_positive and largest_sector_share < 0.40:
            classification = "ROBUST"
        elif (
            economically_positive
            and bad_folds <= 1
            and bad_years <= 1
            and largest_sector_share < 0.50
        ):
            classification = "PROMISING"
        elif economically_positive:
            classification = "FRAGILE"
        else:
            classification = "REJECT"
        rows.append(
            {
                "Rule": rule.name,
                "RuleName": rule.display_name,
                "Classification": classification,
                "PrecisionDiffCiLow": precision_diff["DifferenceCiLow"],
                "PrecisionDiffCiHigh": precision_diff["DifferenceCiHigh"],
                "AverageExcessDiffCiLow": avg_excess_diff["DifferenceCiLow"],
                "AverageExcessDiffCiHigh": avg_excess_diff["DifferenceCiHigh"],
                "MedianExcessDiffCiLow": median_excess_diff["DifferenceCiLow"],
                "MedianExcessDiffCiHigh": median_excess_diff["DifferenceCiHigh"],
                "BadFolds": bad_folds,
                "BadYears": bad_years,
                "BadRegimes": bad_regimes,
                "LargestSectorShare": largest_sector_share,
            }
        )
    return rows


def _interpretation(recommendations: list[dict[str, Any]]) -> str:
    practical = next(row for row in recommendations if row["Rule"] == "practical_buy")
    extreme = next(row for row in recommendations if row["Rule"] == "extreme_buy")
    if practical["Classification"] in {"ROBUST", "PROMISING"} and extreme["Classification"] in {
        "ROBUST",
        "PROMISING",
    }:
        return (
            "Both frozen V2 rules remain economically positive on validation. Prefer the Practical "
            "BUY rule when precision is statistically similar because it has the larger sample."
        )
    if practical["Classification"] in {"ROBUST", "PROMISING"}:
        return "Prefer Practical BUY; Extreme BUY is less stable or too sample-constrained."
    if extreme["Classification"] in {"ROBUST", "PROMISING"}:
        return "Extreme BUY is stronger than Practical BUY, but sample-size risk remains important."
    return "Do not promote V2 BUY until the flagged robustness issues are reviewed."


if __name__ == "__main__":
    main()
