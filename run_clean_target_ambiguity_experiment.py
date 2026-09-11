"""Validation-only cleaner-target ambiguity-zone experiment.

Training rows with +2% < future excess <= +5% are excluded from the experimental
training target only. Validation success remains future excess > +5%.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import _fit_classifier, _positive_probability, _transform_with_medians
from run_stock_specific_feature_variant import _common_evaluation_rows, _frozen_features, _prepare_dataset
from run_v2_backward_forward_selection import CLEAN_DEMERGER_DATASET
from run_v2_model_family_comparison import _selected_26_features

CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"

OUTPUT_COUNTS = REPORT_DIR / "ambiguity_training_counts.csv"
OUTPUT_OOF = REPORT_DIR / "clean_target_oof_predictions.parquet"
OUTPUT_TOPK = REPORT_DIR / "clean_target_topk_results.csv"
OUTPUT_BUY = REPORT_DIR / "clean_target_buy_comparison.csv"
OUTPUT_FOLD = REPORT_DIR / "clean_target_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "clean_target_year_stability.csv"
OUTPUT_SECTOR = REPORT_DIR / "clean_target_sector_audit.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "clean_target_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "clean_target_report.txt"

K_VALUES = (1, 3, 5, 10)
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 42


def main() -> None:
    """Run cleaner-target validation-only experiment."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into clean-target experiment.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    baseline = _load_baseline_oof()
    counts, clean_oof = _clean_target_oof(common, baseline, folds, features_26, features_77)
    clean_oof.to_parquet(OUTPUT_OOF, index=False)

    daily = _daily_topk(clean_oof)
    topk = _aggregate_topk(daily)
    buy = _strict_buy_comparison(clean_oof)
    fold = _scoped_topk(clean_oof, "fold")
    year = _scoped_topk(clean_oof, "year")
    sector = _sector_audit(clean_oof)
    bootstrap = _bootstrap(daily, clean_oof)
    decision = _decision(topk, buy, bootstrap)

    counts.to_csv(OUTPUT_COUNTS, index=False)
    topk.to_csv(OUTPUT_TOPK, index=False)
    buy.to_csv(OUTPUT_BUY, index=False)
    fold.to_csv(OUTPUT_FOLD, index=False)
    year.to_csv(OUTPUT_YEAR, index=False)
    sector.to_csv(OUTPUT_SECTOR, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            counts=counts,
            topk=topk,
            buy=buy,
            fold=fold,
            year=year,
            sector=sector,
            bootstrap=bootstrap,
            clean_oof=clean_oof,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ambiguity_training_counts": str(OUTPUT_COUNTS),
                "clean_target_oof_predictions": str(OUTPUT_OOF),
                "clean_target_topk_results": str(OUTPUT_TOPK),
                "clean_target_buy_comparison": str(OUTPUT_BUY),
                "clean_target_fold_stability": str(OUTPUT_FOLD),
                "clean_target_year_stability": str(OUTPUT_YEAR),
                "clean_target_sector_audit": str(OUTPUT_SECTOR),
                "clean_target_bootstrap": str(OUTPUT_BOOTSTRAP),
                "clean_target_report": str(OUTPUT_REPORT),
                "decision": decision,
                "final_test_used": False,
                "production_artifacts_modified": False,
                "validation_target_changed": False,
            },
            indent=2,
        )
    )


def _load_baseline_oof() -> pd.DataFrame:
    frame = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    frame = frame[frame["Scheme"].eq("EXPANDING_BASELINE")].copy()
    frame = frame[frame["Date"] < FINAL_TEST_START].copy()
    frame["baseline_combined_score"] = (
        frame["classifier_percentile"] + frame["binary_buy_percentile"]
    ) / 2
    frame["baseline_buy_signal"] = (
        (frame["classifier_percentile"] >= 0.99)
        & (frame["binary_buy_percentile"] >= 0.995)
    )
    return frame


def _clean_target_oof(
    common: pd.DataFrame,
    baseline: pd.DataFrame,
    folds: list[dict[str, Any]],
    features_26: list[str],
    features_77: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    counts = []
    key = ["Date", "symbol"]
    for fold in folds:
        train = common[common["Date"].isin(fold["train_dates"])].copy()
        validation = common[common["Date"].isin(fold["validation_dates"])].copy()
        clean_train = train[(train["excess_return"] <= 0.02) | (train["excess_return"] > 0.05)].copy()
        clean_train["clean_buy_target"] = (clean_train["excess_return"] > 0.05).astype(int)
        counts.append(_count_row(train, clean_train, fold["fold"]))

        xgb_model, xgb_medians = _fit_classifier(
            clean_train,
            features_26,
            target="clean_buy_target",
            model_name="xgboost",
            binary=True,
        )
        binary_model, binary_medians = _fit_classifier(
            clean_train,
            features_77,
            target="clean_buy_target",
            model_name="xgboost",
            binary=True,
        )
        output = validation[
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
        output["fold"] = fold["fold"]
        output["year"] = output["Date"].dt.year
        output["clean_xgb26_probability"] = _positive_probability(
            xgb_model, _transform_with_medians(validation, features_26, xgb_medians)
        )
        output["clean_binary77_probability"] = _positive_probability(
            binary_model, _transform_with_medians(validation, features_77, binary_medians)
        )
        output["clean_xgb26_percentile"] = output.groupby("Date")[
            "clean_xgb26_probability"
        ].rank(pct=True)
        output["clean_binary77_percentile"] = output.groupby("Date")[
            "clean_binary77_probability"
        ].rank(pct=True)
        output["clean_combined_percentile"] = (
            output["clean_xgb26_percentile"] + output["clean_binary77_percentile"]
        ) / 2
        frames.append(output)

    oof = pd.concat(frames, ignore_index=True)
    merged = oof.merge(
        baseline[
            [
                "Date",
                "symbol",
                "p_outperform",
                "classifier_percentile",
                "binary_buy_sigmoid_probability",
                "binary_buy_percentile",
                "baseline_combined_score",
                "baseline_buy_signal",
            ]
        ],
        on=key,
        how="inner",
    )
    if len(merged) != len(oof):
        raise AssertionError(f"Baseline merge changed OOF count: {len(oof)} -> {len(merged)}")
    merged["clean_buy_signal"] = (
        (merged["clean_xgb26_percentile"] >= 0.99)
        & (merged["clean_binary77_percentile"] >= 0.995)
    )
    return pd.DataFrame(counts), merged.sort_values(["Date", "symbol"]).reset_index(drop=True)


def _count_row(train: pd.DataFrame, clean_train: pd.DataFrame, fold: int) -> dict[str, Any]:
    ambiguous = train[(train["excess_return"] > 0.02) & (train["excess_return"] <= 0.05)]
    return {
        "fold": fold,
        "TrainingRowsBefore": int(len(train)),
        "AmbiguousRowsRemoved": int(len(ambiguous)),
        "AmbiguousPctRemoved": _safe_float(len(ambiguous) / len(train)) if len(train) else None,
        "TrainingRowsAfter": int(len(clean_train)),
        "PositiveRateBefore": _safe_float((train["excess_return"] > 0.05).mean()),
        "PositiveRateAfter": _safe_float(clean_train["clean_buy_target"].mean()),
        "TrainStart": train["Date"].min(),
        "TrainEnd": train["Date"].max(),
    }


def _daily_topk(frame: pd.DataFrame) -> pd.DataFrame:
    specs = (
        ("baseline_xgb26", "p_outperform", "Baseline XGB26"),
        ("baseline_binary77", "binary_buy_sigmoid_probability", "Baseline Binary77"),
        ("baseline_combined", "baseline_combined_score", "Baseline 50/50 percentile"),
        ("clean_xgb26", "clean_xgb26_probability", "Cleaner-target XGB26"),
        ("clean_binary77", "clean_binary77_probability", "Cleaner-target Binary77"),
        ("clean_combined", "clean_combined_percentile", "Cleaner-target 50/50 percentile"),
    )
    rows = []
    for (date, fold, year, regime), group in frame.groupby(
        ["Date", "fold", "year", "market_regime_label"], sort=True
    ):
        winners = int(group["buy_target"].sum())
        for score_name, score_column, label in specs:
            ranked = group.sort_values(score_column, ascending=False)
            for k in K_VALUES:
                selected = ranked.head(k)
                found = int(selected["buy_target"].sum())
                rows.append(
                    {
                        "Date": date,
                        "fold": fold,
                        "year": year,
                        "market_regime_label": regime,
                        "ScoreName": score_name,
                        "ScoreLabel": label,
                        "K": k,
                        "ActualWinnersOnDate": winners,
                        "PrecisionAtK": found / k,
                        "RecallAtK": found / winners if winners else np.nan,
                        "WinnerCoverageAtK": found > 0,
                        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
                        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
                        "AverageFutureStockReturn": _safe_float(
                            selected["future_stock_return"].mean()
                        ),
                        "Selections": int(len(selected)),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_topk(daily: pd.DataFrame, scope: str = "overall", scope_value: Any = "all") -> pd.DataFrame:
    rows = []
    for (score_name, k), group in daily.groupby(["ScoreName", "K"], sort=False):
        recall_group = group[group["ActualWinnersOnDate"] > 0]
        rows.append(
            {
                "Scope": scope,
                "ScopeValue": scope_value,
                "ScoreName": score_name,
                "K": int(k),
                "Dates": int(group["Date"].nunique()),
                "PrecisionAtK": _safe_float(group["PrecisionAtK"].mean()),
                "RecallAtK": _safe_float(recall_group["RecallAtK"].mean()),
                "WinnerCoverageAtK": _safe_float(group["WinnerCoverageAtK"].mean()),
                "AverageExcessReturn": _safe_float(group["AverageExcessReturn"].mean()),
                "MedianExcessReturn": _safe_float(group["MedianExcessReturn"].median()),
                "NiftyWinRate": _safe_float(group["NiftyWinRate"].mean()),
                "AverageFutureStockReturn": _safe_float(group["AverageFutureStockReturn"].mean()),
                "TotalSelections": int(group["Selections"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _strict_buy_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, column in (
        ("baseline_strict_buy", "baseline_buy_signal"),
        ("clean_target_strict_buy", "clean_buy_signal"),
    ):
        selected = frame[frame[column]].copy()
        coverage = (
            selected.groupby("Date")["buy_target"].sum().reindex(
                frame["Date"].drop_duplicates(), fill_value=0
            )
            > 0
        )
        rows.append(
            {
                "Signal": label,
                "Rows": int(len(frame)),
                "SignalCount": int(len(selected)),
                "SignalDates": int(selected["Date"].nunique()),
                "Precision": _safe_float(selected["buy_target"].mean()),
                "GlobalRecall": _safe_float(selected["buy_target"].sum() / frame["buy_target"].sum()),
                "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
                "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
                "WinnerCoverage": _safe_float(coverage.mean()),
                "LargestSectorShare": _largest_share(selected, "Sector"),
                "LargestStockShare": _largest_share(selected, "symbol"),
            }
        )
    return pd.DataFrame(rows)


def _scoped_topk(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column):
        rows.append(_aggregate_topk(_daily_topk(group), scope=column, scope_value=value))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _sector_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = (
        ("baseline_xgb26", "p_outperform"),
        ("baseline_combined", "baseline_combined_score"),
        ("clean_xgb26", "clean_xgb26_probability"),
        ("clean_combined", "clean_combined_percentile"),
    )
    for score_name, score_column in specs:
        for k in (3, 5):
            selected = (
                frame.sort_values(["Date", score_column], ascending=[True, False])
                .groupby("Date", group_keys=False)
                .head(k)
            )
            total = len(selected)
            for sector, group in selected.groupby("Sector"):
                rows.append(
                    {
                        "ScoreName": score_name,
                        "K": k,
                        "Sector": sector,
                        "Selections": int(len(group)),
                        "SelectionShare": _safe_float(len(group) / total) if total else None,
                        "Precision": _safe_float(group["buy_target"].mean()),
                        "AverageExcessReturn": _safe_float(group["excess_return"].mean()),
                        "MedianExcessReturn": _safe_float(group["excess_return"].median()),
                        "NiftyWinRate": _safe_float((group["excess_return"] > 0).mean()),
                        "UniqueStocks": int(group["symbol"].nunique()),
                        "LargestStockShare": _largest_share(group, "symbol"),
                    }
                )
    return pd.DataFrame(rows)


def _bootstrap(daily: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    pairs = (
        ("top3_precision", "clean_xgb26", "baseline_xgb26", 3, "PrecisionAtK"),
        ("top5_precision", "clean_combined", "baseline_combined", 5, "PrecisionAtK"),
        ("top5_winner_coverage", "clean_combined", "baseline_combined", 5, "WinnerCoverageAtK"),
        ("top5_avg_excess", "clean_combined", "baseline_combined", 5, "AverageExcessReturn"),
    )
    for name, clean_score, base_score, k, metric in pairs:
        rows.append(_bootstrap_daily_pair(daily, name, clean_score, base_score, k, metric, rng))
    rows.append(_bootstrap_strict_buy(frame, rng))
    return pd.DataFrame(rows)


def _bootstrap_daily_pair(
    daily: pd.DataFrame,
    name: str,
    clean_score: str,
    base_score: str,
    k: int,
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    clean = daily[(daily["ScoreName"].eq(clean_score)) & (daily["K"].eq(k))].sort_values("Date")
    base = daily[(daily["ScoreName"].eq(base_score)) & (daily["K"].eq(k))].sort_values("Date")
    if clean["Date"].reset_index(drop=True).tolist() != base["Date"].reset_index(drop=True).tolist():
        raise AssertionError("Clean/baseline bootstrap dates are misaligned.")
    diff = clean[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
    samples = []
    n = len(diff)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        samples.append(float(np.nanmean(diff[idx])))
    arr = np.array(samples)
    return {
        "Comparison": name,
        "CleanScore": clean_score,
        "BaselineScore": base_score,
        "K": k,
        "Metric": metric,
        "MeanDiff": _safe_float(arr.mean()),
        "CI95Low": _safe_float(np.quantile(arr, 0.025)),
        "CI95High": _safe_float(np.quantile(arr, 0.975)),
        "Samples": BOOTSTRAP_SAMPLES,
    }


def _bootstrap_strict_buy(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    dates = frame["Date"].drop_duplicates().sort_values().to_numpy()
    base_by_date = frame[frame["baseline_buy_signal"]].groupby("Date")["buy_target"].agg(["sum", "count"])
    clean_by_date = frame[frame["clean_buy_signal"]].groupby("Date")["buy_target"].agg(["sum", "count"])
    base_tp = base_by_date["sum"].reindex(dates, fill_value=0).to_numpy(dtype=float)
    base_n = base_by_date["count"].reindex(dates, fill_value=0).to_numpy(dtype=float)
    clean_tp = clean_by_date["sum"].reindex(dates, fill_value=0).to_numpy(dtype=float)
    clean_n = clean_by_date["count"].reindex(dates, fill_value=0).to_numpy(dtype=float)
    samples = []
    n = len(dates)
    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, size=n)
        base_precision = base_tp[idx].sum() / base_n[idx].sum() if base_n[idx].sum() else np.nan
        clean_precision = clean_tp[idx].sum() / clean_n[idx].sum() if clean_n[idx].sum() else np.nan
        samples.append(clean_precision - base_precision)
    arr = np.array(samples, dtype=float)
    return {
        "Comparison": "strict_buy_precision",
        "CleanScore": "clean_target_strict_buy",
        "BaselineScore": "baseline_strict_buy",
        "K": None,
        "Metric": "Precision",
        "MeanDiff": _safe_float(np.nanmean(arr)),
        "CI95Low": _safe_float(np.nanquantile(arr, 0.025)),
        "CI95High": _safe_float(np.nanquantile(arr, 0.975)),
        "Samples": BOOTSTRAP_SAMPLES,
    }


def _decision(topk: pd.DataFrame, buy: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    clean_top5 = _metric(topk, "clean_combined", 5)
    base_top5 = _metric(topk, "baseline_combined", 5)
    clean_top3 = _metric(topk, "clean_xgb26", 3)
    base_top3 = _metric(topk, "baseline_xgb26", 3)
    base_buy = buy[buy["Signal"].eq("baseline_strict_buy")].iloc[0]
    clean_buy = buy[buy["Signal"].eq("clean_target_strict_buy")].iloc[0]
    topk_improved = (
        clean_top5["PrecisionAtK"] > base_top5["PrecisionAtK"]
        and clean_top5["WinnerCoverageAtK"] >= base_top5["WinnerCoverageAtK"] - 0.03
        and clean_top3["PrecisionAtK"] >= base_top3["PrecisionAtK"]
        and _bootstrap_low(bootstrap, "top5_precision") > 0
    )
    strict_buy_improved = (
        clean_buy["Precision"] > base_buy["Precision"]
        and clean_buy["SignalCount"] >= 50
        and _bootstrap_low(bootstrap, "strict_buy_precision") > 0
    )
    if topk_improved:
        return "CLEAN_TARGET_IMPROVES_TOPK"
    if strict_buy_improved:
        return "CLEAN_TARGET_IMPROVES_STRICT_BUY_ONLY"
    if (
        clean_top5["PrecisionAtK"] > base_top5["PrecisionAtK"]
        or clean_buy["Precision"] > base_buy["Precision"]
    ):
        return "CLEAN_TARGET_MIXED"
    return "NO_ROBUST_IMPROVEMENT"


def _metric(topk: pd.DataFrame, score_name: str, k: int) -> pd.Series:
    return topk[(topk["ScoreName"].eq(score_name)) & (topk["K"].eq(k))].iloc[0]


def _bootstrap_low(bootstrap: pd.DataFrame, comparison: str) -> float:
    row = bootstrap[bootstrap["Comparison"].eq(comparison)]
    if row.empty:
        return float("-inf")
    return float(row["CI95Low"].iloc[0])


def _report(
    *,
    counts: pd.DataFrame,
    topk: pd.DataFrame,
    buy: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    sector: pd.DataFrame,
    bootstrap: pd.DataFrame,
    clean_oof: pd.DataFrame,
    decision: str,
) -> str:
    diagnostics = _diagnostics(clean_oof)
    return "\n".join(
        [
            "Validation-only Cleaner-target / Ambiguity-zone Experiment",
            "",
            f"Decision: {decision}",
            "Final test used: False",
            "Production artifacts modified: False",
            "Validation target changed: False",
            "Ambiguous +2% to +5% rows removed from training only.",
            "",
            "Training counts:",
            counts.to_string(index=False),
            "",
            "TopK results:",
            topk.to_string(index=False),
            "",
            "Strict BUY comparison:",
            buy.to_string(index=False),
            "",
            "Bootstrap:",
            bootstrap.to_string(index=False),
            "",
            "Diagnostics:",
            diagnostics.to_string(index=False),
            "",
            "Fold stability:",
            fold.to_string(index=False),
            "",
            "Year stability:",
            year.to_string(index=False),
            "",
            "Sector audit sample:",
            sector.sort_values(["ScoreName", "K", "SelectionShare"], ascending=[True, True, False])
            .head(30)
            .to_string(index=False),
        ]
    )


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = (
        ("xgb_probability", "clean_xgb26_probability", "p_outperform"),
        ("binary_probability", "clean_binary77_probability", "binary_buy_sigmoid_probability"),
    )
    for label, clean_col, base_col in pairs:
        rows.append(
            {
                "Diagnostic": label,
                "PearsonCorrelation": _safe_float(frame[clean_col].corr(frame[base_col], method="pearson")),
                "SpearmanCorrelation": _safe_float(frame[clean_col].corr(frame[base_col], method="spearman")),
            }
        )
    for score, clean_col, base_col in (
        ("xgb_top3", "clean_xgb26_probability", "p_outperform"),
        ("combined_top5", "clean_combined_percentile", "baseline_combined_score"),
    ):
        clean_selected = _topk_keys(frame, clean_col, 3 if score == "xgb_top3" else 5)
        base_selected = _topk_keys(frame, base_col, 3 if score == "xgb_top3" else 5)
        intersection = len(clean_selected & base_selected)
        union = len(clean_selected | base_selected)
        rows.append(
            {
                "Diagnostic": f"{score}_overlap",
                "PearsonCorrelation": None,
                "SpearmanCorrelation": None,
                "CleanSelections": len(clean_selected),
                "BaselineSelections": len(base_selected),
                "Intersection": intersection,
                "Jaccard": _safe_float(intersection / union) if union else None,
            }
        )
    return pd.DataFrame(rows)


def _topk_keys(frame: pd.DataFrame, score_column: str, k: int) -> set[tuple[pd.Timestamp, str]]:
    selected = (
        frame.sort_values(["Date", score_column], ascending=[True, False])
        .groupby("Date", group_keys=False)
        .head(k)
    )
    return set(zip(selected["Date"], selected["symbol"], strict=False))


def _largest_share(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty:
        return None
    return _safe_float(frame[column].value_counts(normalize=True).iloc[0])


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


if __name__ == "__main__":
    main()
