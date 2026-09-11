"""Validation-only daily ranking audit for original 150-stock OOF predictions."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START

CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"
TUNED_OOF = REPORT_DIR / "lstm_tuned_binary77_oof_cache.csv"

OUTPUT_DAILY = REPORT_DIR / "daily_ranking_audit.csv"
OUTPUT_PR_AT_K = REPORT_DIR / "precision_recall_at_k.csv"
OUTPUT_COVERAGE = REPORT_DIR / "winner_coverage_at_k.csv"
OUTPUT_FOLD = REPORT_DIR / "ranking_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "ranking_year_stability.csv"
OUTPUT_SECTOR = REPORT_DIR / "ranking_sector_audit.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "ranking_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "ranking_audit_report.txt"

K_VALUES = (1, 3, 5, 10)
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 42

SCORE_SPECS = (
    ("xgb26_probability", "p_outperform", "XGB26 probability"),
    ("binary77_probability", "binary_buy_sigmoid_probability", "Current Binary77 probability"),
    ("combined_percentile", "combined_percentile_score", "0.5 XGB percentile + 0.5 Binary percentile"),
)


def main() -> None:
    """Run the daily ranking audit."""

    ensure_probability_dirs()
    frame = _load_oof()
    daily = _daily_topk_rows(frame)
    pr_at_k = _aggregate_topk(daily, scope="overall", scope_value="all")
    coverage = _coverage_table(pr_at_k)
    fold = _scoped_topk(frame, "fold")
    year = _scoped_topk(frame, "year")
    sector = _sector_audit(frame)
    buy_benchmarks = _buy_benchmark_rows(frame)
    bootstrap = _bootstrap_topk(daily)
    decision = _decision(pr_at_k, buy_benchmarks)

    daily.to_csv(OUTPUT_DAILY, index=False)
    pd.concat([pr_at_k, buy_benchmarks], ignore_index=True).to_csv(OUTPUT_PR_AT_K, index=False)
    coverage.to_csv(OUTPUT_COVERAGE, index=False)
    fold.to_csv(OUTPUT_FOLD, index=False)
    year.to_csv(OUTPUT_YEAR, index=False)
    sector.to_csv(OUTPUT_SECTOR, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            frame=frame,
            pr_at_k=pr_at_k,
            buy_benchmarks=buy_benchmarks,
            fold=fold,
            year=year,
            sector=sector,
            bootstrap=bootstrap,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "daily_ranking_audit": str(OUTPUT_DAILY),
                "precision_recall_at_k": str(OUTPUT_PR_AT_K),
                "winner_coverage_at_k": str(OUTPUT_COVERAGE),
                "ranking_fold_stability": str(OUTPUT_FOLD),
                "ranking_year_stability": str(OUTPUT_YEAR),
                "ranking_sector_audit": str(OUTPUT_SECTOR),
                "ranking_bootstrap": str(OUTPUT_BOOTSTRAP),
                "ranking_audit_report": str(OUTPUT_REPORT),
                "decision": decision,
                "final_test_used": False,
                "models_retrained": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )


def _load_oof() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    current = current[current["Scheme"].eq("EXPANDING_BASELINE")].copy()
    current = current[current["Date"] < FINAL_TEST_START].copy()
    if not current["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into daily ranking audit.")
    current["combined_percentile_score"] = (
        current["classifier_percentile"] + current["binary_buy_percentile"]
    ) / 2
    current["current_buy_signal"] = (
        (current["classifier_percentile"] >= 0.99)
        & (current["binary_buy_percentile"] >= 0.995)
    )
    current["year"] = current["Date"].dt.year
    if TUNED_OOF.exists():
        tuned = pd.read_csv(TUNED_OOF, parse_dates=["Date"])
        current = current.merge(tuned, on=["Date", "symbol", "fold"], how="left")
    else:
        current["tuned_binary_sigmoid_probability"] = np.nan
        current["tuned_binary_percentile"] = np.nan
    current["current_tuned_high_confidence"] = (
        current["current_buy_signal"] & (current["tuned_binary_percentile"] >= 0.995)
    )
    return current.sort_values(["Date", "symbol"]).reset_index(drop=True)


def _daily_topk_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (date, fold, year, regime), group in frame.groupby(
        ["Date", "fold", "year", "market_regime_label"], sort=True
    ):
        winners_on_date = int(group["buy_target"].sum())
        zero_winner_date = winners_on_date == 0
        for score_name, score_column, label in SCORE_SPECS:
            ranked = group.sort_values(score_column, ascending=False)
            for k in K_VALUES:
                selected = ranked.head(k)
                true_winners = int(selected["buy_target"].sum())
                rows.append(
                    {
                        "Date": date,
                        "fold": fold,
                        "year": year,
                        "market_regime_label": regime,
                        "ScoreName": score_name,
                        "ScoreLabel": label,
                        "K": k,
                        "EligibleStocks": int(len(group)),
                        "ActualWinnersOnDate": winners_on_date,
                        "ZeroWinnerDate": zero_winner_date,
                        "TrueWinnersInTopK": true_winners,
                        "PrecisionAtK": true_winners / k,
                        "RecallAtK": (
                            true_winners / winners_on_date if winners_on_date else np.nan
                        ),
                        "WinnerCoverageAtK": true_winners > 0,
                        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
                        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
                        "AverageFutureStockReturn": _safe_float(
                            selected["future_stock_return"].mean()
                        ),
                        "TotalSelections": int(len(selected)),
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_topk(daily: pd.DataFrame, *, scope: str, scope_value: Any) -> pd.DataFrame:
    rows = []
    for (score_name, k), group in daily.groupby(["ScoreName", "K"], sort=False):
        rows.append(_aggregate_row(group, score_name, k, scope, scope_value))
    return pd.DataFrame(rows)


def _aggregate_row(
    group: pd.DataFrame, score_name: str, k: int, scope: str, scope_value: Any
) -> dict[str, Any]:
    recall_group = group[~group["ZeroWinnerDate"]]
    return {
        "Scope": scope,
        "ScopeValue": scope_value,
        "ScoreName": score_name,
        "K": int(k),
        "Dates": int(group["Date"].nunique()),
        "ZeroWinnerDates": int(group["ZeroWinnerDate"].sum()),
        "TotalSelections": int(group["TotalSelections"].sum()),
        "PrecisionAtK": _safe_float(group["PrecisionAtK"].mean()),
        "RecallAtK": _safe_float(recall_group["RecallAtK"].mean()),
        "WinnerCoverageAtK": _safe_float(group["WinnerCoverageAtK"].mean()),
        "AverageExcessReturn": _safe_float(group["AverageExcessReturn"].mean()),
        "MedianExcessReturn": _safe_float(group["MedianExcessReturn"].median()),
        "NiftyWinRate": _safe_float(group["NiftyWinRate"].mean()),
        "AverageFutureStockReturn": _safe_float(group["AverageFutureStockReturn"].mean()),
        "ActualWinnersOnSelectedDates": int(group["ActualWinnersOnDate"].sum()),
    }


def _coverage_table(pr_at_k: pd.DataFrame) -> pd.DataFrame:
    return pr_at_k[
        [
            "Scope",
            "ScopeValue",
            "ScoreName",
            "K",
            "Dates",
            "ZeroWinnerDates",
            "WinnerCoverageAtK",
            "RecallAtK",
            "PrecisionAtK",
        ]
    ].copy()


def _scoped_topk(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column):
        rows.append(_aggregate_topk(_daily_topk_rows(group), scope=column, scope_value=value))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _sector_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_name, score_column, _label in SCORE_SPECS:
        for k in K_VALUES:
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


def _buy_benchmark_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _buy_benchmark_row(frame, "current_buy_signal", "Existing BUY"),
            _buy_benchmark_row(
                frame, "current_tuned_high_confidence", "Current+Tuned HIGH-confidence"
            ),
        ]
    )


def _buy_benchmark_row(frame: pd.DataFrame, signal_column: str, label: str) -> dict[str, Any]:
    selected = frame[frame[signal_column]].copy()
    active_dates = selected["Date"].nunique()
    winner_dates = frame.groupby("Date")["buy_target"].sum()
    active_winner_dates = (
        selected.groupby("Date")["buy_target"].sum().reindex(winner_dates.index, fill_value=0) > 0
    )
    return {
        "Scope": "buy_benchmark",
        "ScopeValue": label,
        "ScoreName": signal_column,
        "K": None,
        "Dates": int(frame["Date"].nunique()),
        "SignalDates": int(active_dates),
        "Signals": int(len(selected)),
        "PrecisionAtK": _safe_float(selected["buy_target"].mean()),
        "GlobalRecall": _safe_float(selected["buy_target"].sum() / frame["buy_target"].sum()),
        "AverageSignalsPerActiveDate": _safe_float(len(selected) / active_dates)
        if active_dates
        else None,
        "WinnerCoverageAtK": _safe_float(active_winner_dates.mean()),
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
        "LargestSectorShare": _largest_share(selected, "Sector"),
        "LargestStockShare": _largest_share(selected, "symbol"),
    }


def _bootstrap_topk(daily: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for score_name in [spec[0] for spec in SCORE_SPECS]:
        for k in (3, 5):
            group = daily[(daily["ScoreName"].eq(score_name)) & (daily["K"].eq(k))].copy()
            precision = group["PrecisionAtK"].to_numpy(dtype=float)
            recall = group["RecallAtK"].to_numpy(dtype=float)
            coverage = group["WinnerCoverageAtK"].astype(float).to_numpy()
            avg_excess = group["AverageExcessReturn"].to_numpy(dtype=float)
            values = {
                "PrecisionAtK": [],
                "RecallAtK": [],
                "WinnerCoverageAtK": [],
                "AverageExcessReturn": [],
            }
            n_dates = len(group)
            for _ in range(BOOTSTRAP_SAMPLES):
                sample_index = rng.integers(0, n_dates, size=n_dates)
                values["PrecisionAtK"].append(float(np.nanmean(precision[sample_index])))
                values["RecallAtK"].append(float(np.nanmean(recall[sample_index])))
                values["WinnerCoverageAtK"].append(float(np.nanmean(coverage[sample_index])))
                values["AverageExcessReturn"].append(float(np.nanmean(avg_excess[sample_index])))
            for metric, samples in values.items():
                arr = np.array(samples, dtype=float)
                rows.append(
                    {
                        "ScoreName": score_name,
                        "K": k,
                        "Metric": metric,
                        "Mean": _safe_float(np.nanmean(arr)),
                        "CI95Low": _safe_float(np.nanquantile(arr, 0.025)),
                        "CI95High": _safe_float(np.nanquantile(arr, 0.975)),
                        "Samples": BOOTSTRAP_SAMPLES,
                    }
                )
    return pd.DataFrame(rows)


def _decision(pr_at_k: pd.DataFrame, buy_benchmarks: pd.DataFrame) -> str:
    combined = pr_at_k[pr_at_k["ScoreName"].eq("combined_percentile")]
    top3 = combined[combined["K"].eq(3)].iloc[0]
    top5 = combined[combined["K"].eq(5)].iloc[0]
    existing = buy_benchmarks[buy_benchmarks["ScoreName"].eq("current_buy_signal")].iloc[0]
    if (
        top3["WinnerCoverageAtK"] >= 0.80
        and top3["PrecisionAtK"] >= 0.40
        and top3["AverageExcessReturn"] >= 0.04
    ):
        return "DAILY_RANKING_SIGNAL_STRONG"
    if (
        top5["WinnerCoverageAtK"] >= 0.65
        and top5["PrecisionAtK"] >= 0.32
        and top5["AverageExcessReturn"] >= 0.025
        and top5["WinnerCoverageAtK"] > existing["WinnerCoverageAtK"]
    ):
        return "DAILY_RANKING_SIGNAL_MODERATE"
    return "DAILY_RANKING_SIGNAL_WEAK"


def _report(
    *,
    frame: pd.DataFrame,
    pr_at_k: pd.DataFrame,
    buy_benchmarks: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    sector: pd.DataFrame,
    bootstrap: pd.DataFrame,
    decision: str,
) -> str:
    return "\n".join(
        [
            "Validation-only Daily Ranking Audit",
            "",
            f"Decision: {decision}",
            f"Rows: {len(frame):,}",
            f"Dates: {frame['Date'].nunique():,}",
            f"Symbols: {frame['symbol'].nunique():,}",
            f"Date range: {frame['Date'].min().date()} to {frame['Date'].max().date()}",
            "Final test used: False",
            "Models retrained: False",
            "Production artifacts modified: False",
            "",
            "Overall Precision/Recall@K:",
            pr_at_k.to_string(index=False),
            "",
            "Existing BUY benchmarks:",
            buy_benchmarks.to_string(index=False),
            "",
            "Bootstrap Top3/Top5:",
            bootstrap.to_string(index=False),
            "",
            "Fold stability:",
            fold.to_string(index=False),
            "",
            "Year stability:",
            year.to_string(index=False),
            "",
            "Sector concentration sample:",
            sector.sort_values(["ScoreName", "K", "SelectionShare"], ascending=[True, True, False])
            .head(30)
            .to_string(index=False),
            "",
            "Leakage checks:",
            "- Used existing EXPANDING_BASELINE strict OOF validation predictions only.",
            "- Restricted to Date < 2025-01-01.",
            "- No model retraining, threshold changes, or production artifact changes.",
            "- Combined score uses predefined 0.5 XGB percentile + 0.5 Binary percentile.",
        ]
    )


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
