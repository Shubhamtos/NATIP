"""Validation-only SSL extreme-tail BUY audit.

This script uses existing strict OOF prediction files only. It does not train,
tune, modify production artifacts, or inspect final-test rows.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START

CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"
SSL_PLUS_OOF = REPORT_DIR / "xgb26_plus_ssl_oof.parquet"
TUNED_BINARY_OOF = REPORT_DIR / "lstm_tuned_binary77_oof_cache.csv"

OUTPUT_COMMON = REPORT_DIR / "ssl_tail_common_scope.csv"
OUTPUT_OVERLAP = REPORT_DIR / "ssl_tail_buy_overlap.csv"
OUTPUT_ADDED_DROPPED = REPORT_DIR / "ssl_tail_added_dropped.csv"
OUTPUT_DATE = REPORT_DIR / "ssl_tail_date_analysis.csv"
OUTPUT_FOLD = REPORT_DIR / "ssl_tail_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "ssl_tail_year_stability.csv"
OUTPUT_CONCENTRATION = REPORT_DIR / "ssl_tail_stock_sector_concentration.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "ssl_tail_bootstrap.csv"
OUTPUT_EQUAL_SIGNAL = REPORT_DIR / "ssl_tail_equal_signal_comparison.csv"
OUTPUT_BUCKETS = REPORT_DIR / "ssl_tail_score_buckets.csv"
OUTPUT_INTERACTION = REPORT_DIR / "ssl_tail_xgb_interaction.csv"
OUTPUT_CORRELATIONS = REPORT_DIR / "ssl_tail_score_correlations.csv"
OUTPUT_TUNED = REPORT_DIR / "ssl_tail_tuned_interaction.csv"
OUTPUT_REPORT = REPORT_DIR / "ssl_tail_audit_report.txt"

CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42


def main() -> None:
    """Run the SSL extreme-tail audit from existing OOF scores."""

    ensure_probability_dirs()
    common = _load_common_scope()
    common.to_csv(OUTPUT_COMMON, index=False)

    overlap = _buy_overlap(common)
    overlap.to_csv(OUTPUT_OVERLAP, index=False)

    added_dropped = _added_dropped(common)
    added_dropped.to_csv(OUTPUT_ADDED_DROPPED, index=False)

    date_analysis = _date_analysis(common)
    date_analysis.to_csv(OUTPUT_DATE, index=False)

    fold = _scope_stability(common, "fold")
    fold.to_csv(OUTPUT_FOLD, index=False)

    year = _scope_stability(common.assign(year=common["Date"].dt.year), "year")
    year.to_csv(OUTPUT_YEAR, index=False)

    concentration = _concentration(common)
    concentration.to_csv(OUTPUT_CONCENTRATION, index=False)

    bootstrap = _bootstrap(common)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)

    equal_signal = _equal_signal_comparison(common)
    equal_signal.to_csv(OUTPUT_EQUAL_SIGNAL, index=False)

    buckets = _ssl_score_buckets(common)
    buckets.to_csv(OUTPUT_BUCKETS, index=False)

    interaction = _xgb_ssl_interaction(common)
    interaction.to_csv(OUTPUT_INTERACTION, index=False)

    correlations = _score_correlations(common)
    correlations.to_csv(OUTPUT_CORRELATIONS, index=False)

    tuned = _tuned_interaction(common)
    tuned.to_csv(OUTPUT_TUNED, index=False)

    decision = _decision(
        common, overlap, bootstrap, equal_signal, buckets, fold, year, concentration
    )
    OUTPUT_REPORT.write_text(
        _report(
            common=common,
            overlap=overlap,
            added_dropped=added_dropped,
            date_analysis=date_analysis,
            fold=fold,
            year=year,
            concentration=concentration,
            bootstrap=bootstrap,
            equal_signal=equal_signal,
            buckets=buckets,
            interaction=interaction,
            correlations=correlations,
            tuned=tuned,
            decision=decision,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "ssl_tail_common_scope": str(OUTPUT_COMMON),
                "ssl_tail_buy_overlap": str(OUTPUT_OVERLAP),
                "ssl_tail_added_dropped": str(OUTPUT_ADDED_DROPPED),
                "ssl_tail_date_analysis": str(OUTPUT_DATE),
                "ssl_tail_fold_stability": str(OUTPUT_FOLD),
                "ssl_tail_year_stability": str(OUTPUT_YEAR),
                "ssl_tail_stock_sector_concentration": str(OUTPUT_CONCENTRATION),
                "ssl_tail_bootstrap": str(OUTPUT_BOOTSTRAP),
                "ssl_tail_equal_signal_comparison": str(OUTPUT_EQUAL_SIGNAL),
                "ssl_tail_score_buckets": str(OUTPUT_BUCKETS),
                "ssl_tail_xgb_interaction": str(OUTPUT_INTERACTION),
                "ssl_tail_score_correlations": str(OUTPUT_CORRELATIONS),
                "ssl_tail_tuned_interaction": str(OUTPUT_TUNED),
                "ssl_tail_audit_report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "models_retrained": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )
    print(decision)


def _load_common_scope() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    current = current[current["Scheme"].eq("EXPANDING_BASELINE")].copy()
    current = current[current["Date"] < FINAL_TEST_START].copy()
    ssl = pd.read_parquet(SSL_PLUS_OOF)
    ssl["Date"] = pd.to_datetime(ssl["Date"])
    ssl = ssl[ssl["Date"] < FINAL_TEST_START].copy()

    key = ["Date", "symbol", "fold"]
    current_cols = [
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
        "p_underperform",
        "p_neutral",
        "p_outperform",
        "binary_buy_sigmoid_probability",
    ]
    ssl_cols = ["Date", "symbol", "fold", "p_outperform"]
    common = current[current_cols].merge(
        ssl[ssl_cols].rename(columns={"p_outperform": "ssl_p_outperform"}),
        on=key,
        how="inner",
        validate="one_to_one",
    )
    common = common.rename(columns={"p_outperform": "xgb_p_outperform"})
    common["xgb_percentile"] = common.groupby("Date")["xgb_p_outperform"].rank(pct=True)
    common["ssl_percentile"] = common.groupby("Date")["ssl_p_outperform"].rank(pct=True)
    common["binary_percentile"] = common.groupby("Date")[
        "binary_buy_sigmoid_probability"
    ].rank(pct=True)
    common["baseline_buy"] = (common["xgb_percentile"] >= 1 - CLASSIFIER_TOP) & (
        common["binary_percentile"] >= 1 - BINARY_TOP
    )
    common["ssl_buy"] = (common["ssl_percentile"] >= 1 - CLASSIFIER_TOP) & (
        common["binary_percentile"] >= 1 - BINARY_TOP
    )
    common["hybrid_confidence"] = 0.5 * common["ssl_percentile"] + 0.5 * common[
        "binary_percentile"
    ]
    common["xgb_ssl_disagreement"] = (
        common["xgb_percentile"].sub(common["ssl_percentile"]).abs()
    )

    if TUNED_BINARY_OOF.exists():
        tuned = pd.read_csv(TUNED_BINARY_OOF, parse_dates=["Date"])
        common = common.merge(tuned, on=key, how="left")
        common["tuned_binary_percentile_exact_scope"] = common.groupby("Date")[
            "tuned_binary_sigmoid_probability"
        ].rank(pct=True)
        common["tuned_confirms"] = common["tuned_binary_percentile_exact_scope"] >= (
            1 - BINARY_TOP
        )
    else:
        common["tuned_binary_sigmoid_probability"] = np.nan
        common["tuned_binary_percentile"] = np.nan
        common["tuned_binary_percentile_exact_scope"] = np.nan
        common["tuned_confirms"] = False

    if common.empty:
        raise AssertionError("No common rows found for XGB26, XGB26_PLUS_SSL, and Binary77.")
    if not common["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into SSL tail audit.")
    duplicates = int(common.duplicated(key).sum())
    if duplicates:
        raise AssertionError(f"Duplicate common-scope keys found: {duplicates}")
    return common.sort_values(key).reset_index(drop=True)


def _metric_row(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "Group": name,
        "Signals": int(len(frame)),
        "UniqueDates": int(frame["Date"].nunique()) if not frame.empty else 0,
        "UniqueStocks": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "Precision": _safe_float(frame["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
        "MeanStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "P10Excess": _safe_float(frame["excess_return"].quantile(0.10)),
        "P25Excess": _safe_float(frame["excess_return"].quantile(0.25)),
        "P75Excess": _safe_float(frame["excess_return"].quantile(0.75)),
        "P90Excess": _safe_float(frame["excess_return"].quantile(0.90)),
        "WorstExcess": _safe_float(frame["excess_return"].min()),
        "BestExcess": _safe_float(frame["excess_return"].max()),
    }


def _buy_overlap(common: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "BOTH": common[common["baseline_buy"] & common["ssl_buy"]],
        "BASELINE_ONLY": common[common["baseline_buy"] & ~common["ssl_buy"]],
        "SSL_ONLY": common[~common["baseline_buy"] & common["ssl_buy"]],
        "NEITHER": common[~common["baseline_buy"] & ~common["ssl_buy"]],
    }
    rows = [_metric_row(name, frame) for name, frame in groups.items()]
    return pd.DataFrame(rows)


def _added_dropped(common: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "DROPPED_BY_SSL": common[common["baseline_buy"] & ~common["ssl_buy"]],
        "ADDED_BY_SSL": common[~common["baseline_buy"] & common["ssl_buy"]],
    }
    return pd.DataFrame([_metric_row(name, frame) for name, frame in groups.items()])


def _date_analysis(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in common.groupby("Date"):
        base = group[group["baseline_buy"]]
        ssl = group[group["ssl_buy"]]
        both = group[group["baseline_buy"] & group["ssl_buy"]]
        added = group[~group["baseline_buy"] & group["ssl_buy"]]
        dropped = group[group["baseline_buy"] & ~group["ssl_buy"]]
        base_precision = _safe_float(base["buy_target"].mean())
        ssl_precision = _safe_float(ssl["buy_target"].mean())
        if base.empty and ssl.empty:
            result = "EQUAL"
        elif base_precision is None:
            result = "SSL_BEATS"
        elif ssl_precision is None:
            result = "SSL_LOSES"
        elif ssl_precision > base_precision:
            result = "SSL_BEATS"
        elif ssl_precision < base_precision:
            result = "SSL_LOSES"
        else:
            result = "EQUAL"
        rows.append(
            {
                "Date": date,
                "BaselineBuyCount": int(len(base)),
                "SSLHybridBuyCount": int(len(ssl)),
                "Overlap": int(len(both)),
                "Added": int(len(added)),
                "Dropped": int(len(dropped)),
                "BaselinePrecision": base_precision,
                "SSLPrecision": ssl_precision,
                "BaselineAvgExcess": _safe_float(base["excess_return"].mean()),
                "SSLAvgExcess": _safe_float(ssl["excess_return"].mean()),
                "DateResult": result,
            }
        )
    return pd.DataFrame(rows)


def _scope_stability(common: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows = []
    scope_col = scope
    for value, group in common.groupby(scope_col):
        base = group[group["baseline_buy"]]
        ssl = group[group["ssl_buy"]]
        for system, selected in [("BASELINE", base), ("SSL_HYBRID", ssl)]:
            row = _metric_row(system, selected)
            row.update({"Scope": scope, "ScopeValue": value})
            rows.append(row)
        delta = _delta_row(value, base, ssl, scope)
        rows.append(delta)
    return pd.DataFrame(rows)


def _delta_row(value: Any, baseline: pd.DataFrame, ssl: pd.DataFrame, scope: str) -> dict[str, Any]:
    base = _metric_row("BASELINE", baseline)
    candidate = _metric_row("SSL_HYBRID", ssl)
    return {
        "Group": "SSL_MINUS_BASELINE",
        "Scope": scope,
        "ScopeValue": value,
        "Signals": int(candidate["Signals"] - base["Signals"]),
        "UniqueDates": None,
        "UniqueStocks": None,
        "Precision": _diff(candidate["Precision"], base["Precision"]),
        "AverageExcessReturn": _diff(
            candidate["AverageExcessReturn"], base["AverageExcessReturn"]
        ),
        "MedianExcessReturn": _diff(
            candidate["MedianExcessReturn"], base["MedianExcessReturn"]
        ),
        "NiftyWinRate": _diff(candidate["NiftyWinRate"], base["NiftyWinRate"]),
        "MeanStockReturn": _diff(candidate["MeanStockReturn"], base["MeanStockReturn"]),
        "P10Excess": None,
        "P25Excess": None,
        "P75Excess": None,
        "P90Excess": None,
        "WorstExcess": None,
        "BestExcess": None,
    }


def _concentration(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "BOTH": common[common["baseline_buy"] & common["ssl_buy"]],
        "SSL_ONLY": common[~common["baseline_buy"] & common["ssl_buy"]],
        "BASELINE_ONLY": common[common["baseline_buy"] & ~common["ssl_buy"]],
    }
    for group_name, frame in groups.items():
        for scope, col in [("ticker", "symbol"), ("sector", "Sector")]:
            counts = frame[col].fillna("UNKNOWN").value_counts()
            total = int(counts.sum())
            top10 = int(counts.head(10).sum()) if total else 0
            hhi = float(((counts / total) ** 2).sum()) if total else None
            metric = _metric_row(group_name, frame)
            rows.append(
                {
                    "Group": group_name,
                    "Scope": scope,
                    "Signals": total,
                    "DistinctValues": int(counts.size),
                    "Top10Share": _safe_float(top10 / total) if total else None,
                    "LargestShare": _safe_float(counts.iloc[0] / total) if total else None,
                    "LargestValue": counts.index[0] if total else None,
                    "Herfindahl": hhi,
                    "Precision": metric["Precision"],
                    "AverageExcessReturn": metric["AverageExcessReturn"],
                    "MedianExcessReturn": metric["MedianExcessReturn"],
                    "NiftyWinRate": metric["NiftyWinRate"],
                    "CountsJson": json.dumps(counts.head(25).to_dict()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap(common: pd.DataFrame) -> pd.DataFrame:
    dates = np.array(sorted(common["Date"].unique()))
    date_stats = [_date_bootstrap_stats(common[common["Date"].eq(date)]) for date in dates]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for sample in range(BOOTSTRAP_SAMPLES):
        sampled_indices = rng.integers(0, len(dates), size=len(dates))
        base = _aggregate_bootstrap_stats(date_stats, sampled_indices, "base")
        ssl = _aggregate_bootstrap_stats(date_stats, sampled_indices, "ssl")
        rows.append(
            {
                "Sample": sample,
                "PrecisionDiff": _dict_metric_diff(ssl, base, "precision"),
                "AverageExcessDiff": _dict_metric_diff(ssl, base, "avg_excess"),
                "MedianExcessDiff": _dict_metric_diff(ssl, base, "median_excess"),
                "NiftyWinRateDiff": _dict_metric_diff(ssl, base, "win"),
                "SignalCountDiff": int(ssl["count"] - base["count"]),
            }
        )
    draws = pd.DataFrame(rows)
    summary = []
    for metric in [
        "PrecisionDiff",
        "AverageExcessDiff",
        "MedianExcessDiff",
        "NiftyWinRateDiff",
        "SignalCountDiff",
    ]:
        series = draws[metric].dropna()
        summary.append(
            {
                "Metric": metric,
                "MedianDiff": _safe_float(series.median()),
                "CI025": _safe_float(series.quantile(0.025)),
                "CI975": _safe_float(series.quantile(0.975)),
                "ProbabilityPositive": _safe_float((series > 0).mean()),
                "BootstrapSamples": int(len(series)),
            }
        )
    return pd.DataFrame(summary)


def _date_bootstrap_stats(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        "base": _bootstrap_stat_row(frame[frame["baseline_buy"]]),
        "ssl": _bootstrap_stat_row(frame[frame["ssl_buy"]]),
    }


def _bootstrap_stat_row(frame: pd.DataFrame) -> dict[str, Any]:
    excess = frame["excess_return"].to_numpy(dtype=float)
    return {
        "count": int(len(frame)),
        "positives": int(frame["buy_target"].sum()) if not frame.empty else 0,
        "excess_sum": float(np.nansum(excess)) if len(excess) else 0.0,
        "wins": int((excess > 0).sum()) if len(excess) else 0,
        "excess_values": excess,
    }


def _aggregate_bootstrap_stats(
    date_stats: list[dict[str, dict[str, Any]]], indices: np.ndarray, key: str
) -> dict[str, Any]:
    count = 0
    positives = 0
    excess_sum = 0.0
    wins = 0
    values = []
    for idx in indices:
        stat = date_stats[int(idx)][key]
        count += stat["count"]
        positives += stat["positives"]
        excess_sum += stat["excess_sum"]
        wins += stat["wins"]
        if stat["count"]:
            values.append(stat["excess_values"])
    excess_values = np.concatenate(values) if values else np.array([], dtype=float)
    return {
        "count": count,
        "precision": positives / count if count else None,
        "avg_excess": excess_sum / count if count else None,
        "median_excess": float(np.nanmedian(excess_values)) if len(excess_values) else None,
        "win": wins / count if count else None,
    }


def _equal_signal_comparison(common: pd.DataFrame) -> pd.DataFrame:
    base = common[common["baseline_buy"]].copy()
    ssl_full = common[common["ssl_buy"]].copy()
    target_count = len(base)
    ssl_matched = (
        ssl_full.sort_values(["hybrid_confidence", "ssl_p_outperform"], ascending=False)
        .head(target_count)
        .copy()
    )
    matched_label = (
        "SSL_HYBRID_MATCHED_BASELINE_COUNT"
        if len(ssl_matched) == target_count
        else "SSL_HYBRID_FULL_BELOW_BASELINE_COUNT"
    )
    rows = [
        _metric_row("BASELINE", base),
        _metric_row(matched_label, ssl_matched),
        _metric_row("SSL_HYBRID_FULL", ssl_full),
    ]
    return pd.DataFrame(rows)


def _ssl_score_buckets(common: pd.DataFrame) -> pd.DataFrame:
    candidates = common[common["binary_percentile"] >= 1 - BINARY_TOP].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates["ssl_rank_desc"] = candidates["ssl_percentile"].rank(
        method="first", ascending=False
    )
    n = len(candidates)
    cuts = [
        ("Top10", 0.00, 0.10),
        ("10-25", 0.10, 0.25),
        ("25-50", 0.25, 0.50),
        ("50-75", 0.50, 0.75),
        ("Bottom25", 0.75, 1.00),
    ]
    rows = []
    for name, lo, hi in cuts:
        start = int(np.floor(lo * n)) + 1
        end = int(np.floor(hi * n))
        if hi == 1.0:
            end = n
        bucket = candidates[
            (candidates["ssl_rank_desc"] >= start) & (candidates["ssl_rank_desc"] <= end)
        ]
        row = _metric_row(name, bucket)
        row["CandidatePopulation"] = "Binary77Top0.5"
        row["RankStart"] = start
        row["RankEnd"] = end
        rows.append(row)
    return pd.DataFrame(rows)


def _xgb_ssl_interaction(common: pd.DataFrame) -> pd.DataFrame:
    candidates = common[common["binary_percentile"] >= 1 - BINARY_TOP].copy()
    candidates["HighXGB26"] = candidates["xgb_percentile"] >= 1 - CLASSIFIER_TOP
    candidates["HighSSL"] = candidates["ssl_percentile"] >= 1 - CLASSIFIER_TOP
    rows = []
    labels = [
        ("High XGB26 + High SSL", True, True),
        ("High XGB26 + Low SSL", True, False),
        ("Low XGB26 + High SSL", False, True),
        ("Low XGB26 + Low SSL", False, False),
    ]
    for label, high_xgb, high_ssl in labels:
        part = candidates[
            candidates["HighXGB26"].eq(high_xgb) & candidates["HighSSL"].eq(high_ssl)
        ]
        row = _metric_row(label, part)
        row["CandidatePopulation"] = "Binary77Top0.5"
        rows.append(row)
    return pd.DataFrame(rows)


def _score_correlations(common: pd.DataFrame) -> pd.DataFrame:
    candidates = common[common["binary_percentile"] >= 1 - BINARY_TOP].copy()
    if candidates.empty:
        return pd.DataFrame()
    xgb_top_decile = set(
        candidates.sort_values("xgb_percentile", ascending=False)
        .head(max(1, int(len(candidates) * 0.10)))[["Date", "symbol"]]
        .itertuples(index=False, name=None)
    )
    ssl_top_decile = set(
        candidates.sort_values("ssl_percentile", ascending=False)
        .head(max(1, int(len(candidates) * 0.10)))[["Date", "symbol"]]
        .itertuples(index=False, name=None)
    )
    intersection = xgb_top_decile & ssl_top_decile
    union = xgb_top_decile | ssl_top_decile
    return pd.DataFrame(
        [
            {
                "CandidatePopulation": "Binary77Top0.5",
                "Rows": int(len(candidates)),
                "PearsonXGBvsSSLScore": _safe_float(
                    candidates["xgb_p_outperform"].corr(candidates["ssl_p_outperform"])
                ),
                "SpearmanXGBvsSSLScore": _safe_float(
                    candidates["xgb_p_outperform"].corr(
                        candidates["ssl_p_outperform"], method="spearman"
                    )
                ),
                "SpearmanPercentileCorrelation": _safe_float(
                    candidates["xgb_percentile"].corr(
                        candidates["ssl_percentile"], method="spearman"
                    )
                ),
                "DisagreementRateTop1": _safe_float(
                    (
                        (candidates["xgb_percentile"] >= 1 - CLASSIFIER_TOP)
                        != (candidates["ssl_percentile"] >= 1 - CLASSIFIER_TOP)
                    ).mean()
                ),
                "TopDecileXGBCount": len(xgb_top_decile),
                "TopDecileSSLCount": len(ssl_top_decile),
                "TopDecileOverlap": len(intersection),
                "TopDecileJaccard": _safe_float(len(intersection) / len(union))
                if union
                else None,
            }
        ]
    )


def _tuned_interaction(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "Current BUY": common[common["baseline_buy"]],
        "Current+Tuned": common[common["baseline_buy"] & common["tuned_confirms"]],
        "SSL hybrid BUY": common[common["ssl_buy"]],
        "SSL hybrid + Tuned": common[common["ssl_buy"] & common["tuned_confirms"]],
    }
    for name, frame in groups.items():
        rows.append(_metric_row(name, frame))
        for fold, fold_frame in frame.groupby("fold"):
            row = _metric_row(name, fold_frame)
            row["Fold"] = fold
            rows.append(row)
    return pd.DataFrame(rows)


def _decision(
    common: pd.DataFrame,
    overlap: pd.DataFrame,
    bootstrap: pd.DataFrame,
    equal_signal: pd.DataFrame,
    buckets: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    concentration: pd.DataFrame,
) -> str:
    base = common[common["baseline_buy"]]
    ssl = common[common["ssl_buy"]]
    ssl_only = overlap[overlap["Group"].eq("SSL_ONLY")].iloc[0]
    matched = equal_signal[
        ~equal_signal["Group"].isin(["BASELINE", "SSL_HYBRID_FULL"])
    ].iloc[0]
    base_precision = _safe_float(base["buy_target"].mean()) or 0.0
    ssl_precision = _safe_float(ssl["buy_target"].mean()) or 0.0
    precision_boot = bootstrap[bootstrap["Metric"].eq("PrecisionDiff")].iloc[0]
    avg_boot = bootstrap[bootstrap["Metric"].eq("AverageExcessDiff")].iloc[0]
    fold_deltas = fold[fold["Group"].eq("SSL_MINUS_BASELINE")]
    year_deltas = year[year["Group"].eq("SSL_MINUS_BASELINE")]
    positive_folds = int((fold_deltas["AverageExcessReturn"].fillna(-1) > 0).sum())
    positive_years = int((year_deltas["AverageExcessReturn"].fillna(-1) > 0).sum())
    top_bucket = buckets[buckets["Group"].eq("Top10")]
    bottom_bucket = buckets[buckets["Group"].eq("Bottom25")]
    bucket_separates = False
    if not top_bucket.empty and not bottom_bucket.empty:
        bucket_separates = (
            (top_bucket.iloc[0]["Precision"] or 0) > (bottom_bucket.iloc[0]["Precision"] or 0)
            and (top_bucket.iloc[0]["AverageExcessReturn"] or 0)
            > (bottom_bucket.iloc[0]["AverageExcessReturn"] or 0)
        )
    concentration_ok = True
    sector = concentration[
        concentration["Group"].eq("SSL_ONLY") & concentration["Scope"].eq("sector")
    ]
    if not sector.empty and (sector.iloc[0]["LargestShare"] or 0) > 0.50:
        concentration_ok = False

    robust = (
        ssl_precision > base_precision
        and (precision_boot["CI025"] or -1) > 0
        and (avg_boot["CI025"] or -1) > 0
        and (ssl_only["Precision"] or 0) >= base_precision
        and (ssl_only["AverageExcessReturn"] or 0) > 0.05
        and (matched["Precision"] or 0) >= base_precision
        and bucket_separates
        and positive_folds >= max(1, int(0.75 * fold_deltas["ScopeValue"].nunique()))
        and positive_years >= max(1, int(0.75 * year_deltas["ScopeValue"].nunique()))
        and concentration_ok
    )
    if robust:
        return "SSL_EXTREME_TAIL_ADDS_ROBUST_VALUE"
    if ssl_precision > base_precision and (ssl_only["AverageExcessReturn"] or 0) > 0:
        if (precision_boot["ProbabilityPositive"] or 0) >= 0.70:
            return "SSL_EXTREME_TAIL_PROMISING_NOT_ROBUST"
        return "SSL_CONFIRMATION_ONLY"
    return "SSL_EXTREME_TAIL_NO_VALUE"


def _report(**frames: Any) -> str:
    decision = frames["decision"]
    common = frames["common"]
    overlap = frames["overlap"]
    bootstrap = frames["bootstrap"]
    equal_signal = frames["equal_signal"]
    date_analysis = frames["date_analysis"]
    correlations = frames["correlations"]
    lines = [
        "SSL EXTREME-TAIL / STRICT-BUY AUDIT",
        "====================================",
        "",
        "Scope and safety",
        f"- Common rows: {len(common):,}",
        f"- Validation dates: {common['Date'].nunique():,}",
        f"- Stocks: {common['symbol'].nunique():,}",
        "- Inputs: existing strict OOF scores only.",
        "- Final test used: NO.",
        "- Retraining/tuning: NO.",
        "- Production artifacts modified: NO.",
        "",
        "Overlap decomposition",
        overlap.to_string(index=False),
        "",
        "Equal signal-count diagnostic",
        equal_signal.to_string(index=False),
        "",
        "Date-level result counts",
        date_analysis["DateResult"].value_counts().to_string(),
        "",
        "Date-block bootstrap summary",
        bootstrap.to_string(index=False),
        "",
        "Binary77-tail XGB/SSL score correlation",
        correlations.to_string(index=False) if not correlations.empty else "No Binary77 tail rows.",
        "",
        f"FINAL DECISION: {decision}",
    ]
    return "\n".join(lines)


def _metric_diff(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> float | None:
    if metric == "precision":
        return _diff(
            _safe_float(left["buy_target"].mean()),
            _safe_float(right["buy_target"].mean()),
        )
    if metric == "avg_excess":
        return _diff(
            _safe_float(left["excess_return"].mean()),
            _safe_float(right["excess_return"].mean()),
        )
    if metric == "median_excess":
        return _diff(
            _safe_float(left["excess_return"].median()),
            _safe_float(right["excess_return"].median()),
        )
    if metric == "win":
        return _diff(
            _safe_float((left["excess_return"] > 0).mean()),
            _safe_float((right["excess_return"] > 0).mean()),
        )
    raise ValueError(metric)


def _dict_metric_diff(left: dict[str, Any], right: dict[str, Any], metric: str) -> float | None:
    return _diff(left.get(metric), right.get(metric))


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


if __name__ == "__main__":
    main()
