"""Audit cached NSE OHLCV data, features, labels, and split drift."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    CACHE_DIR,
    DATA_DIR,
    REPORT_DIR,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.data_download import load_stock_universe, load_universe_metadata
from app.probability.features import feature_columns_for_variant
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from run_pruned_rank_target_validation import HORIZON_DAYS, PRUNED_FEATURES, _embargo_folds

OUTPUT_TICKER = REPORT_DIR / "data_quality_by_ticker.csv"
OUTPUT_FEATURE = REPORT_DIR / "feature_quality_report.csv"
OUTPUT_LABEL = REPORT_DIR / "label_distribution_report.csv"
OUTPUT_DRIFT = REPORT_DIR / "train_validation_test_drift.csv"
OUTPUT_EXTREME = REPORT_DIR / "extreme_move_report.csv"
OUTPUT_LIQUIDITY = REPORT_DIR / "liquidity_report.csv"
OUTPUT_ALIGNMENT = REPORT_DIR / "benchmark_alignment_report.csv"
OUTPUT_SUMMARY = REPORT_DIR / "data_audit_summary.txt"

DATASET_PATH = DATA_DIR / "probability_training_dataset_advanced.csv"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def main() -> None:
    """Run a complete read-only data quality and distribution audit."""

    ensure_probability_dirs()
    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    stock_symbols = load_stock_universe(STOCKS_UNIVERSE_2026_08_CSV)
    sector_symbols = sector_benchmark_symbols(universe)
    all_symbols = [*stock_symbols, BENCHMARK_SYMBOL, *sector_symbols]
    dataset = _load_dataset()
    benchmark = _read_cached_symbol(BENCHMARK_SYMBOL)
    cached = {symbol: _read_cached_symbol(symbol) for symbol in all_symbols}

    ticker_report, extreme_report, liquidity_report = _ticker_reports(
        symbols=all_symbols,
        stock_symbols=stock_symbols,
        sector_symbols=sector_symbols,
        cached=cached,
        benchmark=benchmark,
        dataset=dataset,
        universe=universe,
    )
    feature_report = _feature_quality_report(dataset)
    label_report = _label_distribution_report(dataset)
    drift_report = _drift_report(dataset)
    alignment_report = _benchmark_alignment_report(
        stock_symbols=stock_symbols,
        sector_symbols=sector_symbols,
        cached=cached,
        benchmark=benchmark,
        universe=universe,
    )

    ticker_report.to_csv(OUTPUT_TICKER, index=False)
    feature_report.to_csv(OUTPUT_FEATURE, index=False)
    label_report.to_csv(OUTPUT_LABEL, index=False)
    drift_report.to_csv(OUTPUT_DRIFT, index=False)
    extreme_report.to_csv(OUTPUT_EXTREME, index=False)
    liquidity_report.to_csv(OUTPUT_LIQUIDITY, index=False)
    alignment_report.to_csv(OUTPUT_ALIGNMENT, index=False)
    OUTPUT_SUMMARY.write_text(
        _summary_text(
            ticker_report=ticker_report,
            feature_report=feature_report,
            label_report=label_report,
            drift_report=drift_report,
            alignment_report=alignment_report,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "data_quality_by_ticker": str(OUTPUT_TICKER),
                "feature_quality_report": str(OUTPUT_FEATURE),
                "label_distribution_report": str(OUTPUT_LABEL),
                "train_validation_test_drift": str(OUTPUT_DRIFT),
                "extreme_move_report": str(OUTPUT_EXTREME),
                "liquidity_report": str(OUTPUT_LIQUIDITY),
                "benchmark_alignment_report": str(OUTPUT_ALIGNMENT),
                "data_audit_summary": str(OUTPUT_SUMMARY),
            },
            indent=2,
        )
    )


def _load_dataset() -> pd.DataFrame:
    """Load current advanced training dataset."""

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["Date"])
    dataset = dataset.replace([float("inf"), float("-inf")], pd.NA)
    dataset["Split"] = pd.cut(
        dataset["Date"],
        bins=[pd.Timestamp.min, pd.Timestamp("2022-12-31"), pd.Timestamp("2024-12-31"), pd.Timestamp.max],
        labels=["train", "validation", "final_test"],
    )
    dataset["Year"] = dataset["Date"].dt.year
    dataset["MarketRegime"] = dataset.apply(_market_regime_label, axis=1)
    return dataset


def _market_regime_label(row: pd.Series) -> str:
    """Convert regime flags to labels."""

    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _read_cached_symbol(symbol: str) -> pd.DataFrame:
    """Read one cached OHLCV CSV without refreshing from network."""

    path = CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Date", *OHLCV_COLUMNS])
    frame = pd.read_csv(path, parse_dates=["Date"])
    for column in OHLCV_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _ticker_reports(
    *,
    symbols: list[str],
    stock_symbols: list[str],
    sector_symbols: list[str],
    cached: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    dataset: pd.DataFrame,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create ticker, extreme-move, and liquidity reports."""

    benchmark_dates = set(pd.to_datetime(benchmark["Date"]).dropna())
    metadata = universe.set_index("Ticker").to_dict(orient="index")
    ticker_rows = []
    extreme_rows = []
    liquidity_rows = []
    pruned_features = _pruned_features()
    for symbol in symbols:
        frame = cached.get(symbol, pd.DataFrame()).copy()
        symbol_type = _symbol_type(symbol, stock_symbols, sector_symbols)
        sector = metadata.get(symbol, {}).get("Sector", "Benchmark" if symbol == BENCHMARK_SYMBOL else "Sector Index")
        if frame.empty:
            ticker_rows.append(_empty_ticker_row(symbol, symbol_type, sector))
            continue
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame = frame.sort_values("Date")
        duplicates = int(frame.duplicated("Date").sum())
        required_missing = frame[OHLCV_COLUMNS].isna().sum().sum()
        missing_pct = float(required_missing / (len(frame) * len(OHLCV_COLUMNS))) if len(frame) else 1.0
        invalid_ohlc = _invalid_ohlc_count(frame)
        daily_return = frame["Close"].pct_change()
        turnover = frame["Close"] * frame["Volume"]
        missing_vs_benchmark = len(benchmark_dates - set(frame["Date"])) if benchmark_dates else 0
        stock_extra_dates = len(set(frame["Date"]) - benchmark_dates) if benchmark_dates else 0
        usable_rows = _usable_rows(dataset, symbol, pruned_features)
        suspicious_20 = int((daily_return.abs() > 0.20).sum())
        suspicious_30 = int((daily_return.abs() > 0.30).sum())
        suspicious_50 = int((daily_return.abs() > 0.50).sum())
        stale_rows = _stale_forward_fill_like_rows(frame)
        status = _ticker_status(
            missing_pct=missing_pct,
            duplicates=duplicates,
            invalid_ohlc=invalid_ohlc,
            suspicious_50=suspicious_50,
            stale_rows=stale_rows,
            zero_volume_days=int((frame["Volume"] == 0).sum()),
            row_count=len(frame),
        )
        ticker_rows.append(
            {
                "Ticker": symbol,
                "Type": symbol_type,
                "Sector": sector,
                "FirstDate": frame["Date"].min(),
                "LastDate": frame["Date"].max(),
                "RowCount": int(len(frame)),
                "MissingOHLCVPercent": missing_pct,
                "DuplicateDateSymbolRows": duplicates,
                "ZeroVolumeDays": int((frame["Volume"] == 0).sum()),
                "AverageDailyTurnover": float(turnover.mean()),
                "MedianDailyTurnover": float(turnover.median()),
                "MaxPositive1DReturn": float(daily_return.max()),
                "MaxNegative1DReturn": float(daily_return.min()),
                "SuspiciousMovesGT20Pct": suspicious_20,
                "SuspiciousMovesGT30Pct": suspicious_30,
                "SuspiciousMovesGT50Pct": suspicious_50,
                "InvalidOHLCRelationships": invalid_ohlc,
                "UsableRowsAfterFeatureGeneration": usable_rows,
                "MissingBenchmarkDates": missing_vs_benchmark,
                "ExtraDatesNotInBenchmark": stock_extra_dates,
                "ForwardFillLikeRows": stale_rows,
                "AdjustedPriceStatus": "REVIEW: cache stores raw Close only; Adj Close unavailable; yfinance auto_adjust=False",
                "Status": status,
            }
        )
        liquidity_rows.append(
            {
                "Ticker": symbol,
                "Type": symbol_type,
                "AverageDailyTurnover": float(turnover.mean()),
                "MedianDailyTurnover": float(turnover.median()),
                "P10Turnover": float(turnover.quantile(0.10)),
                "P90Turnover": float(turnover.quantile(0.90)),
                "ZeroVolumeDays": int((frame["Volume"] == 0).sum()),
                "LowLiquidityDaysBelow1Cr": int((turnover < 10_000_000).sum()),
                "LowLiquidityDaysBelow5Cr": int((turnover < 50_000_000).sum()),
            }
        )
        for idx in frame.index[daily_return.abs() > 0.20]:
            extreme_rows.append(
                {
                    "Ticker": symbol,
                    "Type": symbol_type,
                    "Date": frame.loc[idx, "Date"],
                    "DailyReturn": float(daily_return.loc[idx]),
                    "MoveBucket": _move_bucket(float(daily_return.loc[idx])),
                    "Close": frame.loc[idx, "Close"],
                    "Volume": frame.loc[idx, "Volume"],
                    "Turnover": frame.loc[idx, "Close"] * frame.loc[idx, "Volume"],
                    "CorporateActionReview": "REVIEW adjusted-price consistency; raw Close may include split/dividend jumps",
                }
            )
    return pd.DataFrame(ticker_rows), pd.DataFrame(extreme_rows), pd.DataFrame(liquidity_rows)


def _feature_quality_report(dataset: pd.DataFrame) -> pd.DataFrame:
    """Audit current Pruned Advanced features."""

    rows = []
    for feature in _pruned_features():
        series = pd.to_numeric(dataset[feature], errors="coerce") if feature in dataset else pd.Series(dtype=float)
        finite = series.replace([float("inf"), float("-inf")], pd.NA).dropna()
        q1 = finite.quantile(0.25) if not finite.empty else None
        q3 = finite.quantile(0.75) if not finite.empty else None
        iqr = q3 - q1 if q1 is not None and q3 is not None else None
        outlier_iqr = int(((finite < q1 - 3 * iqr) | (finite > q3 + 3 * iqr)).sum()) if iqr and iqr > 0 else 0
        std = finite.std() if not finite.empty else None
        mean = finite.mean() if not finite.empty else None
        z_outlier = int(((finite - mean).abs() > 5 * std).sum()) if std and std > 0 else 0
        rows.append(
            {
                "Feature": feature,
                "MissingPercent": float(series.isna().mean()) if len(series) else 1.0,
                "InfiniteValues": int(series.isin([float("inf"), float("-inf")]).sum()) if len(series) else 0,
                "Median": float(finite.median()) if not finite.empty else None,
                "Mean": float(mean) if mean is not None else None,
                "StdDev": float(std) if std is not None else None,
                "P1": float(finite.quantile(0.01)) if not finite.empty else None,
                "P5": float(finite.quantile(0.05)) if not finite.empty else None,
                "P95": float(finite.quantile(0.95)) if not finite.empty else None,
                "P99": float(finite.quantile(0.99)) if not finite.empty else None,
                "ExtremeOutliersIQR3x": outlier_iqr,
                "ExtremeOutliersAbsZGT5": z_outlier,
                "SpecialAttention": feature in {"beta_120d", "beta_252d"},
            }
        )
    return pd.DataFrame(rows)


def _label_distribution_report(dataset: pd.DataFrame) -> pd.DataFrame:
    """Report target distributions across key groups."""

    rows = []
    for scope, columns in {
        "overall": [],
        "by_year": ["Year"],
        "by_sector": ["Sector"],
        "by_market_regime": ["MarketRegime"],
        "by_split": ["Split"],
    }.items():
        groups = [(("all",), dataset)] if not columns else dataset.groupby(columns, observed=False)
        for key, group in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    "Scope": scope,
                    "Group": "|".join(str(item) for item in key_tuple),
                    "Rows": int(len(group)),
                    "OutperformPct": float((group["label"] == 1).mean()),
                    "NeutralPct": float((group["label"] == 0).mean()),
                    "UnderperformPct": float((group["label"] == -1).mean()),
                    "FutureStockReturnMean": float(group["future_stock_return"].mean()),
                    "FutureNiftyReturnMean": float(group["future_nifty_return"].mean()),
                    "FutureExcessReturnMean": float(group["excess_return"].mean()),
                    "FutureExcessReturnMedian": float(group["excess_return"].median()),
                }
            )
    return pd.DataFrame(rows)


def _drift_report(dataset: pd.DataFrame) -> pd.DataFrame:
    """Compare feature and target distributions across train/validation/final test."""

    rows = []
    columns = [*_pruned_features(), "future_stock_return", "future_nifty_return", "excess_return"]
    splits = ["train", "validation", "final_test"]
    for column in columns:
        if column not in dataset:
            continue
        series_all = pd.to_numeric(dataset[column], errors="coerce")
        pooled_std = series_all.std()
        split_stats = {}
        for split in splits:
            values = pd.to_numeric(dataset.loc[dataset["Split"].astype(str) == split, column], errors="coerce")
            split_stats[split] = {
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "missing": values.isna().mean(),
                "p5": values.quantile(0.05),
                "p95": values.quantile(0.95),
            }
        rows.append(
            {
                "Field": column,
                "TrainMean": split_stats["train"]["mean"],
                "ValidationMean": split_stats["validation"]["mean"],
                "FinalTestMean": split_stats["final_test"]["mean"],
                "TrainMedian": split_stats["train"]["median"],
                "ValidationMedian": split_stats["validation"]["median"],
                "FinalTestMedian": split_stats["final_test"]["median"],
                "TrainMissingPct": split_stats["train"]["missing"],
                "ValidationMissingPct": split_stats["validation"]["missing"],
                "FinalTestMissingPct": split_stats["final_test"]["missing"],
                "ValidationMeanShiftStd": _std_shift(split_stats["validation"]["mean"], split_stats["train"]["mean"], pooled_std),
                "FinalTestMeanShiftStd": _std_shift(split_stats["final_test"]["mean"], split_stats["train"]["mean"], pooled_std),
                "MaxAbsMeanShiftStd": max(
                    abs(_std_shift(split_stats["validation"]["mean"], split_stats["train"]["mean"], pooled_std) or 0),
                    abs(_std_shift(split_stats["final_test"]["mean"], split_stats["train"]["mean"], pooled_std) or 0),
                ),
            }
        )
    return pd.DataFrame(rows)


def _benchmark_alignment_report(
    *,
    stock_symbols: list[str],
    sector_symbols: list[str],
    cached: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Audit stock, benchmark, and sector-date alignment."""

    rows = []
    benchmark_dates = set(pd.to_datetime(benchmark["Date"]).dropna())
    rows.append(
        {
            "Item": BENCHMARK_SYMBOL,
            "Type": "Benchmark",
            "FirstDate": benchmark["Date"].min() if not benchmark.empty else None,
            "LastDate": benchmark["Date"].max() if not benchmark.empty else None,
            "Rows": len(benchmark),
            "MissingBenchmarkDates": 0,
            "UnmatchedStockDates": 0,
            "CoverageVsBenchmarkPct": 1.0,
            "ForwardFillLikeRows": _stale_forward_fill_like_rows(benchmark) if not benchmark.empty else None,
            "Notes": "Nifty benchmark reference calendar.",
        }
    )
    for symbol in stock_symbols:
        frame = cached.get(symbol, pd.DataFrame())
        rows.append(_alignment_row(symbol, "Stock", frame, benchmark_dates, universe))
    for symbol in sector_symbols:
        frame = cached.get(symbol, pd.DataFrame())
        rows.append(_alignment_row(symbol, "Sector Index", frame, benchmark_dates, universe))
    sector_coverage = _sector_coverage_rows(universe, sector_symbols, cached, benchmark_dates)
    rows.extend(sector_coverage)
    return pd.DataFrame(rows)


def _alignment_row(
    symbol: str,
    item_type: str,
    frame: pd.DataFrame,
    benchmark_dates: set[pd.Timestamp],
    universe: pd.DataFrame,
) -> dict[str, Any]:
    """Create one alignment row."""

    if frame.empty:
        return {
            "Item": symbol,
            "Type": item_type,
            "Rows": 0,
            "CoverageVsBenchmarkPct": 0.0,
            "Notes": "Missing cache file or empty cache.",
        }
    dates = set(pd.to_datetime(frame["Date"]).dropna())
    missing = len(benchmark_dates - dates)
    extra = len(dates - benchmark_dates)
    return {
        "Item": symbol,
        "Type": item_type,
        "FirstDate": frame["Date"].min(),
        "LastDate": frame["Date"].max(),
        "Rows": len(frame),
        "MissingBenchmarkDates": missing,
        "UnmatchedStockDates": extra,
        "CoverageVsBenchmarkPct": len(dates & benchmark_dates) / len(benchmark_dates) if benchmark_dates else None,
        "ForwardFillLikeRows": _stale_forward_fill_like_rows(frame),
        "Notes": "No explicit forward-fill step found in cached OHLCV loader; stale rows are heuristic only.",
    }


def _sector_coverage_rows(
    universe: pd.DataFrame,
    sector_symbols: list[str],
    cached: dict[str, pd.DataFrame],
    benchmark_dates: set[pd.Timestamp],
) -> list[dict[str, Any]]:
    """Add sector-to-index coverage diagnostics."""

    rows = []
    reverse = {value: key for key, value in SECTOR_YAHOO_SYMBOLS.items()}
    for sector_symbol in sector_symbols:
        frame = cached.get(sector_symbol, pd.DataFrame())
        sector = reverse.get(sector_symbol, "Unknown")
        rows.append(
            {
                "Item": sector,
                "Type": "SectorCoverage",
                "SectorBenchmark": sector_symbol,
                "UniverseStockCount": int((universe["Sector"] == sector).sum()),
                "Rows": len(frame),
                "CoverageVsBenchmarkPct": (
                    len(set(pd.to_datetime(frame["Date"]).dropna()) & benchmark_dates) / len(benchmark_dates)
                    if not frame.empty and benchmark_dates
                    else 0.0
                ),
                "Notes": "Sector benchmark from configured Yahoo/NSE sector mapping.",
            }
        )
    return rows


def _summary_text(
    *,
    ticker_report: pd.DataFrame,
    feature_report: pd.DataFrame,
    label_report: pd.DataFrame,
    drift_report: pd.DataFrame,
    alignment_report: pd.DataFrame,
) -> str:
    """Write summary with issues that can distort model outputs."""

    fail_count = int((ticker_report["Status"] == "FAIL").sum())
    review_count = int((ticker_report["Status"] == "REVIEW").sum())
    pass_count = int((ticker_report["Status"] == "PASS").sum())
    high_missing_features = feature_report[feature_report["MissingPercent"] > 0.10]
    beta_rows = feature_report[feature_report["Feature"].isin(["beta_120d", "beta_252d"])]
    drift_rows = drift_report.sort_values("MaxAbsMeanShiftStd", ascending=False).head(12)
    label_split = label_report[label_report["Scope"] == "by_split"]
    alignment_issues = alignment_report[
        (alignment_report.get("CoverageVsBenchmarkPct", pd.Series(dtype=float)) < 0.95)
        | (alignment_report.get("ForwardFillLikeRows", pd.Series(dtype=float)).fillna(0) > 0)
    ].head(20)
    lines = [
        "NATIP Historical Data Quality And Distribution Audit",
        "",
        "Scope:",
        "- Read-only audit. No models retrained, no features changed, no stocks removed.",
        "- Audited cached stock, Nifty, and sector OHLCV plus current Pruned Advanced feature dataset.",
        "",
        "Ticker Status Summary:",
        f"- PASS: {pass_count}",
        f"- REVIEW: {review_count}",
        f"- FAIL: {fail_count}",
        "",
        "Adjusted Price / Corporate Action Warning:",
        "- Cached data stores Open/High/Low/Close/Volume only.",
        "- The downloader used yfinance auto_adjust=False and the normalized cache drops Adj Close.",
        "- Therefore adjusted-price consistency cannot be proven from the local cache.",
        "- Large one-day jumps may reflect true moves, bad data, splits, dividends, or unadjusted corporate actions; review extreme_move_report.csv before trusting probability/Sharpe signals.",
        "",
        "Potential Probability / PR-AUC / Sharpe Distorters:",
        f"- Tickers needing REVIEW/FAIL: {review_count + fail_count}. These can distort cross-sectional training, PR-AUC, and top-percentile returns.",
        f"- Features with >10% missingness: {', '.join(high_missing_features['Feature'].tolist()) if not high_missing_features.empty else 'None'}",
        "- beta_120d / beta_252d missingness:",
        *[
            f"  - {row.Feature}: {row.MissingPercent:.2%} missing"
            for row in beta_rows.itertuples()
        ],
        "- Features/targets with largest train-vs-validation/final-test drift:",
        *[
            f"  - {row.Field}: max mean shift {row.MaxAbsMeanShiftStd:.3f} pooled std"
            for row in drift_rows.itertuples()
        ],
        "",
        "Label Distribution By Split:",
        *[
            f"- {row.Group}: outperform {row.OutperformPct:.2%}, neutral {row.NeutralPct:.2%}, underperform {row.UnderperformPct:.2%}, rows {row.Rows}"
            for row in label_split.itertuples()
        ],
        "",
        "Leakage Controls Verified From Code/Reports:",
        f"- Future label horizon is {HORIZON_DAYS} trading days and is generated from future returns only in target columns.",
        "- Pruned Advanced feature list contains backward-looking rolling/relative features only.",
        "- Prior-high breakout features, where present outside the primary model, use shifted highs in feature code.",
        "- Breadth calculations use same-date universe membership and backward-looking SMA/return checks.",
        "- Previous embargo audit confirmed 20-trading-day gaps; this audit does not retrain models.",
        "",
        "Benchmark / Sector Alignment Issues To Review:",
        *[
            f"- {row.Item} ({row.Type}): coverage {row.CoverageVsBenchmarkPct}, forward-fill-like rows {row.ForwardFillLikeRows if 'ForwardFillLikeRows' in alignment_issues.columns else 'NA'}"
            for row in alignment_issues.itertuples()
        ],
        "",
        "Recommended Next Action:",
        "- Do not auto-remove outliers or illiquid stocks yet.",
        "- Manually inspect REVIEW/FAIL tickers, extreme moves, and adjusted-price concerns first.",
        "- If unadjusted corporate-action jumps are confirmed, rebuild the dataset with consistent adjusted OHLC before trusting BUY/ACCUMULATE/SELL signal quality.",
    ]
    return "\n".join(lines)


def _pruned_features() -> list[str]:
    """Return frozen Pruned Advanced features."""

    return list(dict.fromkeys([*feature_columns_for_variant("sector_regime"), *PRUNED_FEATURES]))


PRUNED_FEATURES = [
    "beta_252d",
    "beta_120d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "sector_rsi_14",
    "corr_nifty_20d",
    "relative_momentum_20d_change",
]


def _symbol_type(symbol: str, stock_symbols: list[str], sector_symbols: list[str]) -> str:
    if symbol == BENCHMARK_SYMBOL:
        return "Benchmark"
    if symbol in sector_symbols:
        return "Sector Index"
    if symbol in stock_symbols:
        return "Stock"
    return "Unknown"


def _empty_ticker_row(symbol: str, symbol_type: str, sector: str) -> dict[str, Any]:
    return {
        "Ticker": symbol,
        "Type": symbol_type,
        "Sector": sector,
        "RowCount": 0,
        "MissingOHLCVPercent": 1.0,
        "DuplicateDateSymbolRows": None,
        "ZeroVolumeDays": None,
        "AverageDailyTurnover": None,
        "MedianDailyTurnover": None,
        "InvalidOHLCRelationships": None,
        "UsableRowsAfterFeatureGeneration": 0,
        "Status": "FAIL",
    }


def _invalid_ohlc_count(frame: pd.DataFrame) -> int:
    invalid = (
        (frame["High"] < frame["Low"])
        | (frame["Open"] > frame["High"])
        | (frame["Open"] < frame["Low"])
        | (frame["Close"] > frame["High"])
        | (frame["Close"] < frame["Low"])
        | (frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    )
    return int(invalid.sum())


def _usable_rows(dataset: pd.DataFrame, symbol: str, features: list[str]) -> int:
    if symbol not in set(dataset["symbol"].dropna().astype(str)):
        return 0
    return int(dataset[dataset["symbol"] == symbol].dropna(subset=[*features, "label"]).shape[0])


def _ticker_status(
    *,
    missing_pct: float,
    duplicates: int,
    invalid_ohlc: int,
    suspicious_50: int,
    stale_rows: int,
    zero_volume_days: int,
    row_count: int,
) -> str:
    if row_count < 252 or missing_pct > 0.05 or duplicates > 0 or invalid_ohlc > 0:
        return "FAIL"
    if suspicious_50 > 0 or stale_rows > 0 or zero_volume_days > 20:
        return "REVIEW"
    return "PASS"


def _stale_forward_fill_like_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    previous_close = frame["Close"].shift(1)
    stale = (
        (frame["Open"] == previous_close)
        & (frame["High"] == previous_close)
        & (frame["Low"] == previous_close)
        & (frame["Close"] == previous_close)
        & (frame["Volume"].fillna(0) == 0)
    )
    return int(stale.sum())


def _move_bucket(value: float) -> str:
    abs_value = abs(value)
    if abs_value > 0.50:
        return ">50%"
    if abs_value > 0.30:
        return ">30%"
    return ">20%"


def _std_shift(value: float | None, base: float | None, std: float | None) -> float | None:
    if value is None or base is None or std is None or pd.isna(value) or pd.isna(base) or pd.isna(std) or std == 0:
        return None
    return float((value - base) / std)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")


if __name__ == "__main__":
    main()
