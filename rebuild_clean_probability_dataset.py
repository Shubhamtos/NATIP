"""Rebuild clean adjusted NSE data and compare clean versus old ML research results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import (
    BENCHMARK_SYMBOL,
    DATA_DIR,
    REPORT_DIR,
    START_DATE,
    STOCKS_UNIVERSE_2026_08_CSV,
    ensure_probability_dirs,
)
from app.probability.data_download import load_stock_universe, load_universe_metadata
from app.probability.features import build_feature_dataset, feature_columns_for_variant
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS, sector_benchmark_symbols
from app.probability.targets import add_forward_excess_return_labels
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS, _embargo_folds
from run_ranker_binary_signal_research import (
    _binary_evaluation_rows,
    _fit_calibration_split,
    _fit_classifier,
    _fit_isotonic,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _rank_ic_stability,
    _ranker_results,
    _transform_with_medians,
)

CLEAN_CACHE_DIR = DATA_DIR / "clean_cache_adjusted"
CLEAN_DATASET_PATH = DATA_DIR / "probability_training_dataset_clean.csv"
DOWNLOAD_METADATA_PATH = CLEAN_CACHE_DIR / "download_metadata.json"

OUTPUT_QUALITY = REPORT_DIR / "clean_data_quality_report.csv"
OUTPUT_CA_AUDIT = REPORT_DIR / "corporate_action_audit.csv"
OUTPUT_STALE = REPORT_DIR / "stale_row_report.csv"
OUTPUT_LABELS = REPORT_DIR / "clean_label_distribution.csv"
OUTPUT_EMBARGO = REPORT_DIR / "clean_embargo_audit.csv"
OUTPUT_BETA = REPORT_DIR / "beta_ablation.csv"
OUTPUT_COMPARISON = REPORT_DIR / "clean_vs_old_model_comparison.csv"
OUTPUT_SUMMARY = REPORT_DIR / "clean_data_summary.txt"

AUDIT_ESTIMATORS = 80
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
ADJ_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRUNED_EXTRA_FEATURES = [
    "beta_252d",
    "beta_120d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "sector_rsi_14",
    "corr_nifty_20d",
    "relative_momentum_20d_change",
]


def main() -> None:
    """Rebuild adjusted cache, regenerate dataset, and rerun requested clean comparisons."""

    ensure_probability_dirs()
    CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    universe = load_universe_metadata(STOCKS_UNIVERSE_2026_08_CSV)
    stock_symbols = load_stock_universe(STOCKS_UNIVERSE_2026_08_CSV)
    sector_symbols = sector_benchmark_symbols(universe)
    symbols = [*stock_symbols, *sector_symbols, BENCHMARK_SYMBOL]

    clean_frames, quality, stale, corporate_actions, metadata = _download_and_clean(
        symbols=symbols,
        stock_symbols=stock_symbols,
        sector_symbols=sector_symbols,
    )
    DOWNLOAD_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    stock_data = {
        symbol: frame
        for symbol, frame in clean_frames.items()
        if symbol in [*stock_symbols, BENCHMARK_SYMBOL]
    }
    sector_frames = {
        sector: clean_frames[benchmark]
        for sector, benchmark in SECTOR_YAHOO_SYMBOLS.items()
        if benchmark in clean_frames and not clean_frames[benchmark].empty
    }
    features = build_feature_dataset(
        stock_data,
        benchmark_symbol=BENCHMARK_SYMBOL,
        metadata=universe,
        sector_frames=sector_frames,
        include_sector_features=True,
        include_market_regime_features=True,
    )
    dataset = add_forward_excess_return_labels(features, clean_frames[BENCHMARK_SYMBOL])
    dataset = dataset.merge(
        universe[["Ticker", "Company", "Sector", "MarketCapCategory", "Group"]].rename(
            columns={"Ticker": "symbol"}
        ),
        on="symbol",
        how="left",
    )
    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["Split"] = pd.cut(
        dataset["Date"],
        bins=[pd.Timestamp.min, pd.Timestamp("2022-12-31"), pd.Timestamp("2024-12-31"), pd.Timestamp.max],
        labels=["train", "validation", "final_test"],
    )
    dataset["MarketRegime"] = dataset.apply(_market_regime_label, axis=1)
    dataset["buy_target"] = (dataset["excess_return"] > 0.05).astype(int)
    dataset.to_csv(CLEAN_DATASET_PATH, index=False)

    label_distribution = _label_distribution(dataset)
    embargo_audit = _clean_embargo_audit(dataset)
    comparison, beta = _clean_model_comparisons(dataset)

    quality.to_csv(OUTPUT_QUALITY, index=False)
    corporate_actions.to_csv(OUTPUT_CA_AUDIT, index=False)
    stale.to_csv(OUTPUT_STALE, index=False)
    label_distribution.to_csv(OUTPUT_LABELS, index=False)
    embargo_audit.to_csv(OUTPUT_EMBARGO, index=False)
    beta.to_csv(OUTPUT_BETA, index=False)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    OUTPUT_SUMMARY.write_text(
        _summary_text(quality, stale, corporate_actions, label_distribution, comparison, beta, metadata),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "clean_data_quality_report": str(OUTPUT_QUALITY),
                "corporate_action_audit": str(OUTPUT_CA_AUDIT),
                "stale_row_report": str(OUTPUT_STALE),
                "clean_label_distribution": str(OUTPUT_LABELS),
                "clean_embargo_audit": str(OUTPUT_EMBARGO),
                "beta_ablation": str(OUTPUT_BETA),
                "clean_vs_old_model_comparison": str(OUTPUT_COMPARISON),
                "clean_data_summary": str(OUTPUT_SUMMARY),
                "clean_dataset": str(CLEAN_DATASET_PATH),
                "download_metadata": str(DOWNLOAD_METADATA_PATH),
            },
            indent=2,
        )
    )


def _download_and_clean(
    *,
    symbols: list[str],
    stock_symbols: list[str],
    sector_symbols: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Download raw and adjusted OHLCV, flag stale rows, and return ML-ready frames."""

    clean_frames: dict[str, pd.DataFrame] = {}
    quality_rows = []
    stale_rows = []
    extreme_rows = []
    failures = []
    metadata = {
        "downloaded_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "source": "yfinance",
        "start_date": START_DATE,
        "raw_method": "yf.download(auto_adjust=False, actions=True)",
        "adjusted_method": "yf.download(auto_adjust=True, actions=True)",
        "ml_price_basis": "fully adjusted OHLC from yfinance auto_adjust=True; adjusted Open/High/Low/Close are used together",
        "volume_policy": "raw reported volume preserved; never forward-filled",
        "cache_dir": str(CLEAN_CACHE_DIR),
        "symbols_requested": symbols,
    }
    for index, symbol in enumerate(symbols, start=1):
        print(f"[download-clean] {index}/{len(symbols)} {symbol}", flush=True)
        symbol_type = _symbol_type(symbol, stock_symbols, sector_symbols)
        try:
            raw, adjusted = _download_symbol_pair(symbol)
            combined = _combine_raw_adjusted(raw, adjusted)
            flagged = _flag_rows(combined, symbol_type=symbol_type)
            flagged.to_csv(CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv", index=False)
            stale_part = flagged[
                flagged[["is_zero_volume", "is_stale_row", "is_duplicate_date", "is_invalid_ohlc"]].any(axis=1)
            ].copy()
            if not stale_part.empty:
                stale_part.insert(0, "Ticker", symbol)
                stale_part.insert(1, "Type", symbol_type)
                stale_rows.append(stale_part)
            clean = flagged[flagged["is_tradable_row"]].copy()
            extreme_rows.extend(_extreme_rows(symbol, symbol_type, clean))
            ml_frame = clean[["Date", "adj_Open", "adj_High", "adj_Low", "adj_Close", "Volume"]].rename(
                columns={
                    "adj_Open": "Open",
                    "adj_High": "High",
                    "adj_Low": "Low",
                    "adj_Close": "Close",
                }
            )
            clean_frames[symbol] = ml_frame.reset_index(drop=True)
            quality_rows.append(_quality_row(symbol, symbol_type, flagged, clean))
        except Exception as exc:
            failures.append({"Ticker": symbol, "Error": str(exc)})
            clean_frames[symbol] = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            quality_rows.append({"Ticker": symbol, "Type": symbol_type, "Status": "FAIL", "Error": str(exc)})
    metadata["failures"] = failures
    return (
        clean_frames,
        pd.DataFrame(quality_rows),
        pd.concat(stale_rows, ignore_index=True) if stale_rows else pd.DataFrame(),
        pd.DataFrame(extreme_rows),
        metadata,
    )


def _download_symbol_pair(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download raw and adjusted data for one symbol."""

    import yfinance as yf

    raw = yf.download(
        symbol,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    adjusted = yf.download(
        symbol,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        actions=True,
        progress=False,
        threads=False,
    )
    if raw.empty or adjusted.empty:
        raise ValueError("No raw or adjusted data returned.")
    return _flatten_download(raw), _flatten_download(adjusted)


def _flatten_download(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded yfinance frame."""

    output = frame.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = [str(column[0]) for column in output.columns]
    output = output.reset_index()
    if "Date" not in output.columns and "Datetime" in output.columns:
        output = output.rename(columns={"Datetime": "Date"})
    output["Date"] = pd.to_datetime(output["Date"]).dt.tz_localize(None)
    return output.sort_values("Date").reset_index(drop=True)


def _combine_raw_adjusted(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    """Preserve raw OHLC/Adj Close and adjusted OHLC used for ML."""

    raw_columns = [column for column in RAW_COLUMNS if column in raw.columns]
    adjusted_columns = [column for column in ADJ_COLUMNS if column in adjusted.columns]
    raw_part = raw[["Date", *raw_columns]].rename(
        columns={column: f"raw_{column.replace(' ', '_')}" for column in raw_columns}
    )
    adj_part = adjusted[["Date", *adjusted_columns]].rename(
        columns={column: f"adj_{column}" if column != "Volume" else "Volume" for column in adjusted_columns}
    )
    combined = raw_part.merge(adj_part, on="Date", how="outer").sort_values("Date")
    numeric = [column for column in combined.columns if column != "Date"]
    for column in numeric:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def _flag_rows(frame: pd.DataFrame, *, symbol_type: str) -> pd.DataFrame:
    """Create stale/tradable flags."""

    output = frame.copy().sort_values("Date")
    output["is_duplicate_date"] = output.duplicated("Date", keep="last")
    output["is_zero_volume"] = output["Volume"].fillna(0).eq(0)
    output["is_invalid_ohlc"] = (
        output[["adj_Open", "adj_High", "adj_Low", "adj_Close"]].isna().any(axis=1)
        | (output["adj_High"] < output["adj_Low"])
        | (output["adj_Open"] > output["adj_High"])
        | (output["adj_Open"] < output["adj_Low"])
        | (output["adj_Close"] > output["adj_High"])
        | (output["adj_Close"] < output["adj_Low"])
        | (output[["adj_Open", "adj_High", "adj_Low", "adj_Close"]] <= 0).any(axis=1)
    )
    previous_close = output["adj_Close"].shift(1)
    unchanged_ohlc = (
        output["adj_Open"].eq(previous_close)
        & output["adj_High"].eq(previous_close)
        & output["adj_Low"].eq(previous_close)
        & output["adj_Close"].eq(previous_close)
    )
    output["is_stale_row"] = output["is_zero_volume"] & unchanged_ohlc
    if symbol_type == "Stock":
        non_trading = output["is_zero_volume"] | output["is_stale_row"]
    else:
        non_trading = output["is_stale_row"]
    output["is_tradable_row"] = ~(output["is_duplicate_date"] | output["is_invalid_ohlc"] | non_trading)
    return output


def _quality_row(symbol: str, symbol_type: str, flagged: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    """Create one clean data quality row."""

    returns = clean["adj_Close"].pct_change() if not clean.empty else pd.Series(dtype=float)
    rows_removed = int(len(flagged) - len(clean))
    invalid_rows = int(flagged["is_invalid_ohlc"].sum())
    moves_gt_20 = int((returns.abs() > 0.20).sum())
    moves_gt_50 = int((returns.abs() > 0.50).sum())
    status, status_reason = _quality_status(
        clean_rows=len(clean),
        adjusted_missing_pct=float(flagged[["adj_Open", "adj_High", "adj_Low", "adj_Close"]].isna().mean().mean()),
        rows_removed=rows_removed,
        invalid_rows=invalid_rows,
        moves_gt_20=moves_gt_20,
        moves_gt_50=moves_gt_50,
    )
    return {
        "Ticker": symbol,
        "Type": symbol_type,
        "FirstDate": flagged["Date"].min(),
        "LastDate": flagged["Date"].max(),
        "RawRows": int(len(flagged)),
        "CleanTradableRows": int(len(clean)),
        "RowsRemoved": rows_removed,
        "ZeroVolumeRows": int(flagged["is_zero_volume"].sum()),
        "StaleRows": int(flagged["is_stale_row"].sum()),
        "DuplicateDateRows": int(flagged["is_duplicate_date"].sum()),
        "InvalidOHLCRows": invalid_rows,
        "AdjustedMissingPct": float(flagged[["adj_Open", "adj_High", "adj_Low", "adj_Close"]].isna().mean().mean()),
        "MaxPositive1DReturnAfterAdjustment": float(returns.max()) if not returns.empty else None,
        "MaxNegative1DReturnAfterAdjustment": float(returns.min()) if not returns.empty else None,
        "MovesGT20PctAfterAdjustment": moves_gt_20,
        "MovesGT30PctAfterAdjustment": int((returns.abs() > 0.30).sum()),
        "MovesGT50PctAfterAdjustment": moves_gt_50,
        "AdjustmentMethod": "raw preserved; adjusted OHLC from yfinance auto_adjust=True used for ML",
        "Status": status,
        "StatusReason": status_reason,
    }


def _quality_status(
    *,
    clean_rows: int,
    adjusted_missing_pct: float,
    rows_removed: int,
    invalid_rows: int,
    moves_gt_20: int,
    moves_gt_50: int,
) -> tuple[str, str]:
    """Classify cleaned data quality without penalizing rows already excluded from ML."""

    if clean_rows < 252:
        return "FAIL", "Less than one trading year remains after cleaning."
    if adjusted_missing_pct > 0.05:
        return "FAIL", "Adjusted OHLC missingness above 5%."
    if moves_gt_50 > 0:
        return "REVIEW", "Post-adjustment move above 50% requires manual corporate-action/data review."
    if moves_gt_20 > 0:
        return "REVIEW", "Post-adjustment move above 20% retained but requires review."
    if rows_removed > 10 or invalid_rows > 2:
        return "REVIEW", "Elevated stale/invalid rows were excluded before modeling."
    return "PASS", "Clean adjusted OHLCV is sufficient after stale/non-trading exclusions."


def _extreme_rows(symbol: str, symbol_type: str, clean: pd.DataFrame) -> list[dict[str, Any]]:
    """Classify post-adjustment extreme moves."""

    rows = []
    returns = clean["adj_Close"].pct_change()
    rolling_vol = returns.rolling(60).std()
    for idx in clean.index[returns.abs() > 0.20]:
        value = float(returns.loc[idx])
        rows.append(
            {
                "Ticker": symbol,
                "Type": symbol_type,
                "Date": clean.loc[idx, "Date"],
                "AdjustedDailyReturn": value,
                "MoveBucket": _move_bucket(value),
                "AdjustedClose": clean.loc[idx, "adj_Close"],
                "RawClose": clean.loc[idx, "raw_Close"] if "raw_Close" in clean.columns else None,
                "Volume": clean.loc[idx, "Volume"],
                "Classification": _classify_extreme_move(value, rolling_vol.loc[idx], clean.loc[idx, "Volume"]),
                "ReviewNote": "Post-adjustment move; do not remove unless verified bad data.",
            }
        )
    return rows


def _classify_extreme_move(daily_return: float, rolling_vol: float, volume: float) -> str:
    """Heuristic classification for adjusted extreme moves."""

    if abs(daily_return) > 0.50:
        return "suspicious/bad data"
    if pd.notna(rolling_vol) and rolling_vol > 0 and abs(daily_return) > 8 * rolling_vol:
        return "suspicious/bad data"
    if pd.isna(volume) or volume == 0:
        return "likely corporate-action artifact"
    return "plausible real market move"


def _clean_model_comparisons(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain requested clean models and compare against old validation results."""

    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)
    features = _pruned_features()
    no_beta252 = [feature for feature in features if feature != "beta_252d"]

    pruned_predictions = _classifier_walk_forward(pre_final, folds, features, variant="clean_pruned_advanced")
    no_beta_predictions = _classifier_walk_forward(pre_final, folds, no_beta252, variant="clean_pruned_without_beta_252d")
    ranker_predictions = _ranker_walk_forward(pre_final, folds, features)
    buy_predictions = _binary_buy_walk_forward(pre_final, folds, features)

    comparison_rows = [
        _classifier_metrics(pruned_predictions, "clean_pruned_advanced"),
        _ranker_metric_row(ranker_predictions),
        _binary_buy_metric_row(buy_predictions),
    ]
    comparison_rows.extend(_old_metric_rows())
    beta_rows = [
        _classifier_metrics(pruned_predictions, "clean_pruned_advanced"),
        _classifier_metrics(no_beta_predictions, "clean_pruned_without_beta_252d"),
    ]
    return pd.DataFrame(comparison_rows), pd.DataFrame(beta_rows)


def _clean_embargo_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    """Report leakage-safe 20-trading-day embargo windows for the clean dataset."""

    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    rows = []
    for fold in _embargo_folds(pre_final):
        rows.append(
            {
                "Fold": fold["fold"],
                "TrainStart": fold["train_start"],
                "TrainEnd": fold["train_end"],
                "EmbargoStart": fold["embargo_start"],
                "EmbargoEnd": fold["embargo_end"],
                "ValidationStart": fold["validation_start"],
                "ValidationEnd": fold["validation_end"],
                "TrainTradingDays": len(fold["train_dates"]),
                "EmbargoTradingDays": HORIZON_DAYS,
                "ValidationTradingDays": len(fold["validation_dates"]),
                "TrainRows": int(dataset["Date"].isin(fold["train_dates"]).sum()),
                "ValidationRows": int(dataset["Date"].isin(fold["validation_dates"]).sum()),
                "EmbargoSatisfied": bool(fold["train_end"] < fold["embargo_start"] < fold["validation_start"]),
            }
        )
    return pd.DataFrame(rows)


def _classifier_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    *,
    variant: str,
) -> pd.DataFrame:
    """Walk-forward Pruned Advanced classifier on clean data."""

    frames = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        model, medians = _fit_classifier(train, features, target="label")
        x = _transform_with_medians(validation, features, medians)
        probabilities = model.predict_proba(x)
        output = _base_prediction_frame(validation, fold["fold"])
        output["Variant"] = variant
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["predicted_class"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map({0: -1, 1: 0, 2: 1})
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    """Walk-forward XGBRanker on clean data."""

    frames = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        model, medians = _fit_ranker(train, features)
        x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        output["ranker_score"] = model.predict(x)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_buy_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    """Walk-forward XGBoost BUY model on clean data."""

    frames = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        fit_frame, calibration_frame = _fit_calibration_split(train)
        model, medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
        cal_x = _transform_with_medians(calibration_frame, features, medians)
        val_x = _transform_with_medians(validation, features, medians)
        cal_prob = _positive_probability(model, cal_x)
        val_prob = _positive_probability(model, val_x)
        platt = _fit_platt(cal_prob, calibration_frame["buy_target"])
        isotonic = _fit_isotonic(cal_prob, calibration_frame["buy_target"])
        output = _base_prediction_frame(validation, fold["fold"])
        output["actual"] = validation["buy_target"].astype(int).to_numpy()
        output["p_uncalibrated"] = val_prob
        output["p_platt"] = platt.predict_proba(val_prob.reshape(-1, 1))[:, 1]
        output["p_isotonic"] = isotonic.predict(val_prob)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _classifier_metrics(predictions: pd.DataFrame, variant: str) -> dict[str, Any]:
    """Classifier metrics."""

    from sklearn.metrics import average_precision_score, log_loss

    y_true = predictions["label"].map(LABEL_TO_CLASS)
    probabilities = predictions[["p_underperform", "p_neutral", "p_outperform"]]
    top10 = predictions.sort_values("p_outperform", ascending=False).head(max(1, int(len(predictions) * 0.10)))
    return {
        "Dataset": "clean",
        "Model": variant,
        "Rows": len(predictions),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "BrierOutperform": float(((predictions["p_outperform"] - (predictions["label"] == 1).astype(int)) ** 2).mean()),
        "PRAUCOutperform": float(average_precision_score((predictions["label"] == 1).astype(int), predictions["p_outperform"])),
        "Top10AvgExcessReturn": float(top10["excess_return"].mean()),
        "Top10OutperformRate": float((top10["label"] == 1).mean()),
        "MeanIC": _mean_ic(predictions, "p_outperform"),
    }


def _ranker_metric_row(predictions: pd.DataFrame) -> dict[str, Any]:
    """Ranker metrics."""

    results = _ranker_results(predictions)
    all_ic = results[(results["ReportType"] == "ic_summary") & (results["Group"] == "all")].iloc[0]
    top10 = predictions.sort_values("ranker_score", ascending=False).head(max(1, int(len(predictions) * 0.10)))
    return {
        "Dataset": "clean",
        "Model": "clean_xgbranker",
        "Rows": len(predictions),
        "MeanIC": all_ic["MeanIC"],
        "MedianIC": all_ic["MedianIC"],
        "PositiveICDatePct": all_ic["PositiveICDatePct"],
        "Top10AvgExcessReturn": float(top10["excess_return"].mean()),
        "Top10WinRate": float((top10["excess_return"] > 0).mean()),
    }


def _binary_buy_metric_row(predictions: pd.DataFrame) -> dict[str, Any]:
    """Binary BUY metrics."""

    rows, base = _binary_evaluation_rows(_as_binary_eval_frame(predictions), _BuySpec(), "uncalibrated")
    top1 = next(row for row in rows if row["TopPercentile"] == "Top 1%")
    return {
        "Dataset": "clean",
        "Model": "clean_xgboost_buy",
        "Rows": len(predictions),
        "LogLoss": base["LogLoss"],
        "Brier": base["Brier"],
        "PRAUC": base["PRAUC"],
        "CalibrationMAE": base["CalibrationMAE"],
        "Top1Precision": top1["PrecisionAtPercentile"],
        "Top1AvgDirectionalExcessReturn": top1["AverageDirectionalExcessReturn"],
    }


class _BuySpec:
    model_name = "xgboost"
    target = "buy_target"


def _as_binary_eval_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["Model"] = "xgboost"
    frame["Target"] = "buy_target"
    return frame


def _old_metric_rows() -> list[dict[str, Any]]:
    """Load old model summary rows if available."""

    rows = []
    paths = [
        REPORT_DIR / "target_comparison.csv",
        REPORT_DIR / "xgbranker_results.csv",
        REPORT_DIR / "binary_buy_sell_results.csv",
    ]
    if paths[0].exists():
        old = pd.read_csv(paths[0])
        match = old[(old.get("ModelVariant") == "pruned_fixed_target") & (old.get("Scope") == "walk_forward_validation")]
        if not match.empty:
            row = match.iloc[0]
            rows.append({"Dataset": "old", "Model": "old_pruned_advanced", "Rows": row.get("Rows"), "LogLoss": row.get("LogLoss"), "BrierOutperform": row.get("Brier"), "PRAUCOutperform": row.get("PRAUC"), "MeanIC": row.get("RankICMean")})
    if paths[1].exists():
        old = pd.read_csv(paths[1])
        match = old[(old.get("ReportType") == "ic_summary") & (old.get("Group") == "all")]
        if not match.empty:
            row = match.iloc[0]
            rows.append({"Dataset": "old", "Model": "old_xgbranker", "MeanIC": row.get("MeanIC"), "MedianIC": row.get("MedianIC"), "PositiveICDatePct": row.get("PositiveICDatePct")})
    if paths[2].exists():
        old = pd.read_csv(paths[2])
        match = old[(old.get("Model") == "xgboost") & (old.get("Target") == "buy_target") & (old.get("Calibration") == "uncalibrated")]
        if not match.empty:
            row = match.sort_values("PRAUC", ascending=False).iloc[0]
            rows.append({"Dataset": "old", "Model": "old_xgboost_buy", "LogLoss": row.get("LogLoss"), "Brier": row.get("Brier"), "PRAUC": row.get("PRAUC"), "CalibrationMAE": row.get("CalibrationMAE")})
    return rows


def _base_prediction_frame(frame: pd.DataFrame, fold: Any) -> pd.DataFrame:
    output = frame[["Date", "symbol", "Sector", "MarketCapCategory", "future_stock_return", "future_nifty_return", "excess_return", "label", "buy_target", "MarketRegime"]].copy()
    output["fold"] = fold
    output = output.rename(columns={"MarketRegime": "market_regime_label"})
    return output


def _label_distribution(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, columns in {"overall": [], "by_split": ["Split"], "by_year": [dataset["Date"].dt.year], "by_sector": ["Sector"], "by_market_regime": ["MarketRegime"]}.items():
        groups = [(("all",), dataset)] if not columns else dataset.groupby(columns, observed=False)
        for key, group in groups:
            key_tuple = key if isinstance(key, tuple) else (key,)
            rows.append({"Scope": scope, "Group": "|".join(str(item) for item in key_tuple), "Rows": len(group), "OutperformPct": float((group["label"] == 1).mean()), "NeutralPct": float((group["label"] == 0).mean()), "UnderperformPct": float((group["label"] == -1).mean()), "MeanExcessReturn": float(group["excess_return"].mean()), "MedianExcessReturn": float(group["excess_return"].median())})
    return pd.DataFrame(rows)


def _market_regime_label(row: pd.Series) -> str:
    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _pruned_features() -> list[str]:
    return list(dict.fromkeys([*feature_columns_for_variant("sector_regime"), *PRUNED_EXTRA_FEATURES]))


def _mean_ic(predictions: pd.DataFrame, score_column: str) -> float | None:
    values = []
    for _, group in predictions.groupby("Date"):
        if len(group) < 5 or group[score_column].nunique() < 2:
            continue
        ic = group[score_column].corr(group["excess_return"], method="spearman")
        if pd.notna(ic):
            values.append(ic)
    return float(pd.Series(values).mean()) if values else None


def _move_bucket(value: float) -> str:
    if abs(value) > 0.50:
        return ">50%"
    if abs(value) > 0.30:
        return ">30%"
    return ">20%"


def _symbol_type(symbol: str, stock_symbols: list[str], sector_symbols: list[str]) -> str:
    if symbol == BENCHMARK_SYMBOL:
        return "Benchmark"
    if symbol in sector_symbols:
        return "Sector Index"
    if symbol in stock_symbols:
        return "Stock"
    return "Unknown"


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")


def _summary_text(
    quality: pd.DataFrame,
    stale: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    labels: pd.DataFrame,
    comparison: pd.DataFrame,
    beta: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "NATIP Clean Data Rebuild Summary",
            "",
            "Data actions:",
            "- Redownloaded raw OHLCV with Adj Close and separately redownloaded fully adjusted OHLC with yfinance auto_adjust=True.",
            "- ML features use adjusted Open/High/Low/Close together; raw OHLC is preserved in clean cache for audit.",
            "- Zero-volume/stale/impossible/duplicate rows are flagged. Confirmed stale/non-trading rows are excluded before feature and target generation.",
            "- Volume is never forward-filled.",
            "- Sector-volume-derived features are not used; price-based sector features are preserved.",
            "",
            f"Symbols processed: {len(quality)}",
            f"Rows removed as non-tradable/stale/invalid: {int(quality['RowsRemoved'].fillna(0).sum()) if 'RowsRemoved' in quality else 0}",
            f"Post-adjustment extreme moves >20%: {len(corporate_actions)}",
            "",
            "Clean label distribution:",
            *[
                f"- {row.Scope} {row.Group}: outperform {row.OutperformPct:.2%}, neutral {row.NeutralPct:.2%}, underperform {row.UnderperformPct:.2%}, rows {row.Rows}"
                for row in labels[labels["Scope"].isin(["overall", "by_split"])].itertuples()
            ],
            "",
            "Clean vs old model comparison:",
            comparison.to_string(index=False),
            "",
            "Beta ablation:",
            beta.to_string(index=False),
            "",
            "Do not proceed to BUY/ACCUMULATE/SELL thresholds until clean results are reviewed and accepted.",
            "",
            "Download metadata:",
            json.dumps(metadata, indent=2)[:4000],
        ]
    )


if __name__ == "__main__":
    main()
