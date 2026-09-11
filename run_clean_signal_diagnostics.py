"""Run clean-data anomaly, percentile, tail, and combined-signal diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.features import feature_columns_for_variant
from run_pruned_rank_target_validation import FINAL_TEST_START, PRUNED_FEATURES, _embargo_folds
from run_ranker_binary_signal_research import (
    _base_prediction_frame,
    _fit_calibration_split,
    _fit_classifier,
    _fit_isotonic,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)

CLEAN_DATASET = DATA_DIR / "probability_training_dataset_clean.csv"
CLEAN_CACHE_DIR = DATA_DIR / "clean_cache_adjusted"

OUTPUT_ANOMALIES = REPORT_DIR / "remaining_data_anomalies.csv"
OUTPUT_RANKER_TAIL = REPORT_DIR / "clean_ranker_tail_analysis.csv"
OUTPUT_BUY = REPORT_DIR / "clean_buy_percentile_analysis.csv"
OUTPUT_PRUNED = REPORT_DIR / "clean_pruned_percentile_analysis.csv"
OUTPUT_COMBINED = REPORT_DIR / "clean_combined_signal_analysis.csv"
OUTPUT_RECOMMENDATION = REPORT_DIR / "clean_architecture_recommendation.json"

TOP_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01]
AUDIT_ESTIMATORS = 80


def main() -> None:
    """Generate requested diagnostics from the clean rebuilt dataset only."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)
    features = _pruned_features()

    anomalies = _remaining_anomalies()
    anomalies.to_csv(OUTPUT_ANOMALIES, index=False)

    print("[clean] pruned classifier", flush=True)
    pruned = _classifier_walk_forward(pre_final, folds, features)
    pruned_analysis = _percentile_analysis(
        pruned,
        score_column="p_outperform",
        percentile_column="p_outperform_percentile",
        signal_name="clean_pruned_p_outperform",
        calibrator="none",
    )
    pruned_analysis.to_csv(OUTPUT_PRUNED, index=False)

    print("[clean] XGBRanker", flush=True)
    ranker = _ranker_walk_forward(pre_final, folds, features)
    ranker_tail = _ranker_tail_analysis(ranker)
    ranker_tail.to_csv(OUTPUT_RANKER_TAIL, index=False)

    print("[clean] binary BUY", flush=True)
    buy = _binary_buy_walk_forward(pre_final, folds, features)
    buy_analysis = _buy_percentile_analysis(buy)
    buy_analysis.to_csv(OUTPUT_BUY, index=False)

    combined = _combined_signal_analysis(pruned, buy, ranker)
    combined.to_csv(OUTPUT_COMBINED, index=False)

    recommendation = _recommendation(
        anomalies=anomalies,
        ranker_tail=ranker_tail,
        buy_analysis=buy_analysis,
        pruned_analysis=pruned_analysis,
        combined=combined,
        features=features,
    )
    OUTPUT_RECOMMENDATION.write_text(
        json.dumps(_json_safe(recommendation), indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "remaining_data_anomalies": str(OUTPUT_ANOMALIES),
                "clean_ranker_tail_analysis": str(OUTPUT_RANKER_TAIL),
                "clean_buy_percentile_analysis": str(OUTPUT_BUY),
                "clean_pruned_percentile_analysis": str(OUTPUT_PRUNED),
                "clean_combined_signal_analysis": str(OUTPUT_COMBINED),
                "clean_architecture_recommendation": str(OUTPUT_RECOMMENDATION),
                "source_dataset": str(CLEAN_DATASET),
                "final_test_status": "untouched",
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare clean validation dataset without changing features."""

    output = dataset.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output["buy_target"] = (output["excess_return"] > 0.05).astype(int)
    output["sell_target"] = (output["excess_return"] < -0.05).astype(int)
    output["market_regime_label"] = output.apply(_market_regime_label, axis=1)
    return output.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["label", "excess_return", "buy_target", "sell_target"]
    )


def _market_regime_label(row: pd.Series) -> str:
    """Map numeric market regime flags to readable labels."""

    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _pruned_features() -> list[str]:
    """Return frozen Pruned Advanced feature list with beta_252d retained."""

    return list(dict.fromkeys([*feature_columns_for_variant("sector_regime"), *PRUNED_FEATURES]))


def _remaining_anomalies() -> pd.DataFrame:
    """Audit post-adjustment >20% moves and flagged-but-retained rows."""

    ca_path = REPORT_DIR / "corporate_action_audit.csv"
    stale_path = REPORT_DIR / "stale_row_report.csv"
    quality_path = REPORT_DIR / "clean_data_quality_report.csv"
    corporate = pd.read_csv(ca_path, parse_dates=["Date"])
    stale = pd.read_csv(stale_path, parse_dates=["Date"])
    quality = pd.read_csv(quality_path)
    excluded_by_ticker = quality.set_index("Ticker")["RowsRemoved"].to_dict()
    stale_total_by_ticker = stale.groupby("Ticker").size().to_dict()
    retained_flagged_by_ticker = {
        ticker: int(count - excluded_by_ticker.get(ticker, 0))
        for ticker, count in stale_total_by_ticker.items()
    }

    rows: list[dict[str, Any]] = []
    for row in corporate.itertuples(index=False):
        cache = _read_clean_cache(row.Ticker)
        previous = cache[cache["Date"] < row.Date].sort_values("Date").tail(1)
        previous_close = (
            float(previous["adj_Close"].iloc[0])
            if not previous.empty and pd.notna(previous["adj_Close"].iloc[0])
            else None
        )
        classification = str(row.Classification)
        suspicious = classification == "suspicious/bad data"
        likely_ca = classification == "likely corporate-action artifact"
        action = "REVIEW" if suspicious or likely_ca else "KEEP"
        rows.append(
            {
                "Ticker": row.Ticker,
                "Date": row.Date,
                "PreviousAdjustedClose": previous_close,
                "AdjustedClose": row.AdjustedClose,
                "OneDayReturn": row.AdjustedDailyReturn,
                "Volume": row.Volume,
                "LikelyCorporateAction": likely_ca,
                "SuspiciousDataFlag": suspicious,
                "RecommendedAction": action,
                "Classification": classification,
                "FlaggedRowsInStaleReport": int(stale_total_by_ticker.get(row.Ticker, 0)),
                "RowsExcludedFromML": int(excluded_by_ticker.get(row.Ticker, 0)),
                "FlaggedButRetainedRows": int(retained_flagged_by_ticker.get(row.Ticker, 0)),
                "FlaggedButRetainedStatus": _flagged_retained_status(
                    stale[stale["Ticker"] == row.Ticker],
                    excluded_by_ticker.get(row.Ticker, 0),
                ),
                "Reason": _anomaly_reason(classification, row.AdjustedDailyReturn, row.Volume),
            }
        )
    return pd.DataFrame(rows)


def _flagged_retained_status(stale_rows: pd.DataFrame, excluded_count: int) -> str:
    """Explain flagged rows that remain after the stricter exclusion count."""

    retained_count = len(stale_rows) - int(excluded_count)
    if retained_count <= 0:
        return "No flagged retained rows for this ticker."
    retained = stale_rows[stale_rows["is_tradable_row"].astype(bool)]
    if retained.empty:
        return "Flag count includes overlapping flags; no tradable flagged rows retained."
    if retained["Type"].eq("Sector Index").any():
        return "Sector-index zero-volume rows retained only when OHLC was not stale; sector volume is not used."
    return "Retained rows are non-stale/non-duplicate/non-invalid after overlap accounting; not artificial forward-filled."


def _anomaly_reason(classification: str, daily_return: float, volume: float) -> str:
    """Human-readable anomaly reason."""

    if classification == "suspicious/bad data":
        return "Move remains extreme after adjusted OHLC rebuild; manual source check required before exclusion."
    if classification == "likely corporate-action artifact":
        return "Move has zero/missing volume or corporate-action-like signature; keep out of automatic removal pending source check."
    return (
        "High-volume post-adjustment move with nonzero volume. Treated as plausible market move "
        "and retained unless manually verified as bad data."
    )


def _read_clean_cache(ticker: str) -> pd.DataFrame:
    path = CLEAN_CACHE_DIR / f"{_safe_symbol(ticker)}.csv"
    frame = pd.read_csv(path, parse_dates=["Date"])
    return frame.sort_values("Date")


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace(".", "_").replace("/", "_")


def _classifier_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> pd.DataFrame:
    """Generate clean Pruned Advanced validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, features, target="label")
        x = _transform_with_medians(validation, features, medians)
        probabilities = model.predict_proba(x)
        output = _base_prediction_frame(validation, fold["fold"])
        output["label"] = validation["label"].astype(int).to_numpy()
        output["p_outperform"] = probabilities[:, 2]
        output["p_outperform_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> pd.DataFrame:
    """Generate clean XGBRanker validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_ranker(train, features)
        x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        output["label"] = validation["label"].astype(int).to_numpy()
        output["ranker_score"] = model.predict(x)
        output["ranker_percentile"] = output.groupby("Date")["ranker_score"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_buy_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> pd.DataFrame:
    """Generate clean binary BUY validation predictions with fold-local calibration."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit_frame, calibration_frame = _fit_calibration_split(train)
        model, medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
        cal_x = _transform_with_medians(calibration_frame, features, medians)
        val_x = _transform_with_medians(validation, features, medians)
        cal_prob = _positive_probability(model, cal_x)
        val_prob = _positive_probability(model, val_x)
        platt = _fit_platt(cal_prob, calibration_frame["buy_target"])
        isotonic = _fit_isotonic(cal_prob, calibration_frame["buy_target"])
        output = _base_prediction_frame(validation, fold["fold"])
        output["label"] = validation["label"].astype(int).to_numpy()
        output["actual"] = validation["buy_target"].astype(int).to_numpy()
        output["p_raw"] = val_prob
        output["p_sigmoid"] = platt.predict_proba(val_prob.reshape(-1, 1))[:, 1]
        output["p_isotonic"] = isotonic.predict(val_prob)
        for column in ("p_raw", "p_sigmoid", "p_isotonic"):
            output[f"{column}_percentile"] = output.groupby("Date")[column].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_tail_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report top and bottom ranker tails overall, by fold, and by year."""

    rows = []
    scopes = [("overall", "all", predictions)]
    scopes.extend((f"fold", fold, group) for fold, group in predictions.groupby("fold"))
    scopes.extend((f"year", year, group) for year, group in predictions.groupby(predictions["Date"].dt.year))
    for scope, value, frame in scopes:
        for side in ("Top", "Bottom"):
            for fraction in TOP_FRACTIONS:
                selected = _select_tail(frame, "ranker_percentile", fraction, side)
                rows.append(
                    {
                        "Scope": scope,
                        "Group": value,
                        "Tail": side,
                        "Percentile": f"{int(fraction * 100)}%",
                        **_realized_metrics(selected),
                    }
                )
    return pd.DataFrame(rows)


def _buy_percentile_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report binary BUY percentile performance for raw/sigmoid/isotonic calibration."""

    rows = []
    base_rate = float(predictions["actual"].mean())
    for calibration, score, percentile in [
        ("raw", "p_raw", "p_raw_percentile"),
        ("sigmoid", "p_sigmoid", "p_sigmoid_percentile"),
        ("isotonic", "p_isotonic", "p_isotonic_percentile"),
    ]:
        rows.extend(
            _percentile_analysis(
                predictions,
                score_column=score,
                percentile_column=percentile,
                signal_name="binary_xgboost_buy",
                calibrator=calibration,
                base_rate=base_rate,
                actual_column="actual",
            ).to_dict(orient="records")
        )
    return pd.DataFrame(rows)


def _percentile_analysis(
    predictions: pd.DataFrame,
    *,
    score_column: str,
    percentile_column: str,
    signal_name: str,
    calibrator: str,
    base_rate: float | None = None,
    actual_column: str = "buy_target",
) -> pd.DataFrame:
    """Report top percentile performance overall, by fold, and by year."""

    rows = []
    if base_rate is None:
        base_rate = float(predictions[actual_column].mean())
    scopes = [("overall", "all", predictions)]
    scopes.extend(("fold", fold, group) for fold, group in predictions.groupby("fold"))
    scopes.extend(("year", year, group) for year, group in predictions.groupby(predictions["Date"].dt.year))
    for scope, value, frame in scopes:
        for fraction in TOP_FRACTIONS:
            selected = _select_tail(frame, percentile_column, fraction, "Top")
            metrics = _realized_metrics(selected, actual_column=actual_column)
            rows.append(
                {
                    "Signal": signal_name,
                    "Calibration": calibrator,
                    "Scope": scope,
                    "Group": value,
                    "Percentile": f"Top {int(fraction * 100)}%",
                    "AverageScore": float(selected[score_column].mean()) if not selected.empty else None,
                    "BaseBuyRate": base_rate,
                    "LiftVsBaseRate": metrics["ActualBuyRate"] / base_rate if base_rate else None,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _combined_signal_analysis(
    pruned: pd.DataFrame,
    buy: pd.DataFrame,
    ranker: pd.DataFrame,
) -> pd.DataFrame:
    """Compare combined percentile agreement on validation only."""

    key = ["Date", "symbol", "fold"]
    combined = pruned[
        key
        + [
            "p_outperform_percentile",
            "excess_return",
            "future_stock_return",
            "future_nifty_return",
            "buy_target",
            "sell_target",
            "label",
        ]
    ].merge(
        buy[key + ["p_sigmoid_percentile", "p_raw_percentile", "p_isotonic_percentile"]],
        on=key,
        how="inner",
    )
    combined = combined.merge(ranker[key + ["ranker_percentile"]], on=key, how="inner")
    combined["avg_percentile_pruned_buy_ranker"] = (
        combined["p_outperform_percentile"]
        + combined["p_sigmoid_percentile"]
        + combined["ranker_percentile"]
    ) / 3

    rows = []
    signal_sets = [
        ("pruned_only", ["p_outperform_percentile"]),
        ("binary_buy_only", ["p_sigmoid_percentile"]),
        ("ranker_only", ["ranker_percentile"]),
        ("pruned_plus_binary_buy", ["p_outperform_percentile", "p_sigmoid_percentile"]),
        ("pruned_plus_ranker", ["p_outperform_percentile", "ranker_percentile"]),
        ("binary_buy_plus_ranker", ["p_sigmoid_percentile", "ranker_percentile"]),
        (
            "pruned_plus_binary_buy_plus_ranker",
            ["p_outperform_percentile", "p_sigmoid_percentile", "ranker_percentile"],
        ),
        ("average_percentile_score", ["avg_percentile_pruned_buy_ranker"]),
    ]
    for name, columns in signal_sets:
        for fraction in TOP_FRACTIONS:
            selected = _select_combined(combined, columns, fraction)
            non_overlap = _non_overlapping_metrics(selected, score_columns=columns)
            rows.append(
                {
                    "SignalCombination": name,
                    "SelectionRule": _selection_rule(columns, fraction),
                    "Percentile": f"Top {int(fraction * 100)}%",
                    **_realized_metrics(selected),
                    **non_overlap,
                    "YearConsistencyPositiveAvgExcessPct": _year_consistency(selected),
                }
            )
    return pd.DataFrame(rows)


def _select_combined(frame: pd.DataFrame, percentile_columns: list[str], fraction: float) -> pd.DataFrame:
    """Select rows by percentile agreement or average score."""

    if len(percentile_columns) == 1:
        return _select_tail(frame, percentile_columns[0], fraction, "Top")
    if percentile_columns == ["avg_percentile_pruned_buy_ranker"]:
        return _select_tail(frame, percentile_columns[0], fraction, "Top")
    threshold = 1 - fraction
    mask = pd.Series(True, index=frame.index)
    for column in percentile_columns:
        mask &= frame[column] >= threshold
    return frame[mask].copy()


def _selection_rule(columns: list[str], fraction: float) -> str:
    if len(columns) == 1:
        return f"{columns[0]} in top {int(fraction * 100)}% by date"
    if columns == ["avg_percentile_pruned_buy_ranker"]:
        return f"average percentile score in top {int(fraction * 100)}% by date"
    return " AND ".join(f"{column} top {int(fraction * 100)}%" for column in columns)


def _select_tail(frame: pd.DataFrame, percentile_column: str, fraction: float, side: str) -> pd.DataFrame:
    """Select daily top/bottom percentile rows."""

    if side == "Top":
        return frame[frame[percentile_column] >= 1 - fraction].copy()
    return frame[frame[percentile_column] <= fraction].copy()


def _realized_metrics(frame: pd.DataFrame, *, actual_column: str = "buy_target") -> dict[str, Any]:
    """Common realized forward-return metrics."""

    if frame.empty:
        return {
            "Observations": 0,
            "AverageFutureExcessReturn": None,
            "MedianExcessReturn": None,
            "BenchmarkWinRate": None,
            "ActualBuyRate": None,
            "Precision": None,
            "ActualOutperformRate": None,
            "ActualUnderperformRate": None,
            "WinRate": None,
        }
    buy_rate = float(frame[actual_column].mean())
    return {
        "Observations": int(len(frame)),
        "AverageFutureExcessReturn": float(frame["excess_return"].mean()),
        "MedianExcessReturn": float(frame["excess_return"].median()),
        "BenchmarkWinRate": float((frame["excess_return"] > 0).mean()),
        "ActualBuyRate": buy_rate,
        "Precision": buy_rate,
        "ActualOutperformRate": float((frame["excess_return"] > 0.05).mean()),
        "ActualUnderperformRate": float((frame["excess_return"] < -0.05).mean()),
        "WinRate": float((frame["excess_return"] > 0).mean()),
    }


def _non_overlapping_metrics(frame: pd.DataFrame, *, score_columns: list[str]) -> dict[str, Any]:
    """Approximate non-overlapping 20-trading-day validation performance."""

    if frame.empty:
        return {
            "NonOverlapPeriods": 0,
            "NonOverlapAvgExcessReturn": None,
            "NonOverlapMedianExcessReturn": None,
            "NonOverlapWinRate": None,
        }
    selected_frames = []
    for _, fold_frame in frame.groupby("fold"):
        dates = pd.Series(sorted(fold_frame["Date"].drop_duplicates()))
        rebalance_dates = set(dates.iloc[::20])
        selected_frames.append(fold_frame[fold_frame["Date"].isin(rebalance_dates)])
    non_overlap = pd.concat(selected_frames, ignore_index=True) if selected_frames else frame.iloc[:0]
    if non_overlap.empty:
        return {
            "NonOverlapPeriods": 0,
            "NonOverlapAvgExcessReturn": None,
            "NonOverlapMedianExcessReturn": None,
            "NonOverlapWinRate": None,
        }
    period_returns = non_overlap.groupby("Date")["excess_return"].mean()
    return {
        "NonOverlapPeriods": int(len(period_returns)),
        "NonOverlapAvgExcessReturn": float(period_returns.mean()),
        "NonOverlapMedianExcessReturn": float(period_returns.median()),
        "NonOverlapWinRate": float((period_returns > 0).mean()),
    }


def _year_consistency(frame: pd.DataFrame) -> float | None:
    """Return share of years where selected average excess return is positive."""

    if frame.empty:
        return None
    yearly = frame.groupby(frame["Date"].dt.year)["excess_return"].mean()
    return float((yearly > 0).mean()) if not yearly.empty else None


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
    validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
    return train, validation


def _recommendation(
    *,
    anomalies: pd.DataFrame,
    ranker_tail: pd.DataFrame,
    buy_analysis: pd.DataFrame,
    pruned_analysis: pd.DataFrame,
    combined: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    """Create architecture recommendation without final threshold rules."""

    ranker_top10 = _row(ranker_tail, Scope="overall", Tail="Top", Percentile="10%")
    ranker_bottom10 = _row(ranker_tail, Scope="overall", Tail="Bottom", Percentile="10%")
    buy_top10 = _row(
        buy_analysis,
        Signal="binary_xgboost_buy",
        Calibration="sigmoid",
        Scope="overall",
        Percentile="Top 10%",
    )
    pruned_top10 = _row(
        pruned_analysis,
        Signal="clean_pruned_p_outperform",
        Scope="overall",
        Percentile="Top 10%",
    )
    combined_sorted = combined.sort_values(
        ["AverageFutureExcessReturn", "Precision", "NonOverlapAvgExcessReturn"],
        ascending=False,
        na_position="last",
    )
    best_combined = combined_sorted.head(5).to_dict(orient="records")
    return {
        "source_dataset": str(CLEAN_DATASET),
        "final_test_status": "untouched",
        "feature_policy": {
            "primary_feature_set": "Pruned Advanced frozen",
            "beta_252d": "kept",
            "imputation": "training-fold-only median imputation with missing-value indicators via _transform_with_medians",
            "features_used": features,
        },
        "data_quality": {
            "post_adjustment_moves_gt_20": int(len(anomalies)),
            "actions": anomalies["RecommendedAction"].value_counts().to_dict(),
            "suspicious_data_flags": int(anomalies["SuspiciousDataFlag"].sum()),
            "likely_corporate_actions": int(anomalies["LikelyCorporateAction"].sum()),
            "stale_report_explanation": (
                "stale_row_report records every row with any flag, including overlapping flags and sector-index "
                "zero-volume rows. Only confirmed non-trading/stale/invalid rows are excluded from ML; this is why "
                "4,067 report rows can coexist with 952 excluded rows."
            ),
        },
        "signal_findings": {
            "ranker_top10": ranker_top10,
            "ranker_bottom10": ranker_bottom10,
            "binary_buy_sigmoid_top10": buy_top10,
            "pruned_top10": pruned_top10,
            "best_combined_validation_rows": best_combined,
        },
        "recommendation": _architecture_text(ranker_top10, ranker_bottom10, best_combined),
        "constraints": [
            "No features were added or removed.",
            "No model hyperparameters were tuned.",
            "No final BUY/ACCUMULATE/SELL thresholds were created.",
            "SELL research remains separate.",
        ],
    }


def _architecture_text(
    ranker_top10: dict[str, Any],
    ranker_bottom10: dict[str, Any],
    best_combined: list[dict[str, Any]],
) -> str:
    """Readable architecture recommendation."""

    top = ranker_top10.get("AverageFutureExcessReturn")
    bottom = ranker_bottom10.get("AverageFutureExcessReturn")
    if top is not None and bottom is not None and top > bottom:
        ranker_note = "Ranker has useful tail separation and can remain a confirmation signal."
    else:
        ranker_note = "Ranker tail separation is weak; use only as a monitored secondary diagnostic."
    combined_note = (
        "Combined percentile agreement should be preferred only where it improves realized excess return "
        "and year consistency versus the standalone Pruned Advanced and binary BUY signals."
    )
    return f"{ranker_note} {combined_note}"


def _row(frame: pd.DataFrame, **conditions: Any) -> dict[str, Any]:
    match = frame.copy()
    for column, value in conditions.items():
        if column in match.columns:
            match = match[match[column] == value]
    return match.iloc[0].to_dict() if not match.empty else {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (dict, list)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
