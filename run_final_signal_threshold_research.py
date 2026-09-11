"""Research clean validation BUY/ACCUMULATE/SELL signal thresholds."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.features import feature_columns_for_variant
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS, PRUNED_FEATURES, _embargo_folds
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
CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"

OUTPUT_VEDL = REPORT_DIR / "vedl_demerger_cleanup.csv"
OUTPUT_BUY = REPORT_DIR / "final_buy_threshold_analysis.csv"
OUTPUT_ACCUMULATE = REPORT_DIR / "final_accumulate_threshold_analysis.csv"
OUTPUT_SELL = REPORT_DIR / "final_sell_threshold_analysis.csv"
OUTPUT_STABILITY = REPORT_DIR / "signal_fold_stability.csv"
OUTPUT_RULES = REPORT_DIR / "recommended_signal_rules.json"

VEDL_SYMBOL = "VEDL.NS"
VEDL_EVENT_DATE = pd.Timestamp("2026-04-30")
VEDL_EVENT_TYPE = "CORPORATE_ACTION_STRUCTURAL_BREAK"
MAX_FEATURE_LOOKBACK_DAYS = 252
TOP_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01]


def main() -> None:
    """Run final validation-only threshold research."""

    ensure_probability_dirs()
    clean = _prepare_dataset(pd.read_csv(CLEAN_DATASET, parse_dates=["Date"]))
    filtered, vedl_cleanup = _exclude_vedl_demerger_contamination(clean)
    filtered.to_csv(CLEAN_DEMERGER_DATASET, index=False)
    vedl_cleanup.to_csv(OUTPUT_VEDL, index=False)

    pre_final = filtered[filtered["Date"] < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)
    features = _pruned_features()

    print("[signals] primary pruned classifier", flush=True)
    primary = _primary_walk_forward(pre_final, folds, features)
    print("[signals] binary BUY XGBoost", flush=True)
    buy = _binary_buy_walk_forward(pre_final, folds, features)
    print("[signals] XGBRanker confirmation", flush=True)
    ranker = _ranker_walk_forward(pre_final, folds, features)
    print("[signals] RF SELL diagnostic", flush=True)
    rf_sell = _rf_sell_walk_forward(pre_final, folds, features)

    signals = _merged_signals(primary, buy, ranker, rf_sell)
    buy_analysis = _buy_threshold_analysis(signals)
    accumulate_analysis = _accumulate_threshold_analysis(signals)
    sell_analysis = _sell_threshold_analysis(signals)
    stability = _signal_stability(buy_analysis, accumulate_analysis, sell_analysis)
    rules = _recommended_rules(
        vedl_cleanup=vedl_cleanup,
        buy_analysis=buy_analysis,
        accumulate_analysis=accumulate_analysis,
        sell_analysis=sell_analysis,
        stability=stability,
        features=features,
        old_rows=len(clean),
        new_rows=len(filtered),
    )

    buy_analysis.to_csv(OUTPUT_BUY, index=False)
    accumulate_analysis.to_csv(OUTPUT_ACCUMULATE, index=False)
    sell_analysis.to_csv(OUTPUT_SELL, index=False)
    stability.to_csv(OUTPUT_STABILITY, index=False)
    OUTPUT_RULES.write_text(json.dumps(_json_safe(rules), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "vedl_demerger_cleanup": str(OUTPUT_VEDL),
                "final_buy_threshold_analysis": str(OUTPUT_BUY),
                "final_accumulate_threshold_analysis": str(OUTPUT_ACCUMULATE),
                "final_sell_threshold_analysis": str(OUTPUT_SELL),
                "signal_fold_stability": str(OUTPUT_STABILITY),
                "recommended_signal_rules": str(OUTPUT_RULES),
                "clean_demerger_dataset": str(CLEAN_DEMERGER_DATASET),
                "final_test_status": "not used for threshold selection",
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output["buy_target"] = (output["excess_return"] > 0.05).astype(int)
    output["sell_target"] = (output["excess_return"] < -0.05).astype(int)
    output["market_regime_label"] = output.apply(_market_regime_label, axis=1)
    return output.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["label", "excess_return", "buy_target", "sell_target"]
    )


def _market_regime_label(row: pd.Series) -> str:
    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _exclude_vedl_demerger_contamination(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exclude VEDL rows whose feature/target windows cross the demerger date."""

    vedl = dataset[dataset["symbol"] == VEDL_SYMBOL].sort_values("Date").copy()
    dates = pd.Series(vedl["Date"].drop_duplicates().sort_values().to_list())
    if dates.empty:
        return dataset.copy(), pd.DataFrame()
    event_pos = dates[dates >= VEDL_EVENT_DATE].index.min()
    if pd.isna(event_pos):
        return dataset.copy(), pd.DataFrame()
    event_pos = int(event_pos)
    event_trading_date = dates.iloc[event_pos]
    target_start_pos = max(0, event_pos - HORIZON_DAYS)
    feature_end_pos = min(len(dates) - 1, event_pos + MAX_FEATURE_LOOKBACK_DAYS)

    contaminated_dates = set(dates.iloc[target_start_pos : feature_end_pos + 1])
    target_dates = set(dates.iloc[target_start_pos:event_pos])
    feature_dates = set(dates.iloc[event_pos : feature_end_pos + 1])
    contaminated_mask = dataset["symbol"].eq(VEDL_SYMBOL) & dataset["Date"].isin(contaminated_dates)

    rows = []
    for row in dataset[contaminated_mask].sort_values("Date").itertuples(index=False):
        date = row.Date
        feature_crosses = date in feature_dates
        target_crosses = date in target_dates
        rows.append(
            {
                "Ticker": VEDL_SYMBOL,
                "Date": date,
                "CorporateActionDate": event_trading_date,
                "EventType": VEDL_EVENT_TYPE,
                "FeatureWindowCrossesEvent": feature_crosses,
                "Target20DWindowCrossesEvent": target_crosses,
                "ContaminationType": _contamination_type(feature_crosses, target_crosses),
                "Action": "EXCLUDE",
                "Reason": (
                    "Vedanta demerger structural break is not an ordinary market return. "
                    "Exclude until total-return adjustment including demerged-share value is available."
                ),
                "OriginalDatasetRows": int(len(dataset)),
                "RowsExcludedForVEDLStructuralBreak": int(contaminated_mask.sum()),
                "PostCleanupDatasetRows": int(len(dataset) - contaminated_mask.sum()),
                "ValidationRowsExcluded": int(
                    (
                        contaminated_mask
                        & (dataset["Date"] < FINAL_TEST_START)
                    ).sum()
                ),
                "MetricImpactNote": (
                    "No walk-forward validation metric impact because the structural break is in 2026 "
                    "and threshold selection uses dates before 2025-01-01."
                ),
            }
        )
    return dataset[~contaminated_mask].copy(), pd.DataFrame(rows)


def _contamination_type(feature_crosses: bool, target_crosses: bool) -> str:
    if feature_crosses and target_crosses:
        return "FEATURE_AND_TARGET_WINDOW"
    if feature_crosses:
        return "FEATURE_WINDOW"
    if target_crosses:
        return "TARGET_WINDOW"
    return "NONE"


def _pruned_features() -> list[str]:
    return list(dict.fromkeys([*feature_columns_for_variant("sector_regime"), *PRUNED_FEATURES]))


def _primary_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, features, target="label")
        x = _transform_with_medians(validation, features, medians)
        probabilities = model.predict_proba(x)
        output = _base_prediction_frame(validation, fold["fold"])
        output["label"] = validation["label"].astype(int).to_numpy()
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["p_outperform_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["p_underperform_percentile"] = output.groupby("Date")["p_underperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_buy_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit_frame, calibration_frame = _fit_calibration_split(train)
        model, medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
        cal_x = _transform_with_medians(calibration_frame, features, medians)
        val_x = _transform_with_medians(validation, features, medians)
        cal_prob = _positive_probability(model, cal_x)
        raw_prob = _positive_probability(model, val_x)
        platt = _fit_platt(cal_prob, calibration_frame["buy_target"])
        isotonic = _fit_isotonic(cal_prob, calibration_frame["buy_target"])
        output = _base_prediction_frame(validation, fold["fold"])
        output["buy_raw_probability"] = raw_prob
        output["buy_sigmoid_probability"] = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
        output["buy_isotonic_probability"] = isotonic.predict(raw_prob)
        for column in ("buy_raw_probability", "buy_sigmoid_probability", "buy_isotonic_probability"):
            output[f"{column}_percentile"] = output.groupby("Date")[column].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_ranker(train, features)
        x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        output["ranker_score"] = model.predict(x)
        output["ranker_percentile"] = output.groupby("Date")["ranker_score"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _rf_sell_walk_forward(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(
            train,
            features,
            target="sell_target",
            model_name="random_forest",
            binary=True,
        )
        x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        output["rf_sell_probability"] = _positive_probability(model, x)
        output["rf_sell_percentile"] = output.groupby("Date")["rf_sell_probability"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _merged_signals(
    primary: pd.DataFrame,
    buy: pd.DataFrame,
    ranker: pd.DataFrame,
    rf_sell: pd.DataFrame,
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    columns = [
        "Sector",
        "MarketCapCategory",
        "future_stock_return",
        "future_nifty_return",
        "excess_return",
        "buy_target",
        "sell_target",
        "market_regime_label",
        "label",
        "p_outperform",
        "p_neutral",
        "p_underperform",
        "p_outperform_percentile",
        "p_underperform_percentile",
    ]
    output = primary[key + columns].merge(
        buy[
            key
            + [
                "buy_raw_probability",
                "buy_sigmoid_probability",
                "buy_isotonic_probability",
                "buy_raw_probability_percentile",
                "buy_sigmoid_probability_percentile",
                "buy_isotonic_probability_percentile",
            ]
        ],
        on=key,
        how="inner",
    )
    output = output.merge(ranker[key + ["ranker_score", "ranker_percentile"]], on=key, how="inner")
    output = output.merge(rf_sell[key + ["rf_sell_probability", "rf_sell_percentile"]], on=key, how="inner")
    return output


def _buy_threshold_analysis(signals: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[str, str, pd.Series]] = []
    for fraction in TOP_FRACTIONS:
        candidates.append(
            (
                f"BUY_PRUNED_TOP_{_pct(fraction)}",
                f"Pruned P_Outperform daily top {_pct(fraction)}",
                _top(signals["p_outperform_percentile"], fraction),
            )
        )
        candidates.append(
            (
                f"BUY_BINARY_RAW_TOP_{_pct(fraction)}",
                f"Binary BUY raw probability daily top {_pct(fraction)}",
                _top(signals["buy_raw_probability_percentile"], fraction),
            )
        )
        candidates.append(
            (
                f"BUY_BINARY_SIGMOID_TOP_{_pct(fraction)}",
                f"Binary BUY sigmoid probability daily top {_pct(fraction)}",
                _top(signals["buy_sigmoid_probability_percentile"], fraction),
            )
        )
    for pruned_fraction in TOP_FRACTIONS:
        for buy_fraction in TOP_FRACTIONS:
            mask = _top(signals["p_outperform_percentile"], pruned_fraction) & _top(
                signals["buy_sigmoid_probability_percentile"],
                buy_fraction,
            )
            candidates.append(
                (
                    f"BUY_PRUNED_TOP_{_pct(pruned_fraction)}_AND_BINARY_SIGMOID_TOP_{_pct(buy_fraction)}",
                    f"Pruned top {_pct(pruned_fraction)} AND Binary BUY sigmoid top {_pct(buy_fraction)}",
                    mask,
                )
            )
    return _candidate_analysis(signals, candidates, action="BUY")


def _accumulate_threshold_analysis(signals: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[str, str, pd.Series]] = []
    buy_level = _top(signals["p_outperform_percentile"], 0.05) & _top(
        signals["buy_sigmoid_probability_percentile"],
        0.05,
    )
    for pruned_fraction in (0.20, 0.10):
        for buy_fraction in (0.50, 0.40, 0.30, 0.20, 0.10):
            mask = (
                _top(signals["p_outperform_percentile"], pruned_fraction)
                & _top(signals["buy_sigmoid_probability_percentile"], buy_fraction)
                & ~buy_level
            )
            candidates.append(
                (
                    f"ACCUMULATE_PRUNED_TOP_{_pct(pruned_fraction)}_BINARY_ABOVE_{_pct(buy_fraction)}_EX_BUY5",
                    (
                        f"Pruned positive/top {_pct(pruned_fraction)} and Binary BUY above {_pct(buy_fraction)}, "
                        "excluding validated stronger BUY top-5 agreement zone"
                    ),
                    mask,
                )
            )
    return _candidate_analysis(signals, candidates, action="ACCUMULATE")


def _sell_threshold_analysis(signals: pd.DataFrame) -> pd.DataFrame:
    candidates: list[tuple[str, str, pd.Series]] = []
    for fraction in TOP_FRACTIONS:
        candidates.append(
            (
                f"SELL_PRUNED_UNDER_TOP_{_pct(fraction)}",
                f"Pruned P_Underperform daily top {_pct(fraction)}",
                _top(signals["p_underperform_percentile"], fraction),
            )
        )
        candidates.append(
            (
                f"SELL_LOW_BUY_BOTTOM_{_pct(fraction)}",
                f"Binary BUY sigmoid daily bottom {_pct(fraction)}",
                _bottom(signals["buy_sigmoid_probability_percentile"], fraction),
            )
        )
        candidates.append(
            (
                f"SELL_RF_SELL_TOP_{_pct(fraction)}",
                f"Random Forest SELL probability daily top {_pct(fraction)}",
                _top(signals["rf_sell_percentile"], fraction),
            )
        )
        candidates.append(
            (
                f"SELL_RANKER_BOTTOM_{_pct(fraction)}",
                f"XGBRanker daily bottom {_pct(fraction)}",
                _bottom(signals["ranker_percentile"], fraction),
            )
        )
    for fraction in TOP_FRACTIONS:
        mask = (
            _top(signals["p_underperform_percentile"], fraction)
            & _bottom(signals["buy_sigmoid_probability_percentile"], fraction)
            & _top(signals["rf_sell_percentile"], fraction)
            & _bottom(signals["ranker_percentile"], fraction)
        )
        candidates.append(
            (
                f"SELL_ALL_CONFIRM_TOP_BOTTOM_{_pct(fraction)}",
                (
                    f"P_Underperform top {_pct(fraction)} AND Binary BUY bottom {_pct(fraction)} "
                    f"AND RF SELL top {_pct(fraction)} AND Ranker bottom {_pct(fraction)}"
                ),
                mask,
            )
        )
    return _candidate_analysis(signals, candidates, action="SELL")


def _candidate_analysis(
    signals: pd.DataFrame,
    candidates: list[tuple[str, str, pd.Series]],
    *,
    action: str,
) -> pd.DataFrame:
    rows = []
    for rule_id, description, mask in candidates:
        selected = signals[mask].copy()
        rows.append({"Action": action, "RuleID": rule_id, "RuleDescription": description, "Scope": "overall", "Group": "all", **_metrics(selected, action=action)})
        for fold, group in selected.groupby("fold"):
            rows.append({"Action": action, "RuleID": rule_id, "RuleDescription": description, "Scope": "fold", "Group": fold, **_metrics(group, action=action)})
        for year, group in selected.groupby(selected["Date"].dt.year):
            rows.append({"Action": action, "RuleID": rule_id, "RuleDescription": description, "Scope": "year", "Group": year, **_metrics(group, action=action)})
        for regime, group in selected.groupby("market_regime_label"):
            rows.append({"Action": action, "RuleID": rule_id, "RuleDescription": description, "Scope": "regime", "Group": regime, **_metrics(group, action=action)})
    return pd.DataFrame(rows)


def _metrics(frame: pd.DataFrame, *, action: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "Count": 0,
            "Precision": None,
            "ActualPositiveExcessRate": None,
            "ActualUnderperformRate": None,
            "AverageExcessReturn": None,
            "MedianExcessReturn": None,
            "WinRateVsNifty": None,
            "NonOverlapPeriods": 0,
            "NonOverlapAverageExcessReturn": None,
            "NonOverlapMedianExcessReturn": None,
            "NonOverlapWinRateVsNifty": None,
        }
    precision = float(frame["sell_target"].mean()) if action == "SELL" else float(frame["buy_target"].mean())
    non_overlap = _non_overlap_metrics(frame)
    return {
        "Count": int(len(frame)),
        "Precision": precision,
        "ActualPositiveExcessRate": float((frame["excess_return"] > 0.05).mean()),
        "ActualUnderperformRate": float((frame["excess_return"] < -0.05).mean()),
        "AverageExcessReturn": float(frame["excess_return"].mean()),
        "MedianExcessReturn": float(frame["excess_return"].median()),
        "WinRateVsNifty": float((frame["excess_return"] > 0).mean()),
        **non_overlap,
    }


def _non_overlap_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    selected = []
    for _, fold_frame in frame.groupby("fold"):
        dates = pd.Series(sorted(fold_frame["Date"].drop_duplicates()))
        rebalance_dates = set(dates.iloc[::HORIZON_DAYS])
        selected.append(fold_frame[fold_frame["Date"].isin(rebalance_dates)])
    non_overlap = pd.concat(selected, ignore_index=True) if selected else frame.iloc[:0]
    if non_overlap.empty:
        return {
            "NonOverlapPeriods": 0,
            "NonOverlapAverageExcessReturn": None,
            "NonOverlapMedianExcessReturn": None,
            "NonOverlapWinRateVsNifty": None,
        }
    period_return = non_overlap.groupby("Date")["excess_return"].mean()
    return {
        "NonOverlapPeriods": int(len(period_return)),
        "NonOverlapAverageExcessReturn": float(period_return.mean()),
        "NonOverlapMedianExcessReturn": float(period_return.median()),
        "NonOverlapWinRateVsNifty": float((period_return > 0).mean()),
    }


def _signal_stability(
    buy: pd.DataFrame,
    accumulate: pd.DataFrame,
    sell: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for action, frame in [("BUY", buy), ("ACCUMULATE", accumulate), ("SELL", sell)]:
        overall = frame[frame["Scope"] == "overall"].copy()
        fold = frame[frame["Scope"] == "fold"].copy()
        year = frame[frame["Scope"] == "year"].copy()
        regime = frame[frame["Scope"] == "regime"].copy()
        for row in overall.itertuples(index=False):
            fold_rows = fold[fold["RuleID"] == row.RuleID]
            year_rows = year[year["RuleID"] == row.RuleID]
            regime_rows = regime[regime["RuleID"] == row.RuleID]
            frames.append(
                {
                    "Action": action,
                    "RuleID": row.RuleID,
                    "OverallCount": row.Count,
                    "OverallPrecision": row.Precision,
                    "OverallAverageExcessReturn": row.AverageExcessReturn,
                    "OverallMedianExcessReturn": row.MedianExcessReturn,
                    "FoldCount": int(len(fold_rows)),
                    "FoldsPositiveAverageExcessPct": _positive_pct(fold_rows["AverageExcessReturn"]),
                    "FoldsPositiveMedianExcessPct": _positive_pct(fold_rows["MedianExcessReturn"]),
                    "YearsPositiveAverageExcessPct": _positive_pct(year_rows["AverageExcessReturn"]),
                    "YearsPositiveMedianExcessPct": _positive_pct(year_rows["MedianExcessReturn"]),
                    "RegimesPositiveAverageExcessPct": _positive_pct(regime_rows["AverageExcessReturn"]),
                    "NonOverlapAverageExcessReturn": row.NonOverlapAverageExcessReturn,
                    "NonOverlapWinRateVsNifty": row.NonOverlapWinRateVsNifty,
                    "ValidationStatus": _validation_status(action, row, fold_rows, year_rows),
                }
            )
    return pd.DataFrame(frames)


def _positive_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float((clean > 0).mean()) if not clean.empty else None


def _negative_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float((clean < 0).mean()) if not clean.empty else None


def _validation_status(action: str, row: Any, fold_rows: pd.DataFrame, year_rows: pd.DataFrame) -> str:
    count = int(row.Count or 0)
    avg = row.AverageExcessReturn
    median = row.MedianExcessReturn
    fold_pos = _positive_pct(fold_rows["AverageExcessReturn"])
    year_pos = _positive_pct(year_rows["AverageExcessReturn"])
    fold_neg = _negative_pct(fold_rows["AverageExcessReturn"])
    year_neg = _negative_pct(year_rows["AverageExcessReturn"])
    if action in {"BUY", "ACCUMULATE"}:
        if count >= 500 and avg is not None and median is not None and avg > 0 and median > 0 and (fold_pos or 0) >= 0.67 and (year_pos or 0) >= 0.67:
            return "VALIDATION_CANDIDATE"
        return "NOT_STABLE_ENOUGH"
    if count >= 500 and avg is not None and median is not None and avg < 0 and median < 0 and (fold_neg or 0) >= 0.67 and (year_neg or 0) >= 0.67:
        return "VALIDATION_CANDIDATE"
    return "SELL_RULE_NOT_YET_VALIDATED"


def _recommended_rules(
    *,
    vedl_cleanup: pd.DataFrame,
    buy_analysis: pd.DataFrame,
    accumulate_analysis: pd.DataFrame,
    sell_analysis: pd.DataFrame,
    stability: pd.DataFrame,
    features: list[str],
    old_rows: int,
    new_rows: int,
) -> dict[str, Any]:
    buy_candidates = stability[
        (stability["Action"] == "BUY") & (stability["ValidationStatus"] == "VALIDATION_CANDIDATE")
    ].sort_values(["OverallPrecision", "OverallAverageExcessReturn"], ascending=False)
    accumulate_candidates = stability[
        (stability["Action"] == "ACCUMULATE") & (stability["ValidationStatus"] == "VALIDATION_CANDIDATE")
    ].sort_values(["OverallMedianExcessReturn", "OverallAverageExcessReturn"], ascending=False)
    sell_candidates = stability[
        (stability["Action"] == "SELL") & (stability["ValidationStatus"] == "VALIDATION_CANDIDATE")
    ].sort_values(["OverallAverageExcessReturn", "OverallMedianExcessReturn"])
    best_buy = buy_candidates.head(1).to_dict(orient="records")
    best_accumulate = accumulate_candidates.head(1).to_dict(orient="records")
    best_sell = sell_candidates.head(1).to_dict(orient="records")
    return {
        "selection_basis": "Walk-forward validation only; final test not used for threshold selection.",
        "source_dataset": str(CLEAN_DATASET),
        "demerger_clean_dataset": str(CLEAN_DEMERGER_DATASET),
        "feature_policy": {
            "feature_set": "Clean Pruned Advanced frozen",
            "features": features,
            "beta_252d": "kept with training-fold-only median imputation and missing-value indicator",
            "no_new_features_or_models": True,
        },
        "vedl_structural_break": {
            "event": VEDL_EVENT_TYPE,
            "date": VEDL_EVENT_DATE.date().isoformat(),
            "original_rows": old_rows,
            "post_cleanup_rows": new_rows,
            "rows_excluded": old_rows - new_rows,
            "validation_rows_excluded": int(vedl_cleanup["ValidationRowsExcluded"].max()) if not vedl_cleanup.empty else 0,
            "metric_impact": "Validation metrics unchanged because affected rows are in 2026 final-test period, not threshold-selection validation.",
        },
        "model_hierarchy": {
            "primary": "Clean Pruned Advanced 3-class classifier",
            "buy_confirmation": "XGBoost Binary BUY model; raw/sigmoid percentile ranking validated. Sigmoid retained for calibrated probability display, raw equivalent for ranking.",
            "secondary_confirmation": "XGBRanker confirmation only",
            "sell_diagnostic": "Random Forest SELL diagnostic only",
            "not_equal_weighted": True,
        },
        "buy_research": {
            "reference_rule": "Pruned + Binary BUY Top 1% agreement",
            "best_validation_candidate": best_buy,
            "status": "research_candidate_only_no_live_threshold_built",
        },
        "accumulate_research": {
            "best_validation_candidate": best_accumulate,
            "status": "positive-evidence research candidate only",
        },
        "sell_research": {
            "best_validation_candidate": best_sell,
            "status": "SELL_RULE_NOT_YET_VALIDATED" if not best_sell else "sell_candidate_requires_more_review",
        },
        "report_files": {
            "vedl_demerger_cleanup": str(OUTPUT_VEDL),
            "final_buy_threshold_analysis": str(OUTPUT_BUY),
            "final_accumulate_threshold_analysis": str(OUTPUT_ACCUMULATE),
            "final_sell_threshold_analysis": str(OUTPUT_SELL),
            "signal_fold_stability": str(OUTPUT_STABILITY),
        },
    }


def _top(series: pd.Series, fraction: float) -> pd.Series:
    return series >= 1 - fraction


def _bottom(series: pd.Series, fraction: float) -> pd.Series:
    return series <= fraction


def _pct(fraction: float) -> str:
    return f"{int(fraction * 100)}PCT"


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


if __name__ == "__main__":
    main()
