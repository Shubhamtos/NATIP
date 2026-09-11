"""Research stock-specific feature variant for the clean NSE ML system.

This script is intentionally report-only. It does not overwrite frozen
production models, hashes, configs, rules, or live inference artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _fit_classifier,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)

CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"
FROZEN_CONFIG = REPORT_DIR / "frozen_model_config.json"

OUTPUT_VARIANCE_AUDIT = REPORT_DIR / "flat_feature_variance_audit.csv"
OUTPUT_FEATURE_LIST = REPORT_DIR / "stock_specific_feature_list.json"
OUTPUT_MODEL_COMPARISON = REPORT_DIR / "stock_specific_model_comparison.csv"
OUTPUT_RANKER_COMPARISON = REPORT_DIR / "stock_specific_ranker_comparison.csv"
OUTPUT_BUY_COMPARISON = REPORT_DIR / "stock_specific_buy_comparison.csv"
OUTPUT_SUMMARY = REPORT_DIR / "flat_feature_ablation_summary.txt"

TOP_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01)

NIFTY_FEATURES = {
    "nifty_ret_20d",
    "nifty_ret_60d",
    "nifty_ret_120d",
    "nifty_above_sma20",
    "nifty_above_sma50",
    "nifty_above_sma200",
    "nifty_sma50_sma200_trend",
    "nifty_rsi_14",
    "nifty_adx_14",
    "nifty_atr_14",
    "nifty_atr_pct_14",
    "nifty_volatility_20d",
    "nifty_volatility_60d",
    "nifty_distance_52w_high",
}

MARKET_REGIME_FEATURES = {
    "market_regime_bull_trend",
    "market_regime_bear_trend",
    "market_regime_sideways",
    "market_regime_high_volatility",
}

MARKET_BREADTH_FEATURES = {
    "market_breadth_above_sma50",
    "market_breadth_above_sma200",
    "market_breadth_positive_20d_return",
    "market_breadth_advance_decline_ratio",
}

SECTOR_LEVEL_FEATURES = {
    "sector_momentum_20d",
    "sector_momentum_60d",
    "sector_momentum_120d",
    "sector_strength_vs_nifty_20d",
    "sector_strength_vs_nifty_60d",
    "sector_strength_vs_nifty_120d",
}

MUST_KEEP_FEATURES = {
    "rs_vs_nifty_20d",
    "rs_vs_nifty_60d",
    "rs_vs_nifty_120d",
    "rs_vs_nifty_252d",
    "stock_vs_sector_ret_20d",
    "stock_vs_sector_ret_60d",
    "stock_vs_sector_ret_120d",
    "stock_vs_sector_ret_252d",
    "stock_volatility_relative_sector_20d",
    "stock_volatility_relative_sector_60d",
    "stock_distance_52w_high_relative_sector",
    "beta_120d",
    "beta_252d",
    "corr_nifty_20d",
    "sector_rsi_14",
}


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Named feature-set variant."""

    name: str
    display_name: str
    features: list[str]


def main() -> None:
    """Run stock-specific V2 feature research."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    baseline_features = _frozen_features()
    removed_features = _removed_features(baseline_features)
    stock_specific_features = [
        feature for feature in baseline_features if feature not in removed_features
    ]

    _assert_feature_lists(baseline_features, removed_features, stock_specific_features)
    common = _common_evaluation_rows(pre_final, baseline_features)
    folds = _embargo_folds(common)
    feature_sets = [
        FeatureSet("pruned_77", "Existing 77-feature Pruned benchmark", baseline_features),
        FeatureSet("stock_specific_49", "Stock-Specific ~49-feature V2", stock_specific_features),
    ]

    print("[audit] flat and sector feature variance", flush=True)
    variance_audit = _variance_audit(common, removed_features)
    variance_audit.to_csv(OUTPUT_VARIANCE_AUDIT, index=False)
    _assert_market_features_are_flat(variance_audit)
    _write_feature_list(baseline_features, removed_features, stock_specific_features)

    classifier_outputs = []
    ranker_outputs = []
    buy_outputs = []
    for feature_set in feature_sets:
        print(f"[classifier] {feature_set.name}", flush=True)
        classifier_outputs.append(_classifier_walk_forward(common, folds, feature_set))
        print(f"[ranker] {feature_set.name}", flush=True)
        ranker_outputs.append(_ranker_walk_forward(common, folds, feature_set))
        print(f"[binary-buy] {feature_set.name}", flush=True)
        buy_outputs.append(_binary_buy_walk_forward(common, folds, feature_set))

    classifier_predictions = pd.concat(classifier_outputs, ignore_index=True)
    ranker_predictions = pd.concat(ranker_outputs, ignore_index=True)
    buy_predictions = pd.concat(buy_outputs, ignore_index=True)

    model_comparison = _classifier_comparison(classifier_predictions, feature_sets)
    ranker_comparison = _ranker_comparison(ranker_predictions, feature_sets)
    buy_comparison = _buy_comparison(buy_predictions, feature_sets)

    model_comparison.to_csv(OUTPUT_MODEL_COMPARISON, index=False)
    ranker_comparison.to_csv(OUTPUT_RANKER_COMPARISON, index=False)
    buy_comparison.to_csv(OUTPUT_BUY_COMPARISON, index=False)
    OUTPUT_SUMMARY.write_text(
        _summary_text(
            common=common,
            baseline_features=baseline_features,
            stock_specific_features=stock_specific_features,
            removed_features=removed_features,
            model_comparison=model_comparison,
            ranker_comparison=ranker_comparison,
            buy_comparison=buy_comparison,
            variance_audit=variance_audit,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "flat_feature_variance_audit": str(OUTPUT_VARIANCE_AUDIT),
                "stock_specific_feature_list": str(OUTPUT_FEATURE_LIST),
                "stock_specific_model_comparison": str(OUTPUT_MODEL_COMPARISON),
                "stock_specific_ranker_comparison": str(OUTPUT_RANKER_COMPARISON),
                "stock_specific_buy_comparison": str(OUTPUT_BUY_COMPARISON),
                "flat_feature_ablation_summary": str(OUTPUT_SUMMARY),
                "frozen_artifacts_modified": False,
                "final_test_status": "not_used_for_feature_selection",
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare clean dataset for validation-only research."""

    output = dataset.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output["buy_target"] = (output["excess_return"] > 0.05).astype(int)
    output["market_regime_label"] = output.apply(_market_regime_label, axis=1)
    return output.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=[
            "label",
            "excess_return",
            "future_stock_return",
            "future_nifty_return",
            "buy_target",
        ]
    )


def _market_regime_label(row: pd.Series) -> str:
    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _frozen_features() -> list[str]:
    """Load frozen 77-feature benchmark list without modifying it."""

    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    features = list(config["feature_list"])
    if len(features) != 77:
        raise AssertionError(
            f"Expected frozen benchmark to have 77 features, found {len(features)}."
        )
    return features


def _removed_features(features: list[str]) -> list[str]:
    """Return removed feature list in original frozen feature order."""

    remove = (
        NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES | SECTOR_LEVEL_FEATURES
    )
    missing = sorted(remove.difference(features))
    if missing:
        raise AssertionError(f"Requested removed features missing from frozen list: {missing}")
    accidental_keep_conflict = sorted(remove & MUST_KEEP_FEATURES)
    if accidental_keep_conflict:
        raise AssertionError(f"Remove/keep conflict: {accidental_keep_conflict}")
    return [feature for feature in features if feature in remove]


def _assert_feature_lists(
    baseline_features: list[str],
    removed_features: list[str],
    stock_specific_features: list[str],
) -> None:
    """Validate exact V2 feature-list shape."""

    if len(removed_features) != 28:
        raise AssertionError(
            f"Expected 28 removed flat/sector-level features, found {len(removed_features)}."
        )
    if len(stock_specific_features) != 49:
        raise AssertionError(
            f"Expected 49 stock-specific features from 77 - 28, found {len(stock_specific_features)}."
        )
    missing_kept = sorted(MUST_KEEP_FEATURES.difference(stock_specific_features))
    if missing_kept:
        raise AssertionError(
            f"Required stock-specific features were removed incorrectly: {missing_kept}"
        )
    if len(set(baseline_features)) != len(baseline_features):
        raise AssertionError("Frozen feature list contains duplicates.")


def _common_evaluation_rows(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return common stock/date observations for all variants."""

    required = [
        "Date",
        "symbol",
        "Sector",
        "MarketCapCategory",
        "label",
        "excess_return",
        "future_stock_return",
        "future_nifty_return",
        "buy_target",
        "market_regime_label",
    ]
    missing_columns = [column for column in [*required, *features] if column not in dataset.columns]
    if missing_columns:
        raise AssertionError(f"Dataset missing required columns: {missing_columns}")
    return dataset[required + features].copy()


def _variance_audit(dataset: pd.DataFrame, removed_features: list[str]) -> pd.DataFrame:
    """Audit within-date and within-sector/date variance for removed features."""

    rows = []
    for feature in removed_features:
        within_date = dataset.groupby("Date")[feature].var(ddof=0)
        within_sector_date = dataset.groupby(["Date", "Sector"])[feature].var(ddof=0)
        group = _feature_group(feature)
        rows.append(
            {
                "Feature": feature,
                "RemovalGroup": group,
                "Removed": True,
                "WithinDateVarianceMax": _safe_float(within_date.max()),
                "WithinDateVarianceMean": _safe_float(within_date.mean()),
                "WithinDateNonZeroDates": int((within_date.fillna(0).abs() > 1e-12).sum()),
                "WithinSectorDateVarianceMax": _safe_float(within_sector_date.max()),
                "WithinSectorDateNonZeroGroups": int(
                    (within_sector_date.fillna(0).abs() > 1e-12).sum()
                ),
                "ExpectedFlatAcrossAllStocksOnDate": group
                in {"nifty", "market_regime", "market_breadth"},
                "ExpectedRepeatedWithinSectorOnDate": group == "sector_level",
            }
        )
    return pd.DataFrame(rows)


def _feature_group(feature: str) -> str:
    if feature in NIFTY_FEATURES:
        return "nifty"
    if feature in MARKET_REGIME_FEATURES:
        return "market_regime"
    if feature in MARKET_BREADTH_FEATURES:
        return "market_breadth"
    if feature in SECTOR_LEVEL_FEATURES:
        return "sector_level"
    return "other"


def _assert_market_features_are_flat(audit: pd.DataFrame) -> None:
    """Assert removed market-wide features have zero same-date cross-sectional variance."""

    market = audit[audit["RemovalGroup"].isin(["nifty", "market_regime", "market_breadth"])]
    bad = market[
        pd.to_numeric(market["WithinDateVarianceMax"], errors="coerce").fillna(0).abs() > 1e-12
    ]
    if not bad.empty:
        raise AssertionError(
            "Some market-wide removed features are not flat within date: "
            f"{bad[['Feature', 'WithinDateVarianceMax']].to_dict(orient='records')}"
        )


def _write_feature_list(
    baseline_features: list[str],
    removed_features: list[str],
    stock_specific_features: list[str],
) -> None:
    """Write feature-list manifest for V2 research."""

    payload = {
        "experiment": "stock_specific_feature_variant_v2_research_only",
        "production_frozen_model_modified": False,
        "selection_data_policy": "walk_forward_pre_final_only; final test not used for feature selection",
        "benchmark_feature_count": len(baseline_features),
        "removed_feature_count": len(removed_features),
        "stock_specific_feature_count": len(stock_specific_features),
        "expected_stock_specific_feature_count": 49,
        "removed_features": removed_features,
        "stock_specific_features": stock_specific_features,
        "must_keep_features_confirmed": sorted(MUST_KEEP_FEATURES),
    }
    OUTPUT_FEATURE_LIST.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _classifier_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Generate 3-class classifier validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, feature_set.features, target="label")
        x = _transform_with_medians(validation, feature_set.features, medians)
        probabilities = model.predict_proba(x)
        output = _base_frame(validation, fold["fold"], feature_set)
        output["target_label"] = validation["label"].astype(int).to_numpy()
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["predicted_label"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map(
            {0: -1, 1: 0, 2: 1}
        )
        output["score_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Generate XGBRanker validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_ranker(train, feature_set.features)
        x = _transform_with_medians(validation, feature_set.features, medians)
        output = _base_frame(validation, fold["fold"], feature_set)
        output["ranker_score"] = model.predict(x)
        output["score_percentile"] = output.groupby("Date")["ranker_score"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_buy_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """Generate binary XGBoost BUY validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(
            train,
            feature_set.features,
            target="buy_target",
            model_name="xgboost",
            binary=True,
        )
        x = _transform_with_medians(validation, feature_set.features, medians)
        probability = _positive_probability(model, x)
        output = _base_frame(validation, fold["fold"], feature_set)
        output["actual"] = validation["buy_target"].astype(int).to_numpy()
        output["p_buy"] = probability
        output["p_outperform"] = probability
        output["score_percentile"] = output.groupby("Date")["p_buy"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
    validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
    return train, validation


def _base_frame(frame: pd.DataFrame, fold: Any, feature_set: FeatureSet) -> pd.DataFrame:
    output = frame[
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
    output["FeatureSet"] = feature_set.name
    output["FeatureSetName"] = feature_set.display_name
    output["FeatureCount"] = len(feature_set.features)
    output["fold"] = fold
    return output


def _classifier_comparison(
    predictions: pd.DataFrame, feature_sets: list[FeatureSet]
) -> pd.DataFrame:
    """Build 3-class classifier comparison rows."""

    rows = []
    for feature_set in feature_sets:
        frame = predictions[predictions["FeatureSet"].eq(feature_set.name)].copy()
        rows.extend(_classifier_rows(frame, feature_set))
    return pd.DataFrame(rows)


def _classifier_rows(frame: pd.DataFrame, feature_set: FeatureSet) -> list[dict[str, Any]]:
    from sklearn.metrics import average_precision_score, log_loss, precision_score

    y_true = frame["target_label"].map(LABEL_TO_CLASS)
    probabilities = frame[["p_underperform", "p_neutral", "p_outperform"]]
    y_pred = frame["predicted_label"].map(LABEL_TO_CLASS)
    actual_buy = (frame["target_label"] == 1).astype(int)
    backtest = backtest_top_n_portfolio(frame, top_n=5)
    base = {
        "Experiment": "3class_classifier",
        "FeatureSet": feature_set.name,
        "FeatureSetName": feature_set.display_name,
        "FeatureCount": len(feature_set.features),
        "Rows": int(len(frame)),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(((frame["p_outperform"] - actual_buy) ** 2).mean()),
        "PRAUC": float(average_precision_score(actual_buy, frame["p_outperform"])),
        "Precision": float(
            precision_score(y_true, y_pred, labels=[2], average="micro", zero_division=0)
        ),
        "MeanIC": _ic_summary(frame, "p_outperform")["MeanIC"],
        "MedianIC": _ic_summary(frame, "p_outperform")["MedianIC"],
        "Sharpe": backtest.get("sharpe"),
        "MaxDrawdown": backtest.get("max_drawdown"),
    }
    return [base, *_tail_rows(frame, feature_set, "p_outperform", actual_column="buy_target")]


def _ranker_comparison(predictions: pd.DataFrame, feature_sets: list[FeatureSet]) -> pd.DataFrame:
    """Build XGBRanker comparison rows."""

    rows = []
    for feature_set in feature_sets:
        frame = predictions[predictions["FeatureSet"].eq(feature_set.name)].copy()
        summary = _ic_summary(frame, "ranker_score")
        rows.append(
            {
                "Experiment": "xgbranker",
                "FeatureSet": feature_set.name,
                "FeatureSetName": feature_set.display_name,
                "FeatureCount": len(feature_set.features),
                "Rows": int(len(frame)),
                **summary,
            }
        )
        for year, group in frame.groupby(frame["Date"].dt.year):
            year_summary = _ic_summary(group, "ranker_score")
            rows.append(
                {
                    "Experiment": "xgbranker_year_stability",
                    "FeatureSet": feature_set.name,
                    "FeatureSetName": feature_set.display_name,
                    "FeatureCount": len(feature_set.features),
                    "Year": int(year),
                    "Rows": int(len(group)),
                    **year_summary,
                }
            )
        rows.extend(_tail_rows(frame, feature_set, "ranker_score", actual_column="buy_target"))
    return pd.DataFrame(rows)


def _buy_comparison(predictions: pd.DataFrame, feature_sets: list[FeatureSet]) -> pd.DataFrame:
    """Build binary BUY comparison rows."""

    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, precision_score

    rows = []
    for feature_set in feature_sets:
        frame = predictions[predictions["FeatureSet"].eq(feature_set.name)].copy()
        actual = frame["actual"].astype(int)
        predicted = (frame["p_buy"] >= 0.5).astype(int)
        backtest = backtest_top_n_portfolio(frame, top_n=5)
        rows.append(
            {
                "Experiment": "binary_xgboost_buy",
                "FeatureSet": feature_set.name,
                "FeatureSetName": feature_set.display_name,
                "FeatureCount": len(feature_set.features),
                "Rows": int(len(frame)),
                "LogLoss": float(log_loss(actual, frame["p_buy"], labels=[0, 1])),
                "Brier": float(brier_score_loss(actual, frame["p_buy"])),
                "PRAUC": float(average_precision_score(actual, frame["p_buy"])),
                "Precision": float(precision_score(actual, predicted, zero_division=0)),
                "MeanIC": _ic_summary(frame, "p_buy")["MeanIC"],
                "MedianIC": _ic_summary(frame, "p_buy")["MedianIC"],
                "Sharpe": backtest.get("sharpe"),
                "MaxDrawdown": backtest.get("max_drawdown"),
            }
        )
        rows.extend(_tail_rows(frame, feature_set, "p_buy", actual_column="actual"))
    return pd.DataFrame(rows)


def _tail_rows(
    frame: pd.DataFrame,
    feature_set: FeatureSet,
    score_column: str,
    *,
    actual_column: str,
) -> list[dict[str, Any]]:
    """Return Top20/10/5/2/1 realized performance rows."""

    rows = []
    for fraction in TOP_FRACTIONS:
        selected = _top_fraction(frame, score_column, fraction)
        rows.append(
            {
                "Experiment": "tail_percentile",
                "FeatureSet": feature_set.name,
                "FeatureSetName": feature_set.display_name,
                "FeatureCount": len(feature_set.features),
                "TopPercentile": _top_percentile_label(fraction),
                "Count": int(len(selected)),
                "PrecisionAtPercentile": float(selected[actual_column].astype(int).mean()),
                "AverageFutureReturn": _safe_float(selected["future_stock_return"].mean()),
                "MedianFutureReturn": _safe_float(selected["future_stock_return"].median()),
                "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
            }
        )
    return rows


def _top_percentile_label(fraction: float) -> str:
    """Return a readable top-percentile label."""

    percent = fraction * 100
    return f"Top {percent:g}%"


def _top_fraction(frame: pd.DataFrame, score_column: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(len(frame) * fraction))
    return frame.sort_values(score_column, ascending=False).head(count)


def _ic_summary(frame: pd.DataFrame, score_column: str) -> dict[str, Any]:
    daily = []
    for _, group in frame.groupby("Date"):
        if len(group) < 5 or group[score_column].nunique() < 2:
            continue
        ic = group[score_column].corr(group["excess_return"], method="spearman")
        if pd.notna(ic):
            daily.append(float(ic))
    series = pd.Series(daily, dtype=float)
    std = series.std()
    return {
        "ICDates": int(len(series)),
        "MeanIC": _safe_float(series.mean()),
        "MedianIC": _safe_float(series.median()),
        "ICStd": _safe_float(std),
        "ICIR": _safe_float(series.mean() / std) if len(series) > 1 and std else None,
        "PositiveICDatePct": _safe_float((series > 0).mean()),
    }


def _summary_text(
    *,
    common: pd.DataFrame,
    baseline_features: list[str],
    stock_specific_features: list[str],
    removed_features: list[str],
    model_comparison: pd.DataFrame,
    ranker_comparison: pd.DataFrame,
    buy_comparison: pd.DataFrame,
    variance_audit: pd.DataFrame,
) -> str:
    """Build human-readable ablation summary."""

    classifier = model_comparison[model_comparison["Experiment"].eq("3class_classifier")]
    buy = buy_comparison[buy_comparison["Experiment"].eq("binary_xgboost_buy")]
    ranker = ranker_comparison[ranker_comparison["Experiment"].eq("xgbranker")]
    classifier_decision = _variant_decision(classifier, metric="PRAUC", higher=True)
    buy_decision = _variant_decision(buy, metric="PRAUC", higher=True)
    ranker_decision = _variant_decision(ranker, metric="MeanIC", higher=True)
    market_flat = variance_audit[
        variance_audit["RemovalGroup"].isin(["nifty", "market_regime", "market_breadth"])
    ]
    sector = variance_audit[variance_audit["RemovalGroup"].eq("sector_level")]
    return "\n".join(
        [
            "NATIP Stock-Specific Feature Variant V2",
            "========================================",
            "",
            "Production impact: NONE. Frozen production model/config/rules were not modified.",
            "Final test usage: NOT USED for feature selection or model recommendation.",
            f"Common pre-final evaluation rows: {len(common):,}",
            f"Benchmark feature count: {len(baseline_features)}",
            f"Removed feature count: {len(removed_features)}",
            f"Stock-specific feature count: {len(stock_specific_features)}",
            "",
            "Flat-feature audit:",
            f"- Market-wide removed features audited: {len(market_flat)}",
            f"- Max market-wide within-date variance: {market_flat['WithinDateVarianceMax'].max()}",
            f"- Sector-level removed features audited separately: {len(sector)}",
            f"- Max sector-level within-date variance: {sector['WithinDateVarianceMax'].max()}",
            f"- Max sector-level within-sector/date variance: {sector['WithinSectorDateVarianceMax'].max()}",
            "",
            "Validation-only decision checks:",
            f"- 3-class classifier PRAUC decision: {classifier_decision}",
            f"- XGBRanker Mean IC decision: {ranker_decision}",
            f"- Binary BUY PRAUC decision: {buy_decision}",
            "",
            "Recommendation rule:",
            "Recommend the ~49-feature model only if it preserves or improves out-of-fold "
            "PR-AUC, Mean IC, BUY Top 1%/Top 5% precision, and average/median excess-return stability.",
            "",
            "See CSV reports for Top20/10/5/2/1 precision and excess-return details.",
        ]
    )


def _variant_decision(frame: pd.DataFrame, *, metric: str, higher: bool) -> str:
    if frame.empty or metric not in frame.columns:
        return "insufficient data"
    indexed = frame.set_index("FeatureSet")
    baseline = indexed.loc["pruned_77", metric]
    stock_specific = indexed.loc["stock_specific_49", metric]
    improved = stock_specific >= baseline if higher else stock_specific <= baseline
    return (
        f"stock_specific_49 {'improved/preserved' if improved else 'did_not_preserve'} "
        f"({metric}: {stock_specific:.6f} vs benchmark {baseline:.6f})"
    )


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
