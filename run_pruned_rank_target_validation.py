"""Validate pruned advanced model with embargo, ranks, and quintile targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.features import feature_columns_for_variant
from app.probability.train import LABEL_TO_CLASS
from audit_advanced_probability_features import _build_advanced_dataset

OUTPUT_EMBARGO = REPORT_DIR / "embargo_audit.csv"
OUTPUT_RANK_COMPARISON = REPORT_DIR / "cross_sectional_rank_comparison.csv"
OUTPUT_TARGET_COMPARISON = REPORT_DIR / "target_comparison.csv"
OUTPUT_TOP_PERCENTILE = REPORT_DIR / "top_percentile_performance.csv"
OUTPUT_RANK_IC = REPORT_DIR / "rank_ic_report.csv"
OUTPUT_RECOMMENDATION = REPORT_DIR / "next_model_recommendation.json"

HORIZON_DAYS = 20
AUDIT_ESTIMATORS = 80
FINAL_TEST_START = pd.Timestamp("2025-01-01")

PRUNED_FEATURES = [
    "beta_252d",
    "beta_120d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "sector_rsi_14",
    "corr_nifty_20d",
    "relative_momentum_20d_change",
]

RANK_SOURCE_FEATURES = [
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "rs_vs_nifty_20d",
    "rs_vs_nifty_60d",
    "rs_vs_nifty_120d",
    "volume_to_avg20",
    "rsi_14",
    "volatility_20d",
    "volatility_60d",
    "distance_52w_high",
    "stock_vs_sector_ret_60d",
    "sector_strength_vs_nifty_60d",
    "beta_120d",
    "corr_nifty_20d",
]


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Model variant specification."""

    name: str
    features: list[str]
    target: str
    positive_label: int
    candidate_name: str


def main() -> None:
    """Run leakage-safe pruned model validation."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(_build_advanced_dataset())
    pruned = _pruned_features()
    rank_features = [f"{feature}_cs_pct" for feature in RANK_SOURCE_FEATURES]
    specs = [
        VariantSpec("pruned_fixed_target", pruned, "label", 1, "Pruned Advanced"),
        VariantSpec(
            "pruned_ranks_fixed_target",
            [*pruned, *rank_features],
            "label",
            1,
            "Pruned + cross-sectional ranks",
        ),
        VariantSpec(
            "pruned_quintile_target",
            pruned,
            "quintile_label",
            1,
            "Pruned + quintile target",
        ),
        VariantSpec(
            "pruned_ranks_quintile_target",
            [*pruned, *rank_features],
            "quintile_label",
            1,
            "Pruned + ranks + quintile target",
        ),
    ]
    pre_final = dataset[pd.to_datetime(dataset["Date"]) < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)

    all_predictions = []
    embargo_rows = []
    for spec in specs:
        print(f"[walk-forward] {spec.name}", flush=True)
        predictions, audit = _walk_forward_predictions(pre_final, spec, folds)
        all_predictions.append(predictions)
        embargo_rows.extend(audit)
    embargo = pd.DataFrame(embargo_rows)
    embargo.to_csv(OUTPUT_EMBARGO, index=False)

    walk_forward = pd.concat(all_predictions, ignore_index=True)
    comparison = _comparison_report(walk_forward, specs)
    rank_comparison = comparison[comparison["TargetType"].eq("fixed_threshold")].copy()
    target_comparison = comparison.copy()
    top_percentile = _top_percentile_report(walk_forward, specs)
    rank_ic = _rank_ic_report(walk_forward, specs)

    best_spec = _select_best_spec(comparison)
    print(f"[final-test] {best_spec.name}", flush=True)
    final_predictions, final_audit = _final_test_predictions(dataset, best_spec)
    final_metrics = _metric_row(final_predictions, best_spec, scope="final_test_once")
    final_top = _top_percentile_report(final_predictions, [best_spec], scope="final_test_once")
    final_ic = _rank_ic_report(final_predictions, [best_spec], scope="final_test_once")

    rank_comparison.to_csv(OUTPUT_RANK_COMPARISON, index=False)
    pd.concat([target_comparison, pd.DataFrame([final_metrics])], ignore_index=True).to_csv(
        OUTPUT_TARGET_COMPARISON, index=False
    )
    pd.concat([top_percentile, final_top], ignore_index=True).to_csv(
        OUTPUT_TOP_PERCENTILE, index=False
    )
    pd.concat([rank_ic, final_ic], ignore_index=True).to_csv(OUTPUT_RANK_IC, index=False)
    pd.concat([embargo, pd.DataFrame(final_audit)], ignore_index=True).to_csv(
        OUTPUT_EMBARGO, index=False
    )
    recommendation = _recommendation_payload(
        best_spec,
        comparison,
        final_metrics,
        specs,
    )
    OUTPUT_RECOMMENDATION.write_text(
        json.dumps(_json_safe(recommendation), indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "embargo_audit": str(OUTPUT_EMBARGO),
                "cross_sectional_rank_comparison": str(OUTPUT_RANK_COMPARISON),
                "target_comparison": str(OUTPUT_TARGET_COMPARISON),
                "top_percentile_performance": str(OUTPUT_TOP_PERCENTILE),
                "rank_ic_report": str(OUTPUT_RANK_IC),
                "next_model_recommendation": str(OUTPUT_RECOMMENDATION),
                "selected_validation_candidate": best_spec.candidate_name,
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add same-date rank features and quintile labels."""

    output = dataset.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    for feature in RANK_SOURCE_FEATURES:
        output[f"{feature}_cs_pct"] = output.groupby("Date")[feature].rank(pct=True)
    output["quintile_label"] = output.groupby("Date", group_keys=False)["excess_return"].apply(
        _quintile_labels
    )
    return output.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["label", "quintile_label", "excess_return"]
    )


def _quintile_labels(excess_returns: pd.Series) -> pd.Series:
    """Map same-date future excess returns to loser/neutral/winner labels."""

    lower = excess_returns.quantile(0.20)
    upper = excess_returns.quantile(0.80)
    return pd.Series(
        [
            1 if value >= upper else -1 if value <= lower else 0
            for value in excess_returns
        ],
        index=excess_returns.index,
    )


def _pruned_features() -> list[str]:
    """Return primary pruned advanced feature list."""

    return _dedupe([*feature_columns_for_variant("sector_regime"), *PRUNED_FEATURES])


def _embargo_folds(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    """Create walk-forward folds with a 20-trading-day embargo before validation."""

    dates = pd.Series(sorted(dataset["Date"].drop_duplicates()))
    current = dates.min() + pd.DateOffset(years=3)
    end = min(FINAL_TEST_START, dates.max())
    folds = []
    fold = 1
    while current < end:
        validation_start = dates[dates >= current].min()
        validation_end = min(current + pd.DateOffset(months=6), end)
        validation_dates = dates[(dates >= validation_start) & (dates < validation_end)]
        if validation_dates.empty:
            current = validation_end
            continue
        validation_start = validation_dates.min()
        validation_end_actual = validation_dates.max()
        before_validation = dates[dates < validation_start]
        train_dates = before_validation.iloc[:-HORIZON_DAYS] if len(before_validation) > HORIZON_DAYS else before_validation.iloc[:0]
        embargo_dates = before_validation.iloc[-HORIZON_DAYS:] if len(before_validation) >= HORIZON_DAYS else before_validation
        if not train_dates.empty:
            folds.append(
                {
                    "fold": fold,
                    "train_dates": set(train_dates),
                    "validation_dates": set(validation_dates),
                    "train_start": train_dates.min(),
                    "train_end": train_dates.max(),
                    "embargo_start": embargo_dates.min() if not embargo_dates.empty else pd.NaT,
                    "embargo_end": embargo_dates.max() if not embargo_dates.empty else pd.NaT,
                    "validation_start": validation_start,
                    "validation_end": validation_end_actual,
                }
            )
            fold += 1
        current = validation_end
    return folds


def _walk_forward_predictions(
    dataset: pd.DataFrame,
    spec: VariantSpec,
    folds: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Train/predict folds with fold-local median imputation and embargo checks."""

    frames = []
    audit = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].dropna(subset=[spec.target])
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].dropna(
            subset=[spec.target]
        )
        assert _no_target_overlap(
            train_end=fold["train_end"],
            validation_start=fold["validation_start"],
            all_dates=sorted(dataset["Date"].drop_duplicates()),
        )
        model = _fit_model(train, spec.features, spec.target)
        predictions = _prediction_frame(model, validation, spec, fold["fold"])
        frames.append(predictions)
        audit.append(
            {
                "ModelVariant": spec.name,
                "Fold": fold["fold"],
                "TrainStart": fold["train_start"],
                "TrainEnd": fold["train_end"],
                "EmbargoStart": fold["embargo_start"],
                "EmbargoEnd": fold["embargo_end"],
                "ValidationStart": fold["validation_start"],
                "ValidationEnd": fold["validation_end"],
                "TrainRows": len(train),
                "ValidationRows": len(validation),
                "EmbargoTradingDays": HORIZON_DAYS,
                "LeakageAssertionPassed": True,
            }
        )
    return pd.concat(frames, ignore_index=True), audit


def _final_test_predictions(
    dataset: pd.DataFrame,
    spec: VariantSpec,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate the selected configuration once on untouched final test."""

    all_dates = pd.Series(sorted(dataset["Date"].drop_duplicates()))
    final_dates = all_dates[all_dates >= FINAL_TEST_START]
    before_final = all_dates[all_dates < final_dates.min()]
    train_dates = before_final.iloc[:-HORIZON_DAYS]
    embargo_dates = before_final.iloc[-HORIZON_DAYS:]
    train = dataset[dataset["Date"].isin(set(train_dates))].dropna(subset=[spec.target])
    final = dataset[dataset["Date"].isin(set(final_dates))].dropna(subset=[spec.target])
    assert _no_target_overlap(train_dates.max(), final_dates.min(), list(all_dates))
    model = _fit_model(train, spec.features, spec.target)
    predictions = _prediction_frame(model, final, spec, fold="final")
    audit = [
        {
            "ModelVariant": spec.name,
            "Fold": "final_test_once",
            "TrainStart": train_dates.min(),
            "TrainEnd": train_dates.max(),
            "EmbargoStart": embargo_dates.min(),
            "EmbargoEnd": embargo_dates.max(),
            "ValidationStart": final_dates.min(),
            "ValidationEnd": final_dates.max(),
            "TrainRows": len(train),
            "ValidationRows": len(final),
            "EmbargoTradingDays": HORIZON_DAYS,
            "LeakageAssertionPassed": True,
        }
    ]
    return predictions, audit


def _no_target_overlap(
    train_end: pd.Timestamp,
    validation_start: pd.Timestamp,
    all_dates: list[pd.Timestamp],
) -> bool:
    """Assert the latest training label horizon ends before validation starts."""

    dates = pd.Series(sorted(pd.to_datetime(all_dates)))
    train_idx = int(dates[dates == train_end].index[0])
    target_end_idx = min(train_idx + HORIZON_DAYS, len(dates) - 1)
    target_end = dates.iloc[target_end_idx]
    if target_end >= validation_start:
        raise AssertionError(
            f"Training target overlaps validation: target_end={target_end}, "
            f"validation_start={validation_start}"
        )
    return True


def _fit_model(train: pd.DataFrame, features: list[str], target: str):
    """Fit XGBoost with training-fold-only median imputation and missing indicators."""

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=AUDIT_ESTIMATORS,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    pipeline = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), model)
    pipeline.fit(train[features], train[target].map(LABEL_TO_CLASS))
    return pipeline


def _prediction_frame(model, validation: pd.DataFrame, spec: VariantSpec, fold: Any) -> pd.DataFrame:
    """Return prediction rows."""

    probabilities = model.predict_proba(validation[spec.features])
    output = validation[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "label",
            "quintile_label",
        ]
    ].copy()
    output["ModelVariant"] = spec.name
    output["CandidateName"] = spec.candidate_name
    output["TargetColumn"] = spec.target
    output["TargetType"] = "quintile" if spec.target == "quintile_label" else "fixed_threshold"
    output["fold"] = fold
    output["target_label"] = validation[spec.target].to_numpy()
    output["p_loser_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
    output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
    output["p_winner_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
    output["p_outperform"] = output["p_winner_outperform"]
    output["predicted_label"] = pd.Series(probabilities.argmax(axis=1), index=output.index).map(
        {0: -1, 1: 0, 2: 1}
    )
    output["score_percentile"] = output.groupby("Date")["p_winner_outperform"].rank(pct=True)
    return output


def _comparison_report(predictions: pd.DataFrame, specs: list[VariantSpec]) -> pd.DataFrame:
    """Build validation comparison rows."""

    rows = []
    for spec in specs:
        group = predictions[predictions["ModelVariant"] == spec.name]
        rows.append(_metric_row(group, spec, scope="walk_forward_validation"))
    return pd.DataFrame(rows)


def _metric_row(predictions: pd.DataFrame, spec: VariantSpec, *, scope: str) -> dict[str, Any]:
    """Calculate model comparison metrics."""

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        classification_report,
        log_loss,
    )

    y_true = predictions["target_label"].map(LABEL_TO_CLASS)
    y_pred = predictions["predicted_label"].map(LABEL_TO_CLASS)
    probabilities = predictions[["p_loser_underperform", "p_neutral", "p_winner_outperform"]]
    actual_positive = (predictions["target_label"] == spec.positive_label).astype(int)
    report = classification_report(
        y_true,
        y_pred,
        labels=[LABEL_TO_CLASS[-1], LABEL_TO_CLASS[0], LABEL_TO_CLASS[1]],
        output_dict=True,
        zero_division=0,
    )
    calibration_true, calibration_pred = calibration_curve(
        actual_positive,
        predictions["p_winner_outperform"],
        n_bins=10,
        strategy="quantile",
    )
    top10 = _top_fraction(predictions, 0.10)
    top5 = _top_fraction(predictions, 0.05)
    top1 = _top_fraction(predictions, 0.01)
    backtest = backtest_top_n_portfolio(predictions, top_n=5)
    ic = _rank_ic_summary(predictions)
    return {
        "ModelVariant": spec.name,
        "CandidateName": spec.candidate_name,
        "Scope": scope,
        "TargetType": "quintile" if spec.target == "quintile_label" else "fixed_threshold",
        "Rows": len(predictions),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(((predictions["p_winner_outperform"] - actual_positive) ** 2).mean()),
        "PRAUC": float(average_precision_score(actual_positive, predictions["p_winner_outperform"])),
        "CalibrationMAE": float(abs(calibration_true - calibration_pred).mean()),
        "Precision": report[str(LABEL_TO_CLASS[1])]["precision"],
        "Recall": report[str(LABEL_TO_CLASS[1])]["recall"],
        "MacroF1": report["macro avg"]["f1-score"],
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "PrecisionAtTop10Pct": _positive_rate(top10, spec),
        "PrecisionAtTop5Pct": _positive_rate(top5, spec),
        "PrecisionAtTop1Pct": _positive_rate(top1, spec),
        "AvgExcessTop10Pct": float(top10["excess_return"].mean()),
        "AvgExcessTop5Pct": float(top5["excess_return"].mean()),
        "AvgExcessTop1Pct": float(top1["excess_return"].mean()),
        "CAGR": backtest.get("cagr"),
        "Sharpe": backtest.get("sharpe"),
        "MaxDrawdown": backtest.get("max_drawdown"),
        "RankICMean": ic["MeanIC"],
        "RankICMedian": ic["MedianIC"],
        "RankICPositivePct": ic["PositiveICPct"],
    }


def _top_percentile_report(
    predictions: pd.DataFrame,
    specs: list[VariantSpec],
    *,
    scope: str = "walk_forward_validation",
) -> pd.DataFrame:
    """Report top score percentile realized performance."""

    rows = []
    for spec in specs:
        group = predictions[predictions["ModelVariant"] == spec.name]
        base_rate = (group["target_label"] == spec.positive_label).mean()
        for percentile in (0.20, 0.10, 0.05, 0.02, 0.01):
            selected = _top_fraction(group, percentile)
            actual_rate = _positive_rate(selected, spec)
            rows.append(
                {
                    "ModelVariant": spec.name,
                    "CandidateName": spec.candidate_name,
                    "Scope": scope,
                    "TargetType": "quintile" if spec.target == "quintile_label" else "fixed_threshold",
                    "TopPercentile": f"Top {int(percentile * 100)}%",
                    "Count": int(len(selected)),
                    "ActualPositiveRate": actual_rate,
                    "LiftOverBaseRate": float(actual_rate / base_rate) if base_rate else None,
                    "AverageFutureReturn": float(selected["future_stock_return"].mean()),
                    "MedianFutureReturn": float(selected["future_stock_return"].median()),
                    "AverageExcessReturn": float(selected["excess_return"].mean()),
                    "MedianExcessReturn": float(selected["excess_return"].median()),
                }
            )
    return pd.DataFrame(rows)


def _rank_ic_report(
    predictions: pd.DataFrame,
    specs: list[VariantSpec],
    *,
    scope: str = "walk_forward_validation",
) -> pd.DataFrame:
    """Report Spearman rank IC by variant."""

    rows = []
    for spec in specs:
        group = predictions[predictions["ModelVariant"] == spec.name]
        summary = _rank_ic_summary(group)
        rows.append(
            {
                "ModelVariant": spec.name,
                "CandidateName": spec.candidate_name,
                "Scope": scope,
                "TargetType": "quintile" if spec.target == "quintile_label" else "fixed_threshold",
                **summary,
            }
        )
    return pd.DataFrame(rows)


def _rank_ic_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    """Calculate daily Spearman IC between score and future excess return."""

    ics = []
    for _, group in predictions.groupby("Date"):
        if len(group) < 5 or group["p_winner_outperform"].nunique() < 2:
            continue
        ic = group["p_winner_outperform"].corr(group["excess_return"], method="spearman")
        if pd.notna(ic):
            ics.append(ic)
    series = pd.Series(ics, dtype=float)
    return {
        "Dates": int(len(series)),
        "MeanIC": float(series.mean()) if not series.empty else None,
        "MedianIC": float(series.median()) if not series.empty else None,
        "ICStd": float(series.std()) if not series.empty else None,
        "PositiveICPct": float((series > 0).mean()) if not series.empty else None,
    }


def _select_best_spec(comparison: pd.DataFrame) -> VariantSpec:
    """Select best validation candidate without using final test."""

    ranking = comparison.copy()
    ranking["SelectionScore"] = (
        ranking["PRAUC"].rank(ascending=False)
        + ranking["RankICMean"].rank(ascending=False)
        + ranking["PrecisionAtTop10Pct"].rank(ascending=False)
        + ranking["LogLoss"].rank(ascending=True)
    )
    winner = ranking.sort_values(["SelectionScore", "Sharpe"], ascending=[True, False]).iloc[0]
    specs = {spec.name: spec for spec in _current_specs()}
    return specs[str(winner["ModelVariant"])]


def _current_specs() -> list[VariantSpec]:
    """Recreate current specs for selection lookup."""

    pruned = _pruned_features()
    rank_features = [f"{feature}_cs_pct" for feature in RANK_SOURCE_FEATURES]
    return [
        VariantSpec("pruned_fixed_target", pruned, "label", 1, "Pruned Advanced"),
        VariantSpec(
            "pruned_ranks_fixed_target",
            [*pruned, *rank_features],
            "label",
            1,
            "Pruned + cross-sectional ranks",
        ),
        VariantSpec("pruned_quintile_target", pruned, "quintile_label", 1, "Pruned + quintile target"),
        VariantSpec(
            "pruned_ranks_quintile_target",
            [*pruned, *rank_features],
            "quintile_label",
            1,
            "Pruned + ranks + quintile target",
        ),
    ]


def _recommendation_payload(
    best_spec: VariantSpec,
    comparison: pd.DataFrame,
    final_metrics: dict[str, Any],
    specs: list[VariantSpec],
) -> dict[str, Any]:
    """Build JSON recommendation."""

    return {
        "selection_basis": "Walk-forward validation only; final test evaluated once after selection.",
        "next_production_candidate": best_spec.candidate_name,
        "allowed_candidates": [spec.candidate_name for spec in specs],
        "validation_comparison": comparison.to_dict(orient="records"),
        "final_test_once_for_selected_candidate": final_metrics,
        "embargo_policy": {
            "prediction_horizon_trading_days": HORIZON_DAYS,
            "minimum_embargo_trading_days": HORIZON_DAYS,
            "assertion": "Latest training label horizon must end before validation/test start.",
        },
        "notes": [
            "No BUY / ACCUMULATE / SELL rules were implemented.",
            "No new RSI5, Stochastic, OBV, or other technical indicators were added.",
            "Cross-sectional ranks are same-date percentile ranks only.",
        ],
    }


def _top_fraction(predictions: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Return top same-score rows by global score percentile."""

    count = max(1, int(len(predictions) * fraction))
    return predictions.sort_values("p_winner_outperform", ascending=False).head(count)


def _positive_rate(predictions: pd.DataFrame, spec: VariantSpec) -> float:
    """Return actual positive class rate."""

    return float((predictions["target_label"] == spec.positive_label).mean())


def _dedupe(items: list[str]) -> list[str]:
    """Preserve order while removing duplicates."""

    return list(dict.fromkeys(items))


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy values to JSON-safe values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


if __name__ == "__main__":
    main()
