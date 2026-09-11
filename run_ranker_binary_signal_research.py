"""Research ranker, binary BUY/SELL, and combined signals for NATIP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.features import feature_columns_for_variant
from app.probability.train import LABEL_TO_CLASS
from audit_advanced_probability_features import _build_advanced_dataset
from run_pruned_rank_target_validation import (
    FINAL_TEST_START,
    HORIZON_DAYS,
    PRUNED_FEATURES,
    _embargo_folds,
    _no_target_overlap,
)

OUTPUT_RANKER = REPORT_DIR / "xgbranker_results.csv"
OUTPUT_IC_STABILITY = REPORT_DIR / "rank_ic_stability.csv"
OUTPUT_BINARY = REPORT_DIR / "binary_buy_sell_results.csv"
OUTPUT_BINARY_CALIBRATION = REPORT_DIR / "binary_calibration.csv"
OUTPUT_COMBINED = REPORT_DIR / "combined_signal_analysis.csv"
OUTPUT_RECOMMENDATION = REPORT_DIR / "model_architecture_recommendation.json"

AUDIT_ESTIMATORS = 80
TOP_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01]


@dataclass(frozen=True, slots=True)
class BinarySpec:
    """Binary model specification."""

    model_name: str
    target: str


def main() -> None:
    """Run ranker, binary, and combined-signal research on validation only."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(_build_advanced_dataset())
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)
    features = _pruned_features()

    print("[ranker] XGBRanker", flush=True)
    ranker_predictions = _ranker_walk_forward(pre_final, folds, features)
    ranker_results = _ranker_results(ranker_predictions)
    ranker_results.to_csv(OUTPUT_RANKER, index=False)
    ic_stability = _rank_ic_stability(ranker_predictions, score_column="ranker_score")
    ic_stability.to_csv(OUTPUT_IC_STABILITY, index=False)

    print("[classifier] Pruned P_Outperform", flush=True)
    classifier_predictions = _classifier_walk_forward(pre_final, folds, features)

    binary_predictions = []
    binary_rows = []
    calibration_rows = []
    for spec in _binary_specs():
        print(f"[binary] {spec.model_name} {spec.target}", flush=True)
        predictions = _binary_walk_forward(pre_final, folds, features, spec)
        binary_predictions.append(predictions)
        for calibration in ("uncalibrated", "platt", "isotonic"):
            rows, cal_row = _binary_evaluation_rows(predictions, spec, calibration)
            binary_rows.extend(rows)
            calibration_rows.append(cal_row)
    pd.DataFrame(binary_rows).to_csv(OUTPUT_BINARY, index=False)
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_BINARY_CALIBRATION, index=False)

    combined = _combined_signal_analysis(
        classifier_predictions,
        ranker_predictions,
        binary_predictions,
    )
    combined.to_csv(OUTPUT_COMBINED, index=False)

    recommendation = _architecture_recommendation(
        features=features,
        ranker_results=ranker_results,
        ic_stability=ic_stability,
        binary_results=pd.DataFrame(binary_rows),
        calibration=pd.DataFrame(calibration_rows),
        combined=combined,
    )
    OUTPUT_RECOMMENDATION.write_text(
        json.dumps(_json_safe(recommendation), indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "xgbranker_results": str(OUTPUT_RANKER),
                "rank_ic_stability": str(OUTPUT_IC_STABILITY),
                "binary_buy_sell_results": str(OUTPUT_BINARY),
                "binary_calibration": str(OUTPUT_BINARY_CALIBRATION),
                "combined_signal_analysis": str(OUTPUT_COMBINED),
                "model_architecture_recommendation": str(OUTPUT_RECOMMENDATION),
                "final_test_status": "untouched",
            },
            indent=2,
        )
    )


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare binary targets and validation-only data fields."""

    output = dataset.copy()
    output["Date"] = pd.to_datetime(output["Date"])
    output["buy_target"] = (output["excess_return"] > 0.05).astype(int)
    output["sell_target"] = (output["excess_return"] < -0.05).astype(int)
    output["market_regime_label"] = output.apply(_market_regime_label, axis=1)
    return output.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["label", "excess_return", "buy_target", "sell_target"]
    )


def _market_regime_label(row: pd.Series) -> str:
    """Map numeric market regime flags to a readable label."""

    if row.get("market_regime_high_volatility", 0) == 1:
        return "High Volatility"
    if row.get("market_regime_bull_trend", 0) == 1:
        return "Bull Trend"
    if row.get("market_regime_bear_trend", 0) == 1:
        return "Bear Trend"
    return "Sideways"


def _pruned_features() -> list[str]:
    """Return frozen Pruned Advanced primary feature set."""

    return _dedupe([*feature_columns_for_variant("sector_regime"), *PRUNED_FEATURES])


def _ranker_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> pd.DataFrame:
    """Generate XGBRanker validation predictions grouped by Date."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        assert _no_target_overlap(fold["train_end"], fold["validation_start"], sorted(dataset["Date"].drop_duplicates()))
        model, medians = _fit_ranker(train, features)
        validation_x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        output["ranker_score"] = model.predict(validation_x)
        output["ranker_percentile"] = output.groupby("Date")["ranker_score"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fit_ranker(train: pd.DataFrame, features: list[str]):
    """Fit XGBRanker with Date groups and train-fold median imputation."""

    from xgboost import XGBRanker

    sorted_train = train.sort_values(["Date", "symbol"]).copy()
    medians = sorted_train[features].median(numeric_only=True).fillna(0)
    train_x = _transform_with_medians(sorted_train, features, medians)
    group = sorted_train.groupby("Date", sort=False).size().to_numpy()
    model = XGBRanker(
        n_estimators=AUDIT_ESTIMATORS,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="rank:pairwise",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(train_x, sorted_train["excess_return"], group=group, verbose=False)
    return model, medians


def _classifier_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> pd.DataFrame:
    """Generate primary Pruned Advanced P_Outperform validation predictions."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, features, target="label")
        x = _transform_with_medians(validation, features, medians)
        probabilities = model.predict_proba(x)
        output = _base_prediction_frame(validation, fold["fold"])
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["p_outperform_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    spec: BinarySpec,
) -> pd.DataFrame:
    """Generate binary BUY/SELL validation predictions with fold-local calibration."""

    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit_frame, calibration_frame = _fit_calibration_split(train)
        model, medians = _fit_classifier(
            fit_frame,
            features,
            target=spec.target,
            model_name=spec.model_name,
            binary=True,
        )
        cal_x = _transform_with_medians(calibration_frame, features, medians)
        val_x = _transform_with_medians(validation, features, medians)
        cal_prob = _positive_probability(model, cal_x)
        val_prob = _positive_probability(model, val_x)
        platt = _fit_platt(cal_prob, calibration_frame[spec.target])
        isotonic = _fit_isotonic(cal_prob, calibration_frame[spec.target])
        output = _base_prediction_frame(validation, fold["fold"])
        output["Model"] = spec.model_name
        output["Target"] = spec.target
        output["actual"] = validation[spec.target].astype(int).to_numpy()
        output["p_uncalibrated"] = val_prob
        output["p_platt"] = platt.predict_proba(val_prob.reshape(-1, 1))[:, 1]
        output["p_isotonic"] = isotonic.predict(val_prob)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    model_name: str = "xgboost",
    binary: bool = False,
):
    """Fit classifier with train-fold median imputation plus missing indicators."""

    medians = train[features].median(numeric_only=True).fillna(0)
    x = _transform_with_medians(train, features, medians)
    model = _estimator(model_name, binary=binary)
    y = train[target].astype(int) if binary else train[target].map(LABEL_TO_CLASS)
    model.fit(x, y)
    return model, medians


def _estimator(model_name: str, *, binary: bool):
    """Return requested model family."""

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=AUDIT_ESTIMATORS,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic" if binary else "multi:softprob",
            eval_metric="logloss" if binary else "mlogloss",
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=AUDIT_ESTIMATORS,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary" if binary else "multiclass",
            random_state=42,
            verbose=-1,
        )
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=80,
            max_depth=9,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _transform_with_medians(frame: pd.DataFrame, features: list[str], medians: pd.Series) -> pd.DataFrame:
    """Median-impute and append missing-value indicators."""

    x = frame[features].copy()
    indicators = x.isna().astype(float).add_suffix("__missing")
    filled = x.fillna(medians)
    return pd.concat([filled, indicators], axis=1)


def _fit_calibration_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use latest 20% of past training dates as validation-only calibration data."""

    dates = pd.Series(sorted(train["Date"].drop_duplicates()))
    split = dates.iloc[max(1, int(len(dates) * 0.8))]
    fit = train[train["Date"] < split]
    calibration = train[train["Date"] >= split]
    if fit.empty or calibration.empty:
        return train, train
    return fit, calibration


def _fit_platt(probabilities, actual):
    """Fit sigmoid/Platt calibration."""

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=42)
    model.fit(probabilities.reshape(-1, 1), actual.astype(int))
    return model


def _fit_isotonic(probabilities, actual):
    """Fit isotonic calibration."""

    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probabilities, actual.astype(int))
    return model


def _ranker_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate ranker IC and top-ranked realized performance."""

    rows = [_ic_summary_row(predictions, "all", "all", score_column="ranker_score")]
    for year, group in predictions.groupby(predictions["Date"].dt.year):
        rows.append(_ic_summary_row(group, "year", year, score_column="ranker_score"))
    for regime, group in predictions.groupby("market_regime_label"):
        rows.append(_ic_summary_row(group, "market_regime", regime, score_column="ranker_score"))
    for fraction in TOP_FRACTIONS:
        selected = _top_fraction(predictions, "ranker_score", fraction)
        rows.append(
            {
                "ReportType": "top_percentile",
                "Group": "all",
                "Value": f"Top {int(fraction * 100)}%",
                "Count": int(len(selected)),
                "AverageExcessReturn": float(selected["excess_return"].mean()),
                "MedianExcessReturn": float(selected["excess_return"].median()),
                "WinRate": float((selected["excess_return"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _rank_ic_stability(predictions: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    """Daily IC rows plus year/regime labels."""

    rows = []
    for date, group in predictions.groupby("Date"):
        if len(group) < 5 or group[score_column].nunique() < 2:
            continue
        ic = group[score_column].corr(group["excess_return"], method="spearman")
        if pd.notna(ic):
            rows.append(
                {
                    "Date": date,
                    "Year": date.year,
                    "MarketRegime": group["market_regime_label"].mode().iloc[0],
                    "SpearmanIC": float(ic),
                    "PositiveIC": bool(ic > 0),
                }
            )
    return pd.DataFrame(rows)


def _binary_evaluation_rows(
    predictions: pd.DataFrame,
    spec: BinarySpec,
    calibration: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate binary model for one calibration column."""

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    probability_column = f"p_{calibration}"
    frame = predictions.copy()
    frame["probability"] = frame[probability_column].clip(0, 1)
    actual = frame["actual"].astype(int)
    calibration_true, calibration_pred = calibration_curve(
        actual, frame["probability"], n_bins=10, strategy="quantile"
    )
    base = {
        "Model": spec.model_name,
        "Target": spec.target,
        "Calibration": calibration,
        "Rows": len(frame),
        "LogLoss": float(log_loss(actual, frame["probability"], labels=[0, 1])),
        "Brier": float(brier_score_loss(actual, frame["probability"])),
        "PRAUC": float(average_precision_score(actual, frame["probability"])),
        "CalibrationMAE": float(abs(calibration_true - calibration_pred).mean()),
    }
    rows = []
    for fraction in TOP_FRACTIONS:
        selected = _top_fraction(frame, "probability", fraction)
        rows.append(
            {
                **base,
                "TopPercentile": f"Top {int(fraction * 100)}%",
                "PrecisionAtPercentile": float(selected["actual"].mean()),
                "AverageDirectionalExcessReturn": _directional_excess(selected, spec.target),
                "AverageExcessReturn": float(selected["excess_return"].mean()),
            }
        )
    return rows, base


def _combined_signal_analysis(
    classifier: pd.DataFrame,
    ranker: pd.DataFrame,
    binary_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Test combined validation signals without arbitrary trading thresholds."""

    key = ["Date", "symbol", "fold"]
    base = classifier[key + ["p_outperform", "p_outperform_percentile", "excess_return", "future_stock_return", "future_nifty_return", "market_regime_label"]].copy()
    ranker_part = ranker[key + ["ranker_percentile", "ranker_score"]]
    combined = base.merge(ranker_part, on=key, how="inner")
    buy = _best_binary_frame(binary_frames, target="buy_target")
    sell = _best_binary_frame(binary_frames, target="sell_target")
    combined = combined.merge(buy[key + ["binary_buy_percentile"]], on=key, how="inner")
    combined = combined.merge(sell[key + ["binary_sell_percentile"]], on=key, how="inner")
    combined["inverse_binary_sell_percentile"] = 1 - combined["binary_sell_percentile"]
    combined["combined_score"] = (
        combined["p_outperform_percentile"]
        + combined["ranker_percentile"]
        + combined["binary_buy_percentile"]
        + combined["inverse_binary_sell_percentile"]
    ) / 4
    score_columns = [
        "p_outperform_percentile",
        "ranker_percentile",
        "binary_buy_percentile",
        "inverse_binary_sell_percentile",
        "combined_score",
    ]
    rows = []
    for score in score_columns:
        summary = _ic_summary_row(combined, "signal", score, score_column=score)
        summary["AverageTop10ExcessReturn"] = float(
            _top_fraction(combined, score, 0.10)["excess_return"].mean()
        )
        summary["AverageTop5ExcessReturn"] = float(
            _top_fraction(combined, score, 0.05)["excess_return"].mean()
        )
        summary["AverageTop1ExcessReturn"] = float(
            _top_fraction(combined, score, 0.01)["excess_return"].mean()
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def _best_binary_frame(binary_frames: list[pd.DataFrame], *, target: str) -> pd.DataFrame:
    """Pick XGBoost Platt frame for stable combined-signal research."""

    for frame in binary_frames:
        if frame["Target"].iloc[0] == target and frame["Model"].iloc[0] == "xgboost":
            output = frame.copy()
            column = "binary_buy_percentile" if target == "buy_target" else "binary_sell_percentile"
            output[column] = output.groupby("Date")["p_platt"].rank(pct=True)
            return output
    raise ValueError(f"Missing binary frame for target: {target}")


def _architecture_recommendation(
    *,
    features: list[str],
    ranker_results: pd.DataFrame,
    ic_stability: pd.DataFrame,
    binary_results: pd.DataFrame,
    calibration: pd.DataFrame,
    combined: pd.DataFrame,
) -> dict[str, Any]:
    """Build architecture recommendation without final-test usage."""

    combined_sorted = combined.sort_values("MeanIC", ascending=False)
    best_signal = combined_sorted.iloc[0].to_dict()
    best_buy = (
        binary_results[binary_results["Target"] == "buy_target"]
        .sort_values(["PRAUC", "PrecisionAtPercentile"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )
    best_sell = (
        binary_results[binary_results["Target"] == "sell_target"]
        .sort_values(["PRAUC", "PrecisionAtPercentile"], ascending=[False, False])
        .iloc[0]
        .to_dict()
    )
    ranker_ic = ranker_results[
        (ranker_results["ReportType"] == "ic_summary")
        & (ranker_results["Group"] == "all")
    ].iloc[0].to_dict()
    return {
        "selection_basis": "Walk-forward validation only; final test untouched.",
        "primary_classifier": "Pruned Advanced",
        "frozen_primary_features": features,
        "recommended_architecture": {
            "primary_classifier": "Keep Pruned Advanced as primary classifier.",
            "ranker": "Use XGBRanker as complementary ranking research signal if monitored IC remains positive.",
            "binary_models": "Use calibrated binary BUY/SELL models as auxiliary probability diagnostics, not trade rules.",
            "combined_signal": best_signal,
        },
        "ranker_validation_summary": ranker_ic,
        "best_binary_buy_validation": best_buy,
        "best_binary_sell_validation": best_sell,
        "calibration_summary": calibration.to_dict(orient="records"),
        "final_test_status": "untouched_until_architecture_selected",
        "notes": [
            "No BUY / ACCUMULATE / SELL thresholds were implemented.",
            "No new technical indicators or cross-sectional ranks were added to the primary classifier.",
            "All folds use a 20-trading-day embargo before validation.",
        ],
    }


def _ic_summary_row(
    predictions: pd.DataFrame,
    group: str,
    value: Any,
    *,
    score_column: str,
) -> dict[str, Any]:
    """Create IC summary row."""

    daily = _daily_ic(predictions, score_column)
    return {
        "ReportType": "ic_summary",
        "Group": group,
        "Value": value,
        "Dates": int(len(daily)),
        "MeanIC": float(daily.mean()) if not daily.empty else None,
        "MedianIC": float(daily.median()) if not daily.empty else None,
        "ICStd": float(daily.std()) if not daily.empty else None,
        "ICInformationRatio": float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() else None,
        "PositiveICDatePct": float((daily > 0).mean()) if not daily.empty else None,
    }


def _daily_ic(predictions: pd.DataFrame, score_column: str) -> pd.Series:
    """Calculate daily Spearman IC."""

    values = []
    for _, group in predictions.groupby("Date"):
        if len(group) < 5 or group[score_column].nunique() < 2:
            continue
        ic = group[score_column].corr(group["excess_return"], method="spearman")
        if pd.notna(ic):
            values.append(float(ic))
    return pd.Series(values, dtype=float)


def _top_fraction(frame: pd.DataFrame, score_column: str, fraction: float) -> pd.DataFrame:
    """Return top score rows."""

    count = max(1, int(len(frame) * fraction))
    return frame.sort_values(score_column, ascending=False).head(count)


def _directional_excess(frame: pd.DataFrame, target: str) -> float:
    """Return excess return in the direction implied by target."""

    value = frame["excess_return"].mean()
    return float(-value if target == "sell_target" else value)


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and validation rows for an embargoed fold."""

    train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
    validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
    return train, validation


def _base_prediction_frame(frame: pd.DataFrame, fold: Any) -> pd.DataFrame:
    """Return common validation prediction frame columns."""

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
            "sell_target",
            "market_regime_label",
        ]
    ].copy()
    output["fold"] = fold
    return output


def _positive_probability(model, x: pd.DataFrame):
    """Return positive-class probability with missing-class safety."""

    probabilities = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return pd.Series(0.0, index=x.index).to_numpy()
    return probabilities[:, classes.index(1)]


def _binary_specs() -> list[BinarySpec]:
    """Return binary model specs."""

    return [
        BinarySpec("xgboost", "buy_target"),
        BinarySpec("lightgbm", "buy_target"),
        BinarySpec("random_forest", "buy_target"),
        BinarySpec("xgboost", "sell_target"),
        BinarySpec("lightgbm", "sell_target"),
        BinarySpec("random_forest", "sell_target"),
    ]


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
