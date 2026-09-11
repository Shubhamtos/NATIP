"""Focused V2 model-family comparison for the 26-feature BUY candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import _fit_classifier, _transform_with_medians
from run_stock_specific_feature_variant import (
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _ic_summary,
    _prepare_dataset,
    _safe_float,
)
from run_v2_55_binary_buy_research import _binary_buy_77_oof
from run_v2_backward_forward_selection import (
    BINARY_TOP,
    CLASSIFIER_TOP,
    CLEAN_DEMERGER_DATASET,
    OUTPUT_SELECTED,
    _v2_55_features,
)

MODEL_NAMES = ("xgboost", "lightgbm", "random_forest")
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
}
TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)
AGREEMENT_TOPS = (0.10, 0.05, 0.02, 0.01)
ENSEMBLE_PAIR_WEIGHTS = ((0.5, 0.5), (0.6, 0.4), (0.7, 0.3))

OUTPUT_DIR = REPORT_DIR / "v2_model_family_comparison"
OUTPUT_STANDALONE = OUTPUT_DIR / "v2_three_model_classifier_comparison.csv"
OUTPUT_HYBRID = OUTPUT_DIR / "v2_three_model_hybrid_comparison.csv"
OUTPUT_DIVERSITY = OUTPUT_DIR / "v2_three_model_correlations.csv"
OUTPUT_AGREEMENT = OUTPUT_DIR / "v2_three_model_agreement.csv"
OUTPUT_ENSEMBLES = OUTPUT_DIR / "v2_three_model_ensemble.csv"
OUTPUT_DECISION = OUTPUT_DIR / "v2_three_model_recommendation.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "v2_three_model_report.txt"


@dataclass(frozen=True, slots=True)
class PrimaryModel:
    """Primary 3-class classifier configuration."""

    name: str
    display_name: str
    features: list[str]


def main() -> None:
    """Run focused validation-only model-family comparison."""

    ensure_probability_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into validation research.")

    features_77 = _frozen_features()
    features_55 = _v2_55_features(features_77)
    features_26 = _selected_26_features()
    if len(features_26) != 26:
        raise AssertionError(
            f"Expected selected compact candidate to have 26 features, got {len(features_26)}"
        )

    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    binary = _binary_buy_77_oof(
        common,
        folds,
        FeatureSet("binary_77", "Unchanged 77-feature Binary XGBoost BUY", features_77),
    )
    base_rate = float(common["buy_target"].mean())

    primary = [
        PrimaryModel(name, f"26-feature {MODEL_LABELS[name]}", features_26) for name in MODEL_NAMES
    ]
    benchmark = PrimaryModel("v2_b_55_xgboost", "55-feature V2-B XGBoost benchmark", features_55)

    prediction_frames = []
    for model in [*primary, benchmark]:
        print(f"[primary-oof] {model.display_name}", flush=True)
        model_name = "xgboost" if model.name == "v2_b_55_xgboost" else model.name
        prediction_frames.append(_primary_oof(common, folds, model, model_name=model_name))
    predictions = pd.concat(prediction_frames, ignore_index=True)

    standalone = _standalone_report(predictions[predictions["FeatureCount"].eq(26)])
    standalone.to_csv(OUTPUT_STANDALONE, index=False)
    hybrid = _hybrid_report(predictions, binary, base_rate)
    hybrid.to_csv(OUTPUT_HYBRID, index=False)
    diversity = _diversity_report(predictions[predictions["FeatureCount"].eq(26)])
    diversity.to_csv(OUTPUT_DIVERSITY, index=False)
    agreement = _agreement_report(
        predictions[predictions["FeatureCount"].eq(26)], binary, base_rate
    )
    agreement.to_csv(OUTPUT_AGREEMENT, index=False)
    ensembles = _ensemble_report(predictions[predictions["FeatureCount"].eq(26)], binary, base_rate)
    ensembles.to_csv(OUTPUT_ENSEMBLES, index=False)
    decision = _decision_payload(
        standalone=standalone,
        hybrid=hybrid,
        agreement=agreement,
        ensembles=ensembles,
        features_26=features_26,
    )
    OUTPUT_DECISION.write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")
    OUTPUT_SUMMARY.write_text(
        _summary(standalone, hybrid, diversity, agreement, ensembles, decision),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "standalone": str(OUTPUT_STANDALONE),
                "hybrid_buy": str(OUTPUT_HYBRID),
                "diversity": str(OUTPUT_DIVERSITY),
                "agreement": str(OUTPUT_AGREEMENT),
                "ensembles": str(OUTPUT_ENSEMBLES),
                "decision": str(OUTPUT_DECISION),
                "summary": str(OUTPUT_SUMMARY),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "frozen_buy_thresholds_modified": False,
            },
            indent=2,
        )
    )


def _selected_26_features() -> list[str]:
    payload = json.loads(OUTPUT_SELECTED.read_text(encoding="utf-8"))
    features = payload["selected_features"]
    if payload.get("best_variant", {}).get("Variant") != "progressive_step_1_sma_200":
        raise AssertionError(
            "Selected V2 compact candidate is not the expected lean core + sma_200."
        )
    return features


def _primary_oof(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    model: PrimaryModel,
    *,
    model_name: str,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        estimator, medians = _fit_classifier(
            train,
            model.features,
            target="label",
            model_name=model_name,
            binary=False,
        )
        probabilities = estimator.predict_proba(
            _transform_with_medians(validation, model.features, medians)
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
                "label",
            ]
        ].copy()
        output["fold"] = fold["fold"]
        output["Model"] = model.name
        output["ModelName"] = model.display_name
        output["FeatureCount"] = len(model.features)
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["p_outperform_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _standalone_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in predictions.groupby("Model"):
        rows.append(_classifier_row(frame, model, "overall", "all"))
        for fraction in TOP_FRACTIONS:
            rows.append(_tail_row(frame, model, "overall", "all", fraction, score="p_outperform"))
        for fold, group in frame.groupby("fold"):
            rows.append(_tail_row(group, model, "fold", fold, CLASSIFIER_TOP, score="p_outperform"))
        for year, group in frame.groupby(frame["Date"].dt.year):
            rows.append(
                _tail_row(group, model, "year", int(year), CLASSIFIER_TOP, score="p_outperform")
            )
    return pd.DataFrame(rows)


def _classifier_row(
    frame: pd.DataFrame, model: str, scope: str, scope_value: Any
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    y_true = frame["label"].map(LABEL_TO_CLASS)
    probabilities = _probability_matrix(frame)
    actual_buy = frame["buy_target"].astype(int)
    ic = _ic_summary(frame, "p_outperform")
    backtest = backtest_top_n_portfolio(frame, top_n=5)
    return {
        "RowType": "classifier",
        "Model": model,
        "ModelName": frame["ModelName"].iloc[0],
        "FeatureCount": int(frame["FeatureCount"].iloc[0]),
        "Scope": scope,
        "ScopeValue": scope_value,
        "Rows": int(len(frame)),
        "PRAUC": float(average_precision_score(actual_buy, frame["p_outperform"])),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(brier_score_loss(actual_buy, frame["p_outperform"])),
        "MeanIC": ic["MeanIC"],
        "MedianIC": ic["MedianIC"],
        "Sharpe": backtest.get("sharpe"),
        "MaxDrawdown": backtest.get("max_drawdown"),
    }


def _tail_row(
    frame: pd.DataFrame,
    model: str,
    scope: str,
    scope_value: Any,
    fraction: float,
    *,
    score: str,
) -> dict[str, Any]:
    selected = _top_by_date(frame, score, fraction)
    return {
        "RowType": "tail",
        "Model": model,
        "ModelName": frame["ModelName"].iloc[0] if not frame.empty else model,
        "FeatureCount": int(frame["FeatureCount"].iloc[0]) if not frame.empty else None,
        "Scope": scope,
        "ScopeValue": scope_value,
        "TopFraction": fraction,
        "TopLabel": _top_label(fraction),
        "Count": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
    }


def _hybrid_report(
    predictions: pd.DataFrame, binary: pd.DataFrame, base_rate: float
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    rows = []
    for model, frame in predictions.groupby("Model"):
        merged = frame.merge(binary[key + ["binary_buy_percentile"]], on=key, how="inner")
        selected = _select_hybrid(merged)
        rows.append(
            _signal_row(selected, model, "overall", "all", base_rate, CLASSIFIER_TOP, BINARY_TOP)
        )
        for fold, group in merged.groupby("fold"):
            rows.append(
                _signal_row(
                    _select_hybrid(group),
                    model,
                    "fold",
                    fold,
                    base_rate,
                    CLASSIFIER_TOP,
                    BINARY_TOP,
                )
            )
        for year, group in merged.groupby(merged["Date"].dt.year):
            rows.append(
                _signal_row(
                    _select_hybrid(group),
                    model,
                    "year",
                    int(year),
                    base_rate,
                    CLASSIFIER_TOP,
                    BINARY_TOP,
                )
            )
    return pd.DataFrame(rows)


def _select_hybrid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["p_outperform_percentile"] >= 1 - CLASSIFIER_TOP)
        & (frame["binary_buy_percentile"] >= 1 - BINARY_TOP)
    ].copy()


def _signal_row(
    frame: pd.DataFrame,
    model: str,
    scope: str,
    scope_value: Any,
    base_rate: float,
    classifier_top: float,
    binary_top: float,
    *,
    rule: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    precision = _safe_float(frame["buy_target"].mean()) if not frame.empty else None
    row = {
        "Model": model,
        "Rule": rule or f"classifier_top_{_pct(classifier_top)}__binary_top_{_pct(binary_top)}",
        "ClassifierTop": classifier_top,
        "BinaryTop": binary_top,
        "Scope": scope,
        "ScopeValue": scope_value,
        "SignalCount": int(len(frame)),
        "Precision": precision,
        "BaseRate": base_rate,
        "PrecisionLift": _safe_float(precision / base_rate) if precision is not None else None,
        "AverageFutureStockReturn": _safe_float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": _safe_float(frame["future_stock_return"].median()),
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }
    if extra:
        row.update(extra)
    return row


def _diversity_report(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in combinations(MODEL_NAMES, 2):
        left_frame = predictions[predictions["Model"].eq(left)]
        right_frame = predictions[predictions["Model"].eq(right)]
        merged = left_frame[
            ["Date", "symbol", "fold", "p_outperform", "p_outperform_percentile"]
        ].merge(
            right_frame[["Date", "symbol", "fold", "p_outperform", "p_outperform_percentile"]],
            on=["Date", "symbol", "fold"],
            suffixes=("_left", "_right"),
        )
        left_top = _top_keys(left_frame, "p_outperform", CLASSIFIER_TOP)
        right_top = _top_keys(right_frame, "p_outperform", CLASSIFIER_TOP)
        intersection = left_top & right_top
        union = left_top | right_top
        rows.append(
            {
                "LeftModel": left,
                "RightModel": right,
                "Rows": int(len(merged)),
                "PearsonProbabilityCorrelation": _safe_float(
                    merged["p_outperform_left"].corr(merged["p_outperform_right"], method="pearson")
                ),
                "SpearmanProbabilityCorrelation": _safe_float(
                    merged["p_outperform_left"].corr(
                        merged["p_outperform_right"], method="spearman"
                    )
                ),
                "PercentileCorrelation": _safe_float(
                    merged["p_outperform_percentile_left"].corr(
                        merged["p_outperform_percentile_right"], method="spearman"
                    )
                ),
                "Top1LeftCount": len(left_top),
                "Top1RightCount": len(right_top),
                "Top1Overlap": len(intersection),
                "Top1Jaccard": _safe_float(len(intersection) / len(union)) if union else None,
            }
        )
    return pd.DataFrame(rows)


def _agreement_report(
    predictions: pd.DataFrame, binary: pd.DataFrame, base_rate: float
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    wide = _wide_predictions(predictions)
    merged = wide.merge(binary[key + ["binary_buy_percentile"]], on=key, how="inner")
    rows = []
    for left, right in combinations(MODEL_NAMES, 2):
        for left_top, right_top in product(AGREEMENT_TOPS, AGREEMENT_TOPS):
            selected = merged[
                (merged[f"{left}_percentile"] >= 1 - left_top)
                & (merged[f"{right}_percentile"] >= 1 - right_top)
                & (merged["binary_buy_percentile"] >= 1 - BINARY_TOP)
            ].copy()
            rows.append(
                _signal_row(
                    selected,
                    f"{left}+{right}+binary_xgb",
                    "overall",
                    "all",
                    base_rate,
                    left_top,
                    BINARY_TOP,
                    rule=f"{left}_top_{_pct(left_top)}__{right}_top_{_pct(right_top)}__binary_top_{_pct(BINARY_TOP)}",
                    extra={"SecondaryTop": right_top},
                )
            )
    for top in AGREEMENT_TOPS:
        selected = merged[
            (merged["xgboost_percentile"] >= 1 - top)
            & (merged["lightgbm_percentile"] >= 1 - top)
            & (merged["random_forest_percentile"] >= 1 - top)
            & (merged["binary_buy_percentile"] >= 1 - BINARY_TOP)
        ].copy()
        rows.append(
            _signal_row(
                selected,
                "xgboost+lightgbm+random_forest+binary_xgb",
                "overall",
                "all",
                base_rate,
                top,
                BINARY_TOP,
                rule=f"all_three_top_{_pct(top)}__binary_top_{_pct(BINARY_TOP)}",
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["Precision", "SignalCount", "MedianExcessReturn"], ascending=False
    )


def _ensemble_report(
    predictions: pd.DataFrame, binary: pd.DataFrame, base_rate: float
) -> pd.DataFrame:
    wide = _wide_predictions(predictions)
    key = ["Date", "symbol", "fold"]
    rows = []
    ensemble_frames = []
    for left, right in combinations(MODEL_NAMES, 2):
        for left_weight, right_weight in ENSEMBLE_PAIR_WEIGHTS:
            frame = _ensemble_frame(
                wide,
                f"ensemble_{left}_{int(left_weight * 100)}_{right}_{int(right_weight * 100)}",
                {left: left_weight, right: right_weight},
            )
            ensemble_frames.append(frame)
    ensemble_frames.append(
        _ensemble_frame(
            wide,
            "ensemble_xgboost_lightgbm_random_forest_equal",
            {name: 1 / 3 for name in MODEL_NAMES},
        )
    )
    for frame in ensemble_frames:
        rows.append(_classifier_row(frame, frame["Model"].iloc[0], "overall", "all"))
        merged = frame.merge(binary[key + ["binary_buy_percentile"]], on=key, how="inner")
        rows.append(
            _signal_row(
                _select_hybrid(merged),
                frame["Model"].iloc[0],
                "overall",
                "all",
                base_rate,
                CLASSIFIER_TOP,
                BINARY_TOP,
                extra={"RowType": "hybrid_buy"},
            )
        )
    return pd.DataFrame(rows)


def _ensemble_frame(wide: pd.DataFrame, name: str, weights: dict[str, float]) -> pd.DataFrame:
    frame = wide.copy()
    frame["p_outperform"] = sum(
        frame[f"{model}_p_outperform"] * weight for model, weight in weights.items()
    )
    frame["p_neutral"] = sum(
        frame[f"{model}_p_neutral"] * weight for model, weight in weights.items()
    )
    frame["p_underperform"] = sum(
        frame[f"{model}_p_underperform"] * weight for model, weight in weights.items()
    )
    total = frame[["p_underperform", "p_neutral", "p_outperform"]].sum(axis=1).replace(0, 1)
    frame["p_underperform"] = frame["p_underperform"] / total
    frame["p_neutral"] = frame["p_neutral"] / total
    frame["p_outperform"] = frame["p_outperform"] / total
    frame["p_outperform_percentile"] = frame.groupby("Date")["p_outperform"].rank(pct=True)
    frame["Model"] = name
    frame["ModelName"] = name
    frame["FeatureCount"] = 26
    return frame[
        [
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
            "label",
            "Model",
            "ModelName",
            "FeatureCount",
            "p_underperform",
            "p_neutral",
            "p_outperform",
            "p_outperform_percentile",
        ]
    ].copy()


def _wide_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
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
        "label",
    ]
    wide = predictions[predictions["Model"].eq(MODEL_NAMES[0])][base_cols].copy()
    for model in MODEL_NAMES:
        part = predictions[predictions["Model"].eq(model)][
            [
                "Date",
                "symbol",
                "fold",
                "p_underperform",
                "p_neutral",
                "p_outperform",
                "p_outperform_percentile",
            ]
        ].rename(
            columns={
                "p_underperform": f"{model}_p_underperform",
                "p_neutral": f"{model}_p_neutral",
                "p_outperform": f"{model}_p_outperform",
                "p_outperform_percentile": f"{model}_percentile",
            }
        )
        wide = wide.merge(part, on=["Date", "symbol", "fold"], how="inner")
    return wide


def _top_by_date(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    percentile = frame.groupby("Date")[score].rank(pct=True)
    return frame[percentile >= 1 - fraction].copy()


def _top_keys(frame: pd.DataFrame, score: str, fraction: float) -> set[tuple[pd.Timestamp, str]]:
    selected = _top_by_date(frame, score, fraction)
    return set(zip(selected["Date"], selected["symbol"], strict=False))


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _probability_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    probabilities = frame[["p_underperform", "p_neutral", "p_outperform"]].copy()
    total = probabilities.sum(axis=1).replace(0, 1)
    return probabilities.div(total, axis=0)


def _decision_payload(
    *,
    standalone: pd.DataFrame,
    hybrid: pd.DataFrame,
    agreement: pd.DataFrame,
    ensembles: pd.DataFrame,
    features_26: list[str],
) -> dict[str, Any]:
    standalone_top1 = _tail_overall(standalone, 0.01).sort_values("Precision", ascending=False)
    hybrid_overall = hybrid[hybrid["Scope"].eq("overall")].sort_values(
        ["Precision", "MedianExcessReturn", "AverageExcessReturn", "SignalCount"],
        ascending=False,
    )
    ensemble_hybrid = ensembles[ensembles.get("RowType", "").eq("hybrid_buy")]
    best_ensemble = (
        ensemble_hybrid.sort_values(
            ["Precision", "MedianExcessReturn", "AverageExcessReturn", "SignalCount"],
            ascending=False,
        ).head(1)
        if not ensemble_hybrid.empty
        else pd.DataFrame()
    )
    return {
        "selection_basis": "Validation/out-of-fold only. Final test not used.",
        "v1_production_modified": False,
        "frozen_buy_thresholds_modified": False,
        "primary_feature_set_count": len(features_26),
        "primary_feature_set": features_26,
        "highest_standalone_top1_precision": standalone_top1.head(1).to_dict(orient="records"),
        "best_hybrid_buy": hybrid_overall.head(1).to_dict(orient="records"),
        "best_three_model_agreement": agreement.head(1).to_dict(orient="records"),
        "best_simple_ensemble_hybrid": best_ensemble.to_dict(orient="records"),
        "benchmark_rows": hybrid[
            hybrid["Model"].eq("v2_b_55_xgboost") & hybrid["Scope"].eq("overall")
        ].to_dict(orient="records"),
    }


def _tail_overall(frame: pd.DataFrame, fraction: float) -> pd.DataFrame:
    return frame[
        (frame["RowType"].eq("tail"))
        & (frame["Scope"].eq("overall"))
        & (frame["TopFraction"].eq(fraction))
    ].copy()


def _summary(
    standalone: pd.DataFrame,
    hybrid: pd.DataFrame,
    diversity: pd.DataFrame,
    agreement: pd.DataFrame,
    ensembles: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "NATIP V2 Focused Model-Family Comparison",
            "==========================================",
            "",
            "Models tested: XGBoost, LightGBM, Random Forest only.",
            "Final test usage: NOT USED.",
            "V1 production modified: NO.",
            "Frozen BUY thresholds modified: NO.",
            "",
            "Standalone Top1 precision:",
            _tail_overall(standalone, 0.01).to_string(index=False),
            "",
            "Hybrid BUY overall:",
            hybrid[hybrid["Scope"].eq("overall")].to_string(index=False),
            "",
            "Model diversity:",
            diversity.to_string(index=False),
            "",
            "Top agreement rules:",
            agreement.head(20).to_string(index=False),
            "",
            "Simple ensemble rows:",
            ensembles.to_string(index=False),
            "",
            "Decision payload:",
            json.dumps(_json_safe(decision), indent=2),
        ]
    )


def _pct(value: float) -> str:
    return f"{value * 100:g}%"


def _top_label(fraction: float) -> str:
    return f"Top{fraction * 100:g}%"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
