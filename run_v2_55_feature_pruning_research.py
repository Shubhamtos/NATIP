"""Validation-only feature-pruning research for the V2-B 55-feature classifier."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.probability.backtest import backtest_top_n_portfolio
from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import _fit_classifier, _transform_with_medians
from run_stock_specific_feature_variant import (
    MARKET_BREADTH_FEATURES,
    MARKET_REGIME_FEATURES,
    NIFTY_FEATURES,
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _ic_summary,
    _prepare_dataset,
    _safe_float,
)
from run_v2_55_binary_buy_research import _binary_buy_77_oof

CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"

OUTPUT_IMPORTANCE = REPORT_DIR / "v2_55_feature_importance_detailed.csv"
OUTPUT_ABLATION = REPORT_DIR / "v2_55_feature_ablation_results.csv"
OUTPUT_COMPARISON = REPORT_DIR / "v2_55_pruned_model_comparison.csv"
OUTPUT_BUY_AGREEMENT = REPORT_DIR / "v2_55_pruned_buy_agreement_comparison.csv"
OUTPUT_LABELS = REPORT_DIR / "v2_55_feature_pruning_labels.csv"
OUTPUT_SUMMARY = REPORT_DIR / "v2_55_feature_pruning_summary.txt"

FIRST_PRUNE_REMOVE = [
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "price_to_sma_20",
    "price_to_sma_100",
    "price_to_sma_200",
    "rsi_14",
    "bb_width_20",
    "bb_percent_b_20",
    "volume_to_avg20",
    "plus_di_14",
]

WEAK_ABLATION_FEATURES = [
    "sma_20",
    "sma_200",
    "ema_20",
    "ret_252d",
    "volatility_20d",
    "distance_52w_high",
    "stock_volatility_relative_sector_20d",
    "stock_volatility_relative_sector_60d",
    "stock_vs_sector_ret_252d",
    "sector_strength_vs_nifty_60d",
]

RAW_PRICE_SCALE_REMOVE = [
    "sma_20",
    "sma_50",
    "sma_100",
    "sma_200",
    "ema_20",
    "ema_50",
    "macd",
    "macd_signal",
    "atr_14",
]

TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)
PERMUTATION_SAMPLE_ROWS = 6000
SHAP_SAMPLE_ROWS = 1500
RANDOM_SEED = 20260811


@dataclass(frozen=True, slots=True)
class Variant:
    """Feature-pruning model variant."""

    name: str
    display_name: str
    features: list[str]
    removed_features: list[str]
    variant_type: str


def main() -> None:
    """Run validation-only V2-B feature-pruning research."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_77 = _frozen_features()
    features_55 = _v2_55_features(features_77)
    if len(features_55) != 55:
        raise AssertionError(f"Expected 55 V2-B features, got {len(features_55)}.")
    _assert_requested_features(features_55)
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    print("[importance] V2-B 55-feature classifier", flush=True)
    importance = _importance_audit(common, folds, features_55)
    labels = _feature_labels(importance)
    importance = importance.merge(labels[["Feature", "DecisionLabel"]], on="Feature", how="left")
    importance.to_csv(OUTPUT_IMPORTANCE, index=False)
    labels.to_csv(OUTPUT_LABELS, index=False)

    variants = _variants(features_55, labels)
    print(f"[comparison] {len(variants)} feature variants", flush=True)
    predictions = []
    for variant in variants:
        print(f"[classifier] {variant.name} ({len(variant.features)} features)", flush=True)
        predictions.append(_classifier_oof(common, folds, variant))
    predictions = pd.concat(predictions, ignore_index=True)
    comparison = _comparison(predictions, variants)
    print("[binary-buy] unchanged 77-feature confirmation", flush=True)
    binary_buy = _binary_buy_77_oof(
        common,
        folds,
        FeatureSet("v1_77_binary_confirmation", "77-feature Binary BUY confirmation", features_77),
    )
    buy_agreement = _buy_agreement_comparison(predictions, binary_buy, variants)
    ablation = comparison[comparison["VariantType"].ne("model_comparison")].copy()
    model_comparison = comparison[comparison["VariantType"].eq("model_comparison")].copy()
    ablation.to_csv(OUTPUT_ABLATION, index=False)
    model_comparison.to_csv(OUTPUT_COMPARISON, index=False)
    buy_agreement.to_csv(OUTPUT_BUY_AGREEMENT, index=False)
    OUTPUT_SUMMARY.write_text(
        _summary(importance, labels, model_comparison, ablation, buy_agreement, variants),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_55_feature_importance_detailed": str(OUTPUT_IMPORTANCE),
                "v2_55_feature_pruning_labels": str(OUTPUT_LABELS),
                "v2_55_feature_ablation_results": str(OUTPUT_ABLATION),
                "v2_55_pruned_model_comparison": str(OUTPUT_COMPARISON),
                "v2_55_pruned_buy_agreement_comparison": str(OUTPUT_BUY_AGREEMENT),
                "v2_55_feature_pruning_summary": str(OUTPUT_SUMMARY),
                "frozen_v1_modified": False,
                "frozen_v2_thresholds_modified": False,
                "final_test_status": "not_used",
            },
            indent=2,
        )
    )


def _v2_55_features(features_77: list[str]) -> list[str]:
    remove = NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES
    return [feature for feature in features_77 if feature not in remove]


def _assert_requested_features(features_55: list[str]) -> None:
    requested = set(FIRST_PRUNE_REMOVE) | set(WEAK_ABLATION_FEATURES) | set(RAW_PRICE_SCALE_REMOVE)
    missing = sorted(requested.difference(features_55))
    if missing:
        raise AssertionError(f"Requested pruning features missing from V2-B list: {missing}")


def _importance_audit(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]
) -> pd.DataFrame:
    fold_rows = []
    correlation = _max_correlations(dataset, features)
    missing = dataset[features].isna().mean()
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, features, target="label")
        x_val = _transform_with_medians(validation, features, medians)
        sample_index = _sample_index(x_val, PERMUTATION_SAMPLE_ROWS, fold["fold"])
        perm = _permutation_importance(
            model, x_val.iloc[sample_index], validation.iloc[sample_index]
        )
        shap_values = _shap_importance(model, x_val, fold["fold"])
        shap_rank = _rank_descending(shap_values)
        for feature in features:
            fold_rows.append(
                {
                    "Feature": feature,
                    "Fold": fold["fold"],
                    "PermutationImportance": perm.get(feature),
                    "ShapMeanAbs": shap_values.get(feature),
                    "ShapRank": shap_rank.get(feature),
                    "MaxAbsSpearmanCorrelation": correlation.get(feature),
                    "MissingPercentage": float(missing.get(feature, 0.0)),
                }
            )
    folds_frame = pd.DataFrame(fold_rows)
    summary_rows = []
    for feature, group in folds_frame.groupby("Feature", sort=False):
        positive = group["PermutationImportance"] > 0
        summary_rows.append(
            {
                "Feature": feature,
                "FoldCount": int(group["Fold"].nunique()),
                "MedianPermutationImportance": _safe_float(group["PermutationImportance"].median()),
                "MeanPermutationImportance": _safe_float(group["PermutationImportance"].mean()),
                "PositiveImportanceFolds": int(positive.sum()),
                "PositiveImportanceFoldPercentage": float(positive.mean()),
                "MeanAbsShapValue": _safe_float(group["ShapMeanAbs"].mean()),
                "MedianAbsShapValue": _safe_float(group["ShapMeanAbs"].median()),
                "MedianShapRank": _safe_float(group["ShapRank"].median()),
                "BestShapRank": _safe_float(group["ShapRank"].min()),
                "MaxAbsSpearmanCorrelation": _safe_float(group["MaxAbsSpearmanCorrelation"].max()),
                "MissingPercentage": _safe_float(group["MissingPercentage"].max()),
                "FoldPermutationValues": json.dumps(
                    [None if pd.isna(v) else float(v) for v in group["PermutationImportance"]]
                ),
                "FoldShapMeanAbsValues": json.dumps(
                    [None if pd.isna(v) else float(v) for v in group["ShapMeanAbs"]]
                ),
                "FoldShapRanks": json.dumps(
                    [None if pd.isna(v) else int(v) for v in group["ShapRank"]]
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def _sample_index(frame: pd.DataFrame, max_rows: int, fold: Any) -> np.ndarray:
    if len(frame) <= max_rows:
        return np.arange(len(frame))
    rng = np.random.default_rng(RANDOM_SEED + int(fold))
    return np.sort(rng.choice(np.arange(len(frame)), size=max_rows, replace=False))


def _permutation_importance(
    model: Any, x_val: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, float]:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import log_loss, make_scorer

    y_true = validation["label"].map(LABEL_TO_CLASS).astype(int)
    scorer = make_scorer(
        log_loss, response_method="predict_proba", greater_is_better=False, labels=[0, 1, 2]
    )
    result = permutation_importance(
        model,
        x_val,
        y_true,
        scoring=scorer,
        n_repeats=1,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return dict(zip(x_val.columns, result.importances_mean, strict=True))


def _shap_importance(model: Any, x_val: pd.DataFrame, fold: Any) -> dict[str, float | None]:
    try:
        import shap  # type: ignore[import-untyped]
    except Exception:
        return {feature: None for feature in x_val.columns}
    sample_index = _sample_index(x_val, SHAP_SAMPLE_ROWS, fold)
    sample = x_val.iloc[sample_index]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
        if isinstance(values, list):
            arr = np.mean([np.abs(item) for item in values], axis=0)
        else:
            arr = np.abs(values)
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
        mean_abs = np.asarray(arr).mean(axis=0)
        return dict(zip(x_val.columns, mean_abs.astype(float), strict=True))
    except Exception:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return {feature: None for feature in x_val.columns}
        return dict(zip(x_val.columns, np.asarray(importances).astype(float), strict=True))


def _rank_descending(values: dict[str, float | None]) -> dict[str, int | None]:
    valid = {k: v for k, v in values.items() if v is not None and pd.notna(v)}
    ordered = sorted(valid, key=lambda key: valid[key], reverse=True)
    ranks = {feature: index + 1 for index, feature in enumerate(ordered)}
    return {feature: ranks.get(feature) for feature in values}


def _max_correlations(dataset: pd.DataFrame, features: list[str]) -> dict[str, float]:
    sample = dataset[features].sample(
        n=min(len(dataset), 30000),
        random_state=RANDOM_SEED,
    )
    corr = sample.corr(method="spearman").abs()
    output = {}
    for feature in features:
        values = corr[feature].drop(index=feature, errors="ignore").dropna()
        output[feature] = float(values.max()) if not values.empty else 0.0
    return output


def _feature_labels(importance: pd.DataFrame) -> pd.DataFrame:
    shap_cutoff = importance["MeanAbsShapValue"].fillna(0).quantile(0.25)
    rows = []
    for row in importance.itertuples(index=False):
        low_shap = (
            (row.MeanAbsShapValue is None)
            or pd.isna(row.MeanAbsShapValue)
            or row.MeanAbsShapValue <= shap_cutoff
        )
        weak_perm = (
            (row.MedianPermutationImportance is not None)
            and row.MedianPermutationImportance <= 0
            and row.PositiveImportanceFoldPercentage <= 0.20
        )
        redundant = row.MaxAbsSpearmanCorrelation >= 0.95
        if weak_perm and low_shap:
            label = "DROP_CANDIDATE"
            reason = "median permutation <= 0, <=20% positive folds, low SHAP"
        elif weak_perm or (low_shap and redundant):
            label = "REVIEW"
            reason = "weak importance or low SHAP with high redundancy"
        else:
            label = "KEEP"
            reason = "importance is positive/stable enough for current evidence"
        rows.append(
            {
                "Feature": row.Feature,
                "DecisionLabel": label,
                "DecisionReason": reason,
                "MedianPermutationImportance": row.MedianPermutationImportance,
                "PositiveImportanceFoldPercentage": row.PositiveImportanceFoldPercentage,
                "MeanAbsShapValue": row.MeanAbsShapValue,
                "MaxAbsSpearmanCorrelation": row.MaxAbsSpearmanCorrelation,
            }
        )
    return pd.DataFrame(rows)


def _variants(features_55: list[str], labels: pd.DataFrame) -> list[Variant]:
    first_pruned = _remove(features_55, FIRST_PRUNE_REMOVE)
    drop_candidates = labels[labels["DecisionLabel"].eq("DROP_CANDIDATE")]["Feature"].tolist()
    review_weak = [feature for feature in drop_candidates if feature in WEAK_ABLATION_FEATURES]
    variants = [
        Variant("v2_b_55", "55-feature V2-B baseline", features_55, [], "model_comparison"),
        Variant(
            "v2_first_prune_44",
            "44-feature first pruning candidate",
            first_pruned,
            FIRST_PRUNE_REMOVE,
            "model_comparison",
        ),
        Variant(
            "v2_raw_price_scale_removed",
            "Raw price-scale ablation",
            _remove(features_55, RAW_PRICE_SCALE_REMOVE),
            RAW_PRICE_SCALE_REMOVE,
            "raw_price_scale_ablation",
        ),
    ]
    if review_weak:
        variants.append(
            Variant(
                "v2_best_further_pruned_candidate",
                "Best further-pruned DROP_CANDIDATE subset",
                _remove(features_55, review_weak),
                review_weak,
                "model_comparison",
            )
        )
    else:
        variants.append(
            Variant(
                "v2_best_further_pruned_candidate",
                "Best further-pruned candidate same as first prune",
                first_pruned,
                FIRST_PRUNE_REMOVE,
                "model_comparison",
            )
        )
    for feature in WEAK_ABLATION_FEATURES:
        variants.append(
            Variant(
                f"ablate_{feature}",
                f"Individual ablation: {feature}",
                _remove(features_55, [feature]),
                [feature],
                "individual_weak_ablation",
            )
        )
    variants.append(
        Variant(
            "ablate_weak_group_all",
            "Group ablation: weak candidates",
            _remove(features_55, WEAK_ABLATION_FEATURES),
            WEAK_ABLATION_FEATURES,
            "group_weak_ablation",
        )
    )
    return variants


def _remove(features: list[str], remove: list[str]) -> list[str]:
    missing = sorted(set(remove).difference(features))
    if missing:
        raise AssertionError(f"Cannot remove missing features: {missing}")
    return [feature for feature in features if feature not in set(remove)]


def _classifier_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], variant: Variant
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, variant.features, target="label")
        probabilities = model.predict_proba(
            _transform_with_medians(validation, variant.features, medians)
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
        output["Variant"] = variant.name
        output["VariantName"] = variant.display_name
        output["VariantType"] = variant.variant_type
        output["FeatureCount"] = len(variant.features)
        output["fold"] = fold["fold"]
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["predicted_label"] = [
            {v: k for k, v in LABEL_TO_CLASS.items()}[int(index)]
            for index in probabilities.argmax(axis=1)
        ]
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _comparison(predictions: pd.DataFrame, variants: list[Variant]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        frame = predictions[predictions["Variant"].eq(variant.name)].copy()
        rows.append(_overall_row(frame, variant))
        rows.extend(_tail_rows(frame, variant))
        for fold, group in frame.groupby("fold"):
            rows.append(_stability_row(group, variant, "fold", fold))
        for year, group in frame.groupby(frame["Date"].dt.year):
            rows.append(_stability_row(group, variant, "year", int(year)))
    return pd.DataFrame(rows)


def _buy_agreement_comparison(
    predictions: pd.DataFrame, binary_buy: pd.DataFrame, variants: list[Variant]
) -> pd.DataFrame:
    rows = []
    key = ["Date", "symbol", "fold"]
    for variant in [item for item in variants if item.variant_type == "model_comparison"]:
        frame = predictions[predictions["Variant"].eq(variant.name)].merge(
            binary_buy[key + ["binary_buy_percentile"]],
            on=key,
            how="inner",
        )
        for classifier_top in TOP_FRACTIONS:
            selected = frame[
                (frame["classifier_percentile"] >= 1 - classifier_top)
                & (frame["binary_buy_percentile"] >= 0.995)
            ].copy()
            rows.append(
                {
                    "RowType": "overall",
                    "Variant": variant.name,
                    "VariantName": variant.display_name,
                    "FeatureCount": len(variant.features),
                    "ClassifierTop": classifier_top,
                    "BinaryTop": 0.005,
                    "SignalCount": int(len(selected)),
                    "Precision": (
                        float(selected["buy_target"].mean()) if not selected.empty else None
                    ),
                    "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                    "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
                    "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
                }
            )
        frozen = frame[
            (frame["classifier_percentile"] >= 0.99) & (frame["binary_buy_percentile"] >= 0.995)
        ].copy()
        for fold, group in frozen.groupby("fold"):
            rows.append(_agreement_stability_row(group, variant, "fold", fold))
        for year, group in frozen.groupby(frozen["Date"].dt.year):
            rows.append(_agreement_stability_row(group, variant, "year", int(year)))
    return pd.DataFrame(rows)


def _agreement_stability_row(
    frame: pd.DataFrame, variant: Variant, scope: str, scope_value: Any
) -> dict[str, Any]:
    return {
        "RowType": "stability",
        "Variant": variant.name,
        "VariantName": variant.display_name,
        "FeatureCount": len(variant.features),
        "ClassifierTop": 0.01,
        "BinaryTop": 0.005,
        "Scope": scope,
        "ScopeValue": scope_value,
        "SignalCount": int(len(frame)),
        "Precision": float(frame["buy_target"].mean()) if not frame.empty else None,
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _overall_row(frame: pd.DataFrame, variant: Variant) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, log_loss

    y_true = frame["label"].map(LABEL_TO_CLASS)
    probabilities = frame[["p_underperform", "p_neutral", "p_outperform"]]
    actual_buy = frame["buy_target"].astype(int)
    backtest = backtest_top_n_portfolio(frame, top_n=5)
    return {
        "RowType": "overall",
        "VariantType": (
            "model_comparison"
            if variant.variant_type == "model_comparison"
            else variant.variant_type
        ),
        "Variant": variant.name,
        "VariantName": variant.display_name,
        "FeatureCount": len(variant.features),
        "RemovedFeatureCount": len(variant.removed_features),
        "RemovedFeatures": ",".join(variant.removed_features),
        "Rows": int(len(frame)),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(((frame["p_outperform"] - actual_buy) ** 2).mean()),
        "PRAUC": float(average_precision_score(actual_buy, frame["p_outperform"])),
        "MeanIC": _ic_summary(frame, "p_outperform")["MeanIC"],
        "Sharpe": backtest.get("sharpe"),
        "MaxDrawdown": backtest.get("max_drawdown"),
    }


def _tail_rows(frame: pd.DataFrame, variant: Variant) -> list[dict[str, Any]]:
    rows = []
    for fraction in TOP_FRACTIONS:
        selected = frame.sort_values("p_outperform", ascending=False).head(
            max(1, int(len(frame) * fraction))
        )
        rows.append(
            {
                "RowType": "tail",
                "VariantType": variant.variant_type,
                "Variant": variant.name,
                "VariantName": variant.display_name,
                "FeatureCount": len(variant.features),
                "TopPercentile": f"Top {fraction * 100:g}%",
                "SignalCount": int(len(selected)),
                "Precision": float(selected["buy_target"].mean()),
                "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
                "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
                "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
            }
        )
    return rows


def _stability_row(
    frame: pd.DataFrame, variant: Variant, scope: str, scope_value: Any
) -> dict[str, Any]:
    selected = frame.sort_values("p_outperform", ascending=False).head(
        max(1, int(len(frame) * 0.01))
    )
    return {
        "RowType": "stability",
        "VariantType": variant.variant_type,
        "Variant": variant.name,
        "VariantName": variant.display_name,
        "FeatureCount": len(variant.features),
        "Scope": scope,
        "ScopeValue": scope_value,
        "SignalCount": int(len(selected)),
        "Precision": float(selected["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
    }


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _summary(
    importance: pd.DataFrame,
    labels: pd.DataFrame,
    model_comparison: pd.DataFrame,
    ablation: pd.DataFrame,
    buy_agreement: pd.DataFrame,
    variants: list[Variant],
) -> str:
    overall = model_comparison[model_comparison["RowType"].eq("overall")].copy()
    tails = model_comparison[model_comparison["RowType"].eq("tail")].copy()
    top_tail = tails[tails["TopPercentile"].isin(["Top 1%", "Top 0.5%"])]
    drop = labels[labels["DecisionLabel"].eq("DROP_CANDIDATE")]
    review = labels[labels["DecisionLabel"].eq("REVIEW")]
    best_top1 = (
        top_tail[top_tail["TopPercentile"].eq("Top 1%")]
        .sort_values(["Precision", "MedianExcessReturn", "AverageExcessReturn"], ascending=False)
        .head(1)
    )
    frozen_buy = buy_agreement[
        (buy_agreement["RowType"].eq("overall"))
        & (buy_agreement["ClassifierTop"].eq(0.01))
        & (buy_agreement["BinaryTop"].eq(0.005))
    ].copy()
    return "\n".join(
        [
            "NATIP V2-B 55-Feature Pruning Research",
            "======================================",
            "",
            "Production impact: NONE. V1 production and frozen V2 thresholds were not modified.",
            "Final test usage: NOT USED.",
            "",
            f"Variants tested: {len(variants)}",
            "",
            "Model comparison overall:",
            overall.to_string(index=False),
            "",
            "High-confidence tail comparison:",
            top_tail.to_string(index=False),
            "",
            "Feature labels:",
            labels.sort_values(["DecisionLabel", "MedianPermutationImportance"]).to_string(
                index=False
            ),
            "",
            f"DROP_CANDIDATE count: {len(drop)}",
            f"REVIEW count: {len(review)}",
            "",
            "Best Top1 row:",
            best_top1.to_string(index=False) if not best_top1.empty else "No Top1 row.",
            "",
            "Frozen V2 BUY-rule agreement comparison (classifier Top 1%, Binary Top 0.5%):",
            frozen_buy.to_string(index=False),
            "",
            "Ablation report written to:",
            str(OUTPUT_ABLATION),
            "Importance report written to:",
            str(OUTPUT_IMPORTANCE),
        ]
    )


if __name__ == "__main__":
    main()
