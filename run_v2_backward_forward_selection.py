"""Backward-then-forward V2 feature selection optimized for hybrid BUY validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

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
from run_v2_55_feature_pruning_research import FIRST_PRUNE_REMOVE, RAW_PRICE_SCALE_REMOVE

CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"

OUTPUT_BACKWARD = REPORT_DIR / "v2_backward_elimination.csv"
OUTPUT_35 = REPORT_DIR / "v2_35_feature_results.csv"
OUTPUT_LEAN = REPORT_DIR / "v2_lean_core_results.csv"
OUTPUT_FORWARD = REPORT_DIR / "v2_forward_feature_value.csv"
OUTPUT_PROGRESSIVE = REPORT_DIR / "v2_progressive_forward_selection.csv"
OUTPUT_RANKING = REPORT_DIR / "v2_final_feature_ranking.csv"
OUTPUT_SELECTED = REPORT_DIR / "v2_selected_feature_set.json"
OUTPUT_REPORT = REPORT_DIR / "v2_backward_forward_report.txt"

GROUPS = {
    "H_volatility": [
        "volatility_20d",
        "stock_volatility_relative_sector_20d",
        "stock_volatility_relative_sector_60d",
    ],
    "I_relative_strength_horizons": ["rs_vs_nifty_20d", "rs_vs_nifty_252d"],
    "J_stock_vs_sector": [
        "stock_vs_sector_ret_60d",
        "stock_vs_sector_ret_120d",
        "stock_vs_sector_ret_252d",
    ],
    "K_sector_context": ["sector_strength_vs_nifty_60d", "sector_momentum_20d"],
    "L_technical_location": ["minus_di_14", "distance_52w_high"],
}

PROTECTED_FEATURES = {
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "price_to_sma_50",
    "adx_14",
    "volatility_60d",
    "distance_52w_low",
    "rs_vs_nifty_60d",
    "rs_vs_nifty_120d",
    "stock_vs_sector_ret_20d",
    "sector_strength_vs_nifty_20d",
    "sector_strength_vs_nifty_120d",
    "sector_momentum_60d",
    "sector_momentum_120d",
    "stock_distance_52w_high_relative_sector",
    "beta_120d",
    "beta_252d",
    "corr_nifty_20d",
    "atr_14_to_atr_50",
    "momentum_60d_vs_120d",
    "relative_momentum_20d_change",
    "sector_rsi_14",
}

CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)


@dataclass(frozen=True, slots=True)
class Variant:
    """Feature set variant."""

    name: str
    display_name: str
    features: list[str]
    variant_type: str
    removed_features: list[str]


def main() -> None:
    """Run validation-only backward-then-forward selection."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_77 = _frozen_features()
    features_55 = _v2_55_features(features_77)
    features_44 = _remove(features_55, FIRST_PRUNE_REMOVE)
    features_35 = _remove(features_44, RAW_PRICE_SCALE_REMOVE)
    _assert_setup(features_55, features_44, features_35)
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    binary = _binary_buy_77_oof(
        common,
        folds,
        FeatureSet("binary_77", "Unchanged 77-feature Binary BUY", features_77),
    )

    initial_variants = [
        Variant("v2_b_55", "55-feature V2-B benchmark", features_55, "benchmark", []),
        Variant(
            "v2_candidate_44",
            "44-feature candidate",
            features_44,
            "candidate_44",
            FIRST_PRUNE_REMOVE,
        ),
        Variant(
            "v2_candidate_35",
            "35-feature lean candidate",
            features_35,
            "candidate_35",
            [*FIRST_PRUNE_REMOVE, *RAW_PRICE_SCALE_REMOVE],
        ),
    ]
    group_variants = _group_ablation_variants(features_35)
    print(f"[backward] {len([*initial_variants, *group_variants])} variants", flush=True)
    initial_predictions = _predict_variants(common, folds, [*initial_variants, *group_variants])
    initial_results = _evaluate_variants(
        initial_predictions, binary, [*initial_variants, *group_variants]
    )
    backward = initial_results[
        initial_results["VariantType"].isin(["candidate_35", "group_ablation"])
    ].copy()
    backward.to_csv(OUTPUT_BACKWARD, index=False)
    initial_results[initial_results["Variant"].eq("v2_candidate_35")].to_csv(OUTPUT_35, index=False)

    lean_features, accepted_groups = _build_lean_core(features_35, backward)
    lean_variant = Variant(
        "v2_lean_core",
        "Lean core from accepted backward eliminations",
        lean_features,
        "lean_core",
        [feature for feature in features_55 if feature not in lean_features],
    )
    print(f"[lean] {len(lean_features)} features, accepted groups={accepted_groups}", flush=True)
    lean_predictions = _predict_variants(common, folds, [lean_variant])
    lean_results = _evaluate_variants(lean_predictions, binary, [lean_variant])
    lean_results.to_csv(OUTPUT_LEAN, index=False)

    excluded = [feature for feature in features_55 if feature not in lean_features]
    forward_variants = [
        Variant(
            f"forward_add_{_safe_name(feature)}",
            f"Lean core + {feature}",
            [*lean_features, feature],
            "forward_one_at_a_time",
            [item for item in excluded if item != feature],
        )
        for feature in excluded
    ]
    print(f"[forward] {len(forward_variants)} one-at-a-time additions", flush=True)
    forward_predictions = _predict_variants(common, folds, forward_variants)
    forward_results = _evaluate_variants(forward_predictions, binary, forward_variants)
    forward_value = _forward_value(lean_results, forward_results, excluded)
    forward_value.to_csv(OUTPUT_FORWARD, index=False)

    progressive_variants = _progressive_variants(lean_features, forward_value)
    print(f"[progressive] {len(progressive_variants)} additions", flush=True)
    progressive_predictions = _predict_variants(common, folds, progressive_variants)
    progressive_results = _evaluate_variants(progressive_predictions, binary, progressive_variants)
    progressive_results.to_csv(OUTPUT_PROGRESSIVE, index=False)

    final_ranking = _final_ranking(
        pd.concat([initial_results, lean_results, progressive_results], ignore_index=True)
    )
    final_ranking.to_csv(OUTPUT_RANKING, index=False)
    selected = _selected_payload(
        features_55=features_55,
        features_44=features_44,
        features_35=features_35,
        lean_features=lean_features,
        accepted_groups=accepted_groups,
        forward_value=forward_value,
        progressive=progressive_results,
        progressive_variants=progressive_variants,
        final_ranking=final_ranking,
    )
    OUTPUT_SELECTED.write_text(json.dumps(_json_safe(selected), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(
        _report(
            initial_results,
            backward,
            lean_results,
            forward_value,
            progressive_results,
            final_ranking,
            selected,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "v2_backward_elimination": str(OUTPUT_BACKWARD),
                "v2_35_feature_results": str(OUTPUT_35),
                "v2_lean_core_results": str(OUTPUT_LEAN),
                "v2_forward_feature_value": str(OUTPUT_FORWARD),
                "v2_progressive_forward_selection": str(OUTPUT_PROGRESSIVE),
                "v2_final_feature_ranking": str(OUTPUT_RANKING),
                "v2_selected_feature_set": str(OUTPUT_SELECTED),
                "v2_backward_forward_report": str(OUTPUT_REPORT),
                "final_test_status": "not_used",
                "v1_production_modified": False,
                "frozen_v2_buy_rule_modified": False,
            },
            indent=2,
        )
    )


def _v2_55_features(features_77: list[str]) -> list[str]:
    remove = NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES
    return [feature for feature in features_77 if feature not in remove]


def _assert_setup(features_55: list[str], features_44: list[str], features_35: list[str]) -> None:
    if len(features_55) != 55 or len(features_44) != 44 or len(features_35) != 35:
        raise AssertionError(
            f"Unexpected feature counts: 55={len(features_55)}, 44={len(features_44)}, 35={len(features_35)}"
        )
    protected_removed = PROTECTED_FEATURES.difference(features_35)
    if protected_removed:
        raise AssertionError(
            f"Protected features missing from 35-feature set: {sorted(protected_removed)}"
        )
    missing_group_features = sorted(
        {feature for group in GROUPS.values() for feature in group}.difference(features_35)
    )
    if missing_group_features:
        raise AssertionError(
            f"Group ablation features missing from 35-feature set: {missing_group_features}"
        )


def _remove(features: list[str], remove: list[str]) -> list[str]:
    remove_set = set(remove)
    missing = sorted(remove_set.difference(features))
    if missing:
        raise AssertionError(f"Cannot remove missing features: {missing}")
    return [feature for feature in features if feature not in remove_set]


def _group_ablation_variants(features_35: list[str]) -> list[Variant]:
    return [
        Variant(
            f"ablate_{group_name}",
            f"35-feature minus {group_name}",
            _remove(features_35, group_features),
            "group_ablation",
            group_features,
        )
        for group_name, group_features in GROUPS.items()
    ]


def _predict_variants(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], variants: list[Variant]
) -> pd.DataFrame:
    frames = []
    for variant in variants:
        print(f"[classifier] {variant.name} ({len(variant.features)} features)", flush=True)
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
            output["RemovedFeatures"] = ",".join(variant.removed_features)
            output["fold"] = fold["fold"]
            output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
            output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
            output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
            output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
            frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _evaluate_variants(
    predictions: pd.DataFrame, binary: pd.DataFrame, variants: list[Variant]
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    rows = []
    for variant in variants:
        frame = predictions[predictions["Variant"].eq(variant.name)].merge(
            binary[key + ["binary_buy_percentile"]],
            on=key,
            how="inner",
        )
        rows.append(_overall_classifier_row(frame, variant))
        rows.extend(_hybrid_rows(frame, variant, "overall", "all"))
        frozen = _select_hybrid(frame, CLASSIFIER_TOP, BINARY_TOP)
        for fold, group in frozen.groupby("fold"):
            rows.append(
                _hybrid_metric_row(group, variant, "fold", fold, CLASSIFIER_TOP, BINARY_TOP)
            )
        for year, group in frozen.groupby(frozen["Date"].dt.year):
            rows.append(
                _hybrid_metric_row(group, variant, "year", int(year), CLASSIFIER_TOP, BINARY_TOP)
            )
    return pd.DataFrame(rows)


def _overall_classifier_row(frame: pd.DataFrame, variant: Variant) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, log_loss

    y_true = frame["label"].map(LABEL_TO_CLASS)
    probabilities = frame[["p_underperform", "p_neutral", "p_outperform"]]
    actual_buy = frame["buy_target"].astype(int)
    top1 = frame.sort_values("p_outperform", ascending=False).head(max(1, int(len(frame) * 0.01)))
    backtest = backtest_top_n_portfolio(frame, top_n=5)
    return {
        "RowType": "classifier",
        "Variant": variant.name,
        "VariantName": variant.display_name,
        "VariantType": variant.variant_type,
        "FeatureCount": len(variant.features),
        "RemovedFeatures": ",".join(variant.removed_features),
        "Rows": int(len(frame)),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(((frame["p_outperform"] - actual_buy) ** 2).mean()),
        "PRAUC": float(average_precision_score(actual_buy, frame["p_outperform"])),
        "MeanIC": _ic_summary(frame, "p_outperform")["MeanIC"],
        "Sharpe": backtest.get("sharpe"),
        "ClassifierTop1Precision": float(top1["buy_target"].mean()),
    }


def _hybrid_rows(
    frame: pd.DataFrame, variant: Variant, scope: str, scope_value: Any
) -> list[dict[str, Any]]:
    rows = []
    for fraction in TOP_FRACTIONS:
        selected = _select_hybrid(frame, fraction, BINARY_TOP)
        rows.append(_hybrid_metric_row(selected, variant, scope, scope_value, fraction, BINARY_TOP))
    return rows


def _select_hybrid(frame: pd.DataFrame, classifier_top: float, binary_top: float) -> pd.DataFrame:
    return frame[
        (frame["classifier_percentile"] >= 1 - classifier_top)
        & (frame["binary_buy_percentile"] >= 1 - binary_top)
    ].copy()


def _hybrid_metric_row(
    frame: pd.DataFrame,
    variant: Variant,
    scope: str,
    scope_value: Any,
    classifier_top: float,
    binary_top: float,
) -> dict[str, Any]:
    return {
        "RowType": "hybrid_buy",
        "Variant": variant.name,
        "VariantName": variant.display_name,
        "VariantType": variant.variant_type,
        "FeatureCount": len(variant.features),
        "RemovedFeatures": ",".join(variant.removed_features),
        "ClassifierTop": classifier_top,
        "BinaryTop": binary_top,
        "Scope": scope,
        "ScopeValue": scope_value,
        "SignalCount": int(len(frame)),
        "Precision": float(frame["buy_target"].mean()) if not frame.empty else None,
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()),
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()),
    }


def _build_lean_core(features_35: list[str], backward: pd.DataFrame) -> tuple[list[str], list[str]]:
    base = _frozen_row(backward, "v2_candidate_35")
    accepted_groups: list[str] = []
    remove_features: list[str] = []
    for group_name, group_features in GROUPS.items():
        row = _frozen_row(backward, f"ablate_{group_name}")
        proposed_remove = [*remove_features, *group_features]
        proposed_count = len(_remove(features_35, proposed_remove))
        if proposed_count < 25:
            continue
        if _acceptable_backward(row, base):
            accepted_groups.append(group_name)
            remove_features = proposed_remove
    return _remove(features_35, remove_features), accepted_groups


def _acceptable_backward(row: pd.Series, base: pd.Series) -> bool:
    if row.empty:
        return False
    return bool(
        row["Precision"] >= base["Precision"] - 0.005
        and row["AverageExcessReturn"] >= base["AverageExcessReturn"] - 0.003
        and row["MedianExcessReturn"] >= base["MedianExcessReturn"] - 0.003
        and row["NiftyWinRate"] >= base["NiftyWinRate"] - 0.02
        and row["SignalCount"] >= max(100, base["SignalCount"] * 0.70)
    )


def _forward_value(
    lean_results: pd.DataFrame, forward_results: pd.DataFrame, excluded: list[str]
) -> pd.DataFrame:
    base = _frozen_row(lean_results, "v2_lean_core")
    rows = []
    for feature in excluded:
        variant = f"forward_add_{_safe_name(feature)}"
        row = _frozen_row(forward_results, variant)
        fold_improved = _improvement_count(
            forward_results, lean_results, variant, "fold", "Precision"
        )
        year_improved = _improvement_count(
            forward_results, lean_results, variant, "year", "Precision"
        )
        rows.append(
            {
                "Feature": feature,
                "Variant": variant,
                "SignalCount": row.get("SignalCount"),
                "Precision": row.get("Precision"),
                "AverageExcessReturn": row.get("AverageExcessReturn"),
                "MedianExcessReturn": row.get("MedianExcessReturn"),
                "NiftyWinRate": row.get("NiftyWinRate"),
                "DeltaSignalCount": row.get("SignalCount") - base.get("SignalCount"),
                "DeltaPrecision": row.get("Precision") - base.get("Precision"),
                "DeltaAverageExcessReturn": row.get("AverageExcessReturn")
                - base.get("AverageExcessReturn"),
                "DeltaMedianExcessReturn": row.get("MedianExcessReturn")
                - base.get("MedianExcessReturn"),
                "DeltaNiftyWinRate": row.get("NiftyWinRate") - base.get("NiftyWinRate"),
                "DeltaPRAUC": _classifier_metric(forward_results, variant, "PRAUC")
                - _classifier_metric(lean_results, "v2_lean_core", "PRAUC"),
                "DeltaMeanIC": _classifier_metric(forward_results, variant, "MeanIC")
                - _classifier_metric(lean_results, "v2_lean_core", "MeanIC"),
                "FoldsPrecisionImproved": fold_improved,
                "YearsPrecisionImproved": year_improved,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "DeltaPrecision",
            "DeltaMedianExcessReturn",
            "DeltaAverageExcessReturn",
            "YearsPrecisionImproved",
            "FoldsPrecisionImproved",
        ],
        ascending=False,
    )


def _progressive_variants(lean_features: list[str], forward_value: pd.DataFrame) -> list[Variant]:
    selected = list(lean_features)
    variants = []
    for step, feature in enumerate(forward_value["Feature"].tolist(), start=1):
        if forward_value[forward_value["Feature"].eq(feature)].iloc[0]["DeltaPrecision"] <= 0:
            continue
        selected = [*selected, feature]
        variants.append(
            Variant(
                f"progressive_step_{step}_{_safe_name(feature)}",
                f"Progressive step {step}: add {feature}",
                selected.copy(),
                "progressive_forward",
                [],
            )
        )
        if len(variants) >= 8:
            break
    return variants


def _final_ranking(results: pd.DataFrame) -> pd.DataFrame:
    frozen = results[
        (results["RowType"].eq("hybrid_buy"))
        & (results["Scope"].eq("overall"))
        & (results["ClassifierTop"].eq(CLASSIFIER_TOP))
        & (results["BinaryTop"].eq(BINARY_TOP))
    ].copy()
    classifier = results[results["RowType"].eq("classifier")][
        ["Variant", "PRAUC", "MeanIC", "ClassifierTop1Precision", "LogLoss", "Sharpe"]
    ]
    output = frozen.merge(classifier, on="Variant", how="left")
    output = output.sort_values(
        ["Precision", "MedianExcessReturn", "AverageExcessReturn", "NiftyWinRate"],
        ascending=False,
    )
    output["Rank"] = range(1, len(output) + 1)
    return output


def _selected_payload(
    *,
    features_55: list[str],
    features_44: list[str],
    features_35: list[str],
    lean_features: list[str],
    accepted_groups: list[str],
    forward_value: pd.DataFrame,
    progressive: pd.DataFrame,
    progressive_variants: list[Variant],
    final_ranking: pd.DataFrame,
) -> dict[str, Any]:
    best = final_ranking.iloc[0].to_dict() if not final_ranking.empty else {}
    feature_map = {
        "v2_b_55": features_55,
        "v2_candidate_44": features_44,
        "v2_candidate_35": features_35,
        "v2_lean_core": lean_features,
    }
    feature_map.update({variant.name: variant.features for variant in progressive_variants})
    selected_features = feature_map.get(best.get("Variant"), lean_features)
    selected_feature_set = set(selected_features)
    forward_additions = [
        feature
        for feature in selected_features
        if feature not in lean_features and feature in set(features_55)
    ]
    return {
        "research_version": "v2-backward-forward-validation-only",
        "final_test_used": False,
        "v1_production_modified": False,
        "frozen_v2_buy_rule_modified": False,
        "frozen_buy_rule": {
            "classifier_top": CLASSIFIER_TOP,
            "binary_buy_top": BINARY_TOP,
        },
        "accepted_backward_groups": accepted_groups,
        "best_variant": best,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "excluded_features": [
            feature for feature in features_55 if feature not in selected_feature_set
        ],
        "forward_additions": forward_additions,
        "top_forward_features": forward_value.head(10).to_dict(orient="records"),
        "progressive_rows": progressive.to_dict(orient="records"),
    }


def _frozen_row(frame: pd.DataFrame, variant: str) -> pd.Series:
    rows = frame[
        (frame["Variant"].eq(variant))
        & (frame["RowType"].eq("hybrid_buy"))
        & (frame["Scope"].eq("overall"))
        & (frame["ClassifierTop"].eq(CLASSIFIER_TOP))
        & (frame["BinaryTop"].eq(BINARY_TOP))
    ]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _classifier_metric(frame: pd.DataFrame, variant: str, metric: str) -> float:
    rows = frame[(frame["Variant"].eq(variant)) & (frame["RowType"].eq("classifier"))]
    return float(rows.iloc[0][metric]) if not rows.empty else 0.0


def _improvement_count(
    candidate: pd.DataFrame, base: pd.DataFrame, variant: str, scope: str, metric: str
) -> int:
    candidate_rows = candidate[
        (candidate["Variant"].eq(variant))
        & (candidate["RowType"].eq("hybrid_buy"))
        & (candidate["Scope"].eq(scope))
    ][["ScopeValue", metric]]
    base_rows = base[
        (base["Variant"].eq("v2_lean_core"))
        & (base["RowType"].eq("hybrid_buy"))
        & (base["Scope"].eq(scope))
    ][["ScopeValue", metric]]
    joined = candidate_rows.merge(base_rows, on="ScopeValue", suffixes=("_candidate", "_base"))
    return int((joined[f"{metric}_candidate"] > joined[f"{metric}_base"]).sum())


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _safe_name(feature: str) -> str:
    return feature.replace("/", "_").replace("-", "_").replace(".", "_")


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


def _report(
    initial: pd.DataFrame,
    backward: pd.DataFrame,
    lean: pd.DataFrame,
    forward: pd.DataFrame,
    progressive: pd.DataFrame,
    ranking: pd.DataFrame,
    selected: dict[str, Any],
) -> str:
    initial_frozen = initial[
        (initial["RowType"].eq("hybrid_buy"))
        & (initial["Scope"].eq("overall"))
        & (initial["ClassifierTop"].eq(CLASSIFIER_TOP))
        & (initial["BinaryTop"].eq(BINARY_TOP))
    ]
    lean_frozen = lean[
        (lean["RowType"].eq("hybrid_buy"))
        & (lean["Scope"].eq("overall"))
        & (lean["ClassifierTop"].eq(CLASSIFIER_TOP))
        & (lean["BinaryTop"].eq(BINARY_TOP))
    ]
    return "\n".join(
        [
            "NATIP V2 Backward-Then-Forward Feature Selection",
            "=================================================",
            "",
            "Production impact: NONE. V1 production was not modified.",
            "Final test usage: NOT USED.",
            "Frozen V2 BUY rule modified: NO.",
            "",
            "Initial frozen-rule comparison:",
            initial_frozen.to_string(index=False),
            "",
            "Backward elimination rows:",
            backward[
                (backward["RowType"].eq("hybrid_buy"))
                & (backward["Scope"].eq("overall"))
                & (backward["ClassifierTop"].eq(CLASSIFIER_TOP))
            ].to_string(index=False),
            "",
            "Lean core frozen-rule result:",
            lean_frozen.to_string(index=False),
            "",
            "Top forward feature value:",
            forward.head(20).to_string(index=False),
            "",
            "Progressive forward results:",
            progressive[
                (progressive["RowType"].eq("hybrid_buy"))
                & (progressive["Scope"].eq("overall"))
                & (progressive["ClassifierTop"].eq(CLASSIFIER_TOP))
            ].to_string(index=False),
            "",
            "Final ranking:",
            ranking.to_string(index=False),
            "",
            f"Selected candidate: {selected.get('best_variant', {}).get('Variant')}",
            f"Selected feature count: {selected.get('selected_feature_count')}",
        ]
    )


if __name__ == "__main__":
    main()
