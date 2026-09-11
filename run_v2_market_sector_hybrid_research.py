"""Continue NATIP V2 validation-only market/sector ablation research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

import run_stock_specific_feature_variant as stock_specific_variant
from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_stock_specific_feature_variant import (
    MARKET_BREADTH_FEATURES,
    MARKET_REGIME_FEATURES,
    NIFTY_FEATURES,
    SECTOR_LEVEL_FEATURES,
    FeatureSet,
    _binary_buy_walk_forward,
    _classifier_comparison,
    _classifier_walk_forward,
    _common_evaluation_rows,
    _frozen_features,
    _ic_summary,
    _prepare_dataset,
    _safe_float,
    _top_fraction,
)

CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"

OUTPUT_ABLATION = REPORT_DIR / "v2_market_vs_sector_ablation.csv"
OUTPUT_HYBRID = REPORT_DIR / "v2_hybrid_signal_results.csv"
OUTPUT_INTERACTIONS = REPORT_DIR / "v2_sector_interaction_results.csv"
OUTPUT_CONFIG = REPORT_DIR / "v2_candidate_config.json"
OUTPUT_SUMMARY = REPORT_DIR / "v2_research_summary.txt"

TOP_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
AGREEMENT_PAIRS = ((0.02, 0.02), (0.01, 0.01), (0.01, 0.005), (0.005, 0.01), (0.005, 0.005))
INTERACTION_SPECS = {
    "stock_sector_ret20_x_sector_strength20": (
        "stock_vs_sector_ret_20d",
        "sector_strength_vs_nifty_20d",
    ),
    "stock_sector_ret60_x_sector_strength60": (
        "stock_vs_sector_ret_60d",
        "sector_strength_vs_nifty_60d",
    ),
    "rs_nifty20_x_sector_strength20": (
        "rs_vs_nifty_20d",
        "sector_strength_vs_nifty_20d",
    ),
}


@dataclass(frozen=True, slots=True)
class V2Results:
    """Collected V2 experiment outputs."""

    ablation: pd.DataFrame
    hybrid: pd.DataFrame
    interactions: pd.DataFrame
    candidate: dict[str, Any]


def main() -> None:
    """Run validation-only V2 continuation research."""

    ensure_probability_dirs()
    stock_specific_variant.TOP_FRACTIONS = TOP_FRACTIONS
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    baseline = _frozen_features()
    market_removed = _ordered_remove(
        baseline, NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES
    )
    all_removed = _ordered_remove(
        baseline,
        NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES | SECTOR_LEVEL_FEATURES,
    )
    market_only_features = [feature for feature in baseline if feature not in market_removed]
    stock_specific_features = [feature for feature in baseline if feature not in all_removed]
    _assert_counts(baseline, market_only_features, stock_specific_features)

    common = _common_evaluation_rows(pre_final, baseline)
    folds = _embargo_folds(common)
    feature_sets = [
        FeatureSet("a_full_77", "A. Full 77 features", baseline),
        FeatureSet("b_remove_market_55", "B. Remove market-wide only", market_only_features),
        FeatureSet(
            "c_stock_specific_49", "C. Remove market-wide + sector-level", stock_specific_features
        ),
    ]

    classifier_predictions = []
    for feature_set in feature_sets:
        print(f"[experiment-1 classifier] {feature_set.name}", flush=True)
        classifier_predictions.append(_classifier_walk_forward(common, folds, feature_set))
    classifier_predictions = pd.concat(classifier_predictions, ignore_index=True)
    ablation = _v2_classifier_report(classifier_predictions, feature_sets)
    ablation.to_csv(OUTPUT_ABLATION, index=False)

    best_primary = _select_primary(ablation)
    print(f"[experiment-2 binary-buy 77] agreement with {best_primary.name}", flush=True)
    primary_predictions = classifier_predictions[
        classifier_predictions["FeatureSet"].eq(best_primary.name)
    ].copy()
    buy_77 = _binary_buy_walk_forward(common, folds, feature_sets[0])
    hybrid = _hybrid_agreement_report(primary_predictions, buy_77)
    hybrid.to_csv(OUTPUT_HYBRID, index=False)

    print("[experiment-3 sector interactions]", flush=True)
    interaction_dataset = _add_interactions(common)
    interaction_base = best_primary.features
    interaction_features = [*interaction_base, *INTERACTION_SPECS.keys()]
    interaction_sets = [
        FeatureSet(best_primary.name, f"{best_primary.display_name} baseline", interaction_base),
        FeatureSet(
            f"{best_primary.name}_sector_interactions",
            f"{best_primary.display_name} + 3 sector interactions",
            interaction_features,
        ),
    ]
    interaction_predictions = []
    for feature_set in interaction_sets:
        print(f"[interaction classifier] {feature_set.name}", flush=True)
        interaction_predictions.append(
            _classifier_walk_forward(interaction_dataset, folds, feature_set)
        )
    interactions = _v2_classifier_report(pd.concat(interaction_predictions), interaction_sets)
    interactions.to_csv(OUTPUT_INTERACTIONS, index=False)

    candidate = _candidate_config(
        baseline=baseline,
        market_only_features=market_only_features,
        stock_specific_features=stock_specific_features,
        best_primary=best_primary,
        ablation=ablation,
        hybrid=hybrid,
        interactions=interactions,
    )
    OUTPUT_CONFIG.write_text(json.dumps(_json_safe(candidate), indent=2), encoding="utf-8")
    OUTPUT_SUMMARY.write_text(_summary(candidate, ablation, hybrid, interactions), encoding="utf-8")

    print(
        json.dumps(
            {
                "v2_market_vs_sector_ablation": str(OUTPUT_ABLATION),
                "v2_hybrid_signal_results": str(OUTPUT_HYBRID),
                "v2_sector_interaction_results": str(OUTPUT_INTERACTIONS),
                "v2_candidate_config": str(OUTPUT_CONFIG),
                "v2_research_summary": str(OUTPUT_SUMMARY),
                "frozen_artifacts_modified": False,
                "final_test_status": "not_used",
            },
            indent=2,
        )
    )


def _ordered_remove(features: list[str], remove: set[str]) -> list[str]:
    missing = sorted(remove.difference(features))
    if missing:
        raise AssertionError(f"Requested features missing from frozen list: {missing}")
    return [feature for feature in features if feature in remove]


def _assert_counts(
    baseline: list[str],
    market_only_features: list[str],
    stock_specific_features: list[str],
) -> None:
    if len(baseline) != 77:
        raise AssertionError(f"Expected A=77 features, got {len(baseline)}")
    if len(market_only_features) != 55:
        raise AssertionError(f"Expected B=55 features, got {len(market_only_features)}")
    if len(stock_specific_features) != 49:
        raise AssertionError(f"Expected C=49 features, got {len(stock_specific_features)}")


def _v2_classifier_report(
    predictions: pd.DataFrame, feature_sets: list[FeatureSet]
) -> pd.DataFrame:
    base = _classifier_comparison(predictions, feature_sets)
    stability = []
    for feature_set in feature_sets:
        frame = predictions[predictions["FeatureSet"].eq(feature_set.name)].copy()
        for fold, group in frame.groupby("fold"):
            stability.append(_stability_row(group, feature_set, "fold", fold))
        for year, group in frame.groupby(frame["Date"].dt.year):
            stability.append(_stability_row(group, feature_set, "year", int(year)))
    return pd.concat([base, pd.DataFrame(stability)], ignore_index=True)


def _stability_row(
    frame: pd.DataFrame,
    feature_set: FeatureSet,
    scope: str,
    value: Any,
) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, log_loss

    actual_buy = (frame["target_label"] == 1).astype(int)
    probabilities = frame[["p_underperform", "p_neutral", "p_outperform"]]
    y_true = frame["target_label"].map({-1: 0, 0: 1, 1: 2})
    top1 = _top_fraction(frame, "p_outperform", 0.01)
    return {
        "Experiment": f"{scope}_stability",
        "FeatureSet": feature_set.name,
        "FeatureSetName": feature_set.display_name,
        "FeatureCount": len(feature_set.features),
        "Scope": scope,
        "ScopeValue": value,
        "Rows": int(len(frame)),
        "LogLoss": float(log_loss(y_true, probabilities, labels=[0, 1, 2])),
        "Brier": float(((frame["p_outperform"] - actual_buy) ** 2).mean()),
        "PRAUC": float(average_precision_score(actual_buy, frame["p_outperform"])),
        "MeanIC": _ic_summary(frame, "p_outperform")["MeanIC"],
        "Top1Precision": float(top1["buy_target"].mean()),
        "Top1AvgExcess": _safe_float(top1["excess_return"].mean()),
        "Top1MedianExcess": _safe_float(top1["excess_return"].median()),
    }


def _select_primary(ablation: pd.DataFrame) -> FeatureSet:
    candidates = ablation[ablation["Experiment"].eq("3class_classifier")].copy()
    candidates["SelectionScore"] = (
        candidates["PRAUC"].rank(ascending=False)
        + candidates["MeanIC"].rank(ascending=False)
        + candidates["Sharpe"].rank(ascending=False)
        + candidates["LogLoss"].rank(ascending=True)
    )
    row = candidates.sort_values(["SelectionScore", "PRAUC"], ascending=[True, False]).iloc[0]
    features = _frozen_features()
    if row["FeatureSet"] == "a_full_77":
        selected = features
    elif row["FeatureSet"] == "b_remove_market_55":
        selected = [
            feature
            for feature in features
            if feature not in (NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES)
        ]
    else:
        selected = [
            feature
            for feature in features
            if feature
            not in (
                NIFTY_FEATURES
                | MARKET_REGIME_FEATURES
                | MARKET_BREADTH_FEATURES
                | SECTOR_LEVEL_FEATURES
            )
        ]
    return FeatureSet(str(row["FeatureSet"]), str(row["FeatureSetName"]), selected)


def _hybrid_agreement_report(primary: pd.DataFrame, buy: pd.DataFrame) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = primary[
        key
        + [
            "FeatureSet",
            "FeatureSetName",
            "p_outperform",
            "score_percentile",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "buy_target",
            "market_regime_label",
        ]
    ].merge(
        buy[key + ["p_buy", "score_percentile"]].rename(
            columns={"score_percentile": "binary_buy_percentile"}
        ),
        on=key,
        how="inner",
    )
    base_rate = float(merged["buy_target"].mean())
    rows = []
    for primary_top, buy_top in AGREEMENT_PAIRS:
        mask = (merged["score_percentile"] >= 1 - primary_top) & (
            merged["binary_buy_percentile"] >= 1 - buy_top
        )
        selected = merged[mask].copy()
        rows.append(_hybrid_row(selected, base_rate, "overall", "all", primary_top, buy_top))
        for fold, group in merged.groupby("fold"):
            group_mask = (group["score_percentile"] >= 1 - primary_top) & (
                group["binary_buy_percentile"] >= 1 - buy_top
            )
            rows.append(
                _hybrid_row(group[group_mask], base_rate, "fold", fold, primary_top, buy_top)
            )
        for year, group in merged.groupby(merged["Date"].dt.year):
            group_mask = (group["score_percentile"] >= 1 - primary_top) & (
                group["binary_buy_percentile"] >= 1 - buy_top
            )
            rows.append(
                _hybrid_row(group[group_mask], base_rate, "year", int(year), primary_top, buy_top)
            )
    return pd.DataFrame(rows)


def _hybrid_row(
    frame: pd.DataFrame,
    base_rate: float,
    scope: str,
    value: Any,
    primary_top: float,
    buy_top: float,
) -> dict[str, Any]:
    precision = _safe_float(frame["buy_target"].mean()) if not frame.empty else None
    return {
        "Rule": f"primary_top_{_pct(primary_top)}_binary77_top_{_pct(buy_top)}",
        "PrimaryTop": primary_top,
        "Binary77Top": buy_top,
        "Scope": scope,
        "ScopeValue": value,
        "Count": int(len(frame)),
        "Precision": precision,
        "Lift": _safe_float(precision / base_rate) if precision is not None and base_rate else None,
        "AverageExcessReturn": (
            _safe_float(frame["excess_return"].mean()) if not frame.empty else None
        ),
        "MedianExcessReturn": (
            _safe_float(frame["excess_return"].median()) if not frame.empty else None
        ),
        "NiftyWinRate": (
            _safe_float((frame["excess_return"] > 0).mean()) if not frame.empty else None
        ),
    }


def _pct(value: float) -> str:
    return str(value * 100).replace(".", "p").rstrip("0").rstrip("p")


def _add_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for name, (left, right) in INTERACTION_SPECS.items():
        output[name] = output[left] * output[right]
    return output


def _candidate_config(
    *,
    baseline: list[str],
    market_only_features: list[str],
    stock_specific_features: list[str],
    best_primary: FeatureSet,
    ablation: pd.DataFrame,
    hybrid: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, Any]:
    interaction_keep = _interaction_decision(interactions, best_primary.name)
    return {
        "version": "natip-v2-validation-research",
        "production_frozen_77_model_modified": False,
        "final_test_used_for_selection": False,
        "feature_counts": {
            "A_full_77": len(baseline),
            "B_remove_market_wide_only": len(market_only_features),
            "C_remove_market_and_sector_level": len(stock_specific_features),
        },
        "selected_primary_candidate": {
            "feature_set": best_primary.name,
            "feature_count": len(best_primary.features),
            "selection_basis": "walk-forward validation only",
        },
        "binary_buy_confirmation": "keep 77-feature binary BUY for hybrid agreement research",
        "xgbranker_diagnostic": "keep 77-feature XGBRanker from prior evidence unless separately retested",
        "sector_interactions": interaction_keep,
        "best_hybrid_overall": _best_hybrid(hybrid),
        "ablation_core_rows": ablation[ablation["Experiment"].eq("3class_classifier")].to_dict(
            orient="records"
        ),
    }


def _interaction_decision(interactions: pd.DataFrame, base_name: str) -> dict[str, Any]:
    core = interactions[interactions["Experiment"].eq("3class_classifier")].set_index("FeatureSet")
    interaction_name = f"{base_name}_sector_interactions"
    if interaction_name not in core.index:
        return {"keep": False, "reason": "interaction variant missing"}
    base = core.loc[base_name]
    variant = core.loc[interaction_name]
    improved = (
        variant["PRAUC"] >= base["PRAUC"]
        and variant["MeanIC"] >= base["MeanIC"]
        and variant["Sharpe"] >= base["Sharpe"]
    )
    return {
        "keep": bool(improved),
        "base_feature_set": base_name,
        "interaction_feature_set": interaction_name,
        "features": list(INTERACTION_SPECS.keys()),
        "reason": "kept only if PR-AUC, Mean IC, and Sharpe all improve on validation",
        "base": base.to_dict(),
        "variant": variant.to_dict(),
    }


def _best_hybrid(hybrid: pd.DataFrame) -> dict[str, Any]:
    overall = hybrid[hybrid["Scope"].eq("overall")].copy()
    if overall.empty:
        return {}
    return (
        overall.sort_values(
            ["Precision", "MedianExcessReturn", "AverageExcessReturn", "Count"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )


def _summary(
    candidate: dict[str, Any],
    ablation: pd.DataFrame,
    hybrid: pd.DataFrame,
    interactions: pd.DataFrame,
) -> str:
    core = ablation[ablation["Experiment"].eq("3class_classifier")]
    hybrid_overall = hybrid[hybrid["Scope"].eq("overall")]
    interaction_core = interactions[interactions["Experiment"].eq("3class_classifier")]
    return "\n".join(
        [
            "NATIP V2 Market/Sector/Hybrid Research",
            "======================================",
            "",
            "Production impact: NONE. Frozen 77-feature production model was not modified.",
            "Final test usage: NOT USED.",
            "",
            "Experiment 1 core classifier rows:",
            core.to_string(index=False),
            "",
            "Experiment 2 overall hybrid agreement rows:",
            hybrid_overall.to_string(index=False),
            "",
            "Experiment 3 sector interaction core rows:",
            interaction_core.to_string(index=False),
            "",
            f"Selected primary candidate: {candidate['selected_primary_candidate']}",
            f"Best hybrid: {candidate['best_hybrid_overall']}",
            f"Sector interactions decision: {candidate['sector_interactions'].get('keep')}",
        ]
    )


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
