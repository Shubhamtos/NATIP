"""Validation-only V2 BUY research for 55-feature classifier + 77-feature binary BUY."""

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
    _fit_calibration_split,
    _fit_classifier,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)
from run_stock_specific_feature_variant import (
    MARKET_BREADTH_FEATURES,
    MARKET_REGIME_FEATURES,
    NIFTY_FEATURES,
    SECTOR_LEVEL_FEATURES,
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _ic_summary,
    _prepare_dataset,
    _safe_float,
)

CLEAN_DEMERGER_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"

OUTPUT_GRID = REPORT_DIR / "v2_55_binary_agreement_grid.csv"
OUTPUT_MARGIN = REPORT_DIR / "v2_margin_filter_analysis.csv"
OUTPUT_ARCHITECTURE = REPORT_DIR / "v1_vs_v2_buy_architecture.csv"
OUTPUT_RANKER = REPORT_DIR / "v2_ranker_confirmation.csv"
OUTPUT_CANDIDATE = REPORT_DIR / "v2_high_precision_buy_candidate.json"
OUTPUT_SUMMARY = REPORT_DIR / "v2_buy_research_summary.txt"

AGREEMENT_GRID = (
    (0.05, 0.05),
    (0.02, 0.02),
    (0.02, 0.01),
    (0.01, 0.02),
    (0.01, 0.01),
    (0.01, 0.005),
    (0.005, 0.01),
    (0.005, 0.005),
    (0.005, 0.0025),
    (0.0025, 0.005),
    (0.0025, 0.0025),
)
RANKER_CONFIRMATION_TOPS = (0.20, 0.10, 0.05)
MIN_PROMISING_COUNT = 100


@dataclass(frozen=True, slots=True)
class Architecture:
    """Classifier/Binary architecture under test."""

    name: str
    display_name: str
    classifier: FeatureSet


def main() -> None:
    """Run V2-B high-confidence BUY validation research."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_77 = _frozen_features()
    features_55 = _remove_market_wide(features_77)
    features_49 = _remove_market_and_sector(features_77)
    _assert_feature_counts(features_77, features_55, features_49)
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    feature_sets = {
        "v1_77": FeatureSet("v1_77", "V1 77-feature classifier", features_77),
        "v2_b_55": FeatureSet("v2_b_55", "V2-B 55-feature classifier", features_55),
        "v2_c_49": FeatureSet("v2_c_49", "V2-C 49-feature classifier diagnostic", features_49),
    }
    architectures = [
        Architecture(
            "v1_77_plus_binary77", "V1: 77 classifier + 77 Binary BUY", feature_sets["v1_77"]
        ),
        Architecture(
            "v2_b_55_plus_binary77", "V2-B: 55 classifier + 77 Binary BUY", feature_sets["v2_b_55"]
        ),
        Architecture(
            "v2_c_49_plus_binary77",
            "V2-C diagnostic: 49 classifier + 77 Binary BUY",
            feature_sets["v2_c_49"],
        ),
    ]

    classifier_frames = []
    for feature_set in feature_sets.values():
        print(f"[classifier-oof] {feature_set.name}", flush=True)
        classifier_frames.append(_classifier_oof(common, folds, feature_set))
    classifiers = pd.concat(classifier_frames, ignore_index=True)

    print("[binary-buy-oof] 77-feature Binary BUY", flush=True)
    binary = _binary_buy_77_oof(common, folds, feature_sets["v1_77"])
    print("[ranker-oof] 77-feature XGBRanker", flush=True)
    ranker = _ranker_77_oof(common, folds, feature_sets["v1_77"])

    merged = _merge_all_oof(classifiers, binary, ranker)
    grid = _agreement_grid(merged, architectures)
    grid.to_csv(OUTPUT_GRID, index=False)
    margin = _margin_filter_analysis(merged, grid, architectures)
    margin.to_csv(OUTPUT_MARGIN, index=False)
    architecture = _architecture_comparison(merged, architectures)
    architecture.to_csv(OUTPUT_ARCHITECTURE, index=False)
    ranker_confirmation = _ranker_confirmation(merged, grid, architectures)
    ranker_confirmation.to_csv(OUTPUT_RANKER, index=False)
    candidate = _candidate_payload(grid, margin, architecture, ranker_confirmation, features_55)
    OUTPUT_CANDIDATE.write_text(json.dumps(_json_safe(candidate), indent=2), encoding="utf-8")
    OUTPUT_SUMMARY.write_text(
        _summary(candidate, grid, margin, architecture, ranker_confirmation), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "v2_55_binary_agreement_grid": str(OUTPUT_GRID),
                "v2_margin_filter_analysis": str(OUTPUT_MARGIN),
                "v1_vs_v2_buy_architecture": str(OUTPUT_ARCHITECTURE),
                "v2_ranker_confirmation": str(OUTPUT_RANKER),
                "v2_high_precision_buy_candidate": str(OUTPUT_CANDIDATE),
                "v2_buy_research_summary": str(OUTPUT_SUMMARY),
                "frozen_v1_modified": False,
                "final_test_status": "not_used",
            },
            indent=2,
        )
    )


def _remove_market_wide(features: list[str]) -> list[str]:
    remove = NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES
    return [feature for feature in features if feature not in remove]


def _remove_market_and_sector(features: list[str]) -> list[str]:
    remove = (
        NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES | SECTOR_LEVEL_FEATURES
    )
    return [feature for feature in features if feature not in remove]


def _assert_feature_counts(
    features_77: list[str], features_55: list[str], features_49: list[str]
) -> None:
    if len(features_77) != 77:
        raise AssertionError(f"Expected 77 features, got {len(features_77)}")
    if len(features_55) != 55:
        raise AssertionError(f"Expected 55 features, got {len(features_55)}")
    if len(features_49) != 49:
        raise AssertionError(f"Expected 49 features, got {len(features_49)}")


def _classifier_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], feature_set: FeatureSet
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(train, feature_set.features, target="label")
        probabilities = model.predict_proba(
            _transform_with_medians(validation, feature_set.features, medians)
        )
        output = _base_frame(validation, fold["fold"])
        output["ArchitectureFeatureSet"] = feature_set.name
        output["ArchitectureFeatureSetName"] = feature_set.display_name
        output["FeatureCount"] = len(feature_set.features)
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_margin"] = output["p_outperform"] - output["p_underperform"]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["margin_percentile"] = output.groupby("Date")["classifier_margin"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _binary_buy_77_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], feature_set: FeatureSet
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        fit_frame, calibration_frame = _fit_calibration_split(train)
        model, medians = _fit_classifier(
            fit_frame,
            feature_set.features,
            target="buy_target",
            model_name="xgboost",
            binary=True,
        )
        cal_prob = _positive_probability(
            model, _transform_with_medians(calibration_frame, feature_set.features, medians)
        )
        raw_probability = _positive_probability(
            model, _transform_with_medians(validation, feature_set.features, medians)
        )
        sigmoid = _fit_platt(cal_prob, calibration_frame["buy_target"])
        output = _base_frame(validation, fold["fold"])[["Date", "symbol", "fold"]].copy()
        output["binary_buy_raw_probability"] = raw_probability
        output["binary_buy_sigmoid_probability"] = sigmoid.predict_proba(
            raw_probability.reshape(-1, 1)
        )[:, 1]
        output["binary_buy_percentile"] = output.groupby("Date")[
            "binary_buy_sigmoid_probability"
        ].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _ranker_77_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], feature_set: FeatureSet
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_ranker(train, feature_set.features)
        score = model.predict(_transform_with_medians(validation, feature_set.features, medians))
        output = _base_frame(validation, fold["fold"])[["Date", "symbol", "fold"]].copy()
        output["ranker_score"] = score
        output["ranker_percentile"] = output.groupby("Date")["ranker_score"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        dataset[dataset["Date"].isin(fold["train_dates"])].copy(),
        dataset[dataset["Date"].isin(fold["validation_dates"])].copy(),
    )


def _base_frame(frame: pd.DataFrame, fold: Any) -> pd.DataFrame:
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
    output["fold"] = fold
    return output


def _merge_all_oof(
    classifiers: pd.DataFrame, binary: pd.DataFrame, ranker: pd.DataFrame
) -> pd.DataFrame:
    key = ["Date", "symbol", "fold"]
    merged = classifiers.merge(binary, on=key, how="inner").merge(ranker, on=key, how="inner")
    if len(merged) != len(classifiers):
        raise AssertionError("OOF merge changed classifier row count.")
    return merged


def _agreement_grid(merged: pd.DataFrame, architectures: list[Architecture]) -> pd.DataFrame:
    rows = []
    base_rate = float(merged.drop_duplicates(["Date", "symbol", "fold"])["buy_target"].mean())
    for architecture in architectures:
        frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
        for classifier_top, binary_top in AGREEMENT_GRID:
            selected = _select_agreement(frame, classifier_top, binary_top)
            rows.append(
                _signal_row(
                    selected,
                    base_rate=base_rate,
                    architecture=architecture,
                    rule=_rule_name(classifier_top, binary_top),
                    scope="overall",
                    scope_value="all",
                    classifier_top=classifier_top,
                    binary_top=binary_top,
                )
            )
            rows.extend(_scoped_rows(frame, architecture, classifier_top, binary_top, base_rate))
    return pd.DataFrame(rows)


def _scoped_rows(
    frame: pd.DataFrame,
    architecture: Architecture,
    classifier_top: float,
    binary_top: float,
    base_rate: float,
) -> list[dict[str, Any]]:
    rows = []
    rule = _rule_name(classifier_top, binary_top)
    for fold, group in frame.groupby("fold"):
        rows.append(
            _signal_row(
                _select_agreement(group, classifier_top, binary_top),
                base_rate=base_rate,
                architecture=architecture,
                rule=rule,
                scope="fold",
                scope_value=fold,
                classifier_top=classifier_top,
                binary_top=binary_top,
            )
        )
    for year, group in frame.groupby(frame["Date"].dt.year):
        rows.append(
            _signal_row(
                _select_agreement(group, classifier_top, binary_top),
                base_rate=base_rate,
                architecture=architecture,
                rule=rule,
                scope="year",
                scope_value=int(year),
                classifier_top=classifier_top,
                binary_top=binary_top,
            )
        )
    for regime, group in frame.groupby("market_regime_label"):
        rows.append(
            _signal_row(
                _select_agreement(group, classifier_top, binary_top),
                base_rate=base_rate,
                architecture=architecture,
                rule=rule,
                scope="market_regime",
                scope_value=regime,
                classifier_top=classifier_top,
                binary_top=binary_top,
            )
        )
    return rows


def _select_agreement(
    frame: pd.DataFrame, classifier_top: float, binary_top: float
) -> pd.DataFrame:
    return frame[
        (frame["classifier_percentile"] >= 1 - classifier_top)
        & (frame["binary_buy_percentile"] >= 1 - binary_top)
    ].copy()


def _signal_row(
    frame: pd.DataFrame,
    *,
    base_rate: float,
    architecture: Architecture,
    rule: str,
    scope: str,
    scope_value: Any,
    classifier_top: float,
    binary_top: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    precision = _safe_float(frame["buy_target"].mean()) if not frame.empty else None
    row = {
        "Architecture": architecture.name,
        "ArchitectureName": architecture.display_name,
        "ClassifierFeatureSet": architecture.classifier.name,
        "ClassifierFeatureCount": len(architecture.classifier.features),
        "Rule": rule,
        "ClassifierTop": classifier_top,
        "BinaryTop": binary_top,
        "Scope": scope,
        "ScopeValue": scope_value,
        "Count": int(len(frame)),
        "Precision": precision,
        "BaseRate": base_rate,
        "PrecisionLift": (
            _safe_float(precision / base_rate) if precision is not None and base_rate else None
        ),
        "AverageFutureStockReturn": (
            _safe_float(frame["future_stock_return"].mean()) if not frame.empty else None
        ),
        "MedianFutureStockReturn": (
            _safe_float(frame["future_stock_return"].median()) if not frame.empty else None
        ),
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
    if extra:
        row.update(extra)
    return row


def _margin_filter_analysis(
    merged: pd.DataFrame,
    grid: pd.DataFrame,
    architectures: list[Architecture],
) -> pd.DataFrame:
    rows = []
    base_rate = float(merged.drop_duplicates(["Date", "symbol", "fold"])["buy_target"].mean())
    promising = _promising_rules(grid)
    for item in promising.itertuples(index=False):
        architecture = next(arch for arch in architectures if arch.name == item.Architecture)
        frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
        base_selection = _select_agreement(frame, float(item.ClassifierTop), float(item.BinaryTop))
        if base_selection.empty:
            continue
        cutoffs = (
            base_selection["classifier_margin"].quantile([0.50, 0.60, 0.70, 0.80, 0.90]).dropna()
        )
        for quantile, cutoff in cutoffs.items():
            selected = base_selection[base_selection["classifier_margin"] >= cutoff].copy()
            rows.append(
                _signal_row(
                    selected,
                    base_rate=base_rate,
                    architecture=architecture,
                    rule=f"{item.Rule}_margin_q{int(quantile * 100)}",
                    scope="overall",
                    scope_value="all",
                    classifier_top=float(item.ClassifierTop),
                    binary_top=float(item.BinaryTop),
                    extra={
                        "BaseRule": item.Rule,
                        "MarginQuantile": float(quantile),
                        "MarginCutoff": float(cutoff),
                        "BaseRuleCount": int(item.Count),
                        "BaseRulePrecision": float(item.Precision),
                    },
                )
            )
    return pd.DataFrame(rows)


def _promising_rules(grid: pd.DataFrame) -> pd.DataFrame:
    overall = grid[(grid["Scope"].eq("overall")) & (grid["Count"] >= MIN_PROMISING_COUNT)].copy()
    if overall.empty:
        return overall
    threshold = overall["Precision"].quantile(0.60)
    return overall[overall["Precision"] >= threshold].copy()


def _architecture_comparison(
    merged: pd.DataFrame, architectures: list[Architecture]
) -> pd.DataFrame:
    rows = []
    base_rate = float(merged.drop_duplicates(["Date", "symbol", "fold"])["buy_target"].mean())
    for architecture in architectures:
        frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
        rows.append(_architecture_core_row(frame, architecture))
        for classifier_top, binary_top in AGREEMENT_GRID:
            selected = _select_agreement(frame, classifier_top, binary_top)
            rows.append(
                _signal_row(
                    selected,
                    base_rate=base_rate,
                    architecture=architecture,
                    rule=_rule_name(classifier_top, binary_top),
                    scope="overall",
                    scope_value="all",
                    classifier_top=classifier_top,
                    binary_top=binary_top,
                    extra={"ComparisonType": "tail_agreement"},
                )
            )
    return pd.DataFrame(rows)


def _architecture_core_row(frame: pd.DataFrame, architecture: Architecture) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, log_loss

    actual = frame["buy_target"].astype(int)
    backtest = backtest_top_n_portfolio(frame, top_n=5)
    return {
        "ComparisonType": "classifier_core",
        "Architecture": architecture.name,
        "ArchitectureName": architecture.display_name,
        "ClassifierFeatureSet": architecture.classifier.name,
        "ClassifierFeatureCount": len(architecture.classifier.features),
        "Rows": int(len(frame)),
        "ClassifierBuyLogLoss": float(log_loss(actual, frame["p_outperform"], labels=[0, 1])),
        "ClassifierPRAUC": float(average_precision_score(actual, frame["p_outperform"])),
        "MeanIC": _ic_summary(frame, "p_outperform")["MeanIC"],
        "Sharpe": backtest.get("sharpe"),
        "MaxDrawdown": backtest.get("max_drawdown"),
        "BaseRate": _safe_float(actual.mean()),
    }


def _ranker_confirmation(
    merged: pd.DataFrame,
    grid: pd.DataFrame,
    architectures: list[Architecture],
) -> pd.DataFrame:
    rows = []
    base_rate = float(merged.drop_duplicates(["Date", "symbol", "fold"])["buy_target"].mean())
    strongest = _promising_rules(grid)
    if strongest.empty:
        strongest = (
            grid[grid["Scope"].eq("overall")].sort_values("Precision", ascending=False).head(5)
        )
    for item in strongest.itertuples(index=False):
        architecture = next(arch for arch in architectures if arch.name == item.Architecture)
        frame = merged[merged["ArchitectureFeatureSet"].eq(architecture.classifier.name)].copy()
        base_selection = _select_agreement(frame, float(item.ClassifierTop), float(item.BinaryTop))
        for ranker_top in RANKER_CONFIRMATION_TOPS:
            selected = base_selection[base_selection["ranker_percentile"] >= 1 - ranker_top].copy()
            rows.append(
                _signal_row(
                    selected,
                    base_rate=base_rate,
                    architecture=architecture,
                    rule=f"{item.Rule}_ranker_top_{_pct(ranker_top)}",
                    scope="overall",
                    scope_value="all",
                    classifier_top=float(item.ClassifierTop),
                    binary_top=float(item.BinaryTop),
                    extra={
                        "BaseRule": item.Rule,
                        "RankerTop": ranker_top,
                        "BaseRuleCount": int(item.Count),
                        "BaseRulePrecision": float(item.Precision),
                    },
                )
            )
    return pd.DataFrame(rows)


def _candidate_payload(
    grid: pd.DataFrame,
    margin: pd.DataFrame,
    architecture: pd.DataFrame,
    ranker: pd.DataFrame,
    features_55: list[str],
) -> dict[str, Any]:
    overall = grid[grid["Scope"].eq("overall")].copy()
    viable = overall[
        (overall["Architecture"].eq("v2_b_55_plus_binary77"))
        & (overall["Count"] >= MIN_PROMISING_COUNT)
    ]
    best = _best_rule(viable if not viable.empty else overall)
    margin_v2_b = (
        margin[margin["Architecture"].eq("v2_b_55_plus_binary77")] if not margin.empty else margin
    )
    ranker_v2_b = (
        ranker[ranker["Architecture"].eq("v2_b_55_plus_binary77")] if not ranker.empty else ranker
    )
    margin_best = _best_rule(margin_v2_b) if not margin_v2_b.empty else {}
    ranker_best = _best_rule(ranker_v2_b) if not ranker_v2_b.empty else {}
    return {
        "version": "natip-v2-buy-validation-research",
        "production_v1_frozen_model_modified": False,
        "final_test_used": False,
        "selected_primary_classifier": "V2-B 55-feature classifier",
        "primary_feature_count": len(features_55),
        "binary_buy_confirmation": "Existing 77-feature Binary BUY sigmoid percentile",
        "xgbranker_status": "diagnostic only; retain only if validation precision improves without destroying sample size",
        "best_v2_b_rule": best,
        "best_v2_b_margin_filter_rule": margin_best,
        "best_v2_b_ranker_confirmation_rule": ranker_best,
        "diagnostic_best_margin_filter_rule_any_architecture": (
            _best_rule(margin) if not margin.empty else {}
        ),
        "diagnostic_best_ranker_confirmation_rule_any_architecture": (
            _best_rule(ranker) if not ranker.empty else {}
        ),
        "decision_note": _decision_note(best, margin_best, ranker_best),
        "outputs": {
            "agreement_grid": str(OUTPUT_GRID),
            "margin_filter_analysis": str(OUTPUT_MARGIN),
            "architecture_comparison": str(OUTPUT_ARCHITECTURE),
            "ranker_confirmation": str(OUTPUT_RANKER),
        },
    }


def _best_rule(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return (
        frame.sort_values(
            ["Precision", "MedianExcessReturn", "AverageExcessReturn", "Count"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
        .to_dict()
    )


def _decision_note(
    best: dict[str, Any], margin_best: dict[str, Any], ranker_best: dict[str, Any]
) -> str:
    if not best:
        return "No viable V2-B validation rule found."
    notes = [
        "Candidate selected from walk-forward validation only.",
        "Prefer rules with adequate sample count, positive median excess, and fold/year stability.",
    ]
    if margin_best and margin_best.get("Precision", 0) > best.get("Precision", 0):
        notes.append(
            "Margin filter improved headline precision; inspect sample count/stability before promotion."
        )
    if ranker_best and ranker_best.get("Precision", 0) > best.get("Precision", 0):
        notes.append(
            "Ranker confirmation improved headline precision; inspect sample-size tradeoff."
        )
    return " ".join(notes)


def _summary(
    candidate: dict[str, Any],
    grid: pd.DataFrame,
    margin: pd.DataFrame,
    architecture: pd.DataFrame,
    ranker: pd.DataFrame,
) -> str:
    overall_grid = grid[grid["Scope"].eq("overall")]
    architecture_core = architecture[architecture["ComparisonType"].eq("classifier_core")]
    margin_top = (
        margin.sort_values("Precision", ascending=False).head(10)
        if not margin.empty
        else pd.DataFrame()
    )
    ranker_top = (
        ranker.sort_values("Precision", ascending=False).head(10)
        if not ranker.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "NATIP V2 55-Feature Binary BUY Research",
            "=======================================",
            "",
            "Production impact: NONE. Frozen V1 model/config/rules were not modified.",
            "Final test usage: NOT USED for tuning or selection.",
            "",
            "Architecture core comparison:",
            architecture_core.to_string(index=False),
            "",
            "Overall agreement grid:",
            overall_grid.to_string(index=False),
            "",
            "Top margin-filter rows:",
            margin_top.to_string(index=False) if not margin_top.empty else "No margin rows.",
            "",
            "Top ranker-confirmation rows:",
            ranker_top.to_string(index=False) if not ranker_top.empty else "No ranker rows.",
            "",
            f"Candidate payload: {candidate}",
        ]
    )


def _rule_name(classifier_top: float, binary_top: float) -> str:
    return f"classifier_top_{_pct(classifier_top)}_binary_top_{_pct(binary_top)}"


def _pct(value: float) -> str:
    return f"{value * 100:g}pct".replace(".", "p")


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
