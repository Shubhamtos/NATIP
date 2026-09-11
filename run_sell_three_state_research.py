"""Research SELL/downside models and produce frozen three-state NATIP rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.probability.config import DATA_DIR, MODEL_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_final_signal_threshold_research import CLEAN_DEMERGER_DATASET, _prepare_dataset, _pruned_features
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS, _embargo_folds
from run_ranker_binary_signal_research import (
    _base_prediction_frame,
    _fit_calibration_split,
    _fit_classifier,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)

OUTPUT_MODEL_COMPARISON = REPORT_DIR / "sell_model_comparison.csv"
OUTPUT_TAIL = REPORT_DIR / "sell_tail_analysis.csv"
OUTPUT_META = REPORT_DIR / "sell_meta_signal_analysis.csv"
OUTPUT_RULES = REPORT_DIR / "final_three_state_rules.json"
OUTPUT_REPORT = REPORT_DIR / "three_state_validation_report.txt"
SELL_ARTIFACT_PATH = MODEL_DIR / "frozen_downside_sell_v1.joblib"

TOP_FRACTIONS = [0.20, 0.10, 0.05, 0.02, 0.01]
CONFIG_HASH = "11edc1f63c08f8b20d0448c34d44c201f6088caca0898de4576bf6f47a60b4df"
RULES_HASH = "cee5662fb707f4eb2bcbf9fdcab19555a8e80f2185468c8f2bd76e64035e2c27"


@dataclass(frozen=True, slots=True)
class SellSpec:
    """SELL model research spec."""

    model_name: str
    target: str


def main() -> None:
    """Run validation-only SELL research and freeze selected three-state rules."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    dataset = _add_downside_targets(dataset)
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)
    features = _pruned_features()

    print("[sell] primary/buy/ranker OOF diagnostics", flush=True)
    diagnostics = _diagnostic_oof(pre_final, folds, features)

    model_predictions = []
    comparison_rows = []
    tail_rows = []
    for spec in _sell_specs():
        print(f"[sell] {spec.model_name} {spec.target}", flush=True)
        predictions = _sell_oof(pre_final, folds, features, spec)
        model_predictions.append(predictions)
        comparison_rows.append(_model_metrics(predictions, spec))
        tail_rows.extend(_tail_rows(predictions, spec))

    comparison = pd.DataFrame(comparison_rows)
    tail = pd.DataFrame(tail_rows)
    meta = _meta_signal_analysis(diagnostics, model_predictions)
    selected = _select_sell_rule(meta)
    rules = _three_state_rules(selected, comparison, features)

    comparison.to_csv(OUTPUT_MODEL_COMPARISON, index=False)
    tail.to_csv(OUTPUT_TAIL, index=False)
    meta.to_csv(OUTPUT_META, index=False)
    OUTPUT_RULES.write_text(json.dumps(_json_safe(rules), indent=2), encoding="utf-8")
    _fit_selected_sell_artifact(dataset, features, rules)
    OUTPUT_REPORT.write_text(_report_text(rules, comparison, meta), encoding="utf-8")

    print(
        json.dumps(
            {
                "sell_model_comparison": str(OUTPUT_MODEL_COMPARISON),
                "sell_tail_analysis": str(OUTPUT_TAIL),
                "sell_meta_signal_analysis": str(OUTPUT_META),
                "final_three_state_rules": str(OUTPUT_RULES),
                "three_state_validation_report": str(OUTPUT_REPORT),
                "sell_artifact": str(SELL_ARTIFACT_PATH),
            },
            indent=2,
        )
    )


def _sell_specs() -> list[SellSpec]:
    return [
        SellSpec(model, target)
        for target in ("sell_target_a", "sell_target_b", "sell_target_c", "sell_target_combined")
        for model in ("xgboost", "lightgbm", "random_forest")
    ]


def _add_downside_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy().sort_values(["symbol", "Date"])
    output["future_20d_max_drawdown"] = _future_max_drawdown(output)
    output["sell_target_a"] = (output["future_stock_return"] < -0.05).astype(int)
    output["sell_target_b"] = (
        (output["future_stock_return"] < -0.05) & (output["excess_return"] < 0)
    ).astype(int)
    output["sell_target_c"] = (output["future_20d_max_drawdown"] < -0.08).astype(int)
    output["sell_target_combined"] = (
        (output["sell_target_a"].eq(1) & output["sell_target_b"].eq(1)) | output["sell_target_c"].eq(1)
    ).astype(int)
    return output.dropna(subset=["future_20d_max_drawdown"]).reset_index(drop=True)


def _future_max_drawdown(dataset: pd.DataFrame) -> pd.Series:
    pieces = []
    for _, group in dataset.sort_values(["symbol", "Date"]).groupby("symbol", sort=False):
        future_min = group["Close"].shift(-1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).min().shift(
            -(HORIZON_DAYS - 1)
        )
        pieces.append(future_min / group["Close"] - 1)
    return pd.concat(pieces).sort_index()


def _diagnostic_oof(dataset: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        primary, primary_medians = _fit_classifier(train, features, target="label")
        primary_prob = primary.predict_proba(_transform_with_medians(validation, features, primary_medians))
        fit_frame, calibration_frame = _fit_calibration_split(train)
        buy, buy_medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
        cal_prob = _positive_probability(buy, _transform_with_medians(calibration_frame, features, buy_medians))
        buy_raw = _positive_probability(buy, _transform_with_medians(validation, features, buy_medians))
        platt = _fit_platt(cal_prob, calibration_frame["buy_target"])
        ranker, ranker_medians = _fit_ranker(train, features)
        output = _base_prediction_frame(validation, fold["fold"])
        output["label"] = validation["label"].astype(int).to_numpy()
        output["p_outperform"] = primary_prob[:, LABEL_TO_CLASS[1]]
        output["p_underperform"] = primary_prob[:, LABEL_TO_CLASS[-1]]
        output["p_margin_under_minus_out"] = output["p_underperform"] - output["p_outperform"]
        output["buy_sigmoid_probability"] = platt.predict_proba(buy_raw.reshape(-1, 1))[:, 1]
        output["ranker_score"] = ranker.predict(_transform_with_medians(validation, features, ranker_medians))
        output["sector_strength_vs_nifty_60d"] = validation["sector_strength_vs_nifty_60d"].to_numpy()
        output["future_20d_max_drawdown"] = validation["future_20d_max_drawdown"].to_numpy()
        output["market_regime_label"] = validation["market_regime_label"].to_numpy()
        for column in (
            "p_underperform",
            "p_margin_under_minus_out",
            "buy_sigmoid_probability",
            "ranker_score",
            "sector_strength_vs_nifty_60d",
        ):
            ascending = column in ("buy_sigmoid_probability", "ranker_score", "sector_strength_vs_nifty_60d")
            output[f"{column}_risk_percentile"] = output.groupby("Date")[column].rank(
                pct=True,
                ascending=ascending,
            )
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _sell_oof(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    spec: SellSpec,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(
            train,
            features,
            target=spec.target,
            model_name=spec.model_name,
            binary=True,
        )
        output = _base_prediction_frame(validation, fold["fold"])
        output["actual"] = validation[spec.target].astype(int).to_numpy()
        output["future_20d_max_drawdown"] = validation["future_20d_max_drawdown"].to_numpy()
        output["probability"] = _positive_probability(model, _transform_with_medians(validation, features, medians))
        output["probability_percentile"] = output.groupby("Date")["probability"].rank(pct=True)
        output["Model"] = spec.model_name
        output["Target"] = spec.target
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _model_metrics(predictions: pd.DataFrame, spec: SellSpec) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

    actual = predictions["actual"].astype(int)
    probability = predictions["probability"].clip(0, 1)
    return {
        "Model": spec.model_name,
        "Target": spec.target,
        "Rows": int(len(predictions)),
        "BaseRate": float(actual.mean()),
        "PRAUC": float(average_precision_score(actual, probability)),
        "LogLoss": float(log_loss(actual, probability, labels=[0, 1])),
        "Brier": float(brier_score_loss(actual, probability)),
    }


def _tail_rows(predictions: pd.DataFrame, spec: SellSpec) -> list[dict[str, Any]]:
    rows = []
    for fraction in TOP_FRACTIONS:
        selected = predictions[predictions["probability_percentile"] >= 1 - fraction]
        rows.append(
            {
                "Model": spec.model_name,
                "Target": spec.target,
                "Percentile": f"Top {int(fraction * 100)}%",
                **_sell_metrics(selected),
            }
        )
        for fold, group in selected.groupby("fold"):
            rows.append(
                {
                    "Model": spec.model_name,
                    "Target": spec.target,
                    "Percentile": f"Top {int(fraction * 100)}%",
                    "Scope": "fold",
                    "Group": fold,
                    **_sell_metrics(group),
                }
            )
        for year, group in selected.groupby(selected["Date"].dt.year):
            rows.append(
                {
                    "Model": spec.model_name,
                    "Target": spec.target,
                    "Percentile": f"Top {int(fraction * 100)}%",
                    "Scope": "year",
                    "Group": year,
                    **_sell_metrics(group),
                }
            )
    return rows


def _meta_signal_analysis(diagnostics: pd.DataFrame, sell_predictions: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    key = ["Date", "symbol", "fold"]
    for predictions in sell_predictions:
        merged = diagnostics.merge(
            predictions[key + ["probability", "probability_percentile", "Model", "Target"]],
            on=key,
            how="inner",
        )
        model = str(merged["Model"].iloc[0])
        target = str(merged["Target"].iloc[0])
        for sell_fraction in TOP_FRACTIONS:
            base = merged["probability_percentile"] >= 1 - sell_fraction
            _append_meta(rows, merged, base, model, target, f"SELL_PROB_TOP_{int(sell_fraction*100)}")
            combo = (
                base
                & (merged["p_underperform_risk_percentile"] >= 0.80)
                & (merged["buy_sigmoid_probability_risk_percentile"] >= 0.80)
            )
            _append_meta(rows, merged, combo, model, target, f"SELL_PROB_TOP_{int(sell_fraction*100)}_UNDER80_LOWBUY80")
            combo_rank = combo & (merged["ranker_score_risk_percentile"] >= 0.80)
            _append_meta(
                rows,
                merged,
                combo_rank,
                model,
                target,
                f"SELL_PROB_TOP_{int(sell_fraction*100)}_UNDER80_LOWBUY80_RANKBOTTOM80",
            )
            combo_margin = combo & (merged["p_margin_under_minus_out_risk_percentile"] >= 0.80)
            _append_meta(
                rows,
                merged,
                combo_margin,
                model,
                target,
                f"SELL_PROB_TOP_{int(sell_fraction*100)}_UNDER80_LOWBUY80_MARGIN80",
            )
    return pd.DataFrame(rows)


def _append_meta(
    rows: list[dict[str, Any]],
    merged: pd.DataFrame,
    mask: pd.Series,
    model: str,
    target: str,
    rule: str,
) -> None:
    selected = merged[mask].copy()
    metrics = _sell_metrics(selected)
    fold = selected.groupby("fold").apply(lambda g: g["future_stock_return"].mean(), include_groups=False) if not selected.empty else pd.Series(dtype=float)
    year = selected.groupby(selected["Date"].dt.year).apply(lambda g: g["future_stock_return"].mean(), include_groups=False) if not selected.empty else pd.Series(dtype=float)
    rows.append(
        {
            "Model": model,
            "Target": target,
            "MetaRule": rule,
            **metrics,
            "FoldsNegativeAvgStockReturnPct": float((fold < 0).mean()) if not fold.empty else None,
            "YearsNegativeAvgStockReturnPct": float((year < 0).mean()) if not year.empty else None,
            "ValidationStatus": _meta_status(metrics, fold, year),
        }
    )


def _sell_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "Scope": "overall",
            "Group": "all",
            "Count": 0,
            "Precision": None,
            "AverageFutureStockReturn": None,
            "MedianFutureStockReturn": None,
            "AverageFutureExcessReturn": None,
            "MedianFutureExcessReturn": None,
            "DownsideHitRate": None,
            "MaxDrawdownHitRate": None,
        }
    return {
        "Scope": "overall",
        "Group": "all",
        "Count": int(len(frame)),
        "Precision": float(frame.get("actual", frame["future_stock_return"] < -0.05).mean()),
        "AverageFutureStockReturn": float(frame["future_stock_return"].mean()),
        "MedianFutureStockReturn": float(frame["future_stock_return"].median()),
        "AverageFutureExcessReturn": float(frame["excess_return"].mean()),
        "MedianFutureExcessReturn": float(frame["excess_return"].median()),
        "DownsideHitRate": float((frame["future_stock_return"] < -0.05).mean()),
        "MaxDrawdownHitRate": float((frame["future_20d_max_drawdown"] < -0.08).mean()),
    }


def _meta_status(metrics: dict[str, Any], fold: pd.Series, year: pd.Series) -> str:
    if (
        (metrics.get("Count") or 0) >= 300
        and (metrics.get("AverageFutureStockReturn") or 1) < 0
        and (metrics.get("MedianFutureStockReturn") or 1) < 0
        and (metrics.get("AverageFutureExcessReturn") or 1) < 0
        and (metrics.get("DownsideHitRate") or 0) >= 0.35
        and (float((fold < 0).mean()) if not fold.empty else 0) >= 0.5
        and (float((year < 0).mean()) if not year.empty else 0) >= 0.5
    ):
        return "SELL_VALIDATION_CANDIDATE"
    return "SELL_RULE_NOT_VALIDATED"


def _select_sell_rule(meta: pd.DataFrame) -> dict[str, Any]:
    candidates = meta[meta["ValidationStatus"] == "SELL_VALIDATION_CANDIDATE"].copy()
    if candidates.empty:
        best = meta.sort_values(["AverageFutureStockReturn", "MedianFutureStockReturn"]).head(1)
        return {"validated": False, "selected": best.iloc[0].to_dict(), "status": "SELL_RULE_NOT_VALIDATED"}
    candidates = candidates.sort_values(
        ["Precision", "AverageFutureStockReturn", "MedianFutureStockReturn"],
        ascending=[False, True, True],
    )
    return {"validated": True, "selected": candidates.iloc[0].to_dict(), "status": "SELL_VALIDATED"}


def _three_state_rules(selected: dict[str, Any], comparison: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    chosen = selected["selected"]
    return {
        "rules_version": "natip-three-state-v1.0",
        "config_hash": CONFIG_HASH,
        "buy_rules_hash": RULES_HASH,
        "feature_set": "Frozen Clean Pruned Advanced",
        "feature_list": features,
        "buy": {
            "rule_id": "FROZEN_BUY_PRUNED_TOP_1PCT_AND_BINARY_SIGMOID_TOP_1PCT",
            "p_outperform_percentile_min": 0.99,
            "buy_sigmoid_percentile_min": 0.99,
        },
        "sell": {
            "validated": bool(selected["validated"]),
            "status": selected["status"],
            "model": chosen.get("Model"),
            "target": chosen.get("Target"),
            "meta_rule": chosen.get("MetaRule"),
            "probability_percentile_min": _prob_from_rule(str(chosen.get("MetaRule"))),
            "requires_p_underperform_risk_percentile_min": 0.80 if "UNDER80" in str(chosen.get("MetaRule")) else None,
            "requires_low_buy_risk_percentile_min": 0.80 if "LOWBUY80" in str(chosen.get("MetaRule")) else None,
            "requires_ranker_bottom_risk_percentile_min": 0.80 if "RANKBOTTOM80" in str(chosen.get("MetaRule")) else None,
            "requires_margin_risk_percentile_min": 0.80 if "MARGIN80" in str(chosen.get("MetaRule")) else None,
            "validation_evidence": chosen,
        },
        "accumulate": {
            "rule_id": "DEFAULT_BETWEEN_BUY_AND_SELL",
            "description": "Return ACCUMULATE for all observations that do not satisfy frozen BUY or validated SELL.",
        },
        "model_comparison_summary": comparison.to_dict(orient="records"),
    }


def _prob_from_rule(rule: str) -> float | None:
    for value in (20, 10, 5, 2, 1):
        if f"TOP_{value}" in rule:
            return 1 - value / 100
    return None


def _fit_selected_sell_artifact(dataset: pd.DataFrame, features: list[str], rules: dict[str, Any]) -> None:
    train = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    train = _add_downside_targets(train)
    model, medians = _fit_classifier(
        train,
        features,
        target=rules["sell"]["target"],
        model_name=rules["sell"]["model"],
        binary=True,
    )
    artifact = {
        "artifact_version": "natip-frozen-downside-sell-v1.0",
        "config_hash": CONFIG_HASH,
        "buy_rules_hash": RULES_HASH,
        "validated_for_live_sell": rules["sell"]["validated"],
        "three_state_rules": rules,
        "feature_list": features,
        "sell_model": model,
        "sell_medians": medians,
    }
    from joblib import dump

    dump(artifact, SELL_ARTIFACT_PATH)


def _report_text(rules: dict[str, Any], comparison: pd.DataFrame, meta: pd.DataFrame) -> str:
    selected = rules["sell"]["validation_evidence"]
    return "\n".join(
        [
            "NATIP Three-State Validation Report",
            "",
            "BUY architecture remains frozen. No BUY threshold was changed.",
            f"SELL status: {rules['sell']['status']}",
            f"Selected SELL model: {rules['sell']['model']} / {rules['sell']['target']}",
            f"Selected SELL meta-rule: {rules['sell']['meta_rule']}",
            "",
            "Selected SELL validation evidence:",
            json.dumps(_json_safe(selected), indent=2),
            "",
            "Model comparison:",
            comparison.to_string(index=False),
            "",
            "Top meta candidates:",
            meta.sort_values(["ValidationStatus", "AverageFutureStockReturn"]).head(12).to_string(index=False),
            "",
            "Final live recommendation layer returns exactly BUY, ACCUMULATE, or SELL.",
        ]
    )


def _fold_frames(dataset: pd.DataFrame, fold: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return dataset[dataset["Date"].isin(fold["train_dates"])].copy(), dataset[
        dataset["Date"].isin(fold["validation_dates"])
    ].copy()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


if __name__ == "__main__":
    main()
