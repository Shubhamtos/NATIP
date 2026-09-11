"""Freeze clean model config and run one-time final-test signal evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_final_signal_threshold_research import (
    CLEAN_DEMERGER_DATASET,
    HORIZON_DAYS,
    TOP_FRACTIONS,
    _bottom,
    _fold_frames,
    _market_regime_label,
    _metrics,
    _non_overlap_metrics,
    _prepare_dataset,
    _pruned_features,
    _top,
)
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _base_prediction_frame,
    _fit_calibration_split,
    _fit_classifier,
    _fit_platt,
    _positive_probability,
    _transform_with_medians,
)

OUTPUT_FINAL = REPORT_DIR / "final_test_signal_results.csv"
OUTPUT_COMPARE = REPORT_DIR / "validation_vs_final_test.csv"
OUTPUT_SELL = REPORT_DIR / "sell_target_research.csv"
OUTPUT_CONFIG = REPORT_DIR / "frozen_model_config.json"
OUTPUT_RULES = REPORT_DIR / "frozen_signal_rules.json"
OUTPUT_READINESS = REPORT_DIR / "deployment_readiness_report.txt"

BUY_RULE_ID = "FROZEN_BUY_PRUNED_TOP_1PCT_AND_BINARY_SIGMOID_TOP_1PCT"
ACCUMULATE_RULE_ID = "FROZEN_ACCUMULATE_PRUNED_TOP_20PCT_BINARY_SIGMOID_ABOVE_50TH_EX_BUY"
SELL_STATUS = "SELL_RULE_NOT_YET_VALIDATED"


def main() -> None:
    """Freeze configs, run final-test evaluation once, and run validation-only SELL research."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    features = _pruned_features()
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    final = dataset[dataset["Date"] >= FINAL_TEST_START].copy()
    folds = _embargo_folds(pre_final)

    config = _frozen_config(dataset, pre_final, final, folds, features)
    rules = _frozen_rules(config)
    OUTPUT_CONFIG.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    OUTPUT_RULES.write_text(json.dumps(_json_safe(rules), indent=2), encoding="utf-8")

    print("[final] one-time BUY/ACCUMULATE evaluation", flush=True)
    final_signals = _final_test_predictions(dataset, final, features)
    final_results = _final_signal_results(final_signals)
    final_results.to_csv(OUTPUT_FINAL, index=False)

    comparison = _validation_vs_final(final_results)
    comparison.to_csv(OUTPUT_COMPARE, index=False)

    print("[sell] validation-only target research", flush=True)
    sell_research = _sell_target_research(pre_final, folds, features)
    sell_research.to_csv(OUTPUT_SELL, index=False)

    OUTPUT_READINESS.write_text(
        _readiness_report(config, rules, final_results, comparison, sell_research),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "final_test_signal_results": str(OUTPUT_FINAL),
                "validation_vs_final_test": str(OUTPUT_COMPARE),
                "sell_target_research": str(OUTPUT_SELL),
                "frozen_model_config": str(OUTPUT_CONFIG),
                "frozen_signal_rules": str(OUTPUT_RULES),
                "deployment_readiness_report": str(OUTPUT_READINESS),
                "final_test_policy": "evaluated once; thresholds not changed",
            },
            indent=2,
        )
    )


def _frozen_config(
    dataset: pd.DataFrame,
    pre_final: pd.DataFrame,
    final: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
) -> dict[str, Any]:
    """Build immutable model config with hashes for silent-change detection."""

    train_dates_for_final, embargo_dates = _final_train_and_embargo_dates(dataset)
    model_hyperparameters = {
        "primary_pruned_xgboost_3class": _xgb_hyperparameters(binary=False),
        "binary_xgboost_buy": _xgb_hyperparameters(binary=True),
        "random_forest_sell_diagnostic": {
            "n_estimators": 80,
            "max_depth": 9,
            "min_samples_leaf": 20,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": 42,
        },
    }
    split_rows = [
        {
            "fold": fold["fold"],
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "embargo_start": fold["embargo_start"],
            "embargo_end": fold["embargo_end"],
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "train_dates": len(fold["train_dates"]),
            "validation_dates": len(fold["validation_dates"]),
        }
        for fold in folds
    ]
    config = {
        "config_version": "natip-clean-pruned-v1.0",
        "source_dataset": str(CLEAN_DEMERGER_DATASET),
        "dataset_rows": int(len(dataset)),
        "pre_final_rows": int(len(pre_final)),
        "final_test_rows": int(len(final)),
        "final_test_start": FINAL_TEST_START.isoformat(),
        "prediction_horizon_trading_days": HORIZON_DAYS,
        "embargo_trading_days": HORIZON_DAYS,
        "feature_list": features,
        "feature_order": {feature: index for index, feature in enumerate(features)},
        "preprocessing": {
            "missing_values": "fit training-fold median only",
            "missing_indicators": "append __missing indicator columns after median imputation",
            "no_global_row_drop_for_beta": True,
            "beta_252d": "kept",
        },
        "model_hyperparameters": model_hyperparameters,
        "calibration": {
            "binary_buy_sigmoid": "Platt LogisticRegression fitted only on latest 20% of training dates inside each fold/final training window",
            "final_calibration_training_policy": "fit primary/binary on pre-final data excluding 20-trading-day final embargo; split latest 20% of train dates for Platt calibration",
            "stored_objects": "not serialized in JSON; object provenance fixed by code/config/version/hash",
        },
        "validation_folds": split_rows,
        "final_training_window": {
            "train_start": min(train_dates_for_final).isoformat(),
            "train_end": max(train_dates_for_final).isoformat(),
            "embargo_start": min(embargo_dates).isoformat(),
            "embargo_end": max(embargo_dates).isoformat(),
            "final_test_start": FINAL_TEST_START.isoformat(),
            "train_trading_days": len(train_dates_for_final),
            "embargo_trading_days": len(embargo_dates),
        },
    }
    config["feature_hash"] = _hash_json(features)
    config["hyperparameter_hash"] = _hash_json(model_hyperparameters)
    config["split_hash"] = _hash_json(split_rows)
    config["config_hash"] = _hash_json(config)
    return config


def _frozen_rules(config: dict[str, Any]) -> dict[str, Any]:
    rules = {
        "rules_version": "natip-clean-signals-v1.0",
        "config_hash": config["config_hash"],
        "buy": {
            "rule_id": BUY_RULE_ID,
            "conditions": [
                "Pruned Advanced P_Outperform daily percentile >= 0.99",
                "Binary XGBoost BUY sigmoid probability daily percentile >= 0.99",
            ],
            "validation_evidence": {
                "count": 775,
                "precision": 0.45806451612903226,
                "average_20d_excess_return": 0.038180026358291624,
                "median_20d_excess_return": 0.0406920111292454,
                "non_overlapping_average_excess_return": 0.050193654816914915,
            },
        },
        "accumulate": {
            "rule_id": ACCUMULATE_RULE_ID,
            "conditions": [
                "Pruned Advanced P_Outperform daily percentile >= 0.80",
                "Binary XGBoost BUY sigmoid probability daily percentile >= 0.50",
                "Exclude observations satisfying BUY rule",
            ],
            "validation_evidence": {
                "average_20d_excess_return": 0.01485871206340852,
                "median_20d_excess_return": 0.0064866315477856,
            },
        },
        "sell": {
            "status": SELL_STATUS,
            "live_rule": None,
        },
        "threshold_freeze_policy": "Do not tune BUY or ACCUMULATE from final-test results.",
    }
    rules["rules_hash"] = _hash_json(rules)
    return rules


def _final_test_predictions(dataset: pd.DataFrame, final: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    train_dates, _ = _final_train_and_embargo_dates(dataset)
    train = dataset[dataset["Date"].isin(train_dates)].copy()

    primary_model, primary_medians = _fit_classifier(train, features, target="label")
    x_final = _transform_with_medians(final, features, primary_medians)
    primary_prob = primary_model.predict_proba(x_final)

    fit_frame, calibration_frame = _fit_calibration_split(train)
    buy_model, buy_medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
    cal_x = _transform_with_medians(calibration_frame, features, buy_medians)
    final_x = _transform_with_medians(final, features, buy_medians)
    cal_prob = _positive_probability(buy_model, cal_x)
    raw_prob = _positive_probability(buy_model, final_x)
    platt = _fit_platt(cal_prob, calibration_frame["buy_target"])

    output = _base_prediction_frame(final, "final_test")
    output["label"] = final["label"].astype(int).to_numpy()
    output["p_outperform"] = primary_prob[:, LABEL_TO_CLASS[1]]
    output["p_underperform"] = primary_prob[:, LABEL_TO_CLASS[-1]]
    output["buy_raw_probability"] = raw_prob
    output["buy_sigmoid_probability"] = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
    output["p_outperform_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
    output["buy_sigmoid_probability_percentile"] = output.groupby("Date")["buy_sigmoid_probability"].rank(pct=True)
    output["Signal"] = "NONE"
    buy_mask = _buy_mask(output)
    accumulate_mask = _accumulate_mask(output) & ~buy_mask
    output.loc[accumulate_mask, "Signal"] = "ACCUMULATE"
    output.loc[buy_mask, "Signal"] = "BUY"
    return output


def _final_signal_results(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in ("BUY", "ACCUMULATE"):
        selected = signals[signals["Signal"] == signal].copy()
        rows.append({"Signal": signal, "Scope": "overall", "Group": "all", **_final_metrics(selected, signal)})
        for year, group in selected.groupby(selected["Date"].dt.year):
            rows.append({"Signal": signal, "Scope": "year", "Group": year, **_final_metrics(group, signal)})
        for regime, group in selected.groupby("market_regime_label"):
            rows.append({"Signal": signal, "Scope": "regime", "Group": regime, **_final_metrics(group, signal)})
        for sector, group in selected.groupby("Sector"):
            if len(group) >= 10:
                rows.append({"Signal": signal, "Scope": "sector", "Group": sector, **_final_metrics(group, signal)})
    return pd.DataFrame(rows)


def _final_metrics(frame: pd.DataFrame, signal: str) -> dict[str, Any]:
    base = _metrics(frame, action="BUY")
    if frame.empty:
        return {
            **base,
            "AverageFuture20DReturn": None,
            "MedianFuture20DReturn": None,
            "PassCautionFail": "FAIL",
        }
    avg_excess = base["AverageExcessReturn"]
    median_excess = base["MedianExcessReturn"]
    precision = base["Precision"]
    count = base["Count"]
    status = _pass_caution_fail(signal, count, precision, avg_excess, median_excess)
    return {
        **base,
        "AverageFuture20DReturn": float(frame["future_stock_return"].mean()),
        "MedianFuture20DReturn": float(frame["future_stock_return"].median()),
        "PassCautionFail": status,
    }


def _validation_vs_final(final_results: pd.DataFrame) -> pd.DataFrame:
    validation = {
        "BUY": {
            "ValidationCount": 775,
            "ValidationPrecision": 0.45806451612903226,
            "ValidationAverageExcessReturn": 0.038180026358291624,
            "ValidationMedianExcessReturn": 0.0406920111292454,
            "ValidationNonOverlapAverageExcessReturn": 0.050193654816914915,
        },
        "ACCUMULATE": {
            "ValidationCount": 17297,
            "ValidationPrecision": 0.3014395559923686,
            "ValidationAverageExcessReturn": 0.01485871206340852,
            "ValidationMedianExcessReturn": 0.0064866315477856,
            "ValidationNonOverlapAverageExcessReturn": 0.01799289640222231,
        },
    }
    rows = []
    overall = final_results[final_results["Scope"] == "overall"]
    for signal, val in validation.items():
        final_row = overall[overall["Signal"] == signal]
        record = {"Signal": signal, **val}
        if final_row.empty:
            record.update(
                {
                    "FinalCount": 0,
                    "FinalPrecision": None,
                    "FinalAverageExcessReturn": None,
                    "FinalMedianExcessReturn": None,
                    "FinalNonOverlapAverageExcessReturn": None,
                    "PrecisionDeterioration": None,
                    "AverageExcessDeterioration": None,
                    "MedianExcessDeterioration": None,
                    "PassCautionFail": "FAIL",
                }
            )
        else:
            row = final_row.iloc[0]
            record.update(
                {
                    "FinalCount": row["Count"],
                    "FinalPrecision": row["Precision"],
                    "FinalAverageExcessReturn": row["AverageExcessReturn"],
                    "FinalMedianExcessReturn": row["MedianExcessReturn"],
                    "FinalNonOverlapAverageExcessReturn": row["NonOverlapAverageExcessReturn"],
                    "PrecisionDeterioration": row["Precision"] - val["ValidationPrecision"],
                    "AverageExcessDeterioration": row["AverageExcessReturn"] - val["ValidationAverageExcessReturn"],
                    "MedianExcessDeterioration": row["MedianExcessReturn"] - val["ValidationMedianExcessReturn"],
                    "PassCautionFail": row["PassCautionFail"],
                }
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _sell_target_research(pre_final: pd.DataFrame, folds: list[dict[str, Any]], features: list[str]) -> pd.DataFrame:
    dataset = _add_sell_targets(pre_final)
    rows = []
    for target in ("sell_target_a", "sell_target_b", "sell_target_c"):
        predictions = _rf_binary_walk_forward(dataset, folds, features, target)
        base_rate = float(predictions["actual"].mean())
        pr_auc = _pr_auc(predictions["actual"], predictions["probability"])
        for fraction in TOP_FRACTIONS:
            selected = predictions[_top(predictions["probability_percentile"], fraction)].copy()
            metrics = _sell_metrics(selected)
            rows.append(
                {
                    "Target": target,
                    "TargetDefinition": _sell_target_definition(target),
                    "Scope": "overall",
                    "Group": "all",
                    "Percentile": f"Top {int(fraction * 100)}%",
                    "BaseRate": base_rate,
                    "PRAUC": pr_auc,
                    **metrics,
                }
            )
            for fold, group in selected.groupby("fold"):
                rows.append(
                    {
                        "Target": target,
                        "TargetDefinition": _sell_target_definition(target),
                        "Scope": "fold",
                        "Group": fold,
                        "Percentile": f"Top {int(fraction * 100)}%",
                        "BaseRate": base_rate,
                        "PRAUC": pr_auc,
                        **_sell_metrics(group),
                    }
                )
            for year, group in selected.groupby(selected["Date"].dt.year):
                rows.append(
                    {
                        "Target": target,
                        "TargetDefinition": _sell_target_definition(target),
                        "Scope": "year",
                        "Group": year,
                        "Percentile": f"Top {int(fraction * 100)}%",
                        "BaseRate": base_rate,
                        "PRAUC": pr_auc,
                        **_sell_metrics(group),
                    }
                )
    return pd.DataFrame(rows)


def _add_sell_targets(dataset: pd.DataFrame) -> pd.DataFrame:
    output = dataset.copy()
    output["sell_target_a"] = (output["future_stock_return"] < -0.05).astype(int)
    output["sell_target_b"] = (
        (output["future_stock_return"] < -0.05) & (output["excess_return"] < 0)
    ).astype(int)
    output["future_20d_max_drawdown"] = _future_max_drawdown(output)
    output["sell_target_c"] = (output["future_20d_max_drawdown"] < -0.08).astype(int)
    return output


def _future_max_drawdown(dataset: pd.DataFrame) -> pd.Series:
    """Calculate future 20-trading-day downside from current close for SELL labels."""

    pieces = []
    for _, group in dataset.sort_values(["symbol", "Date"]).groupby("symbol", sort=False):
        future_min = group["Close"].shift(-1).rolling(HORIZON_DAYS, min_periods=HORIZON_DAYS).min().shift(
            -(HORIZON_DAYS - 1)
        )
        pieces.append(future_min / group["Close"] - 1)
    return pd.concat(pieces).sort_index()


def _rf_binary_walk_forward(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features: list[str],
    target: str,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        train, validation = _fold_frames(dataset, fold)
        model, medians = _fit_classifier(
            train,
            features,
            target=target,
            model_name="random_forest",
            binary=True,
        )
        x = _transform_with_medians(validation, features, medians)
        output = _base_prediction_frame(validation, fold["fold"])
        if "future_20d_max_drawdown" in validation.columns:
            output["future_20d_max_drawdown"] = validation["future_20d_max_drawdown"].to_numpy()
        output["actual"] = validation[target].astype(int).to_numpy()
        output["probability"] = _positive_probability(model, x)
        output["probability_percentile"] = output.groupby("Date")["probability"].rank(pct=True)
        rows.append(output)
    return pd.concat(rows, ignore_index=True)


def _sell_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "Count": 0,
            "Precision": None,
            "PRAUCProxyBasePrecision": None,
            "AverageFutureStockReturn": None,
            "AverageFutureExcessReturn": None,
            "MedianFutureStockReturn": None,
            "MedianFutureExcessReturn": None,
            "DownsideHitRate": None,
            "SampleAdequate": False,
            "ValidationStatus": SELL_STATUS,
        }
    return {
        "Count": int(len(frame)),
        "Precision": float(frame["actual"].mean()),
        "PRAUCProxyBasePrecision": float(frame["actual"].mean()),
        "AverageFutureStockReturn": float(frame["future_stock_return"].mean()),
        "AverageFutureExcessReturn": float(frame["excess_return"].mean()),
        "MedianFutureStockReturn": float(frame["future_stock_return"].median()),
        "MedianFutureExcessReturn": float(frame["excess_return"].median()),
        "DownsideHitRate": float((frame["future_stock_return"] < -0.05).mean()),
        "MaxDrawdownHitRate": float((frame.get("future_20d_max_drawdown", pd.Series(index=frame.index)) < -0.08).mean())
        if "future_20d_max_drawdown" in frame
        else None,
        "SampleAdequate": bool(len(frame) >= 500),
        "ValidationStatus": _sell_status(frame),
    }


def _sell_status(frame: pd.DataFrame) -> str:
    if (
        len(frame) >= 500
        and frame["future_stock_return"].mean() < 0
        and frame["excess_return"].mean() < 0
        and frame["future_stock_return"].median() < 0
        and frame["excess_return"].median() < 0
    ):
        return "SELL_RESEARCH_CANDIDATE_REQUIRES_MORE_REVIEW"
    return SELL_STATUS


def _sell_target_definition(target: str) -> str:
    return {
        "sell_target_a": "future 20D absolute stock return < -5%",
        "sell_target_b": "future 20D stock return < -5% AND future excess return < 0",
        "sell_target_c": "future 20D max drawdown from current close exceeds -8%",
    }[target]


def _pr_auc(actual: pd.Series, probability: pd.Series) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(actual.astype(int), probability))


def _pass_caution_fail(
    signal: str,
    count: int,
    precision: float | None,
    avg_excess: float | None,
    median_excess: float | None,
) -> str:
    if not count or precision is None or avg_excess is None or median_excess is None:
        return "FAIL"
    if avg_excess <= 0 or median_excess <= 0:
        return "FAIL"
    if signal == "BUY" and count >= 50 and precision >= 0.30:
        return "PASS"
    if signal == "ACCUMULATE" and count >= 500 and avg_excess > 0 and median_excess > 0:
        return "PASS"
    return "CAUTION"


def _buy_mask(frame: pd.DataFrame) -> pd.Series:
    return _top(frame["p_outperform_percentile"], 0.01) & _top(
        frame["buy_sigmoid_probability_percentile"],
        0.01,
    )


def _accumulate_mask(frame: pd.DataFrame) -> pd.Series:
    return _top(frame["p_outperform_percentile"], 0.20) & (
        frame["buy_sigmoid_probability_percentile"] >= 0.50
    )


def _final_train_and_embargo_dates(dataset: pd.DataFrame) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    dates = pd.Series(sorted(dataset[dataset["Date"] < FINAL_TEST_START]["Date"].drop_duplicates()))
    train_dates = dates.iloc[:-HORIZON_DAYS].to_list()
    embargo_dates = dates.iloc[-HORIZON_DAYS:].to_list()
    return train_dates, embargo_dates


def _xgb_hyperparameters(*, binary: bool) -> dict[str, Any]:
    return {
        "n_estimators": 80,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "binary:logistic" if binary else "multi:softprob",
        "eval_metric": "logloss" if binary else "mlogloss",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }


def _readiness_report(
    config: dict[str, Any],
    rules: dict[str, Any],
    final_results: pd.DataFrame,
    comparison: pd.DataFrame,
    sell_research: pd.DataFrame,
) -> str:
    overall = final_results[final_results["Scope"] == "overall"]
    lines = [
        "NATIP Frozen Clean Model Deployment Readiness Report",
        "",
        f"Config hash: {config['config_hash']}",
        f"Rules hash: {rules['rules_hash']}",
        "",
        "Policy:",
        "- Clean Pruned Advanced feature set is frozen.",
        "- BUY and ACCUMULATE thresholds were selected from walk-forward validation only.",
        "- Final test was evaluated once and thresholds were not changed.",
        "- SELL remains validation-only research.",
        "",
        "Final-test signal results:",
        overall.to_string(index=False),
        "",
        "Validation vs final test:",
        comparison.to_string(index=False),
        "",
        f"SELL research status: {SELL_STATUS}",
        "Reason: no SELL threshold is promoted unless validation shows negative average and median future returns with adequate sample size and fold/year stability.",
        "",
        "Do not build live predict_stock.py until these final-test results are reviewed.",
    ]
    return "\n".join(lines)


def _hash_json(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    return value


if __name__ == "__main__":
    main()
