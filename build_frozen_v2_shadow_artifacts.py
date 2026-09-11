"""Build frozen V2 shadow BUY artifacts without modifying V1 production files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import MODEL_DIR, REPORT_DIR, ensure_probability_dirs
from run_final_signal_threshold_research import CLEAN_DEMERGER_DATASET, _prepare_dataset
from run_frozen_final_evaluation import (
    _final_train_and_embargo_dates,
    _hash_json,
    _json_safe,
    _xgb_hyperparameters,
)
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS
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
    _frozen_features,
)

V2_ARTIFACT_PATH = MODEL_DIR / "frozen_v2_buy_shadow.joblib"
V2_CONFIG_PATH = REPORT_DIR / "frozen_v2_model_config.json"
V2_RULES_PATH = REPORT_DIR / "frozen_v2_buy_rules.json"
V2_AUDIT_PATH = REPORT_DIR / "v2_shadow_mode_audit.txt"


def main() -> None:
    """Train and freeze the V2 shadow BUY artifact."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    features_77 = _frozen_features()
    features_55 = _v2_55_features(features_77)
    if len(features_55) != 55:
        raise RuntimeError(f"Expected 55 V2 classifier features, got {len(features_55)}.")

    train_dates, embargo_dates = _final_train_and_embargo_dates(dataset)
    train = dataset[dataset["Date"].isin(train_dates)].copy()

    primary_model, primary_medians = _fit_classifier(train, features_55, target="label")
    fit_frame, calibration_frame = _fit_calibration_split(train)
    buy_model, buy_medians = _fit_classifier(
        fit_frame,
        features_77,
        target="buy_target",
        model_name="xgboost",
        binary=True,
    )
    cal_x = _transform_with_medians(calibration_frame, features_77, buy_medians)
    cal_prob = _positive_probability(buy_model, cal_x)
    buy_sigmoid_calibrator = _fit_platt(cal_prob, calibration_frame["buy_target"])
    ranker_model, ranker_medians = _fit_ranker(train, features_77)

    config = _v2_config(
        dataset=dataset,
        features_55=features_55,
        features_77=features_77,
        train_dates=train_dates,
        embargo_dates=embargo_dates,
    )
    rules = _v2_rules(config)

    artifact = {
        "artifact_version": "natip-frozen-v2-buy-shadow-v1.0",
        "config_hash": config["config_hash"],
        "rules_hash": rules["rules_hash"],
        "primary_feature_list": features_55,
        "binary_feature_list": features_77,
        "ranker_feature_list": features_77,
        "primary_model": primary_model,
        "primary_medians": primary_medians,
        "buy_model": buy_model,
        "buy_medians": buy_medians,
        "buy_sigmoid_calibrator": buy_sigmoid_calibrator,
        "ranker_model": ranker_model,
        "ranker_medians": ranker_medians,
        "training_start": min(train_dates).isoformat(),
        "training_end": max(train_dates).isoformat(),
        "final_test_start": FINAL_TEST_START.isoformat(),
        "embargo_trading_days": HORIZON_DAYS,
        "created_from": str(CLEAN_DEMERGER_DATASET),
    }

    try:
        from joblib import dump
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install joblib to save frozen V2 artifacts.") from exc

    config["config_hash"] = _stable_config_hash(config)
    rules["config_hash"] = config["config_hash"]
    rules["rules_hash"] = _stable_rules_hash(rules)
    artifact["config_hash"] = config["config_hash"]
    artifact["rules_hash"] = rules["rules_hash"]
    dump(artifact, V2_ARTIFACT_PATH)
    checksum = _file_sha256(V2_ARTIFACT_PATH)
    config["model_artifacts"]["v2_shadow_artifact"]["sha256"] = checksum
    rules["artifact_sha256"] = checksum

    V2_CONFIG_PATH.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    V2_RULES_PATH.write_text(json.dumps(_json_safe(rules), indent=2), encoding="utf-8")
    V2_AUDIT_PATH.write_text(_audit_text(config, rules), encoding="utf-8")
    print(
        json.dumps(
            {
                "frozen_v2_model_config": str(V2_CONFIG_PATH),
                "frozen_v2_buy_rules": str(V2_RULES_PATH),
                "frozen_v2_artifact": str(V2_ARTIFACT_PATH),
                "v2_shadow_mode_audit": str(V2_AUDIT_PATH),
                "production_v1_modified": False,
            },
            indent=2,
        )
    )


def _v2_55_features(features_77: list[str]) -> list[str]:
    remove = NIFTY_FEATURES | MARKET_REGIME_FEATURES | MARKET_BREADTH_FEATURES
    return [feature for feature in features_77 if feature not in remove]


def _v2_config(
    *,
    dataset: pd.DataFrame,
    features_55: list[str],
    features_77: list[str],
    train_dates: list[pd.Timestamp],
    embargo_dates: list[pd.Timestamp],
) -> dict[str, Any]:
    model_hyperparameters = {
        "primary_v2_55_xgboost_3class": _xgb_hyperparameters(binary=False),
        "binary_v1_77_xgboost_buy_confirmation": _xgb_hyperparameters(binary=True),
        "xgbranker_v1_77_diagnostic": {
            "source": "run_ranker_binary_signal_research._fit_ranker",
            "objective": "rank:pairwise",
            "random_state": 42,
            "diagnostic_only": True,
        },
    }
    config = {
        "config_version": "natip-v2-buy-shadow-v1.0",
        "source_dataset": str(CLEAN_DEMERGER_DATASET),
        "dataset_rows": int(len(dataset)),
        "final_test_start": FINAL_TEST_START.isoformat(),
        "final_test_used_for_v2_selection": False,
        "prediction_horizon_trading_days": HORIZON_DAYS,
        "embargo_trading_days": HORIZON_DAYS,
        "primary_classifier": "55-feature Variant B",
        "primary_feature_list": features_55,
        "primary_feature_order": {feature: index for index, feature in enumerate(features_55)},
        "binary_buy_confirmation": "77-feature Binary XGBoost BUY model",
        "binary_feature_list": features_77,
        "binary_feature_order": {feature: index for index, feature in enumerate(features_77)},
        "ranker_diagnostic_feature_list": features_77,
        "preprocessing": {
            "missing_values": "training-window median imputation only",
            "missing_indicators": "appended by _transform_with_medians",
            "feature_order_asserted": True,
            "no_retraining_during_prediction": True,
        },
        "training_medians": {
            "stored_in_artifact": True,
            "primary_key": "primary_medians",
            "binary_key": "buy_medians",
            "ranker_key": "ranker_medians",
        },
        "calibration": {
            "binary_buy_sigmoid": "Platt LogisticRegression fitted only on latest 20% of training dates",
            "stored_in_artifact_key": "buy_sigmoid_calibrator",
        },
        "model_hyperparameters": model_hyperparameters,
        "training_window": {
            "train_start": min(train_dates).isoformat(),
            "train_end": max(train_dates).isoformat(),
            "embargo_start": min(embargo_dates).isoformat(),
            "embargo_end": max(embargo_dates).isoformat(),
            "train_trading_days": len(train_dates),
            "embargo_trading_days": len(embargo_dates),
        },
        "walk_forward_methodology": {
            "source": "run_pruned_rank_target_validation._embargo_folds",
            "purge_embargo_trading_days": HORIZON_DAYS,
            "validation_only_selection": True,
        },
        "percentile_calculation": {
            "method": "same-date reference universe rank(pct=True)",
            "rank_direction": "higher score is stronger BUY",
            "tie_handling": "pandas rank average tie handling",
            "minimum_reference_universe_size": 50,
        },
        "model_artifacts": {
            "v2_shadow_artifact": {
                "path": str(V2_ARTIFACT_PATH),
                "sha256": None,
            }
        },
    }
    config["primary_feature_hash"] = _hash_json(features_55)
    config["binary_feature_hash"] = _hash_json(features_77)
    config["hyperparameter_hash"] = _hash_json(model_hyperparameters)
    config["config_hash"] = _stable_config_hash(config)
    return config


def _v2_rules(config: dict[str, Any]) -> dict[str, Any]:
    rules = {
        "rules_version": "natip-v2-buy-shadow-rules-v1.0",
        "config_hash": config["config_hash"],
        "buy": {
            "rule_id": "V2_BUY_55_TOP_1PCT_AND_BINARY_77_TOP_0P5PCT",
            "official_production": False,
            "shadow_only": True,
            "conditions": [
                "55-feature classifier P_Outperform same-date percentile >= 0.99",
                "77-feature Binary BUY sigmoid probability same-date percentile >= 0.995",
            ],
            "classifier_percentile_min": 0.99,
            "binary_buy_percentile_min": 0.995,
            "validation_evidence": {
                "signals": 465,
                "precision": 0.503226,
                "average_20d_excess_return": 0.051657,
                "median_20d_excess_return": 0.050970,
                "nifty_win_rate": 0.690323,
                "folds_positive": True,
                "years_positive": True,
                "robustness_classification": "PROMISING",
            },
        },
        "ranker": {
            "status": "diagnostic_only",
            "mandatory_for_buy": False,
        },
        "freeze_policy": "Do not tune V2 thresholds from live/shadow results.",
        "artifact_sha256": None,
    }
    rules["rules_hash"] = _stable_rules_hash(rules)
    return rules


def _stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(_json_safe(config)))
    payload.pop("config_hash", None)
    artifact = payload.get("model_artifacts", {}).get("v2_shadow_artifact", {})
    artifact["sha256"] = None
    return _hash_json(payload)


def _stable_rules_hash(rules: dict[str, Any]) -> str:
    payload = json.loads(json.dumps(_json_safe(rules)))
    payload.pop("rules_hash", None)
    payload["artifact_sha256"] = None
    return _hash_json(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_text(config: dict[str, Any], rules: dict[str, Any]) -> str:
    return "\n".join(
        [
            "NATIP V2 Shadow Mode Audit",
            "===========================",
            "",
            "Production Version 1 artifacts modified: NO",
            "Final test used for V2 selection/tuning: NO",
            "V2 live behavior: shadow only; official recommendation remains V1.",
            "",
            f"V2 config hash: {config['config_hash']}",
            f"V2 rules hash: {rules['rules_hash']}",
            f"V2 artifact: {V2_ARTIFACT_PATH}",
            f"V2 artifact sha256: {rules['artifact_sha256']}",
            "",
            "Frozen BUY rule:",
            "55-feature classifier Top 1% AND 77-feature Binary BUY Top 0.5%.",
            "",
            "Safety assertions:",
            "- Prediction loads artifacts only and never retrains.",
            "- Feature order is asserted from JSON config against artifact metadata.",
            "- Percentiles are computed against the same-date reference universe.",
            "- XGBRanker remains diagnostic only.",
            "- Shadow results must not be used to tune thresholds.",
        ]
    )


if __name__ == "__main__":
    main()
