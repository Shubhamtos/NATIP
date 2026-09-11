"""Build tuned Binary77 confidence-upgrader artifacts for inference.

This creates a separate research/deployment candidate artifact and does not
overwrite the frozen V1 or V2 production/shadow artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import MODEL_DIR, REPORT_DIR, ensure_probability_dirs
from run_final_signal_threshold_research import CLEAN_DEMERGER_DATASET
from run_frozen_final_evaluation import _final_train_and_embargo_dates, _hash_json, _json_safe
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)
from run_stock_specific_feature_variant import _frozen_features, _prepare_dataset
from run_v2_model_family_comparison import _selected_26_features
from run_v2_xgb_hyperparameter_optimization import TrialSpec, _fit_xgb

ARTIFACT_PATH = MODEL_DIR / "frozen_tuned_binary77_confidence_candidate.joblib"
CONFIG_PATH = REPORT_DIR / "frozen_tuned_binary77_confidence_config.json"
RULES_PATH = REPORT_DIR / "frozen_tuned_binary77_confidence_rules.json"
AUDIT_PATH = REPORT_DIR / "tuned_binary77_confidence_audit.txt"
TUNED_CONFIG_PATH = REPORT_DIR / "v2_tuned_xgb_config.json"


def main() -> None:
    """Train and freeze the confidence-upgrader candidate artifact."""

    ensure_probability_dirs()
    tuned_research = json.loads(TUNED_CONFIG_PATH.read_text(encoding="utf-8"))
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    if len(features_26) != 26:
        raise RuntimeError(f"Expected 26 primary features, got {len(features_26)}.")

    train_dates, embargo_dates = _final_train_and_embargo_dates(dataset)
    train = dataset[dataset["Date"].isin(train_dates)].copy()
    fit_frame, calibration_frame = _fit_calibration_split(train)

    primary_spec = TrialSpec(
        trial_id=tuned_research["selected_primary_trial"],
        phase="deployment_candidate_primary",
        params=tuned_research["selected_primary_params"],
    )
    current_binary_params = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 1.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    current_binary_spec = TrialSpec(
        trial_id="binary_baseline",
        phase="deployment_candidate_current_binary",
        params=current_binary_params,
        is_baseline=True,
    )
    primary_model, primary_medians = _fit_xgb(
        fit_frame,
        calibration_frame,
        features_26,
        target="label",
        spec=primary_spec,
        binary=False,
    )
    current_buy_model, current_buy_medians = _fit_xgb(
        fit_frame,
        calibration_frame,
        features_77,
        target="buy_target",
        spec=current_binary_spec,
        binary=True,
    )
    current_cal_x = _transform_with_medians(calibration_frame, features_77, current_buy_medians)
    current_cal_prob = _positive_probability(current_buy_model, current_cal_x)
    current_buy_sigmoid_calibrator = _fit_platt(
        current_cal_prob, calibration_frame["buy_target"]
    )

    tuned_spec = TrialSpec(
        trial_id=tuned_research["selected_binary_trial"],
        phase="deployment_candidate_confidence",
        params=tuned_research["selected_binary_params"],
        scale_pos_weight=tuned_research["selected_binary_scale_pos_weight"],
    )
    tuned_buy_model, tuned_buy_medians = _fit_xgb(
        fit_frame,
        calibration_frame,
        features_77,
        target="buy_target",
        spec=tuned_spec,
        binary=True,
    )
    tuned_cal_x = _transform_with_medians(calibration_frame, features_77, tuned_buy_medians)
    tuned_cal_prob = _positive_probability(tuned_buy_model, tuned_cal_x)
    tuned_buy_sigmoid_calibrator = _fit_platt(tuned_cal_prob, calibration_frame["buy_target"])
    ranker_model, ranker_medians = _fit_ranker(train, features_77)

    config = _candidate_config(
        dataset=dataset,
        features_26=features_26,
        features_77=features_77,
        train_dates=train_dates,
        embargo_dates=embargo_dates,
        tuned_research=tuned_research,
    )
    rules = _candidate_rules(config, tuned_research)
    artifact = {
        "artifact_version": "natip-tuned-binary77-confidence-candidate-v1.0",
        "config_hash": config["config_hash"],
        "rules_hash": rules["rules_hash"],
        "primary_feature_list": features_26,
        "binary_feature_list": features_77,
        "ranker_feature_list": features_77,
        "primary_model": primary_model,
        "primary_medians": primary_medians,
        "current_buy_model": current_buy_model,
        "current_buy_medians": current_buy_medians,
        "current_buy_sigmoid_calibrator": current_buy_sigmoid_calibrator,
        "tuned_buy_model": tuned_buy_model,
        "tuned_buy_medians": tuned_buy_medians,
        "tuned_buy_sigmoid_calibrator": tuned_buy_sigmoid_calibrator,
        "ranker_model": ranker_model,
        "ranker_medians": ranker_medians,
        "training_start": min(train_dates).isoformat(),
        "training_end": max(train_dates).isoformat(),
        "final_test_start": FINAL_TEST_START.isoformat(),
        "embargo_trading_days": HORIZON_DAYS,
        "created_from": str(CLEAN_DEMERGER_DATASET),
    }

    from joblib import dump

    dump(artifact, ARTIFACT_PATH)
    checksum = _file_sha256(ARTIFACT_PATH)
    config["model_artifacts"]["confidence_candidate_artifact"]["sha256"] = checksum
    config["model_artifacts"]["tuned_binary_model"]["sha256"] = checksum
    rules["artifact_sha256"] = checksum
    CONFIG_PATH.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
    RULES_PATH.write_text(json.dumps(_json_safe(rules), indent=2), encoding="utf-8")
    AUDIT_PATH.write_text(_audit_text(config, rules), encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(ARTIFACT_PATH),
                "config": str(CONFIG_PATH),
                "rules": str(RULES_PATH),
                "audit": str(AUDIT_PATH),
                "artifact_sha256": checksum,
                "v1_production_modified": False,
                "v2_shadow_modified": False,
            },
            indent=2,
        )
    )


def _candidate_config(
    *,
    dataset: pd.DataFrame,
    features_26: list[str],
    features_77: list[str],
    train_dates: list[pd.Timestamp],
    embargo_dates: list[pd.Timestamp],
    tuned_research: dict[str, Any],
) -> dict[str, Any]:
    config = {
        "config_version": "natip-tuned-binary77-confidence-candidate-v1.0",
        "source_dataset": str(CLEAN_DEMERGER_DATASET),
        "dataset_rows": int(len(dataset)),
        "final_test_start": FINAL_TEST_START.isoformat(),
        "final_test_used_for_threshold_selection": False,
        "prediction_horizon_trading_days": HORIZON_DAYS,
        "embargo_trading_days": HORIZON_DAYS,
        "primary_classifier": "26-feature XGBoost classifier",
        "primary_feature_list": features_26,
        "primary_feature_order": {feature: index for index, feature in enumerate(features_26)},
        "current_binary_buy_confirmation": "77-feature current Binary XGBoost BUY model",
        "tuned_binary_confidence_upgrader": "77-feature tuned Binary XGBoost BUY model",
        "binary_feature_list": features_77,
        "binary_feature_order": {feature: index for index, feature in enumerate(features_77)},
        "preprocessing": {
            "missing_values": "training-window median imputation only",
            "missing_indicators": "appended by _transform_with_medians",
            "feature_order_asserted": True,
            "no_retraining_during_prediction": True,
        },
        "calibration": {
            "current_binary_sigmoid": "Platt LogisticRegression fitted on calibration split only",
            "tuned_binary_sigmoid": "Platt LogisticRegression fitted on calibration split only",
        },
        "model_hyperparameters": {
            "primary_26_xgboost": tuned_research["selected_primary_params"],
            "current_binary77_xgboost": {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 1.0,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": 42,
                "tree_method": "hist",
                "n_jobs": -1,
                "early_stopping_rounds": 50,
            },
            "tuned_binary77_xgboost": tuned_research["selected_binary_params"],
            "tuned_binary77_scale_pos_weight": tuned_research[
                "selected_binary_scale_pos_weight"
            ],
        },
        "training_window": {
            "train_start": min(train_dates).isoformat(),
            "train_end": max(train_dates).isoformat(),
            "embargo_start": min(embargo_dates).isoformat(),
            "embargo_end": max(embargo_dates).isoformat(),
            "train_trading_days": len(train_dates),
            "embargo_trading_days": len(embargo_dates),
        },
        "percentile_calculation": {
            "method": "same-date reference universe rank(pct=True)",
            "rank_direction": "higher score is stronger BUY",
            "tie_handling": "pandas rank average tie handling",
            "minimum_reference_universe_size": 50,
        },
        "confirmation_percentile_logic": {
            "primary_xgb_pass": "p_outperform_percentile >= 0.99",
            "current_binary77_pass": "current_binary77_percentile >= 0.995",
            "tuned_binary77_confirmation_pass": "tuned_binary77_percentile >= 0.995",
            "tuned_only_status": "RESEARCH_ONLY_TUNED_SIGNAL",
        },
        "confidence_mapping": {
            "BUY_and_tuned_confirms": "HIGH",
            "BUY_without_tuned_confirm": "MEDIUM",
            "non_BUY": "unchanged from existing non-BUY recommendation",
        },
        "validation_evidence": tuned_research,
        "model_artifacts": {
            "confidence_candidate_artifact": {"path": str(ARTIFACT_PATH), "sha256": None},
            "tuned_binary_model": {"path": str(ARTIFACT_PATH), "sha256": None},
        },
    }
    config["primary_feature_hash"] = _hash_json(features_26)
    config["binary_feature_hash"] = _hash_json(features_77)
    config["hyperparameter_hash"] = _hash_json(config["model_hyperparameters"])
    config["config_hash"] = _stable_hash(config, hash_key="config_hash")
    return config


def _candidate_rules(config: dict[str, Any], tuned_research: dict[str, Any]) -> dict[str, Any]:
    rules = {
        "rules_version": "natip-tuned-binary77-confidence-rules-v1.0",
        "config_hash": config["config_hash"],
        "buy": {
            "rule_id": "CURRENT_BUY_XGB26_TOP_1PCT_AND_CURRENT_BINARY77_TOP_0P5PCT",
            "changed_by_tuned_binary": False,
            "primary_classifier_percentile_min": 0.99,
            "current_binary77_percentile_min": 0.995,
        },
        "tuned_binary77_confidence": {
            "role": "confidence_upgrader_only",
            "must_not_create_buy": True,
            "confirmation_percentile_min": 0.995,
            "if_current_buy_and_tuned_confirms": {
                "recommendation": "BUY",
                "confidence": "HIGH",
            },
            "if_current_buy_and_tuned_does_not_confirm": {
                "recommendation": "BUY",
                "confidence": "MEDIUM",
            },
            "if_tuned_qualifies_but_current_buy_fails": {
                "recommendation": "unchanged_non_buy",
                "tuned_status": "RESEARCH_ONLY_TUNED_SIGNAL",
            },
            "validation_evidence": {
                "signals": 153,
                "precision": 0.6732,
                "average_excess_return": 0.0797,
                "median_excess_return": 0.0824,
                "nifty_win_rate": 0.8693,
                "bootstrap_precision_lift_ci": [0.0952, 0.2215],
                "source_decision": tuned_research.get("recommendation"),
            },
        },
        "artifact_sha256": None,
    }
    rules["rules_hash"] = _stable_hash(rules, hash_key="rules_hash")
    return rules


def _stable_hash(payload: dict[str, Any], *, hash_key: str) -> str:
    normalized = json.loads(json.dumps(_json_safe(payload)))
    normalized.pop(hash_key, None)
    normalized["artifact_sha256"] = None
    artifacts = normalized.get("model_artifacts", {})
    for artifact in artifacts.values():
        if isinstance(artifact, dict):
            artifact["sha256"] = None
    return _hash_json(normalized)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_text(config: dict[str, Any], rules: dict[str, Any]) -> str:
    return "\n".join(
        [
            "NATIP Tuned Binary77 Confidence Upgrader Audit",
            "================================================",
            "",
            "Production V1 artifacts modified: NO",
            "Frozen V2 shadow artifacts modified: NO",
            "Final test used for threshold selection: NO",
            "",
            f"Config hash: {config['config_hash']}",
            f"Rules hash: {rules['rules_hash']}",
            f"Artifact path: {ARTIFACT_PATH}",
            f"Artifact sha256: {rules['artifact_sha256']}",
            "",
            "Official BUY qualification remains:",
            "26-feature XGB classifier Top 1% AND Current Binary77 Top 0.5%.",
            "",
            "Tuned Binary77 behavior:",
            "- Current BUY + tuned confirms => BUY / HIGH confidence.",
            "- Current BUY + tuned does not confirm => BUY / MEDIUM confidence.",
            "- Tuned-only signal => RESEARCH_ONLY_TUNED_SIGNAL; never BUY.",
        ]
    )


if __name__ == "__main__":
    main()
