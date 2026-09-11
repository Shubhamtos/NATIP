"""Build frozen NATIP clean model artifacts for inference.

This script is intentionally separate from ``predict_stock.py``. Prediction code
must never retrain; it only loads the artifacts produced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.probability.config import MODEL_DIR, ensure_probability_dirs
from run_frozen_final_evaluation import OUTPUT_CONFIG, OUTPUT_RULES
from run_final_signal_threshold_research import CLEAN_DEMERGER_DATASET, _prepare_dataset, _pruned_features
from run_pruned_rank_target_validation import FINAL_TEST_START, HORIZON_DAYS
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_classifier,
    _fit_platt,
    _fit_ranker,
    _positive_probability,
    _transform_with_medians,
)

FROZEN_ARTIFACT_PATH = MODEL_DIR / "frozen_clean_pruned_v1.joblib"


def main() -> None:
    """Train once from the frozen clean pre-final window and save inference artifacts."""

    ensure_probability_dirs()
    config = json.loads(OUTPUT_CONFIG.read_text(encoding="utf-8"))
    rules = json.loads(OUTPUT_RULES.read_text(encoding="utf-8"))
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    features = _pruned_features()
    if features != config["feature_list"]:
        raise RuntimeError("Frozen feature list does not match frozen_model_config.json.")

    train_dates = _final_train_dates(dataset)
    train = dataset[dataset["Date"].isin(train_dates)].copy()

    primary_model, primary_medians = _fit_classifier(train, features, target="label")
    fit_frame, calibration_frame = _fit_calibration_split(train)
    buy_model, buy_medians = _fit_classifier(fit_frame, features, target="buy_target", binary=True)
    cal_x = _transform_with_medians(calibration_frame, features, buy_medians)
    cal_prob = _positive_probability(buy_model, cal_x)
    buy_sigmoid_calibrator = _fit_platt(cal_prob, calibration_frame["buy_target"])
    ranker_model, ranker_medians = _fit_ranker(train, features)

    artifact = {
        "artifact_version": "natip-frozen-clean-pruned-v1.0",
        "config_hash": config["config_hash"],
        "rules_hash": rules["rules_hash"],
        "feature_list": features,
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
        raise RuntimeError("Install joblib to save frozen model artifacts.") from exc

    dump(artifact, FROZEN_ARTIFACT_PATH)
    print(json.dumps({"frozen_artifact": str(FROZEN_ARTIFACT_PATH)}, indent=2))


def _final_train_dates(dataset: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.Series(sorted(dataset[dataset["Date"] < FINAL_TEST_START]["Date"].drop_duplicates()))
    return dates.iloc[:-HORIZON_DAYS].to_list()


if __name__ == "__main__":
    main()
