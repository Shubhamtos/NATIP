"""Train the XGBoost stock outperformance model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.probability.config import FEATURES_PATH, MODEL_PATH, ModelConfig, ensure_probability_dirs
from app.probability.features import FEATURE_COLUMNS

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


def train_xgb_classifier(
    dataset: pd.DataFrame,
    *,
    model_path: Path = MODEL_PATH,
    features_path: Path = FEATURES_PATH,
    config: ModelConfig | None = None,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Train an XGBClassifier with time-based validation and early stopping."""

    try:
        from joblib import dump
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "Install scikit-learn, xgboost, and joblib before training the probability model."
        ) from exc

    ensure_probability_dirs()
    config = config or ModelConfig()
    train_frame, validation_frame, test_frame = time_based_split(dataset)
    feature_columns = feature_columns or [
        column for column in FEATURE_COLUMNS if column in dataset.columns
    ]
    train_frame = train_frame.dropna(subset=feature_columns + ["label"])
    validation_frame = validation_frame.dropna(subset=feature_columns + ["label"])
    test_frame = test_frame.dropna(subset=feature_columns + ["label"])
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Training and validation splits must both contain data.")

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        objective=config.objective,
        eval_metric=config.eval_metric,
        random_state=config.random_state,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    model.fit(
        train_frame[feature_columns],
        train_frame["label"].map(LABEL_TO_CLASS),
        eval_set=[
            (validation_frame[feature_columns], validation_frame["label"].map(LABEL_TO_CLASS))
        ],
        verbose=False,
    )
    model.natip_feature_columns = feature_columns
    dump(model, model_path)
    features_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    return {
        "model_path": str(model_path),
        "features_path": str(features_path),
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "test_rows": len(test_frame),
        "feature_count": len(feature_columns),
    }


def time_based_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by calendar time instead of random sampling."""

    dates = pd.to_datetime(dataset["Date"])
    train = dataset[dates <= "2022-12-31"].copy()
    validation = dataset[(dates >= "2023-01-01") & (dates <= "2024-12-31")].copy()
    test = dataset[dates >= "2025-01-01"].copy()
    return train, validation, test
