"""Validation utilities for stock probability models."""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.probability.features import FEATURE_COLUMNS
from app.probability.train import CLASS_TO_LABEL, LABEL_TO_CLASS


def evaluate_classifier(model: Any, dataset: pd.DataFrame) -> dict[str, Any]:
    """Evaluate classifier metrics on a labeled dataset."""

    try:
        from sklearn.calibration import calibration_curve
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            log_loss,
            roc_auc_score,
        )
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("Install scikit-learn before evaluating the model.") from exc

    feature_columns = getattr(model, "natip_feature_columns", None) or [
        column for column in FEATURE_COLUMNS if column in dataset.columns
    ]
    clean = dataset.dropna(subset=feature_columns + ["label"]).copy()
    y_true = clean["label"].map(LABEL_TO_CLASS)
    probabilities = model.predict_proba(clean[feature_columns])
    predictions = probabilities.argmax(axis=1)
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    importance_sample = clean.tail(min(len(clean), 5000))
    permutation = permutation_importance(
        model,
        importance_sample[feature_columns],
        importance_sample["label"].map(LABEL_TO_CLASS),
        n_repeats=3,
        random_state=42,
        scoring="balanced_accuracy",
    )
    calibration_true, calibration_pred = calibration_curve(
        (y_true == LABEL_TO_CLASS[1]).astype(int),
        probabilities[:, LABEL_TO_CLASS[1]],
        n_bins=10,
        strategy="quantile",
    )
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "log_loss": log_loss(y_true, probabilities),
        "feature_importance": _feature_importance(model, feature_columns),
        "permutation_importance": dict(
            zip(feature_columns, permutation.importances_mean.tolist(), strict=True)
        ),
        "calibration": {
            "predicted_probability": calibration_pred.tolist(),
            "observed_frequency": calibration_true.tolist(),
        },
        "label_mapping": CLASS_TO_LABEL,
    }
    try:
        metrics["roc_auc_ovr_weighted"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        metrics["roc_auc_ovr_weighted"] = None
    return metrics


def walk_forward_validation(
    model_factory: Any,
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    train_years: int = 3,
    test_months: int = 6,
) -> list[dict[str, Any]]:
    """Run expanding walk-forward validation windows."""

    try:
        from sklearn.metrics import balanced_accuracy_score, log_loss
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("Install scikit-learn before walk-forward validation.") from exc

    dates = pd.to_datetime(dataset["Date"])
    start = dates.min() + pd.DateOffset(years=train_years)
    end = dates.max()
    results: list[dict[str, Any]] = []
    current = start
    while current < end:
        train = dataset[dates < current]
        test_end = current + pd.DateOffset(months=test_months)
        test = dataset[(dates >= current) & (dates < test_end)]
        if not train.empty and not test.empty:
            model = model_factory()
            features = feature_columns or [
                column for column in FEATURE_COLUMNS if column in dataset.columns
            ]
            train = train.dropna(subset=features + ["label"])
            test = test.dropna(subset=features + ["label"])
            if train.empty or test.empty:
                current = test_end
                continue
            model.fit(train[features], train["label"].map(LABEL_TO_CLASS))
            y_true = test["label"].map(LABEL_TO_CLASS)
            probabilities = model.predict_proba(test[features])
            predictions = probabilities.argmax(axis=1)
            results.append(
                {
                    "train_until": current.date().isoformat(),
                    "test_until": test_end.date().isoformat(),
                    "rows": len(test),
                    "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                    "log_loss": log_loss(y_true, probabilities),
                }
            )
        current = test_end
    return results


def _feature_importance(model: Any, feature_columns: list[str]) -> dict[str, float]:
    """Return model feature importance if available."""

    values = getattr(model, "feature_importances_", None)
    if values is None:
        return {}
    return dict(zip(feature_columns, [float(value) for value in values], strict=True))


def shap_summary_sample(
    model: Any,
    dataset: pd.DataFrame,
    *,
    max_rows: int = 500,
) -> dict[str, float]:
    """Return optional mean absolute SHAP importance for a sample.

    SHAP can be heavy on local machines, so this helper is intentionally opt-in.
    """

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Install shap before running SHAP analysis.") from exc

    feature_columns = [column for column in FEATURE_COLUMNS if column in dataset.columns]
    sample = dataset.dropna(subset=feature_columns).tail(max_rows)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample[feature_columns])
    if isinstance(values, list):
        importance = sum(abs(value).mean(axis=0) for value in values) / len(values)
    else:
        importance = abs(values).mean(axis=0)
    return dict(zip(feature_columns, [float(value) for value in importance], strict=True))
