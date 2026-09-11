"""Focused recency-aware training experiment on the original 150-stock universe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.train import LABEL_TO_CLASS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_ranker_binary_signal_research import (
    _fit_calibration_split,
    _fit_platt,
    _positive_probability,
    _transform_with_medians,
)
from run_stock_specific_feature_variant import (
    FeatureSet,
    _common_evaluation_rows,
    _frozen_features,
    _prepare_dataset,
    _safe_float,
)

CLEAN_150_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"

OUTPUT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"
OUTPUT_COMPARISON = REPORT_DIR / "recency_training_comparison.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "recency_training_bootstrap.csv"
OUTPUT_DECISION = REPORT_DIR / "recency_training_decision.json"
OUTPUT_REPORT = REPORT_DIR / "recency_training_report.txt"

CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
RECENCY_HALFLIFE_DAYS = 365.25 * 2


@dataclass(frozen=True, slots=True)
class Scheme:
    """Recency-aware training scheme."""

    name: str
    lookback_years: int | None = None
    use_time_decay_weights: bool = False


SCHEMES = [
    Scheme("EXPANDING_BASELINE"),
    Scheme("ROLLING_5Y", lookback_years=5),
    Scheme("ROLLING_3Y", lookback_years=3),
    Scheme("RECENCY_WEIGHTED", use_time_decay_weights=True),
]


def main() -> None:
    """Run recency-aware validation-only training comparison."""

    ensure_probability_dirs()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_150_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    features_26 = _selected_26_features()
    features_77 = _frozen_features()
    _assert_features(pre_final, features_26, features_77)
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    frames = []
    for scheme in SCHEMES:
        print(f"[recency] {scheme.name}", flush=True)
        frames.append(_scheme_oof(common, folds, features_26, features_77, scheme))
    oof = pd.concat(frames, ignore_index=True)
    oof.to_csv(OUTPUT_OOF, index=False)
    comparison = _comparison(oof)
    comparison.to_csv(OUTPUT_COMPARISON, index=False)
    bootstrap = _bootstrap(oof)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    decision = _decision_payload(comparison, bootstrap)
    OUTPUT_DECISION.write_text(json.dumps(_json_safe(decision), indent=2), encoding="utf-8")
    OUTPUT_REPORT.write_text(_report(comparison, bootstrap, decision), encoding="utf-8")
    print(
        json.dumps(
            {
                "oof_predictions": str(OUTPUT_OOF),
                "comparison": str(OUTPUT_COMPARISON),
                "bootstrap": str(OUTPUT_BOOTSTRAP),
                "decision": str(OUTPUT_DECISION),
                "report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "v1_v2_modified": False,
            },
            indent=2,
        )
    )


def _scheme_oof(
    dataset: pd.DataFrame,
    folds: list[dict[str, Any]],
    features_26: list[str],
    features_77: list[str],
    scheme: Scheme,
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        full_train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        train = _training_window(full_train, validation["Date"].min(), scheme)
        if train.empty:
            raise ValueError(f"{scheme.name} produced empty training data for fold {fold['fold']}")

        classifier, medians_26 = _fit_classifier(
            train,
            features_26,
            target="label",
            binary=False,
            sample_weight=_sample_weights(train, scheme),
        )
        probabilities = classifier.predict_proba(
            _transform_with_medians(validation, features_26, medians_26)
        )

        fit_frame, calibration_frame = _fit_calibration_split(train)
        binary, medians_77 = _fit_classifier(
            fit_frame,
            features_77,
            target="buy_target",
            binary=True,
            sample_weight=_sample_weights(fit_frame, scheme),
        )
        cal_prob = _positive_probability(
            binary, _transform_with_medians(calibration_frame, features_77, medians_77)
        )
        raw_probability = _positive_probability(
            binary, _transform_with_medians(validation, features_77, medians_77)
        )
        sigmoid = _fit_platt(cal_prob, calibration_frame["buy_target"])

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
            ]
        ].copy()
        output["Scheme"] = scheme.name
        output["fold"] = fold["fold"]
        output["TrainingRows"] = len(train)
        output["TrainingStart"] = train["Date"].min()
        output["TrainingEnd"] = train["Date"].max()
        output["TrainingBuyBaseRate"] = float(train["buy_target"].mean())
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["classifier_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["binary_buy_raw_probability"] = raw_probability
        output["binary_buy_sigmoid_probability"] = sigmoid.predict_proba(
            raw_probability.reshape(-1, 1)
        )[:, 1]
        output["binary_buy_percentile"] = output.groupby("Date")[
            "binary_buy_sigmoid_probability"
        ].rank(pct=True)
        output["buy_signal"] = (
            (output["classifier_percentile"] >= 1 - CLASSIFIER_TOP)
            & (output["binary_buy_percentile"] >= 1 - BINARY_TOP)
        )
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _training_window(train: pd.DataFrame, validation_start: pd.Timestamp, scheme: Scheme) -> pd.DataFrame:
    if scheme.lookback_years is None:
        return train.copy()
    cutoff = validation_start - pd.DateOffset(years=scheme.lookback_years)
    return train[train["Date"] >= cutoff].copy()


def _sample_weights(train: pd.DataFrame, scheme: Scheme) -> np.ndarray | None:
    if not scheme.use_time_decay_weights:
        return None
    newest = pd.to_datetime(train["Date"]).max()
    age_days = (newest - pd.to_datetime(train["Date"])).dt.days.clip(lower=0).to_numpy(dtype=float)
    weights = np.power(0.5, age_days / RECENCY_HALFLIFE_DAYS)
    return weights / np.mean(weights)


def _fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    binary: bool,
    sample_weight: np.ndarray | None,
):
    from xgboost import XGBClassifier

    medians = train[features].median(numeric_only=True).fillna(0)
    x = _transform_with_medians(train, features, medians)
    y = train[target].astype(int) if binary else train[target].map(LABEL_TO_CLASS)
    model = XGBClassifier(
        n_estimators=80,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic" if binary else "multi:softprob",
        eval_metric="logloss" if binary else "mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return model, medians


def _comparison(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, frame in oof.groupby("Scheme"):
        selected = frame[frame["buy_signal"]].copy()
        rows.append(_metric_row(scheme, "overall", "all", selected, frame))
        for fold, group in frame.groupby("fold"):
            rows.append(_metric_row(scheme, "fold", fold, group[group["buy_signal"]], group))
        for year, group in frame.groupby(frame["Date"].dt.year):
            rows.append(_metric_row(scheme, "year", int(year), group[group["buy_signal"]], group))
        for sector, group in frame.groupby("Sector"):
            rows.append(_metric_row(scheme, "sector", sector, group[group["buy_signal"]], group))
    return pd.DataFrame(rows)


def _metric_row(
    scheme: str,
    scope: str,
    scope_value: Any,
    selected: pd.DataFrame,
    population: pd.DataFrame,
) -> dict[str, Any]:
    try:
        from sklearn.metrics import average_precision_score

        pr_auc = float(average_precision_score(population["buy_target"], population["p_outperform"]))
    except Exception:
        pr_auc = np.nan
    sector_counts = selected["Sector"].value_counts(normalize=True) if not selected.empty else pd.Series(dtype=float)
    return {
        "Scheme": scheme,
        "Scope": scope,
        "ScopeValue": scope_value,
        "PopulationRows": int(len(population)),
        "AverageTrainingRows": _safe_float(population["TrainingRows"].mean()),
        "AverageTrainingBuyBaseRate": _safe_float(population["TrainingBuyBaseRate"].mean()),
        "PRAUC": pr_auc,
        "BuySignals": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean())
        if not selected.empty
        else None,
        "MedianExcessReturn": _safe_float(selected["excess_return"].median())
        if not selected.empty
        else None,
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean())
        if not selected.empty
        else None,
        "LargestSectorShare": _safe_float(sector_counts.iloc[0]) if not sector_counts.empty else None,
        "LargestSector": sector_counts.index[0] if not sector_counts.empty else None,
    }


def _bootstrap(oof: pd.DataFrame) -> pd.DataFrame:
    baseline = oof[oof["Scheme"].eq("EXPANDING_BASELINE") & oof["buy_signal"]].copy()
    baseline_blocks = _date_blocks(baseline)
    all_dates = sorted(oof["Date"].drop_duplicates())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for scheme in [scheme.name for scheme in SCHEMES if scheme.name != "EXPANDING_BASELINE"]:
        candidate = oof[oof["Scheme"].eq(scheme) & oof["buy_signal"]].copy()
        candidate_blocks = _date_blocks(candidate)
        for iteration in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(all_dates, size=len(all_dates), replace=True)
            base = _bootstrap_metrics(baseline_blocks, sampled)
            cand = _bootstrap_metrics(candidate_blocks, sampled)
            rows.append(
                {
                    "Scheme": scheme,
                    "Iteration": iteration + 1,
                    "PrecisionDiffVsBaseline": cand["precision"] - base["precision"],
                    "AverageExcessDiffVsBaseline": cand["average_excess"] - base["average_excess"],
                    "MedianExcessDiffVsBaseline": cand["median_excess"] - base["median_excess"],
                    "NiftyWinRateDiffVsBaseline": cand["win_rate"] - base["win_rate"],
                    "SignalCountBaseline": int(base["count"]),
                    "SignalCountCandidate": int(cand["count"]),
                }
            )
    return pd.DataFrame(rows)


def _date_blocks(frame: pd.DataFrame) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    blocks = {}
    for date, group in frame.groupby("Date"):
        excess = group["excess_return"].to_numpy(dtype=float)
        blocks[pd.Timestamp(date)] = {
            "target": group["buy_target"].to_numpy(dtype=float),
            "excess": excess,
            "win": (excess > 0).astype(float),
        }
    return blocks


def _bootstrap_metrics(
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]], sampled_dates: np.ndarray
) -> dict[str, float]:
    targets = []
    excesses = []
    wins = []
    for date in sampled_dates:
        block = blocks.get(pd.Timestamp(date))
        if block is None:
            continue
        targets.append(block["target"])
        excesses.append(block["excess"])
        wins.append(block["win"])
    if not targets:
        return {
            "count": 0.0,
            "precision": np.nan,
            "average_excess": np.nan,
            "median_excess": np.nan,
            "win_rate": np.nan,
        }
    target = np.concatenate(targets)
    excess = np.concatenate(excesses)
    win = np.concatenate(wins)
    return {
        "count": float(len(target)),
        "precision": float(np.mean(target)),
        "average_excess": float(np.mean(excess)),
        "median_excess": float(np.median(excess)),
        "win_rate": float(np.mean(win)),
    }


def _decision_payload(comparison: pd.DataFrame, bootstrap: pd.DataFrame) -> dict[str, Any]:
    overall = comparison[comparison["Scope"].eq("overall")].set_index("Scheme")
    candidates = {}
    for scheme in [item.name for item in SCHEMES]:
        row = overall.loc[scheme]
        candidates[scheme] = row.to_dict()
    bootstrap_ci = {}
    for scheme, group in bootstrap.groupby("Scheme"):
        bootstrap_ci[scheme] = {
            "precision_diff_p025_p50_p975": _ci(group["PrecisionDiffVsBaseline"]),
            "average_excess_diff_p025_p50_p975": _ci(group["AverageExcessDiffVsBaseline"]),
            "median_excess_diff_p025_p50_p975": _ci(group["MedianExcessDiffVsBaseline"]),
            "nifty_win_rate_diff_p025_p50_p975": _ci(group["NiftyWinRateDiffVsBaseline"]),
        }

    baseline = overall.loc["EXPANDING_BASELINE"]
    best_name = max(
        overall.index,
        key=lambda name: (
            _nan_to_neg(overall.loc[name, "Precision"]),
            _nan_to_neg(overall.loc[name, "MedianExcessReturn"]),
            _nan_to_neg(overall.loc[name, "AverageExcessReturn"]),
        ),
    )
    if best_name == "EXPANDING_BASELINE":
        decision = "EXPANDING_WINDOW_BEST"
    else:
        ci = bootstrap_ci.get(best_name, {})
        precision_ci = ci.get("precision_diff_p025_p50_p975", [np.nan, np.nan, np.nan])
        excess_ci = ci.get("average_excess_diff_p025_p50_p975", [np.nan, np.nan, np.nan])
        best = overall.loc[best_name]
        materially_better = (
            best["Precision"] > baseline["Precision"]
            and best["AverageExcessReturn"] > baseline["AverageExcessReturn"]
            and precision_ci[0] > 0
            and excess_ci[0] >= 0
        )
        decision = _decision_name(best_name) if materially_better else "NO_CLEAR_IMPROVEMENT"
    return {
        "decision": decision,
        "best_point_estimate_scheme": best_name,
        "final_test_used": False,
        "v1_v2_production_modified": False,
        "recency_weighting_formula": (
            "weight = 0.5 ** (age_days / 730.5), normalized to mean 1 within the train fold"
        ),
        "candidates": candidates,
        "bootstrap_ci": bootstrap_ci,
    }


def _decision_name(scheme: str) -> str:
    return {
        "EXPANDING_BASELINE": "EXPANDING_WINDOW_BEST",
        "ROLLING_5Y": "ROLLING_5Y_BEST",
        "ROLLING_3Y": "ROLLING_3Y_BEST",
        "RECENCY_WEIGHTED": "RECENCY_WEIGHTED_BEST",
    }[scheme]


def _report(comparison: pd.DataFrame, bootstrap: pd.DataFrame, decision: dict[str, Any]) -> str:
    overall = comparison[comparison["Scope"].eq("overall")].copy()
    folds = comparison[comparison["Scope"].eq("fold")].copy()
    years = comparison[comparison["Scope"].eq("year")].copy()
    return "\n".join(
        [
            "NATIP Recency-Aware Training Experiment",
            "",
            f"Decision: {decision['decision']}",
            "Scope: original 150-stock universe, validation/out-of-fold only; final test not used.",
            "BUY rule unchanged: XGB26 Top1% AND Binary77 Top0.5%.",
            "Recency weighted formula: weight = 0.5 ** (age_days / 730.5), normalized to train-fold mean 1.",
            "",
            "Overall comparison:",
            overall.to_string(index=False),
            "",
            "Fold stability:",
            folds.to_string(index=False),
            "",
            "Year stability:",
            years.to_string(index=False),
            "",
            "Bootstrap confidence intervals vs EXPANDING_BASELINE:",
            json.dumps(decision["bootstrap_ci"], indent=2),
            "",
            f"Bootstrap rows: {len(bootstrap)}",
            "Production V1/V2 artifacts were not modified.",
        ]
    )


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 selected features, got {len(features)}")
    return features


def _assert_features(dataset: pd.DataFrame, features_26: list[str], features_77: list[str]) -> None:
    missing = [feature for feature in [*features_26, *features_77] if feature not in dataset.columns]
    if missing:
        raise AssertionError(f"Missing features: {missing}")
    if len(features_77) != 77:
        raise AssertionError(f"Expected 77 frozen Binary77 features, got {len(features_77)}")


def _ci(series: pd.Series) -> list[float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return [np.nan, np.nan, np.nan]
    return [float(clean.quantile(q)) for q in [0.025, 0.5, 0.975]]


def _nan_to_neg(value: Any) -> float:
    return -999.0 if value is None or pd.isna(value) else float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


if __name__ == "__main__":
    main()
