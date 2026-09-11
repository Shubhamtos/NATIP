"""Validation-only point-in-time fundamental ML experiment for NATIP.

This script is deliberately research-only. It reads the clean technical
dataset and the ML-safe point-in-time fundamental parquet files, generates
walk-forward OOF predictions, and writes diagnostic reports without touching
production V1/V2 artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
from run_stock_specific_feature_variant import _common_evaluation_rows, _prepare_dataset

CLEAN_150_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"
FROZEN_CONFIG = REPORT_DIR / "frozen_model_config.json"
CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"
TUNED_OOF = REPORT_DIR / "lstm_tuned_binary77_oof_cache.csv"

PIT_VERIFIED = Path("data/fundamentals/exchange_publication_dates/PIT_verified_only.parquet")
PIT_MAX_SAFE = Path("data/fundamentals/exchange_publication_dates/PIT_max_safe.parquet")
SAFE_RESTATEMENTS = Path("outputs/safe_late_restatements.csv")
RAW_QUARTERLY = Path("data/fundamentals/raw/screener_quarterly.parquet")
AVAILABILITY_MASTER = Path("outputs/publication_availability_master.csv")

OUTPUT_RESTATEMENT = REPORT_DIR / "restatement_resolution.csv"
OUTPUT_COVERAGE = REPORT_DIR / "fundamental_feature_coverage.csv"
OUTPUT_SELECTED = REPORT_DIR / "fundamental_selected_features.json"
OUTPUT_FUND_VERIFIED = REPORT_DIR / "fundamental_only_verified_oof.parquet"
OUTPUT_FUND_MAXSAFE = REPORT_DIR / "fundamental_only_maxsafe_oof.parquet"
OUTPUT_XGB_VERIFIED = REPORT_DIR / "xgb26_fund_verified_oof.parquet"
OUTPUT_XGB_MAXSAFE = REPORT_DIR / "xgb26_fund_maxsafe_oof.parquet"
OUTPUT_TOPK = REPORT_DIR / "fundamental_topk_comparison.csv"
OUTPUT_BUY_CONFIRMATION = REPORT_DIR / "fundamental_buy_confirmation.csv"
OUTPUT_FALSE_POSITIVE = REPORT_DIR / "fundamental_false_positive_analysis.csv"
OUTPUT_VERIFIED_VS_MAXSAFE = REPORT_DIR / "fundamental_verified_vs_maxsafe.csv"
OUTPUT_FOLD = REPORT_DIR / "fundamental_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "fundamental_year_stability.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "fundamental_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "fundamental_ml_report.txt"

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
TOP_K = (1, 3, 5, 10)
CONFIRMATION_BUCKETS = (0.50, 0.30, 0.20)

PREFERRED_FUNDAMENTALS = [
    "sales_growth_yoy",
    "profit_growth_yoy",
    "eps_growth_yoy",
    "sales_growth_acceleration",
    "profit_growth_acceleration",
    "operating_margin",
    "operating_margin_change_yoy",
    "roe",
    "roce",
    "debt_to_equity",
    "interest_coverage",
    "operating_cashflow_to_net_profit",
    "fundamental_data_age_days",
]


@dataclass(frozen=True, slots=True)
class Variant:
    """Experiment variant."""

    name: str
    display_name: str
    pit_path: Path
    technical_features: list[str]
    fundamental_features: list[str]
    use_technical: bool
    output_path: Path

    @property
    def features(self) -> list[str]:
        """Return ordered model features."""

        return [*self.technical_features, *self.fundamental_features]


def main() -> None:
    """Run the first PIT fundamental ML experiment."""

    ensure_probability_dirs()
    restatement = _resolve_restatements()
    restatement.to_csv(OUTPUT_RESTATEMENT, index=False)

    technical = _load_technical_dataset()
    pre_final = technical[technical["Date"] < FINAL_TEST_START].copy()
    features_26 = _selected_26_features()
    features_77 = _frozen_77_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)

    verified = _merge_pit(pre_final, PIT_VERIFIED)
    maxsafe = _merge_pit(pre_final, PIT_MAX_SAFE)
    coverage = _fundamental_coverage(verified, maxsafe)
    coverage.to_csv(OUTPUT_COVERAGE, index=False)
    selected_fundamentals = _select_fundamentals(coverage)
    _write_selected_features(selected_fundamentals, features_26)

    variants = [
        Variant(
            "FUND_ONLY_VERIFIED",
            "Fundamental-only VERIFIED_ONLY",
            PIT_VERIFIED,
            [],
            selected_fundamentals,
            False,
            OUTPUT_FUND_VERIFIED,
        ),
        Variant(
            "FUND_ONLY_MAX_SAFE",
            "Fundamental-only MAX_SAFE",
            PIT_MAX_SAFE,
            [],
            selected_fundamentals,
            False,
            OUTPUT_FUND_MAXSAFE,
        ),
        Variant(
            "XGB26_PLUS_FUND_VERIFIED",
            "XGB26 + fundamentals VERIFIED_ONLY",
            PIT_VERIFIED,
            features_26,
            selected_fundamentals,
            True,
            OUTPUT_XGB_VERIFIED,
        ),
        Variant(
            "XGB26_PLUS_FUND_MAX_SAFE",
            "XGB26 + fundamentals MAX_SAFE",
            PIT_MAX_SAFE,
            features_26,
            selected_fundamentals,
            True,
            OUTPUT_XGB_MAXSAFE,
        ),
    ]

    predictions = []
    for variant in variants:
        print(f"[fundamental-ml] {variant.name}", flush=True)
        dataset = verified if variant.pit_path == PIT_VERIFIED else maxsafe
        pred = _walk_forward_oof(dataset, folds, variant)
        pred.to_parquet(variant.output_path, index=False)
        predictions.append(pred)

    all_predictions = pd.concat(predictions, ignore_index=True)
    baseline = _load_current_oof()
    topk = _topk_comparison(all_predictions, baseline)
    fold = _scoped_stability(all_predictions, baseline, "fold")
    year = _scoped_stability(all_predictions, baseline, "year")
    buy_confirmation = _buy_confirmation(all_predictions, baseline)
    false_positive = _false_positive_analysis(maxsafe, baseline, selected_fundamentals)
    verified_vs_maxsafe = _verified_vs_maxsafe(topk, buy_confirmation)
    bootstrap = _bootstrap(all_predictions, baseline)
    decision = _decision(topk, buy_confirmation, bootstrap)

    topk.to_csv(OUTPUT_TOPK, index=False)
    buy_confirmation.to_csv(OUTPUT_BUY_CONFIRMATION, index=False)
    false_positive.to_csv(OUTPUT_FALSE_POSITIVE, index=False)
    verified_vs_maxsafe.to_csv(OUTPUT_VERIFIED_VS_MAXSAFE, index=False)
    fold.to_csv(OUTPUT_FOLD, index=False)
    year.to_csv(OUTPUT_YEAR, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            decision=decision,
            restatement=restatement,
            coverage=coverage,
            selected_fundamentals=selected_fundamentals,
            topk=topk,
            buy_confirmation=buy_confirmation,
            false_positive=false_positive,
            verified_vs_maxsafe=verified_vs_maxsafe,
            bootstrap=bootstrap,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "restatement_resolution": str(OUTPUT_RESTATEMENT),
                "fundamental_feature_coverage": str(OUTPUT_COVERAGE),
                "fundamental_selected_features": str(OUTPUT_SELECTED),
                "fundamental_topk_comparison": str(OUTPUT_TOPK),
                "fundamental_buy_confirmation": str(OUTPUT_BUY_CONFIRMATION),
                "fundamental_ml_report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )


def _load_technical_dataset() -> pd.DataFrame:
    dataset = _prepare_dataset(pd.read_csv(CLEAN_150_DATASET, parse_dates=["Date"]))
    dataset = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    return dataset


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 selected features, got {len(features)}")
    return features


def _frozen_77_features() -> list[str]:
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    features = list(payload["feature_list"])
    if len(features) != 77:
        raise AssertionError(f"Expected 77 frozen features, got {len(features)}")
    return features


def _merge_pit(technical: pd.DataFrame, pit_path: Path) -> pd.DataFrame:
    pit = pd.read_parquet(pit_path)
    pit = pit.rename(columns={"ticker": "symbol"})
    pit["Date"] = pd.to_datetime(pit["Date"])
    pit["period_end"] = pd.to_datetime(pit["period_end"], errors="coerce")
    pit["publication_date"] = pd.to_datetime(pit["publication_date"], errors="coerce")
    columns = [
        "Date",
        "symbol",
        "period_end",
        "publication_date",
        "company_financial_type",
        "statement_basis",
        "publication_date_quality",
        "restatement_risk_flag",
        *[feature for feature in PREFERRED_FUNDAMENTALS if feature in pit.columns],
    ]
    merged = technical.merge(pit[columns], on=["Date", "symbol"], how="left")
    dated = merged["publication_date"].notna()
    if (merged.loc[dated, "publication_date"] > merged.loc[dated, "Date"]).any():
        raise AssertionError(f"PIT leakage detected in {pit_path}")
    return merged


def _resolve_restatements() -> pd.DataFrame:
    conflicts = pd.read_csv(SAFE_RESTATEMENTS) if SAFE_RESTATEMENTS.exists() else pd.DataFrame()
    raw = pd.read_parquet(RAW_QUARTERLY) if RAW_QUARTERLY.exists() else pd.DataFrame()
    availability = (
        pd.read_csv(AVAILABILITY_MASTER, parse_dates=["period_end"])
        if AVAILABILITY_MASTER.exists()
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    if conflicts.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "period_end",
                "result_type",
                "version_id",
                "version_type",
                "published_values",
                "revised_or_current_values",
                "publication_date",
                "ml_effective_date",
                "ml_eligible",
                "resolution_status",
                "notes",
                "source_reference",
            ]
        )

    for item in conflicts.itertuples(index=False):
        period_end = pd.Timestamp(item.period_end)
        raw_values = _raw_values_for_period(raw, item.ticker, period_end)
        exact = availability[
            (availability["ticker"].eq(item.ticker))
            & (pd.to_datetime(availability["period_end"]).eq(period_end))
            & (availability["availability_method"].eq("VERIFIED_PUBLICATION"))
        ]
        if exact.empty:
            rows.append(
                {
                    "ticker": item.ticker,
                    "period_end": period_end.date().isoformat(),
                    "result_type": item.result_type,
                    "version_id": 1,
                    "version_type": "CURRENT_SCREENER_VALUES_UNVERIFIED_VERSION",
                    "published_values": None,
                    "revised_or_current_values": json.dumps(raw_values, sort_keys=True),
                    "publication_date": None,
                    "ml_effective_date": None,
                    "ml_eligible": False,
                    "resolution_status": "EXCLUDED_NO_DEFENSIBLE_TIME_VERSION",
                    "notes": (
                        "SAFE_LATE candidate conflicted or values were not found in the "
                        "official document. Current Screener values are retained only for "
                        "audit, not backdated into ML."
                    ),
                    "source_reference": item.source_reference,
                }
            )
        else:
            for version, exact_row in enumerate(exact.itertuples(index=False), start=1):
                rows.append(
                    {
                        "ticker": item.ticker,
                        "period_end": period_end.date().isoformat(),
                        "result_type": item.result_type,
                        "version_id": version,
                        "version_type": "VERIFIED_PUBLICATION",
                        "published_values": json.dumps(raw_values, sort_keys=True),
                        "revised_or_current_values": None,
                        "publication_date": exact_row.publication_date_exact,
                        "ml_effective_date": exact_row.ml_effective_date,
                        "ml_eligible": True,
                        "resolution_status": "EXACT_VERSION_AVAILABLE",
                        "notes": "Exact verified publication remains ML-eligible.",
                        "source_reference": exact_row.source_reference,
                    }
                )
    return pd.DataFrame(rows)


def _raw_values_for_period(
    raw: pd.DataFrame, ticker: str, period_end: pd.Timestamp
) -> dict[str, Any]:
    if raw.empty:
        return {}
    subset = raw[
        (raw["ticker"].eq(ticker))
        & (pd.to_datetime(raw["period_end"]).eq(period_end))
        & (
            raw["normalized_field_name"].isin(
                PREFERRED_FUNDAMENTALS + ["sales", "net_profit", "eps"]
            )
        )
    ]
    if subset.empty:
        return {}
    return {
        str(row.normalized_field_name): _json_safe(row.numeric_value)
        for row in subset.itertuples(index=False)
    }


def _fundamental_coverage(verified: pd.DataFrame, maxsafe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in PREFERRED_FUNDAMENTALS:
        if feature not in verified.columns or feature not in maxsafe.columns:
            continue
        for name, frame in [("VERIFIED_ONLY", verified), ("MAX_SAFE", maxsafe)]:
            non_null = frame[feature].notna()
            usable = frame[non_null]
            report_age = pd.to_numeric(frame.get("fundamental_data_age_days"), errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "dataset": name,
                    "rows": int(len(frame)),
                    "non_null_rows": int(non_null.sum()),
                    "non_null_pct": _safe_float(non_null.mean()),
                    "stocks_covered": int(usable["symbol"].nunique()),
                    "years_covered": (
                        int(usable["Date"].dt.year.nunique()) if not usable.empty else 0
                    ),
                    "median_report_age": _safe_float(report_age[non_null].median()),
                    "p90_report_age": _safe_float(report_age[non_null].quantile(0.90)),
                }
            )
    pivot = pd.DataFrame(rows)
    if pivot.empty:
        return pivot
    verified_cov = pivot[pivot["dataset"].eq("VERIFIED_ONLY")][["feature", "non_null_pct"]]
    maxsafe_cov = pivot[pivot["dataset"].eq("MAX_SAFE")][["feature", "non_null_pct"]]
    verified_cov = verified_cov.rename(columns={"non_null_pct": "coverage_verified_only"})
    maxsafe_cov = maxsafe_cov.rename(columns={"non_null_pct": "coverage_max_safe"})
    return pivot.merge(verified_cov, on="feature", how="left").merge(
        maxsafe_cov, on="feature", how="left"
    )


def _select_fundamentals(coverage: pd.DataFrame) -> list[str]:
    if coverage.empty:
        raise RuntimeError("No fundamental coverage rows available.")
    maxsafe = coverage[coverage["dataset"].eq("MAX_SAFE")].copy()
    selected = [
        row.feature
        for row in maxsafe.itertuples(index=False)
        if pd.notna(row.non_null_pct) and float(row.non_null_pct) >= 0.05
    ]
    selected = [feature for feature in PREFERRED_FUNDAMENTALS if feature in selected]
    if len(selected) < 3:
        selected = [
            feature for feature in PREFERRED_FUNDAMENTALS if feature in maxsafe["feature"].values
        ]
    return selected[:15]


def _write_selected_features(fundamentals: list[str], technical: list[str]) -> None:
    payload = {
        "experiment": "v4_point_in_time_fundamental_ml_research",
        "final_test_used": False,
        "production_artifacts_modified": False,
        "technical_feature_count": len(technical),
        "technical_features": technical,
        "fundamental_feature_count": len(fundamentals),
        "fundamental_features": fundamentals,
        "pit_inputs": {
            "VERIFIED_ONLY": str(PIT_VERIFIED),
            "MAX_SAFE": str(PIT_MAX_SAFE),
        },
        "unresolved_period_policy": "excluded",
        "safe_late_policy": "available only from conservative ml_effective_date in MAX_SAFE",
    }
    OUTPUT_SELECTED.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _walk_forward_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], variant: Variant
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        if variant.use_technical:
            model, medians = _fit_classifier(train, variant.features, target="label", binary=False)
            probabilities = model.predict_proba(
                _transform_with_medians(validation, variant.features, medians)
            )
            output = _base_prediction_frame(validation, fold["fold"], variant)
            output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
            output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
            output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        else:
            fit_frame, calibration_frame = _fit_calibration_split(train)
            model, medians = _fit_classifier(
                fit_frame,
                variant.features,
                target="buy_target",
                binary=True,
            )
            cal_prob = _positive_probability(
                model, _transform_with_medians(calibration_frame, variant.features, medians)
            )
            raw_prob = _positive_probability(
                model, _transform_with_medians(validation, variant.features, medians)
            )
            platt = _fit_platt(cal_prob, calibration_frame["buy_target"])
            output = _base_prediction_frame(validation, fold["fold"], variant)
            output["p_outperform"] = platt.predict_proba(raw_prob.reshape(-1, 1))[:, 1]
            output["p_neutral"] = np.nan
            output["p_underperform"] = np.nan
        output["score_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        output["fundamental_non_null_count"] = (
            validation[variant.fundamental_features].notna().sum(axis=1).to_numpy()
        )
        output["fundamental_report_age_days"] = validation.get(
            "fundamental_data_age_days", pd.Series(index=validation.index, dtype=float)
        ).to_numpy()
        output["strict_buy_with_existing_binary"] = False
        frames.append(output)
    return pd.concat(frames, ignore_index=True)


def _fit_classifier(
    train: pd.DataFrame,
    features: list[str],
    *,
    target: str,
    binary: bool,
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
    model.fit(x, y)
    return model, medians


def _base_prediction_frame(frame: pd.DataFrame, fold: Any, variant: Variant) -> pd.DataFrame:
    output = frame[
        [
            "Date",
            "symbol",
            "Sector",
            "MarketCapCategory",
            "future_stock_return",
            "future_nifty_return",
            "excess_return",
            "label",
            "buy_target",
            "market_regime_label",
        ]
    ].copy()
    output["fold"] = fold
    output["Variant"] = variant.name
    output["VariantName"] = variant.display_name
    output["FeatureCount"] = len(variant.features)
    output["FundamentalFeatureCount"] = len(variant.fundamental_features)
    return output


def _load_current_oof() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    current = current[current["Scheme"].eq("EXPANDING_BASELINE")].copy()
    current = current[current["Date"] < FINAL_TEST_START].copy()
    current["current_buy_signal"] = (current["classifier_percentile"] >= 1 - CLASSIFIER_TOP) & (
        current["binary_buy_percentile"] >= 1 - BINARY_TOP
    )
    tuned = pd.read_csv(TUNED_OOF, parse_dates=["Date"]) if TUNED_OOF.exists() else pd.DataFrame()
    if not tuned.empty:
        current = current.merge(tuned, on=["Date", "symbol", "fold"], how="left")
    else:
        current["tuned_binary_percentile"] = np.nan
    current["current_tuned_high_confidence"] = current["current_buy_signal"] & (
        current["tuned_binary_percentile"] >= 1 - BINARY_TOP
    )
    return current


def _topk_comparison(predictions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(_topk_rows(baseline, "EXISTING_XGB26_BASELINE", "p_outperform"))
    for variant, group in predictions.groupby("Variant"):
        rows.extend(_topk_rows(group, variant, "p_outperform"))
    merged = _merge_with_binary(predictions, baseline)
    for variant, group in merged.groupby("Variant"):
        group = group.copy()
        group["strict_buy_with_existing_binary"] = (
            group["score_percentile"] >= 1 - CLASSIFIER_TOP
        ) & (group["binary_buy_percentile"] >= 1 - BINARY_TOP)
        selected = group[group["strict_buy_with_existing_binary"]]
        rows.append(_selection_row(variant, "strict_buy_existing_binary", selected, group))
    return pd.DataFrame(rows)


def _topk_rows(frame: pd.DataFrame, model: str, score: str) -> list[dict[str, Any]]:
    rows = []
    for k in TOP_K:
        daily = []
        for date, group in frame.groupby("Date"):
            selected = group.sort_values(score, ascending=False).head(k)
            winners = int(group["buy_target"].sum())
            daily.append(
                {
                    "Date": date,
                    "precision": selected["buy_target"].mean(),
                    "recall": selected["buy_target"].sum() / winners if winners else np.nan,
                    "winner_coverage": selected["buy_target"].sum() > 0,
                    "avg_excess": selected["excess_return"].mean(),
                    "median_excess": selected["excess_return"].median(),
                    "nifty_win": (selected["excess_return"] > 0).mean(),
                    "abs_return": selected["future_stock_return"].mean(),
                }
            )
        daily_frame = pd.DataFrame(daily)
        rows.append(
            {
                "Model": model,
                "MetricType": "daily_topk",
                "K": k,
                "Rows": int(len(frame)),
                "Dates": int(frame["Date"].nunique()),
                "Selections": int(frame["Date"].nunique() * k),
                "Precision": _safe_float(daily_frame["precision"].mean()),
                "Recall": _safe_float(daily_frame["recall"].mean()),
                "WinnerCoverage": _safe_float(daily_frame["winner_coverage"].mean()),
                "AverageExcessReturn": _safe_float(daily_frame["avg_excess"].mean()),
                "MedianExcessReturn": _safe_float(daily_frame["median_excess"].median()),
                "NiftyWinRate": _safe_float(daily_frame["nifty_win"].mean()),
                "AverageFutureStockReturn": _safe_float(daily_frame["abs_return"].mean()),
                "PRAUC": _pr_auc(frame, score),
            }
        )
    return rows


def _selection_row(
    model: str, metric_type: str, selected: pd.DataFrame, population: pd.DataFrame
) -> dict[str, Any]:
    return {
        "Model": model,
        "MetricType": metric_type,
        "K": np.nan,
        "Rows": int(len(population)),
        "Dates": int(population["Date"].nunique()),
        "Selections": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
        "Recall": (
            _safe_float(selected["buy_target"].sum() / population["buy_target"].sum())
            if population["buy_target"].sum()
            else None
        ),
        "WinnerCoverage": (
            _safe_float(selected.groupby("Date")["buy_target"].sum().gt(0).mean())
            if not selected.empty
            else None
        ),
        "AverageExcessReturn": (
            _safe_float(selected["excess_return"].mean()) if not selected.empty else None
        ),
        "MedianExcessReturn": (
            _safe_float(selected["excess_return"].median()) if not selected.empty else None
        ),
        "NiftyWinRate": (
            _safe_float((selected["excess_return"] > 0).mean()) if not selected.empty else None
        ),
        "AverageFutureStockReturn": (
            _safe_float(selected["future_stock_return"].mean()) if not selected.empty else None
        ),
        "PRAUC": _pr_auc(population, "p_outperform") if "p_outperform" in population else None,
    }


def _merge_with_binary(predictions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date",
        "symbol",
        "fold",
        "binary_buy_percentile",
        "current_buy_signal",
        "current_tuned_high_confidence",
    ]
    return predictions.merge(baseline[columns], on=["Date", "symbol", "fold"], how="inner")


def _scoped_stability(
    predictions: pd.DataFrame, baseline: pd.DataFrame, scope: str
) -> pd.DataFrame:
    rows = []
    baseline = baseline.copy()
    baseline["Variant"] = "EXISTING_XGB26_BASELINE"
    for value, group in baseline.groupby(scope if scope != "year" else baseline["Date"].dt.year):
        rows.extend(
            _scoped_topk_rows(group, "EXISTING_XGB26_BASELINE", "p_outperform", scope, value)
        )
    for variant, frame in predictions.groupby("Variant"):
        for value, group in frame.groupby(scope if scope != "year" else frame["Date"].dt.year):
            rows.extend(_scoped_topk_rows(group, variant, "p_outperform", scope, value))
    return pd.DataFrame(rows)


def _scoped_topk_rows(
    frame: pd.DataFrame, model: str, score: str, scope: str, value: Any
) -> list[dict[str, Any]]:
    rows = _topk_rows(frame, model, score)
    for row in rows:
        row["Scope"] = scope
        row["ScopeValue"] = value
    return rows


def _buy_confirmation(predictions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    merged = predictions.merge(
        baseline[
            [
                "Date",
                "symbol",
                "fold",
                "current_buy_signal",
                "current_tuned_high_confidence",
                "buy_target",
                "excess_return",
                "future_stock_return",
            ]
        ],
        on=["Date", "symbol", "fold"],
        suffixes=("", "_baseline"),
        how="inner",
    )
    for variant, frame in merged.groupby("Variant"):
        for signal_col, signal_name in [
            ("current_buy_signal", "Current BUY"),
            ("current_tuned_high_confidence", "Current+Tuned HIGH"),
        ]:
            signal = frame[frame[signal_col]].copy()
            rows.extend(_fundamental_probability_buckets(variant, signal_name, signal))
            for fold, group in signal.groupby("fold"):
                rows.extend(
                    _fundamental_probability_buckets(variant, signal_name, group, "fold", fold)
                )
            for year, group in signal.groupby(signal["Date"].dt.year):
                rows.extend(
                    _fundamental_probability_buckets(variant, signal_name, group, "year", int(year))
                )
    return pd.DataFrame(rows)


def _fundamental_probability_buckets(
    variant: str,
    signal_name: str,
    frame: pd.DataFrame,
    scope: str = "overall",
    scope_value: Any = "all",
) -> list[dict[str, Any]]:
    rows = []
    if frame.empty:
        return rows
    ranked = frame.sort_values("p_outperform", ascending=False).copy()
    for fraction in CONFIRMATION_BUCKETS:
        selected = ranked.head(max(1, int(len(ranked) * fraction)))
        rows.append(
            _confirmation_row(
                variant, signal_name, f"Top{fraction * 100:g}%", selected, scope, scope_value
            )
        )
    bottom = ranked.tail(max(1, int(len(ranked) * 0.50)))
    rows.append(_confirmation_row(variant, signal_name, "Bottom50%", bottom, scope, scope_value))
    return rows


def _confirmation_row(
    variant: str,
    signal_name: str,
    bucket: str,
    selected: pd.DataFrame,
    scope: str,
    scope_value: Any,
) -> dict[str, Any]:
    return {
        "Variant": variant,
        "SignalSet": signal_name,
        "Bucket": bucket,
        "Scope": scope,
        "ScopeValue": scope_value,
        "Count": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
    }


def _false_positive_analysis(
    maxsafe: pd.DataFrame, baseline: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    current_buy = baseline[baseline["current_buy_signal"]][["Date", "symbol", "buy_target"]]
    frame = maxsafe.merge(current_buy, on=["Date", "symbol"], how="inner", suffixes=("", "_base"))
    rows = []
    for feature in features:
        true = pd.to_numeric(frame.loc[frame["buy_target_base"].eq(1), feature], errors="coerce")
        false = pd.to_numeric(frame.loc[frame["buy_target_base"].eq(0), feature], errors="coerce")
        values = pd.to_numeric(frame[feature], errors="coerce")
        valid = values.notna()
        auc = None
        if valid.sum() and frame.loc[valid, "buy_target_base"].nunique() == 2:
            try:
                auc = float(roc_auc_score(frame.loc[valid, "buy_target_base"], values[valid]))
            except ValueError:
                auc = None
        pooled_std = values.std()
        rows.append(
            {
                "feature": feature,
                "true_count": int(true.notna().sum()),
                "false_count": int(false.notna().sum()),
                "median_true": _safe_float(true.median()),
                "median_false": _safe_float(false.median()),
                "standardized_difference": (
                    _safe_float((true.median() - false.median()) / pooled_std)
                    if pooled_std and pd.notna(pooled_std)
                    else None
                ),
                "univariate_auc": auc,
                "missing_true_pct": _safe_float(true.isna().mean()),
                "missing_false_pct": _safe_float(false.isna().mean()),
                "missingness_difference": _safe_float(true.isna().mean() - false.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("univariate_auc", ascending=False, na_position="last")


def _verified_vs_maxsafe(topk: pd.DataFrame, buy_confirmation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("FUND_ONLY_VERIFIED", "FUND_ONLY_MAX_SAFE"),
        ("XGB26_PLUS_FUND_VERIFIED", "XGB26_PLUS_FUND_MAX_SAFE"),
    ]
    for verified, maxsafe in pairs:
        for metric_type in ["daily_topk", "strict_buy_existing_binary"]:
            v = topk[(topk["Model"].eq(verified)) & (topk["MetricType"].eq(metric_type))]
            m = topk[(topk["Model"].eq(maxsafe)) & (topk["MetricType"].eq(metric_type))]
            for k in sorted(set(v["K"].dropna()) | set(m["K"].dropna())) or [np.nan]:
                vr = v[v["K"].eq(k)].head(1) if pd.notna(k) else v.head(1)
                mr = m[m["K"].eq(k)].head(1) if pd.notna(k) else m.head(1)
                if vr.empty or mr.empty:
                    continue
                rows.append(
                    {
                        "Comparison": f"{maxsafe}_minus_{verified}",
                        "MetricType": metric_type,
                        "K": k,
                        "PrecisionDiff": _diff(mr, vr, "Precision"),
                        "WinnerCoverageDiff": _diff(mr, vr, "WinnerCoverage"),
                        "AverageExcessDiff": _diff(mr, vr, "AverageExcessReturn"),
                        "MedianExcessDiff": _diff(mr, vr, "MedianExcessReturn"),
                        "NiftyWinRateDiff": _diff(mr, vr, "NiftyWinRate"),
                    }
                )
    confirm_overall = buy_confirmation[buy_confirmation["Scope"].eq("overall")]
    for signal in confirm_overall["SignalSet"].drop_duplicates():
        for bucket in confirm_overall["Bucket"].drop_duplicates():
            v = confirm_overall[
                confirm_overall["Variant"].eq("FUND_ONLY_VERIFIED")
                & confirm_overall["SignalSet"].eq(signal)
                & confirm_overall["Bucket"].eq(bucket)
            ]
            m = confirm_overall[
                confirm_overall["Variant"].eq("FUND_ONLY_MAX_SAFE")
                & confirm_overall["SignalSet"].eq(signal)
                & confirm_overall["Bucket"].eq(bucket)
            ]
            if not v.empty and not m.empty:
                rows.append(
                    {
                        "Comparison": "FUND_ONLY_MAX_SAFE_minus_FUND_ONLY_VERIFIED",
                        "MetricType": f"buy_confirmation_{signal}_{bucket}",
                        "K": np.nan,
                        "PrecisionDiff": _diff(m, v, "Precision"),
                        "WinnerCoverageDiff": np.nan,
                        "AverageExcessDiff": _diff(m, v, "AverageExcessReturn"),
                        "MedianExcessDiff": _diff(m, v, "MedianExcessReturn"),
                        "NiftyWinRateDiff": _diff(m, v, "NiftyWinRate"),
                    }
                )
    return pd.DataFrame(rows)


def _diff(a: pd.DataFrame, b: pd.DataFrame, column: str) -> float | None:
    av = a.iloc[0].get(column)
    bv = b.iloc[0].get(column)
    if pd.isna(av) or pd.isna(bv):
        return None
    return float(av - bv)


def _bootstrap(predictions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    dates = sorted(baseline["Date"].drop_duplicates())
    baseline_topk = {k: _daily_topk_arrays(baseline, "p_outperform", k) for k in (3, 5)}
    merged = _merge_with_binary(predictions, baseline)
    for variant, group in predictions.groupby("Variant"):
        topk_blocks = {k: _daily_topk_arrays(group, "p_outperform", k) for k in (3, 5)}
        for metric in ["Top3", "Top5"]:
            k = int(metric.replace("Top", ""))
            for iteration in range(BOOTSTRAP_SAMPLES):
                sampled = rng.choice(dates, size=len(dates), replace=True)
                base = _bootstrap_array_metric(baseline_topk[k], sampled)
                cand = _bootstrap_array_metric(topk_blocks[k], sampled)
                rows.append(_bootstrap_row(variant, metric, iteration, cand, base))
    current_selected = baseline[baseline["current_buy_signal"]].copy()
    current_blocks = _selection_arrays(current_selected)
    for variant, group in merged.groupby("Variant"):
        group = group.copy()
        selected = group[
            (group["score_percentile"] >= 1 - CLASSIFIER_TOP)
            & (group["binary_buy_percentile"] >= 1 - BINARY_TOP)
        ]
        cand_blocks = _selection_arrays(selected)
        for iteration in range(BOOTSTRAP_SAMPLES):
            sampled = rng.choice(dates, size=len(dates), replace=True)
            base = _bootstrap_array_metric(current_blocks, sampled)
            cand = _bootstrap_array_metric(cand_blocks, sampled)
            rows.append(_bootstrap_row(variant, "StrictBUY", iteration, cand, base))
    return pd.DataFrame(rows)


def _daily_topk_arrays(
    frame: pd.DataFrame, score: str, k: int
) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    blocks = {}
    for date, group in frame.groupby("Date"):
        selected = group.sort_values(score, ascending=False).head(k)
        blocks[pd.Timestamp(date)] = {
            "target": selected["buy_target"].to_numpy(dtype=float),
            "excess": selected["excess_return"].to_numpy(dtype=float),
        }
    return blocks


def _selection_arrays(frame: pd.DataFrame) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    blocks = {}
    for date, group in frame.groupby("Date"):
        blocks[pd.Timestamp(date)] = {
            "target": group["buy_target"].to_numpy(dtype=float),
            "excess": group["excess_return"].to_numpy(dtype=float),
        }
    return blocks


def _bootstrap_array_metric(
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]], dates: np.ndarray
) -> dict[str, float]:
    targets = []
    excesses = []
    coverages = []
    for date in dates:
        group = blocks.get(pd.Timestamp(date))
        if group is not None and len(group["target"]):
            targets.append(group["target"])
            excesses.append(group["excess"])
            coverages.append(float(np.sum(group["target"]) > 0))
    if not targets:
        return {
            "precision": np.nan,
            "avg_excess": np.nan,
            "median_excess": np.nan,
            "winner_coverage": np.nan,
        }
    target = np.concatenate(targets)
    excess = np.concatenate(excesses)
    return {
        "precision": float(np.mean(target)),
        "avg_excess": float(np.mean(excess)),
        "median_excess": float(np.median(excess)),
        "winner_coverage": float(np.mean(coverages)),
    }


def _bootstrap_row(
    variant: str,
    metric: str,
    iteration: int,
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, Any]:
    return {
        "Variant": variant,
        "Metric": metric,
        "Iteration": iteration + 1,
        "PrecisionDiff": candidate["precision"] - baseline["precision"],
        "Top5WinnerCoverageDiff": (
            candidate["winner_coverage"] - baseline["winner_coverage"]
            if metric == "Top5"
            else np.nan
        ),
        "AverageExcessDiff": candidate["avg_excess"] - baseline["avg_excess"],
        "MedianExcessDiff": candidate["median_excess"] - baseline["median_excess"],
    }


def _decision(topk: pd.DataFrame, buy_confirmation: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    strict = topk[topk["MetricType"].eq("strict_buy_existing_binary")]
    baseline_top5 = topk[topk["Model"].eq("EXISTING_XGB26_BASELINE") & topk["K"].eq(5)].head(1)
    best_plus = strict[strict["Model"].str.contains("XGB26_PLUS", na=False)].copy()
    if not best_plus.empty:
        best_plus = best_plus.sort_values(
            ["Precision", "MedianExcessReturn", "AverageExcessReturn"],
            ascending=False,
        ).head(1)
    top5_plus = topk[topk["Model"].str.contains("XGB26_PLUS", na=False) & topk["K"].eq(5)]
    confirmation = buy_confirmation[
        (buy_confirmation["Scope"].eq("overall"))
        & (buy_confirmation["SignalSet"].eq("Current BUY"))
        & (buy_confirmation["Bucket"].isin(["Top20%", "Top30%", "Top50%", "Bottom50%"]))
    ]
    confirms_separation = False
    if not confirmation.empty:
        pivot = confirmation.pivot_table(
            index="Variant", columns="Bucket", values="Precision", aggfunc="first"
        )
        confirms_separation = bool(
            (
                (
                    pivot.get("Top20%", pd.Series(dtype=float))
                    - pivot.get("Bottom50%", pd.Series(dtype=float))
                )
                > 0.03
            ).any()
        )
    robust_top5 = False
    if not top5_plus.empty and not baseline_top5.empty:
        candidate = top5_plus.sort_values("Precision", ascending=False).head(1)
        robust_top5 = (
            float(candidate.iloc[0]["Precision"]) > float(baseline_top5.iloc[0]["Precision"])
            and _ci_low(bootstrap, candidate.iloc[0]["Model"], "Top5", "PrecisionDiff") > 0
        )
    robust_strict = False
    if not best_plus.empty:
        robust_strict = (
            _ci_low(bootstrap, best_plus.iloc[0]["Model"], "StrictBUY", "PrecisionDiff") > 0
        )
    maxsafe = topk[topk["Model"].str.contains("MAX_SAFE", na=False)]
    verified = topk[topk["Model"].str.contains("VERIFIED", na=False)]
    if not maxsafe.empty and not verified.empty:
        safe_late_help = maxsafe["Precision"].mean() > verified["Precision"].mean()
    else:
        safe_late_help = False
    if robust_top5 and robust_strict:
        return "FUNDAMENTALS_ADD_ROBUST_VALUE"
    if confirms_separation:
        return "FUNDAMENTALS_ADD_VALUE_AS_CONFIRMATION"
    if not safe_late_help:
        return "SAFE_LATE_DOES_NOT_HELP"
    if not top5_plus.empty:
        return "FUNDAMENTALS_DIAGNOSTIC_ONLY"
    return "FUNDAMENTALS_DO_NOT_ADD_VALUE"


def _ci_low(bootstrap: pd.DataFrame, variant: str, metric: str, column: str) -> float:
    subset = bootstrap[bootstrap["Variant"].eq(variant) & bootstrap["Metric"].eq(metric)]
    if subset.empty:
        return -np.inf
    return float(pd.to_numeric(subset[column], errors="coerce").quantile(0.025))


def _pr_auc(frame: pd.DataFrame, score: str) -> float | None:
    from sklearn.metrics import average_precision_score

    if frame.empty or frame["buy_target"].nunique() < 2:
        return None
    return float(average_precision_score(frame["buy_target"], frame[score]))


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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


def _report(
    *,
    decision: str,
    restatement: pd.DataFrame,
    coverage: pd.DataFrame,
    selected_fundamentals: list[str],
    topk: pd.DataFrame,
    buy_confirmation: pd.DataFrame,
    false_positive: pd.DataFrame,
    verified_vs_maxsafe: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    overall_topk = topk.copy()
    confirmation = buy_confirmation[buy_confirmation["Scope"].eq("overall")].copy()
    bootstrap_summary = (
        bootstrap.groupby(["Variant", "Metric"])
        .agg(
            PrecisionDiffLow=("PrecisionDiff", lambda x: x.quantile(0.025)),
            PrecisionDiffMedian=("PrecisionDiff", "median"),
            PrecisionDiffHigh=("PrecisionDiff", lambda x: x.quantile(0.975)),
            AverageExcessDiffLow=("AverageExcessDiff", lambda x: x.quantile(0.025)),
            AverageExcessDiffMedian=("AverageExcessDiff", "median"),
            AverageExcessDiffHigh=("AverageExcessDiff", lambda x: x.quantile(0.975)),
        )
        .reset_index()
        if not bootstrap.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "NATIP Point-In-Time Fundamental ML Experiment",
            "",
            f"Decision: {decision}",
            "Scope: original 150-stock universe, validation / OOF only. Final test was not used.",
            "Production V1/V2 artifacts were not modified.",
            "",
            "Restatement handling:",
            restatement.to_string(index=False),
            "",
            f"Selected fundamentals ({len(selected_fundamentals)}):",
            ", ".join(selected_fundamentals),
            "",
            "Feature coverage:",
            coverage.to_string(index=False),
            "",
            "Top-K and strict-BUY comparison:",
            overall_topk.to_string(index=False),
            "",
            "Existing BUY confirmation buckets:",
            confirmation.to_string(index=False),
            "",
            "False-positive feature separation:",
            false_positive.to_string(index=False),
            "",
            "VERIFIED_ONLY versus MAX_SAFE:",
            verified_vs_maxsafe.to_string(index=False),
            "",
            "Bootstrap summary:",
            bootstrap_summary.to_string(index=False),
            "",
            (
                "Leakage policy: PIT parquet inputs assert publication_date <= Date; "
                "SAFE_LATE rows are available only from conservative ml_effective_date; "
                "unresolved rows are excluded."
            ),
        ]
    )


if __name__ == "__main__":
    main()
