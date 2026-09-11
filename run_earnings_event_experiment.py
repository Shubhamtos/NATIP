"""Validation-only earnings-event / fundamental-change experiment.

The experiment uses only HIGH/MEDIUM verified point-in-time publication dates.
It excludes SAFE_LATE and unresolved dates, keeps final-test rows untouched,
and writes research reports without modifying production artifacts.
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
from run_ranker_binary_signal_research import _transform_with_medians
from run_stock_specific_feature_variant import _common_evaluation_rows, _prepare_dataset

CLEAN_150_DATASET = DATA_DIR / "probability_training_dataset_clean_vedl_demerger_excluded.csv"
PIT_VERIFIED = Path("data/fundamentals/exchange_publication_dates/PIT_verified_only.parquet")
CACHE_DIR = DATA_DIR / "clean_cache_adjusted"
NIFTY_CACHE = CACHE_DIR / "INDEX_NSEI.csv"
V2_SELECTED_FEATURES = REPORT_DIR / "v2_selected_feature_set.json"
FROZEN_CONFIG = REPORT_DIR / "frozen_model_config.json"
CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"

OUTPUT_DATASET = REPORT_DIR / "earnings_event_dataset.parquet"
OUTPUT_OOF = REPORT_DIR / "earnings_event_oof_predictions.parquet"
OUTPUT_TOPK = REPORT_DIR / "earnings_event_topk_comparison.csv"
OUTPUT_BUY_SPLIT = REPORT_DIR / "earnings_event_buy_signal_split.csv"
OUTPUT_SEGMENT = REPORT_DIR / "earnings_event_segment_analysis.csv"
OUTPUT_FOLD = REPORT_DIR / "earnings_event_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "earnings_event_year_stability.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "earnings_event_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "earnings_event_report.txt"

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 42
CLASSIFIER_TOP = 0.01
BINARY_TOP = 0.005
TOP_K = (3, 5)

BASE_EVENT_FEATURES = [
    "trading_days_since_result",
    "sales_growth_yoy",
    "profit_growth_yoy",
    "eps_growth_yoy",
    "sales_growth_acceleration",
    "profit_growth_acceleration",
    "operating_margin",
    "operating_margin_change_yoy",
    "roce",
    "debt_to_equity",
    "operating_cashflow_to_net_profit",
    "sales_yoy_change_vs_prev_q",
    "profit_yoy_change_vs_prev_q",
    "eps_yoy_change_vs_prev_q",
    "margin_change_vs_prev_q",
    "roce_change_vs_prev_report",
    "debt_equity_change_vs_prev_report",
    "cashflow_profit_change_vs_prev_report",
    "result_day_return",
    "result_day_excess_vs_nifty",
    "post_result_2d_excess",
    "post_result_5d_excess",
    "result_day_volume_to_avg20",
    "post_result_gap",
    "post_result_volatility",
]

WINDOWS = {
    "D0_5": (0, 5),
    "D6_10": (6, 10),
    "D11_20": (11, 20),
    "D0_20": (0, 20),
}


@dataclass(frozen=True, slots=True)
class EventVariant:
    """Event model variant."""

    name: str
    display_name: str
    features: list[str]
    use_technical: bool


def main() -> None:
    """Run the event-relative fundamental-change experiment."""

    ensure_probability_dirs()
    technical = _load_technical_dataset()
    features_26 = _selected_26_features()
    features_77 = _frozen_77_features()
    common = _common_evaluation_rows(technical, features_77)
    folds = _embargo_folds(common)
    current_oof = _load_current_oof()

    event_data = _build_event_dataset(technical)
    event_data = event_data[event_data["Date"] < FINAL_TEST_START].copy()
    event_data.to_parquet(OUTPUT_DATASET, index=False)

    event_features = [feature for feature in BASE_EVENT_FEATURES if feature in event_data.columns]
    variants = [
        EventVariant("EVENT_FUND_ONLY", "Event fundamentals only", event_features, False),
        EventVariant(
            "XGB26_PLUS_EVENT_FUND",
            "XGB26 + event/change fundamentals",
            [*features_26, *event_features],
            True,
        ),
    ]
    predictions = []
    for variant in variants:
        print(f"[earnings-event] {variant.name}", flush=True)
        predictions.append(_walk_forward_oof(event_data, folds, variant))
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_parquet(OUTPUT_OOF, index=False)

    topk = _topk_comparison(event_data, oof, current_oof)
    buy_split = _current_buy_split(oof, current_oof)
    segment = _segment_analysis(event_data, oof, current_oof)
    fold = _scope_stability(event_data, oof, current_oof, "fold")
    year = _scope_stability(event_data, oof, current_oof, "year")
    bootstrap = _bootstrap(event_data, oof, current_oof)
    decision = _decision(topk, buy_split, bootstrap)

    topk.to_csv(OUTPUT_TOPK, index=False)
    buy_split.to_csv(OUTPUT_BUY_SPLIT, index=False)
    segment.to_csv(OUTPUT_SEGMENT, index=False)
    fold.to_csv(OUTPUT_FOLD, index=False)
    year.to_csv(OUTPUT_YEAR, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report(
            decision=decision,
            event_rows=event_data,
            event_features=event_features,
            topk=topk,
            buy_split=buy_split,
            segment=segment,
            fold=fold,
            year=year,
            bootstrap=bootstrap,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision,
                "event_rows": int(len(event_data)),
                "event_stocks": int(event_data["symbol"].nunique()),
                "event_features": len(event_features),
                "earnings_event_topk_comparison": str(OUTPUT_TOPK),
                "earnings_event_buy_signal_split": str(OUTPUT_BUY_SPLIT),
                "earnings_event_report": str(OUTPUT_REPORT),
                "final_test_used": False,
                "production_artifacts_modified": False,
            },
            indent=2,
        )
    )


def _load_technical_dataset() -> pd.DataFrame:
    dataset = _prepare_dataset(pd.read_csv(CLEAN_150_DATASET, parse_dates=["Date"]))
    return dataset[dataset["Date"] < FINAL_TEST_START].copy()


def _selected_26_features() -> list[str]:
    payload = json.loads(V2_SELECTED_FEATURES.read_text(encoding="utf-8"))
    features = list(payload["selected_features"])
    if len(features) != 26:
        raise AssertionError(f"Expected 26 XGB features, got {len(features)}")
    return features


def _frozen_77_features() -> list[str]:
    payload = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    features = list(payload["feature_list"])
    if len(features) != 77:
        raise AssertionError(f"Expected 77 frozen features, got {len(features)}")
    return features


def _load_current_oof() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    current = current[current["Scheme"].eq("EXPANDING_BASELINE")].copy()
    current = current[current["Date"] < FINAL_TEST_START].copy()
    current["current_buy_signal"] = (current["classifier_percentile"] >= 1 - CLASSIFIER_TOP) & (
        current["binary_buy_percentile"] >= 1 - BINARY_TOP
    )
    return current


def _build_event_dataset(technical: pd.DataFrame) -> pd.DataFrame:
    pit = pd.read_parquet(PIT_VERIFIED).rename(columns={"ticker": "symbol"})
    pit["Date"] = pd.to_datetime(pit["Date"])
    pit["period_end"] = pd.to_datetime(pit["period_end"], errors="coerce")
    pit["publication_date"] = pd.to_datetime(pit["publication_date"], errors="coerce")
    pit = pit[pit["publication_date_quality"].isin(["VERIFIED_HIGH", "VERIFIED_MEDIUM"])].copy()
    pit = pit[pit["period_end"].notna() & pit["publication_date"].notna()].copy()

    columns = [
        "Date",
        "symbol",
        "period_end",
        "publication_date",
        "sales_growth_yoy",
        "profit_growth_yoy",
        "eps_growth_yoy",
        "sales_growth_acceleration",
        "profit_growth_acceleration",
        "operating_margin",
        "operating_margin_change_yoy",
        "roce",
        "debt_to_equity",
        "operating_cashflow_to_net_profit",
        "publication_date_quality",
    ]
    merged = technical.merge(pit[columns], on=["Date", "symbol"], how="inner")
    merged = merged.sort_values(["symbol", "period_end", "Date"]).copy()
    merged["trading_days_since_result"] = merged.groupby(["symbol", "period_end"]).cumcount()
    merged = merged[merged["trading_days_since_result"].between(0, 20)].copy()
    if (merged["publication_date"] > merged["Date"]).any():
        raise AssertionError("Publication date leaked beyond event Date.")
    merged = _add_history_change_features(merged)
    merged = _add_market_reaction_features(merged)
    merged["event_window"] = merged["trading_days_since_result"].map(_window_label)
    return merged.replace([np.inf, -np.inf], np.nan)


def _add_history_change_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    period = (
        output.sort_values(["symbol", "period_end"])
        .drop_duplicates(["symbol", "period_end"])
        .copy()
    )
    change_map = {
        "sales_yoy_change_vs_prev_q": "sales_growth_yoy",
        "profit_yoy_change_vs_prev_q": "profit_growth_yoy",
        "eps_yoy_change_vs_prev_q": "eps_growth_yoy",
        "margin_change_vs_prev_q": "operating_margin",
        "roce_change_vs_prev_report": "roce",
        "debt_equity_change_vs_prev_report": "debt_to_equity",
        "cashflow_profit_change_vs_prev_report": "operating_cashflow_to_net_profit",
    }
    for new_column, source in change_map.items():
        period[new_column] = period[source] - period.groupby("symbol")[source].shift(1)
    return output.merge(
        period[["symbol", "period_end", *change_map.keys()]],
        on=["symbol", "period_end"],
        how="left",
    )


def _add_market_reaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    cache = _load_reaction_cache(frame["symbol"].unique())
    nifty = _load_nifty_returns()
    rows = []
    for event_key, group in frame.groupby(["symbol", "period_end"], sort=False):
        symbol, _period_end = event_key
        prices = cache.get(symbol)
        if prices is None or prices.empty:
            rows.append(group)
            continue
        event_date = pd.Timestamp(group["Date"].min())
        prices = prices.sort_values("Date").reset_index(drop=True)
        event_pos = prices.index[prices["Date"].eq(event_date)]
        if len(event_pos) == 0:
            rows.append(group)
            continue
        event_idx = int(event_pos[0])
        enriched = group.copy()
        event_row = prices.iloc[event_idx]
        prev_close = prices.iloc[event_idx - 1]["adj_Close"] if event_idx > 0 else np.nan
        event_return = event_row["adj_Close"] / prev_close - 1 if prev_close else np.nan
        nifty_event = nifty.get(event_date, np.nan)
        avg20_volume = (
            prices.loc[max(0, event_idx - 20) : event_idx - 1, "Volume"].mean()
            if event_idx > 0
            else np.nan
        )
        enriched["result_day_return"] = event_return
        enriched["result_day_excess_vs_nifty"] = event_return - nifty_event
        enriched["result_day_volume_to_avg20"] = (
            event_row["Volume"] / avg20_volume
            if avg20_volume and pd.notna(avg20_volume)
            else np.nan
        )
        enriched["post_result_gap"] = (
            event_row["adj_Open"] / prev_close - 1
            if prev_close and pd.notna(prev_close)
            else np.nan
        )
        enriched["post_result_2d_excess"] = _post_event_excess(
            prices, nifty, event_idx, event_date, days=2
        )
        enriched["post_result_5d_excess"] = _post_event_excess(
            prices, nifty, event_idx, event_date, days=5
        )
        for idx in enriched.index:
            days_since = int(enriched.at[idx, "trading_days_since_result"])
            if days_since < 2:
                enriched.at[idx, "post_result_2d_excess"] = np.nan
            if days_since < 5:
                enriched.at[idx, "post_result_5d_excess"] = np.nan
            end_idx = min(event_idx + days_since, len(prices) - 1)
            returns = prices.loc[event_idx:end_idx, "daily_return"]
            enriched.at[idx, "post_result_volatility"] = (
                returns.std(ddof=0) if len(returns.dropna()) >= 2 else np.nan
            )
        rows.append(enriched)
    return pd.concat(rows, ignore_index=True)


def _load_reaction_cache(symbols: np.ndarray) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols:
        path = CACHE_DIR / f"{symbol.replace('.', '_')}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame[frame.get("is_tradable_row", True).astype(bool)].copy()
        frame["daily_return"] = frame["adj_Close"].pct_change()
        frames[str(symbol)] = frame
    return frames


def _load_nifty_returns() -> dict[pd.Timestamp, float]:
    nifty = pd.read_csv(NIFTY_CACHE, parse_dates=["Date"])
    nifty = nifty[nifty.get("is_tradable_row", True).astype(bool)].copy()
    nifty["return"] = nifty["adj_Close"].pct_change()
    return {
        pd.Timestamp(row.Date): float(row.return_)
        for row in nifty.rename(columns={"return": "return_"}).itertuples(index=False)
    }


def _post_event_excess(
    prices: pd.DataFrame,
    nifty_returns: dict[pd.Timestamp, float],
    event_idx: int,
    event_date: pd.Timestamp,
    *,
    days: int,
) -> float:
    end_idx = event_idx + days
    if end_idx >= len(prices):
        return np.nan
    stock_return = prices.iloc[end_idx]["adj_Close"] / prices.iloc[event_idx]["adj_Close"] - 1
    dates = prices.iloc[event_idx + 1 : end_idx + 1]["Date"]
    nifty_path = [nifty_returns.get(pd.Timestamp(date), np.nan) for date in dates]
    if any(pd.isna(item) for item in nifty_path):
        return np.nan
    nifty_return = float(np.prod([1 + item for item in nifty_path]) - 1)
    return float(stock_return - nifty_return)


def _window_label(days_since: int) -> str:
    if 0 <= days_since <= 5:
        return "D0_5"
    if 6 <= days_since <= 10:
        return "D6_10"
    if 11 <= days_since <= 20:
        return "D11_20"
    return "OUTSIDE"


def _walk_forward_oof(
    dataset: pd.DataFrame, folds: list[dict[str, Any]], variant: EventVariant
) -> pd.DataFrame:
    frames = []
    for fold in folds:
        train = dataset[dataset["Date"].isin(fold["train_dates"])].copy()
        validation = dataset[dataset["Date"].isin(fold["validation_dates"])].copy()
        if train.empty or validation.empty:
            continue
        model, medians = _fit_classifier(train, variant.features, target="label")
        probabilities = model.predict_proba(
            _transform_with_medians(validation, variant.features, medians)
        )
        output = _base_frame(validation, fold["fold"], variant)
        output["p_underperform"] = probabilities[:, LABEL_TO_CLASS[-1]]
        output["p_neutral"] = probabilities[:, LABEL_TO_CLASS[0]]
        output["p_outperform"] = probabilities[:, LABEL_TO_CLASS[1]]
        output["score_percentile"] = output.groupby("Date")["p_outperform"].rank(pct=True)
        frames.append(output)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fit_classifier(train: pd.DataFrame, features: list[str], *, target: str):
    from xgboost import XGBClassifier

    medians = train[features].median(numeric_only=True).fillna(0)
    x = _transform_with_medians(train, features, medians)
    model = XGBClassifier(
        n_estimators=80,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(x, train[target].map(LABEL_TO_CLASS))
    return model, medians


def _base_frame(frame: pd.DataFrame, fold: Any, variant: EventVariant) -> pd.DataFrame:
    columns = [
        "Date",
        "symbol",
        "Sector",
        "MarketCapCategory",
        "future_stock_return",
        "future_nifty_return",
        "excess_return",
        "buy_target",
        "market_regime_label",
        "event_window",
        "trading_days_since_result",
        "profit_growth_acceleration",
        "operating_margin_change_yoy",
    ]
    output = frame[columns].copy()
    output["fold"] = fold
    output["Variant"] = variant.name
    output["VariantName"] = variant.display_name
    output["FeatureCount"] = len(variant.features)
    return output


def _topk_comparison(
    event_data: pd.DataFrame, oof: pd.DataFrame, current_oof: pd.DataFrame
) -> pd.DataFrame:
    current = _matched_current(event_data, current_oof)
    rows = []
    for window, (start, end) in WINDOWS.items():
        cur = current[current["trading_days_since_result"].between(start, end)].copy()
        rows.extend(_topk_rows(cur, "EXISTING_XGB26", "p_outperform", window))
        for variant, group in oof.groupby("Variant"):
            subset = group[group["trading_days_since_result"].between(start, end)].copy()
            rows.extend(_topk_rows(subset, variant, "p_outperform", window))
            with_binary = subset.merge(
                cur[["Date", "symbol", "fold", "binary_buy_percentile"]],
                on=["Date", "symbol", "fold"],
                how="inner",
            )
            selected = with_binary[
                (with_binary["score_percentile"] >= 1 - CLASSIFIER_TOP)
                & (with_binary["binary_buy_percentile"] >= 1 - BINARY_TOP)
            ]
            rows.append(
                _selection_row(variant, "strict_buy_precision", window, selected, with_binary)
            )
        selected_current = cur[cur["current_buy_signal"]]
        rows.append(
            _selection_row(
                "EXISTING_CURRENT_BUY", "strict_buy_precision", window, selected_current, cur
            )
        )
    return pd.DataFrame(rows)


def _matched_current(event_data: pd.DataFrame, current_oof: pd.DataFrame) -> pd.DataFrame:
    event_cols = [
        "Date",
        "symbol",
        "event_window",
        "trading_days_since_result",
        "profit_growth_acceleration",
        "operating_margin_change_yoy",
    ]
    cols = [
        "Date",
        "symbol",
        "fold",
        "Sector",
        "MarketCapCategory",
        "future_stock_return",
        "future_nifty_return",
        "excess_return",
        "buy_target",
        "market_regime_label",
        "p_outperform",
        "classifier_percentile",
        "binary_buy_percentile",
        "current_buy_signal",
    ]
    return current_oof[cols].merge(event_data[event_cols], on=["Date", "symbol"], how="inner")


def _topk_rows(frame: pd.DataFrame, model: str, score: str, window: str) -> list[dict[str, Any]]:
    rows = []
    for k in TOP_K:
        daily = []
        for _date, group in frame.groupby("Date"):
            selected = group.sort_values(score, ascending=False).head(k)
            winners = int(group["buy_target"].sum())
            daily.append(
                {
                    "precision": selected["buy_target"].mean(),
                    "recall": selected["buy_target"].sum() / winners if winners else np.nan,
                    "coverage": selected["buy_target"].sum() > 0,
                    "avg_excess": selected["excess_return"].mean(),
                    "median_excess": selected["excess_return"].median(),
                    "win": (selected["excess_return"] > 0).mean(),
                }
            )
        daily_frame = pd.DataFrame(daily)
        rows.append(
            {
                "Model": model,
                "MetricType": "daily_topk",
                "Window": window,
                "K": k,
                "Rows": int(len(frame)),
                "StockCount": int(frame["symbol"].nunique()) if not frame.empty else 0,
                "PositiveBaseRate": (
                    _safe_float(frame["buy_target"].mean()) if not frame.empty else None
                ),
                "PRAUC": _pr_auc(frame, score),
                "Precision": _safe_float(daily_frame["precision"].mean()),
                "Recall": _safe_float(daily_frame["recall"].mean()),
                "WinnerCoverage": _safe_float(daily_frame["coverage"].mean()),
                "AverageExcessReturn": _safe_float(daily_frame["avg_excess"].mean()),
                "MedianExcessReturn": _safe_float(daily_frame["median_excess"].median()),
                "NiftyWinRate": _safe_float(daily_frame["win"].mean()),
            }
        )
    return rows


def _selection_row(
    model: str, metric_type: str, window: str, selected: pd.DataFrame, population: pd.DataFrame
) -> dict[str, Any]:
    return {
        "Model": model,
        "MetricType": metric_type,
        "Window": window,
        "K": np.nan,
        "Rows": int(len(population)),
        "StockCount": int(population["symbol"].nunique()) if not population.empty else 0,
        "PositiveBaseRate": (
            _safe_float(population["buy_target"].mean()) if not population.empty else None
        ),
        "PRAUC": _pr_auc(population, "p_outperform") if "p_outperform" in population else None,
        "Precision": _safe_float(selected["buy_target"].mean()) if not selected.empty else None,
        "Recall": (
            _safe_float(selected["buy_target"].sum() / population["buy_target"].sum())
            if not population.empty and population["buy_target"].sum()
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
    }


def _current_buy_split(oof: pd.DataFrame, current_oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in oof.groupby("Variant"):
        merged = group.merge(
            current_oof[["Date", "symbol", "fold", "current_buy_signal"]],
            on=["Date", "symbol", "fold"],
            how="inner",
        )
        signals = merged[merged["current_buy_signal"]].copy()
        if signals.empty:
            continue
        ranked = signals.sort_values("p_outperform", ascending=False)
        for bucket, fraction in [
            ("StrongTop50", 0.50),
            ("StrongTop30", 0.30),
            ("StrongTop20", 0.20),
        ]:
            selected = ranked.head(max(1, int(len(ranked) * fraction)))
            rows.append(_buy_split_row(variant, bucket, "overall", "all", selected))
        weak = ranked.tail(max(1, int(len(ranked) * 0.50)))
        rows.append(_buy_split_row(variant, "WeakBottom50", "overall", "all", weak))
    return pd.DataFrame(rows)


def _buy_split_row(
    variant: str, bucket: str, scope: str, scope_value: Any, selected: pd.DataFrame
) -> dict[str, Any]:
    return {
        "Variant": variant,
        "Bucket": bucket,
        "Scope": scope,
        "ScopeValue": scope_value,
        "Count": int(len(selected)),
        "Precision": _safe_float(selected["buy_target"].mean()),
        "AverageExcessReturn": _safe_float(selected["excess_return"].mean()),
        "MedianExcessReturn": _safe_float(selected["excess_return"].median()),
        "NiftyWinRate": _safe_float((selected["excess_return"] > 0).mean()),
    }


def _segment_analysis(
    event_data: pd.DataFrame, oof: pd.DataFrame, current_oof: pd.DataFrame
) -> pd.DataFrame:
    current = _matched_current(event_data, current_oof)
    rows = []
    segments = {
        "positive_profit_acceleration": event_data["profit_growth_acceleration"] > 0,
        "negative_profit_acceleration": event_data["profit_growth_acceleration"] < 0,
        "margin_expansion": event_data["operating_margin_change_yoy"] > 0,
        "margin_contraction": event_data["operating_margin_change_yoy"] < 0,
    }
    for segment, mask in segments.items():
        symbols_dates = event_data.loc[mask, ["Date", "symbol"]]
        cur = current.merge(symbols_dates, on=["Date", "symbol"], how="inner")
        rows.append(_segment_row(segment, "EXISTING_XGB26", cur))
        for variant, group in oof.groupby("Variant"):
            subset = group.merge(symbols_dates, on=["Date", "symbol"], how="inner")
            rows.append(_segment_row(segment, variant, subset))
    return pd.DataFrame(rows)


def _segment_row(segment: str, model: str, frame: pd.DataFrame) -> dict[str, Any]:
    selected = _top_fraction(frame, "p_outperform", 0.05) if not frame.empty else frame
    return {
        "Segment": segment,
        "Model": model,
        "Rows": int(len(frame)),
        "StockCount": int(frame["symbol"].nunique()) if not frame.empty else 0,
        "PositiveBaseRate": _safe_float(frame["buy_target"].mean()) if not frame.empty else None,
        "Top5PctPrecision": (
            _safe_float(selected["buy_target"].mean()) if not selected.empty else None
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
    }


def _scope_stability(
    event_data: pd.DataFrame, oof: pd.DataFrame, current_oof: pd.DataFrame, scope: str
) -> pd.DataFrame:
    current = _matched_current(event_data, current_oof)
    rows = []
    group_key = current["Date"].dt.year if scope == "year" else current[scope]
    for value, group in current.groupby(group_key):
        for row in _topk_rows(group, "EXISTING_XGB26", "p_outperform", "D0_20"):
            row["Scope"] = scope
            row["ScopeValue"] = value
            rows.append(row)
    for variant, frame in oof.groupby("Variant"):
        group_key = frame["Date"].dt.year if scope == "year" else frame[scope]
        for value, group in frame.groupby(group_key):
            for row in _topk_rows(group, variant, "p_outperform", "D0_20"):
                row["Scope"] = scope
                row["ScopeValue"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(
    event_data: pd.DataFrame, oof: pd.DataFrame, current_oof: pd.DataFrame
) -> pd.DataFrame:
    current = _matched_current(event_data, current_oof)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    dates = sorted(current["Date"].drop_duplicates())
    baseline_blocks = {k: _daily_topk_arrays(current, "p_outperform", k) for k in TOP_K}
    rows = []
    for variant, group in oof.groupby("Variant"):
        blocks = {k: _daily_topk_arrays(group, "p_outperform", k) for k in TOP_K}
        for k in TOP_K:
            for iteration in range(BOOTSTRAP_SAMPLES):
                sampled = rng.choice(dates, size=len(dates), replace=True)
                base = _bootstrap_metric(baseline_blocks[k], sampled)
                cand = _bootstrap_metric(blocks[k], sampled)
                rows.append(
                    {
                        "Variant": variant,
                        "K": k,
                        "Iteration": iteration + 1,
                        "PrecisionDiff": cand["precision"] - base["precision"],
                        "WinnerCoverageDiff": cand["coverage"] - base["coverage"],
                        "AverageExcessDiff": cand["avg_excess"] - base["avg_excess"],
                        "MedianExcessDiff": cand["median_excess"] - base["median_excess"],
                    }
                )
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


def _bootstrap_metric(
    blocks: dict[pd.Timestamp, dict[str, np.ndarray]], dates: np.ndarray
) -> dict[str, float]:
    targets = []
    excesses = []
    coverage = []
    for date in dates:
        block = blocks.get(pd.Timestamp(date))
        if block is None:
            continue
        targets.append(block["target"])
        excesses.append(block["excess"])
        coverage.append(float(np.sum(block["target"]) > 0))
    if not targets:
        return {
            "precision": np.nan,
            "coverage": np.nan,
            "avg_excess": np.nan,
            "median_excess": np.nan,
        }
    target = np.concatenate(targets)
    excess = np.concatenate(excesses)
    return {
        "precision": float(np.mean(target)),
        "coverage": float(np.mean(coverage)),
        "avg_excess": float(np.mean(excess)),
        "median_excess": float(np.median(excess)),
    }


def _top_fraction(frame: pd.DataFrame, score: str, fraction: float) -> pd.DataFrame:
    count = max(1, int(len(frame) * fraction))
    return frame.sort_values(score, ascending=False).head(count)


def _decision(topk: pd.DataFrame, buy_split: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    top5 = topk[
        (topk["Window"].eq("D0_20")) & (topk["MetricType"].eq("daily_topk")) & (topk["K"].eq(5))
    ]
    baseline = top5[top5["Model"].eq("EXISTING_XGB26")].head(1)
    candidates = top5[top5["Model"].isin(["EVENT_FUND_ONLY", "XGB26_PLUS_EVENT_FUND"])]
    robust_add = False
    if not baseline.empty and not candidates.empty:
        best = candidates.sort_values("Precision", ascending=False).head(1).iloc[0]
        low = _ci_low(bootstrap, best["Model"], 5, "PrecisionDiff")
        robust_add = best["Precision"] > baseline.iloc[0]["Precision"] and low > 0
    confirmation = False
    if not buy_split.empty:
        pivot = buy_split.pivot_table(
            index="Variant", columns="Bucket", values="Precision", aggfunc="first"
        )
        if "StrongTop20" in pivot and "WeakBottom50" in pivot:
            confirmation = bool(((pivot["StrongTop20"] - pivot["WeakBottom50"]) > 0.03).any())
    if robust_add:
        return "EARNINGS_EVENT_SIGNAL_ADDS_VALUE"
    if confirmation:
        return "EARNINGS_EVENT_SIGNAL_CONFIRMATION_ONLY"
    return "EARNINGS_EVENT_SIGNAL_WEAK"


def _ci_low(bootstrap: pd.DataFrame, variant: str, k: int, column: str) -> float:
    subset = bootstrap[bootstrap["Variant"].eq(variant) & bootstrap["K"].eq(k)]
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


def _report(
    *,
    decision: str,
    event_rows: pd.DataFrame,
    event_features: list[str],
    topk: pd.DataFrame,
    buy_split: pd.DataFrame,
    segment: pd.DataFrame,
    fold: pd.DataFrame,
    year: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> str:
    bootstrap_summary = (
        bootstrap.groupby(["Variant", "K"])
        .agg(
            PrecisionDiffLow=("PrecisionDiff", lambda x: x.quantile(0.025)),
            PrecisionDiffMedian=("PrecisionDiff", "median"),
            PrecisionDiffHigh=("PrecisionDiff", lambda x: x.quantile(0.975)),
            CoverageDiffMedian=("WinnerCoverageDiff", "median"),
            AverageExcessDiffMedian=("AverageExcessDiff", "median"),
        )
        .reset_index()
        if not bootstrap.empty
        else pd.DataFrame()
    )
    return "\n".join(
        [
            "NATIP Earnings-Event / Fundamental-Change Experiment",
            "",
            f"Decision: {decision}",
            "Scope: validation/out-of-fold only; final test not used.",
            "Input fundamentals: HIGH/MEDIUM verified PIT only; SAFE_LATE and unresolved excluded.",
            "",
            f"Event rows: {len(event_rows)}",
            f"Stocks: {event_rows['symbol'].nunique()}",
            f"Positive base rate: {_safe_float(event_rows['buy_target'].mean())}",
            "",
            f"Event features ({len(event_features)}):",
            ", ".join(event_features),
            "",
            "Top-K comparison:",
            topk.to_string(index=False),
            "",
            "Existing BUY split by event model score:",
            buy_split.to_string(index=False),
            "",
            "Segment analysis:",
            segment.to_string(index=False),
            "",
            "Fold stability:",
            fold.to_string(index=False),
            "",
            "Year stability:",
            year.to_string(index=False),
            "",
            "Bootstrap summary:",
            bootstrap_summary.to_string(index=False),
            "",
            "Production models were not modified.",
        ]
    )


if __name__ == "__main__":
    main()
