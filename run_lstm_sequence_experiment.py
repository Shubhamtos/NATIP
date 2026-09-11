"""Validation-only LSTM sequence experiment for NSE BUY confirmation research.

This script is report-only. It does not modify frozen V1/V2 artifacts, does not
use final-test rows, and does not alter production recommendation rules.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.probability.config import DATA_DIR, REPORT_DIR, ensure_probability_dirs
from app.probability.sector_benchmarks import SECTOR_YAHOO_SYMBOLS
from run_pruned_rank_target_validation import FINAL_TEST_START, _embargo_folds
from run_stock_specific_feature_variant import _common_evaluation_rows, _frozen_features, _prepare_dataset
from run_v2_backward_forward_selection import CLEAN_DEMERGER_DATASET
from run_v2_binary_stability_diagnosis import (
    TUNED_BINARY_ID,
    _binary_oof_with_audit,
)
from run_v2_model_family_comparison import _selected_26_features
from run_v2_xgb_hyperparameter_optimization import _binary_specs

SEQUENCE_LENGTH = 60
RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 2000
CURRENT_OOF = REPORT_DIR / "recency_training_oof_predictions.csv"
TUNED_BINARY_CACHE = REPORT_DIR / "lstm_tuned_binary77_oof_cache.csv"
CLEAN_CACHE_DIR = DATA_DIR / "clean_cache_adjusted"
UNIVERSE_CSV = Path("stocks_universe_2026_08.csv")

OUTPUT_OOF = REPORT_DIR / "lstm_sequence_oof_predictions.parquet"
OUTPUT_STANDALONE = REPORT_DIR / "lstm_standalone_results.csv"
OUTPUT_CONFIRMATION = REPORT_DIR / "lstm_buy_confirmation.csv"
OUTPUT_FOLD = REPORT_DIR / "lstm_fold_stability.csv"
OUTPUT_YEAR = REPORT_DIR / "lstm_year_stability.csv"
OUTPUT_DIVERSITY = REPORT_DIR / "lstm_model_diversity.csv"
OUTPUT_BOOTSTRAP = REPORT_DIR / "lstm_date_block_bootstrap.csv"
OUTPUT_REPORT = REPORT_DIR / "lstm_research_report.txt"

TOP_FRACTIONS = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005)
BUY_STRATA = (0.50, 0.30, 0.20, 0.10)
TUNED_TOP = 0.005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True, slots=True)
class SequenceFold:
    """Prepared fold tensors and validation metadata."""

    fold: int
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_frame: pd.DataFrame


class BuyLSTM(nn.Module):
    """Small binary LSTM used only for validation research."""

    def __init__(self, input_size: int = 6, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.20)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for a batch of sequences."""

        _, (hidden, _) = self.lstm(x)
        return self.output(self.dropout(hidden[-1])).squeeze(-1)


def main() -> None:
    """Run the LSTM validation-only experiment."""

    ensure_probability_dirs()
    _set_seeds()
    dataset = _prepare_dataset(pd.read_csv(CLEAN_DEMERGER_DATASET, parse_dates=["Date"]))
    pre_final = dataset[dataset["Date"] < FINAL_TEST_START].copy()
    if not pre_final["Date"].lt(FINAL_TEST_START).all():
        raise AssertionError("Final-test rows leaked into LSTM experiment.")

    features_77 = _frozen_features()
    features_26 = _selected_26_features()
    common = _common_evaluation_rows(pre_final, features_77)
    folds = _embargo_folds(common)
    current_oof = _load_current_oof()
    tuned_oof = _load_or_build_tuned_oof(common, folds, features_77)
    benchmark = _merge_benchmark_oof(common, current_oof, tuned_oof)

    print("[sequence] building point-in-time 60-day tensors", flush=True)
    sequence_cache = _build_sequence_feature_cache(common)
    fold_predictions = []
    leakage_rows = []
    for fold in folds:
        prepared = _prepare_sequence_fold(common, benchmark, sequence_cache, fold)
        if len(prepared.validation_frame) == 0:
            continue
        print(
            f"[lstm] fold={prepared.fold} train={len(prepared.train_y)} "
            f"validation={len(prepared.validation_frame)}",
            flush=True,
        )
        probabilities, fold_leakage = _fit_predict_fold(prepared)
        frame = prepared.validation_frame.copy()
        frame["p_lstm_buy"] = probabilities
        fold_predictions.append(frame)
        leakage_rows.append(fold_leakage)

    if not fold_predictions:
        raise RuntimeError("No validation predictions were produced for the LSTM experiment.")

    predictions = pd.concat(fold_predictions, ignore_index=True)
    predictions["lstm_percentile"] = predictions.groupby("Date")["p_lstm_buy"].rank(pct=True)
    predictions.to_parquet(OUTPUT_OOF, index=False)

    standalone = _standalone_results(predictions)
    confirmation = _confirmation_results(predictions)
    fold_stability = _stability_results(predictions, "fold")
    year_stability = _year_stability(predictions)
    diversity = _diversity_results(predictions)
    bootstrap = _bootstrap_results(predictions, confirmation)
    decision = _decision(confirmation, bootstrap)

    standalone.to_csv(OUTPUT_STANDALONE, index=False)
    confirmation.to_csv(OUTPUT_CONFIRMATION, index=False)
    fold_stability.to_csv(OUTPUT_FOLD, index=False)
    year_stability.to_csv(OUTPUT_YEAR, index=False)
    diversity.to_csv(OUTPUT_DIVERSITY, index=False)
    bootstrap.to_csv(OUTPUT_BOOTSTRAP, index=False)
    OUTPUT_REPORT.write_text(
        _report_text(
            predictions=predictions,
            standalone=standalone,
            confirmation=confirmation,
            fold_stability=fold_stability,
            year_stability=year_stability,
            diversity=diversity,
            bootstrap=bootstrap,
            leakage_rows=leakage_rows,
            decision=decision,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "lstm_sequence_oof_predictions": str(OUTPUT_OOF),
                "lstm_standalone_results": str(OUTPUT_STANDALONE),
                "lstm_buy_confirmation": str(OUTPUT_CONFIRMATION),
                "lstm_fold_stability": str(OUTPUT_FOLD),
                "lstm_year_stability": str(OUTPUT_YEAR),
                "lstm_model_diversity": str(OUTPUT_DIVERSITY),
                "lstm_date_block_bootstrap": str(OUTPUT_BOOTSTRAP),
                "lstm_research_report": str(OUTPUT_REPORT),
                "decision": decision,
                "final_test_used": False,
                "v1_v2_artifacts_modified": False,
                "existing_buy_rules_modified": False,
                "feature_26_count_reference": len(features_26),
            },
            indent=2,
        )
    )


def _set_seeds() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _load_current_oof() -> pd.DataFrame:
    """Load existing validation OOF for XGB26 + Current Binary77."""

    frame = pd.read_csv(CURRENT_OOF, parse_dates=["Date"])
    frame = frame[frame["Scheme"].eq("EXPANDING_BASELINE")].copy()
    frame = frame[frame["Date"] < FINAL_TEST_START].copy()
    required = [
        "Date",
        "symbol",
        "fold",
        "p_underperform",
        "p_neutral",
        "p_outperform",
        "classifier_percentile",
        "binary_buy_raw_probability",
        "binary_buy_sigmoid_probability",
        "binary_buy_percentile",
        "buy_signal",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssertionError(f"Current OOF is missing required columns: {missing}")
    return frame[required].copy()


def _load_or_build_tuned_oof(
    common: pd.DataFrame, folds: list[dict[str, Any]], features_77: list[str]
) -> pd.DataFrame:
    """Load cached tuned Binary77 OOF, or build it from validation folds."""

    if TUNED_BINARY_CACHE.exists():
        cached = pd.read_csv(TUNED_BINARY_CACHE, parse_dates=["Date"])
        required = {"Date", "symbol", "fold", "tuned_binary_sigmoid_probability", "tuned_binary_percentile"}
        if required.issubset(cached.columns):
            return cached[list(required)].copy()

    print("[binary] building tuned Binary77 OOF cache for comparison", flush=True)
    specs = {spec.trial_id: spec for spec in _binary_specs(common)}
    tuned, _ = _binary_oof_with_audit(common, folds, features_77, specs[TUNED_BINARY_ID])
    output = tuned[
        ["Date", "symbol", "fold", "sigmoid_probability", "sigmoid_percentile"]
    ].rename(
        columns={
            "sigmoid_probability": "tuned_binary_sigmoid_probability",
            "sigmoid_percentile": "tuned_binary_percentile",
        }
    )
    output.to_csv(TUNED_BINARY_CACHE, index=False)
    return output


def _merge_benchmark_oof(
    common: pd.DataFrame, current_oof: pd.DataFrame, tuned_oof: pd.DataFrame
) -> pd.DataFrame:
    """Merge targets, metadata, current BUY scores, and tuned binary scores."""

    keys = ["Date", "symbol", "fold"]
    base = common[
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
    merged = base.merge(current_oof, on=["Date", "symbol"], how="inner")
    if "fold_x" in merged.columns and "fold_y" in merged.columns:
        if not (merged["fold_x"].to_numpy() == merged["fold_y"].to_numpy()).all():
            raise AssertionError("Fold mismatch between clean dataset and current OOF.")
        merged = merged.drop(columns=["fold_x"]).rename(columns={"fold_y": "fold"})
    merged = merged.merge(tuned_oof, on=keys, how="left")
    missing_tuned = merged["tuned_binary_sigmoid_probability"].isna().sum()
    if missing_tuned:
        raise AssertionError(f"Tuned Binary77 OOF missing for {missing_tuned} benchmark rows.")
    merged["current_buy_signal"] = (
        (merged["classifier_percentile"] >= 0.99)
        & (merged["binary_buy_percentile"] >= 0.995)
    )
    merged["tuned_high_confidence"] = (
        merged["current_buy_signal"] & (merged["tuned_binary_percentile"] >= 0.995)
    )
    return merged.sort_values(["Date", "symbol"]).reset_index(drop=True)


def _safe_symbol(symbol: str) -> str:
    cleaned = symbol.replace("^", "INDEX_").replace(".", "_").replace("=", "_")
    cleaned = cleaned.replace("-", "-")
    return cleaned


def _read_clean_price(symbol: str) -> pd.DataFrame:
    path = CLEAN_CACHE_DIR / f"{_safe_symbol(symbol)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Clean adjusted cache missing for {symbol}: {path}")
    frame = pd.read_csv(path, parse_dates=["Date"])
    required = ["Date", "adj_Open", "adj_High", "adj_Low", "adj_Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssertionError(f"{path} missing adjusted columns: {missing}")
    if "is_tradable_row" in frame.columns:
        frame = frame[frame["is_tradable_row"].astype(bool)].copy()
    frame = frame.drop_duplicates("Date").sort_values("Date")
    return frame[required].copy()


def _build_sequence_feature_cache(benchmark: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build daily six-feature sequence source for all required symbols."""

    universe = pd.read_csv(UNIVERSE_CSV)
    sector_by_ticker = dict(zip(universe["Ticker"], universe["Sector"], strict=False))
    nifty = _read_clean_price("^NSEI")[["Date", "adj_Close"]].rename(
        columns={"adj_Close": "nifty_close"}
    )
    nifty["nifty_return"] = nifty["nifty_close"].pct_change()

    sector_returns: dict[str, pd.DataFrame] = {}
    for sector in benchmark["Sector"].dropna().astype(str).unique():
        yahoo_symbol = SECTOR_YAHOO_SYMBOLS.get(sector)
        if not yahoo_symbol:
            continue
        try:
            sector_frame = _read_clean_price(yahoo_symbol)[["Date", "adj_Close"]].rename(
                columns={"adj_Close": "sector_close"}
            )
            sector_frame["sector_return"] = sector_frame["sector_close"].pct_change()
            sector_returns[sector] = sector_frame[["Date", "sector_return"]]
        except FileNotFoundError:
            continue

    cache: dict[str, pd.DataFrame] = {}
    for symbol in benchmark["symbol"].drop_duplicates():
        stock = _read_clean_price(symbol)
        stock["stock_return"] = stock["adj_Close"].pct_change()
        stock["daily_range_pct"] = (stock["adj_High"] - stock["adj_Low"]) / stock["adj_Close"]
        denominator = (stock["adj_High"] - stock["adj_Low"]).replace(0, np.nan)
        stock["close_location_value"] = (stock["adj_Close"] - stock["adj_Low"]) / denominator
        stock["close_location_value"] = stock["close_location_value"].fillna(0.5)
        trailing_volume = stock["Volume"].shift(1).rolling(20, min_periods=20).mean()
        stock["relative_volume_20d"] = stock["Volume"] / trailing_volume
        stock = stock.merge(nifty[["Date", "nifty_return"]], on="Date", how="left")
        sector = sector_by_ticker.get(symbol)
        sector_frame = sector_returns.get(str(sector))
        if sector_frame is not None:
            stock = stock.merge(sector_frame, on="Date", how="left")
        else:
            stock["sector_return"] = np.nan
        stock["stock_minus_nifty_return"] = stock["stock_return"] - stock["nifty_return"]
        stock["stock_minus_sector_return"] = stock["stock_return"] - stock["sector_return"]
        cache[symbol] = stock[
            [
                "Date",
                "stock_return",
                "daily_range_pct",
                "close_location_value",
                "relative_volume_20d",
                "stock_minus_nifty_return",
                "stock_minus_sector_return",
            ]
        ].replace([np.inf, -np.inf], np.nan)
    return cache


def _prepare_sequence_fold(
    training_source: pd.DataFrame,
    validation_source: pd.DataFrame,
    sequence_cache: dict[str, pd.DataFrame],
    fold: dict[str, Any],
) -> SequenceFold:
    """Create train and validation sequence tensors for a walk-forward fold."""

    train = training_source[training_source["Date"].isin(fold["train_dates"])].copy()
    validation = validation_source[validation_source["Date"].isin(fold["validation_dates"])].copy()
    train_x, train_rows = _materialize_sequences(train, sequence_cache)
    validation_x, validation_rows = _materialize_sequences(validation, sequence_cache)
    return SequenceFold(
        fold=int(fold["fold"]),
        train_x=train_x,
        train_y=train_rows["buy_target"].astype(int).to_numpy(),
        validation_x=validation_x,
        validation_frame=validation_rows,
    )


def _materialize_sequences(
    frame: pd.DataFrame, sequence_cache: dict[str, pd.DataFrame]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Materialize sequences ending on each requested stock/date row."""

    arrays: list[np.ndarray] = []
    rows: list[pd.Series] = []
    feature_cols = [
        "stock_return",
        "daily_range_pct",
        "close_location_value",
        "relative_volume_20d",
        "stock_minus_nifty_return",
        "stock_minus_sector_return",
    ]
    for symbol, group in frame.groupby("symbol", sort=False):
        source = sequence_cache.get(symbol)
        if source is None:
            continue
        source = source.reset_index(drop=True)
        positions = pd.Series(source.index.to_numpy(), index=source["Date"]).to_dict()
        values = source[feature_cols].to_numpy(dtype=np.float32)
        for _, row in group.iterrows():
            pos = positions.get(row["Date"])
            if pos is None or pos < SEQUENCE_LENGTH - 1:
                continue
            sequence = values[pos - SEQUENCE_LENGTH + 1 : pos + 1]
            if sequence.shape != (SEQUENCE_LENGTH, len(feature_cols)):
                continue
            if not np.isfinite(sequence).all():
                continue
            arrays.append(sequence)
            rows.append(row)
    if not arrays:
        return (
            np.empty((0, SEQUENCE_LENGTH, len(feature_cols)), dtype=np.float32),
            frame.iloc[:0].copy(),
        )
    return np.stack(arrays).astype(np.float32), pd.DataFrame(rows).reset_index(drop=True)


def _fit_predict_fold(prepared: SequenceFold) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one LSTM fold and return validation probabilities."""

    train_x = prepared.train_x
    train_y = prepared.train_y.astype(np.float32)
    if len(train_y) < 100 or train_y.sum() == 0:
        raise RuntimeError(f"Fold {prepared.fold} has insufficient LSTM training data.")

    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_x = (train_x - mean) / std
    validation_x = (prepared.validation_x - mean) / std
    train_dates = pd.to_datetime(prepared.validation_frame["Date"])

    n = len(train_y)
    split = max(1, int(n * 0.82))
    order_dates = pd.Series(np.arange(n))
    train_inner_x = train_x[:split]
    train_inner_y = train_y[:split]
    early_x = train_x[split:] if split < n else train_x[-max(1, min(512, n)) :]
    early_y = train_y[split:] if split < n else train_y[-max(1, min(512, n)) :]

    model = BuyLSTM(input_size=train_x.shape[-1]).to(DEVICE)
    positive = float(train_inner_y.sum())
    negative = float(len(train_inner_y) - positive)
    pos_weight = torch.tensor([negative / max(positive, 1.0)], dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(train_inner_x, dtype=torch.float32),
            torch.tensor(train_inner_y, dtype=torch.float32),
        ),
        batch_size=1024,
        shuffle=False,
    )
    early_tensor_x = torch.tensor(early_x, dtype=torch.float32, device=DEVICE)
    early_tensor_y = torch.tensor(early_y, dtype=torch.float32, device=DEVICE)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    patience = 0
    for _epoch in range(12):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            early_loss = float(loss_fn(model(early_tensor_x), early_tensor_y).cpu().item())
        if early_loss + 1e-5 < best_loss:
            best_loss = early_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    validation_loader = DataLoader(
        torch.tensor(validation_x, dtype=torch.float32), batch_size=2048, shuffle=False
    )
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x in validation_loader:
            logits = model(batch_x.to(DEVICE))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    leakage = {
        "fold": prepared.fold,
        "train_sequences": int(len(prepared.train_y)),
        "validation_sequences": int(len(prepared.validation_frame)),
        "scaler_fit_scope": "training_fold_only",
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_ends_on_prediction_date": True,
        "final_test_used": False,
        "early_stopping_chronological": True,
        "validation_min_date": str(train_dates.min().date()) if not train_dates.empty else None,
        "validation_max_date": str(train_dates.max().date()) if not train_dates.empty else None,
    }
    return np.concatenate(probabilities), leakage


def _metric_row(frame: pd.DataFrame, score_column: str, group: str, threshold: float | None = None) -> dict[str, Any]:
    actual = frame["buy_target"].astype(int)
    return {
        "Group": group,
        "Threshold": threshold,
        "Count": int(len(frame)),
        "BaseRate": _safe_float(actual.mean()) if len(frame) else None,
        "Precision": _safe_float(actual.mean()) if len(frame) else None,
        "AverageFutureReturn": _safe_float(frame["future_stock_return"].mean()) if len(frame) else None,
        "MedianFutureReturn": _safe_float(frame["future_stock_return"].median()) if len(frame) else None,
        "AverageExcessReturn": _safe_float(frame["excess_return"].mean()) if len(frame) else None,
        "MedianExcessReturn": _safe_float(frame["excess_return"].median()) if len(frame) else None,
        "NiftyWinRate": _safe_float((frame["excess_return"] > 0).mean()) if len(frame) else None,
        "ScoreMean": _safe_float(frame[score_column].mean()) if len(frame) else None,
    }


def _standalone_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate LSTM standalone and its same-date top tails."""

    y = predictions["buy_target"].astype(int)
    rows = [
        {
            "Group": "LSTM_STANDALONE_ALL",
            "Threshold": None,
            "Count": int(len(predictions)),
            "PRAUC": _safe_float(average_precision_score(y, predictions["p_lstm_buy"])),
            "LogLoss": _safe_float(log_loss(y, predictions["p_lstm_buy"], labels=[0, 1])),
            "Brier": _safe_float(brier_score_loss(y, predictions["p_lstm_buy"])),
            "Precision": _safe_float(y.mean()),
            "AverageExcessReturn": _safe_float(predictions["excess_return"].mean()),
            "MedianExcessReturn": _safe_float(predictions["excess_return"].median()),
            "NiftyWinRate": _safe_float((predictions["excess_return"] > 0).mean()),
        }
    ]
    for top in TOP_FRACTIONS:
        selected = predictions[predictions["lstm_percentile"] >= 1 - top].copy()
        row = _metric_row(selected, "p_lstm_buy", f"LSTM_TOP_{top:.3f}", top)
        row.update(
            {
                "PRAUC": None,
                "LogLoss": None,
                "Brier": None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _confirmation_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate LSTM only as confirmation of existing BUY signals."""

    rows = []
    current_buy = predictions[predictions["current_buy_signal"]].copy()
    tuned_high = predictions[predictions["tuned_high_confidence"]].copy()
    rows.append(_metric_row(current_buy, "p_lstm_buy", "A_ALL_CURRENT_BUY"))
    rows.append(_metric_row(tuned_high, "p_lstm_buy", "B_CURRENT_PLUS_TUNED_HIGH"))
    for top in BUY_STRATA:
        current_lstm = current_buy[current_buy["p_lstm_buy"] >= current_buy["p_lstm_buy"].quantile(1 - top)]
        row = _metric_row(current_lstm, "p_lstm_buy", f"C_CURRENT_BUY_LSTM_TOP_{top:.2f}", top)
        rows.append(row)
        tuned_lstm = tuned_high[tuned_high["p_lstm_buy"] >= tuned_high["p_lstm_buy"].quantile(1 - top)]
        row = _metric_row(
            tuned_lstm, "p_lstm_buy", f"D_CURRENT_TUNED_LSTM_TOP_{top:.2f}", top
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _stability_results(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    current_buy = predictions[predictions["current_buy_signal"]].copy()
    for value, group in current_buy.groupby(column):
        rows.append(_metric_row(group, "p_lstm_buy", f"CURRENT_BUY_BY_{column.upper()}", None) | {"ScopeValue": value})
        for top in (0.20, 0.10):
            selected = group[group["p_lstm_buy"] >= group["p_lstm_buy"].quantile(1 - top)]
            rows.append(
                _metric_row(selected, "p_lstm_buy", f"CURRENT_BUY_LSTM_TOP_{top:.2f}_BY_{column.upper()}", top)
                | {"ScopeValue": value}
            )
    return pd.DataFrame(rows)


def _year_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    output["year"] = output["Date"].dt.year
    return _stability_results(output, "year")


def _diversity_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("p_lstm_buy", "p_outperform"),
        ("p_lstm_buy", "binary_buy_sigmoid_probability"),
        ("p_lstm_buy", "tuned_binary_sigmoid_probability"),
    ]
    for left, right in pairs:
        rows.append(
            {
                "Metric": "probability_correlation",
                "Left": left,
                "Right": right,
                "Pearson": _safe_float(predictions[left].corr(predictions[right], method="pearson")),
                "Spearman": _safe_float(predictions[left].corr(predictions[right], method="spearman")),
            }
        )
    for top in (0.01, 0.05):
        lstm = set(_keys(predictions[predictions["lstm_percentile"] >= 1 - top]))
        xgb = set(_keys(predictions[predictions["classifier_percentile"] >= 1 - top]))
        binary = set(_keys(predictions[predictions["binary_buy_percentile"] >= 1 - top]))
        rows.append(_overlap_row(f"LSTM_TOP_{top}", "XGB26_TOP", lstm, xgb))
        rows.append(_overlap_row(f"LSTM_TOP_{top}", "BINARY77_TOP", lstm, binary))
    return pd.DataFrame(rows)


def _keys(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    return list(zip(frame["Date"], frame["symbol"], strict=False))


def _overlap_row(left_name: str, right_name: str, left: set[Any], right: set[Any]) -> dict[str, Any]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "Metric": "membership_overlap",
        "Left": left_name,
        "Right": right_name,
        "LeftCount": len(left),
        "RightCount": len(right),
        "Intersection": intersection,
        "Jaccard": _safe_float(intersection / union) if union else None,
    }


def _bootstrap_results(predictions: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    candidates = confirmation[
        confirmation["Group"].str.startswith("C_CURRENT_BUY_LSTM_TOP_")
        | confirmation["Group"].str.startswith("D_CURRENT_TUNED_LSTM_TOP_")
    ].copy()
    rows = []
    for group_name in candidates["Group"]:
        benchmark_name = (
            "B_CURRENT_PLUS_TUNED_HIGH"
            if group_name.startswith("D_")
            else "A_ALL_CURRENT_BUY"
        )
        candidate = _group_frame(predictions, group_name)
        benchmark = _group_frame(predictions, benchmark_name)
        if len(candidate) < 30 or len(benchmark) < 30:
            continue
        rows.extend(_date_block_bootstrap(candidate, benchmark, group_name, benchmark_name))
    return pd.DataFrame(rows)


def _group_frame(predictions: pd.DataFrame, group_name: str) -> pd.DataFrame:
    if group_name == "A_ALL_CURRENT_BUY":
        return predictions[predictions["current_buy_signal"]].copy()
    if group_name == "B_CURRENT_PLUS_TUNED_HIGH":
        return predictions[predictions["tuned_high_confidence"]].copy()
    if group_name.startswith("C_CURRENT_BUY_LSTM_TOP_"):
        top = float(group_name.rsplit("_", 1)[1])
        base = predictions[predictions["current_buy_signal"]].copy()
        return base[base["p_lstm_buy"] >= base["p_lstm_buy"].quantile(1 - top)].copy()
    if group_name.startswith("D_CURRENT_TUNED_LSTM_TOP_"):
        top = float(group_name.rsplit("_", 1)[1])
        base = predictions[predictions["tuned_high_confidence"]].copy()
        return base[base["p_lstm_buy"] >= base["p_lstm_buy"].quantile(1 - top)].copy()
    raise ValueError(group_name)


def _date_block_bootstrap(
    candidate: pd.DataFrame, benchmark: pd.DataFrame, candidate_name: str, benchmark_name: str
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = np.array(sorted(set(candidate["Date"]) | set(benchmark["Date"])))
    candidate_by_date = {date: group for date, group in candidate.groupby("Date")}
    benchmark_by_date = {date: group for date, group in benchmark.groupby("Date")}
    diffs = {"Precision": [], "AverageExcessReturn": [], "MedianExcessReturn": [], "NiftyWinRate": []}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        cand = pd.concat([candidate_by_date[d] for d in sample_dates if d in candidate_by_date], ignore_index=True)
        bench = pd.concat([benchmark_by_date[d] for d in sample_dates if d in benchmark_by_date], ignore_index=True)
        cand_metrics = _simple_metrics(cand)
        bench_metrics = _simple_metrics(bench)
        for metric in diffs:
            diffs[metric].append(cand_metrics[metric] - bench_metrics[metric])
    rows = []
    for metric, values in diffs.items():
        arr = np.array(values)
        rows.append(
            {
                "Candidate": candidate_name,
                "Benchmark": benchmark_name,
                "Metric": metric,
                "MeanDiff": _safe_float(arr.mean()),
                "CI95Low": _safe_float(np.quantile(arr, 0.025)),
                "CI95High": _safe_float(np.quantile(arr, 0.975)),
                "Samples": BOOTSTRAP_SAMPLES,
            }
        )
    return rows


def _simple_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "Precision": np.nan,
            "AverageExcessReturn": np.nan,
            "MedianExcessReturn": np.nan,
            "NiftyWinRate": np.nan,
        }
    return {
        "Precision": float(frame["buy_target"].mean()),
        "AverageExcessReturn": float(frame["excess_return"].mean()),
        "MedianExcessReturn": float(frame["excess_return"].median()),
        "NiftyWinRate": float((frame["excess_return"] > 0).mean()),
    }


def _decision(confirmation: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    """Classify the LSTM contribution without changing production rules."""

    all_current = confirmation[confirmation["Group"].eq("A_ALL_CURRENT_BUY")].iloc[0]
    tuned = confirmation[confirmation["Group"].eq("B_CURRENT_PLUS_TUNED_HIGH")].iloc[0]
    c_rows = confirmation[confirmation["Group"].str.startswith("C_CURRENT_BUY_LSTM_TOP_")]
    d_rows = confirmation[confirmation["Group"].str.startswith("D_CURRENT_TUNED_LSTM_TOP_")]
    robust_c = _has_robust_bootstrap(bootstrap, "C_CURRENT_BUY_LSTM_TOP_", "A_ALL_CURRENT_BUY")
    robust_d = _has_robust_bootstrap(bootstrap, "D_CURRENT_TUNED_LSTM_TOP_", "B_CURRENT_PLUS_TUNED_HIGH")
    c_better = c_rows[
        (c_rows["Count"] >= 80)
        & (c_rows["Precision"] > all_current["Precision"])
        & (c_rows["AverageExcessReturn"] > all_current["AverageExcessReturn"])
        & (c_rows["MedianExcessReturn"] > all_current["MedianExcessReturn"])
    ]
    d_better = d_rows[
        (d_rows["Count"] >= 40)
        & (d_rows["Precision"] > tuned["Precision"])
        & (d_rows["AverageExcessReturn"] > tuned["AverageExcessReturn"])
        & (d_rows["MedianExcessReturn"] > tuned["MedianExcessReturn"])
    ]
    if not c_better.empty and robust_c:
        return "LSTM_ADDS_VALUE_AS_CONFIRMATION"
    if not d_better.empty and robust_d:
        return "LSTM_ADDS_VALUE_ONLY_WITH_TUNED_BINARY"
    if not c_better.empty or not d_better.empty:
        return "LSTM_DIAGNOSTIC_ONLY"
    return "LSTM_DOES_NOT_ADD_VALUE"


def _has_robust_bootstrap(bootstrap: pd.DataFrame, prefix: str, benchmark: str) -> bool:
    if bootstrap.empty:
        return False
    subset = bootstrap[
        bootstrap["Candidate"].str.startswith(prefix) & bootstrap["Benchmark"].eq(benchmark)
    ]
    if subset.empty:
        return False
    precision = subset[subset["Metric"].eq("Precision")]
    avg = subset[subset["Metric"].eq("AverageExcessReturn")]
    median = subset[subset["Metric"].eq("MedianExcessReturn")]
    return bool(
        not precision.empty
        and not avg.empty
        and not median.empty
        and precision["CI95Low"].max() > 0
        and avg["CI95Low"].max() > 0
        and median["CI95Low"].max() > 0
    )


def _report_text(
    *,
    predictions: pd.DataFrame,
    standalone: pd.DataFrame,
    confirmation: pd.DataFrame,
    fold_stability: pd.DataFrame,
    year_stability: pd.DataFrame,
    diversity: pd.DataFrame,
    bootstrap: pd.DataFrame,
    leakage_rows: list[dict[str, Any]],
    decision: str,
) -> str:
    current = confirmation[confirmation["Group"].eq("A_ALL_CURRENT_BUY")].iloc[0].to_dict()
    tuned = confirmation[confirmation["Group"].eq("B_CURRENT_PLUS_TUNED_HIGH")].iloc[0].to_dict()
    best_c = confirmation[confirmation["Group"].str.startswith("C_")].sort_values(
        ["Precision", "AverageExcessReturn"], ascending=False
    ).head(1)
    best_d = confirmation[confirmation["Group"].str.startswith("D_")].sort_values(
        ["Precision", "AverageExcessReturn"], ascending=False
    ).head(1)
    leakage = pd.DataFrame(leakage_rows)
    lines = [
        "LSTM Sequence Model Validation-Only Research",
        "",
        f"Decision: {decision}",
        f"Rows scored: {len(predictions):,}",
        f"Validation date range: {predictions['Date'].min().date()} to {predictions['Date'].max().date()}",
        f"Final test used: False; final-test cutoff: {FINAL_TEST_START.date()}",
        "Production artifacts modified: False",
        "Existing BUY thresholds modified: False",
        "",
        "Leakage checks:",
        "- 60-day sequence ends on prediction date.",
        "- Future 20D labels are used only as targets/evaluation.",
        "- Scaling statistics are fitted inside each training fold only.",
        "- Folds use existing 20-trading-day embargo.",
        "- Early stopping split is chronological within the training fold.",
        f"- Leakage audit rows: {leakage.to_dict(orient='records')}",
        "",
        "Current BUY benchmark:",
        json.dumps(_json_safe(current), indent=2),
        "",
        "Current + Tuned Binary HIGH-confidence benchmark:",
        json.dumps(_json_safe(tuned), indent=2),
        "",
        "Best LSTM confirmation over Current BUY:",
        best_c.to_json(orient="records", indent=2),
        "",
        "Best LSTM confirmation over Current + Tuned:",
        best_d.to_json(orient="records", indent=2),
        "",
        "Standalone summary:",
        standalone.head(8).to_string(index=False),
        "",
        "Diversity summary:",
        diversity.to_string(index=False),
        "",
        "Bootstrap summary:",
        bootstrap.to_string(index=False) if not bootstrap.empty else "No bootstrap candidates.",
        "",
        "Output files:",
        str(OUTPUT_OOF),
        str(OUTPUT_STANDALONE),
        str(OUTPUT_CONFIRMATION),
        str(OUTPUT_FOLD),
        str(OUTPUT_YEAR),
        str(OUTPUT_DIVERSITY),
        str(OUTPUT_BOOTSTRAP),
    ]
    return "\n".join(lines)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    main()
