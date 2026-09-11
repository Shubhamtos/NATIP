"""Evaluate prospective V1/V2 shadow predictions after the 20-trading-day horizon."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.probability.config import BENCHMARK_SYMBOL, PROJECT_ROOT
from app.probability.frozen_inference import V1_V2_SHADOW_HISTORY, load_clean_frames

OUTCOME_PATH = PROJECT_ROOT / "outputs" / "v1_v2_shadow_outcomes.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "v1_v2_shadow_evaluation_summary.json"


def main() -> None:
    """Evaluate mature shadow predictions without changing any signal rule."""

    if not V1_V2_SHADOW_HISTORY.exists():
        raise FileNotFoundError(
            f"No shadow predictions found at {V1_V2_SHADOW_HISTORY}. Run predict_stock.py first."
        )
    predictions = pd.read_csv(V1_V2_SHADOW_HISTORY, parse_dates=["signal_date"])
    outcomes = _evaluate_predictions(predictions)
    OUTCOME_PATH.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(OUTCOME_PATH, index=False)
    summary = _summary(outcomes)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"outcomes": str(OUTCOME_PATH), "summary": summary}, indent=2))


def _evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tickers = sorted(set(predictions["ticker"].dropna().astype(str)))
    symbols = sorted(set([*tickers, BENCHMARK_SYMBOL]))
    frames = load_clean_frames(symbols, stock_symbols=tickers, sector_symbols=[])
    nifty = frames[BENCHMARK_SYMBOL].sort_values("Date").reset_index(drop=True)
    for item in predictions.itertuples(index=False):
        ticker = str(item.ticker)
        stock = frames.get(ticker)
        if stock is None or stock.empty:
            continue
        outcome = _future_outcome(
            stock.sort_values("Date").reset_index(drop=True),
            nifty,
            pd.Timestamp(item.signal_date),
        )
        if outcome is None:
            continue
        rows.append(
            {
                **item._asdict(),
                **outcome,
                "excess_gt_5pct_hit": bool(outcome["future_excess_return"] > 0.05),
                "nifty_win": bool(outcome["future_excess_return"] > 0),
                "v1_buy_signal": str(item.v1_recommendation).upper() == "BUY",
                "v2_buy_signal": str(item.v2_candidate_signal).upper() == "BUY",
            }
        )
    return pd.DataFrame(rows)


def _future_outcome(
    stock: pd.DataFrame, nifty: pd.DataFrame, signal_date: pd.Timestamp
) -> dict[str, Any] | None:
    stock_dates = pd.to_datetime(stock["Date"])
    nifty_dates = pd.to_datetime(nifty["Date"])
    stock_match = stock_dates[stock_dates == signal_date]
    nifty_match = nifty_dates[nifty_dates == signal_date]
    if stock_match.empty or nifty_match.empty:
        return None
    stock_pos = int(stock_match.index[0])
    nifty_pos = int(nifty_match.index[0])
    if stock_pos + 20 >= len(stock) or nifty_pos + 20 >= len(nifty):
        return None
    stock_entry = float(stock.loc[stock_pos, "Close"])
    stock_exit = float(stock.loc[stock_pos + 20, "Close"])
    nifty_entry = float(nifty.loc[nifty_pos, "Close"])
    nifty_exit = float(nifty.loc[nifty_pos + 20, "Close"])
    stock_return = stock_exit / stock_entry - 1
    nifty_return = nifty_exit / nifty_entry - 1
    return {
        "outcome_date": pd.Timestamp(stock.loc[stock_pos + 20, "Date"]).date().isoformat(),
        "future_stock_return": stock_return,
        "future_nifty_return": nifty_return,
        "future_excess_return": stock_return - nifty_return,
    }


def _summary(outcomes: pd.DataFrame) -> dict[str, Any]:
    if outcomes.empty:
        return {
            "evaluated_predictions": 0,
            "status": "No predictions have reached the 20-trading-day horizon yet.",
            "threshold_policy": "Do not tune V2 from shadow results.",
        }
    base_rate = float(outcomes["excess_gt_5pct_hit"].mean())
    v1_buy = outcomes[outcomes["v1_buy_signal"]].copy()
    v2_buy = outcomes[outcomes["v2_buy_signal"]].copy()
    overlap = outcomes[outcomes["v1_buy_signal"] & outcomes["v2_buy_signal"]]
    v1_only = outcomes[outcomes["v1_buy_signal"] & ~outcomes["v2_buy_signal"]]
    v2_only = outcomes[~outcomes["v1_buy_signal"] & outcomes["v2_buy_signal"]]
    return {
        "evaluated_predictions": int(len(outcomes)),
        "base_excess_gt_5pct_rate": base_rate,
        "v1": _metrics(v1_buy, base_rate),
        "v2": _metrics(v2_buy, base_rate),
        "overlap_signals": _metrics(overlap, base_rate),
        "v1_only_signals": _metrics(v1_only, base_rate),
        "v2_only_signals": _metrics(v2_only, base_rate),
        "threshold_policy": "Do not tune V2 from shadow results.",
    }


def _metrics(frame: pd.DataFrame, base_rate: float) -> dict[str, Any]:
    if frame.empty:
        return {
            "buy_signal_count": 0,
            "precision": None,
            "average_excess": None,
            "median_excess": None,
            "nifty_win_rate": None,
            "precision_lift": None,
        }
    precision = float(frame["excess_gt_5pct_hit"].mean())
    return {
        "buy_signal_count": int(len(frame)),
        "precision": precision,
        "average_excess": float(frame["future_excess_return"].mean()),
        "median_excess": float(frame["future_excess_return"].median()),
        "nifty_win_rate": float(frame["nifty_win"].mean()),
        "precision_lift": precision / base_rate if base_rate else None,
    }


if __name__ == "__main__":
    main()
