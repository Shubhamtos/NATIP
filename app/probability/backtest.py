"""Backtest probability-ranked portfolios against Nifty."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def backtest_top_n_portfolio(
    predictions: pd.DataFrame,
    *,
    top_n: int = 5,
    transaction_cost_bps: float = 10.0,
    rebalance_every_days: int = 20,
    leverage_enabled: bool = False,
) -> dict[str, Any]:
    """Backtest a non-overlapping top-ranked portfolio.

    The prediction target is a 20-trading-day future return. Rebalancing every
    row/date would compound overlapping 20-day outcomes and can distort drawdown
    and CAGR. This backtest therefore samples every ``rebalance_every_days``
    available dates and applies one equal-weight 20-day holding-period return.
    """

    required = {"Date", "symbol", "p_outperform", "future_stock_return", "future_nifty_return"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns for backtest: {sorted(missing)}")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if rebalance_every_days <= 0:
        raise ValueError("rebalance_every_days must be positive.")

    clean = predictions.copy()
    clean["Date"] = pd.to_datetime(clean["Date"])
    clean = clean.replace([float("inf"), float("-inf")], pd.NA)
    clean = clean.dropna(
        subset=["Date", "symbol", "p_outperform", "future_stock_return", "future_nifty_return"]
    )
    clean = clean.sort_values(["Date", "p_outperform"], ascending=[True, False])
    duplicate_rows = int(clean.duplicated(subset=["Date", "symbol"]).sum())
    clean = clean.drop_duplicates(subset=["Date", "symbol"], keep="first")

    unique_dates = list(clean["Date"].drop_duplicates().sort_values())
    rebalance_dates = unique_dates[::rebalance_every_days]

    rows: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    max_weight_sum = 0.0
    skipped_dates = 0
    for date in rebalance_dates:
        group = clean[clean["Date"] == date]
        selected = group.sort_values("p_outperform", ascending=False).head(top_n)
        if selected.empty:
            skipped_dates += 1
            continue

        weight = 1.0 / len(selected)
        current_weights = {str(symbol): weight for symbol in selected["symbol"]}
        weight_sum = sum(current_weights.values())
        max_weight_sum = max(max_weight_sum, weight_sum)
        if not leverage_enabled and weight_sum > 1.0 + 1e-9:
            raise AssertionError(f"Portfolio weights exceed 100% on {date}: {weight_sum:.6f}")

        turnover = _portfolio_turnover(previous_weights, current_weights)
        transaction_cost = turnover * transaction_cost_bps / 10_000
        portfolio_return = float((selected["future_stock_return"] * weight).sum() - transaction_cost)
        benchmark_return = float(selected["future_nifty_return"].mean())
        rows.append(
            {
                "Date": date,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "turnover": turnover,
                "weight_sum": weight_sum,
                "positions": int(len(selected)),
            }
        )
        previous_weights = current_weights

    returns = pd.DataFrame(rows).dropna()
    if returns.empty:
        return {}
    non_finite_returns = int(
        (~returns["portfolio_return"].map(lambda value: math.isfinite(float(value)))).sum()
    )
    if non_finite_returns:
        raise ValueError(f"Backtest contains {non_finite_returns} non-finite portfolio returns.")

    return {
        "top_n": top_n,
        "rebalance_every_days": rebalance_every_days,
        "leverage_enabled": leverage_enabled,
        "cagr": _annualized_return(returns["portfolio_return"]),
        "benchmark_cagr": _annualized_return(returns["benchmark_return"]),
        "sharpe": _sharpe(returns["portfolio_return"]),
        "sortino": _sortino(returns["portfolio_return"]),
        "max_drawdown": _max_drawdown(returns["portfolio_return"]),
        "volatility": returns["portfolio_return"].std() * math.sqrt(252 / 20),
        "win_rate": float((returns["portfolio_return"] > returns["benchmark_return"]).mean()),
        "turnover": float(returns["turnover"].mean()),
        "max_weight_sum": float(max_weight_sum),
        "duplicate_prediction_rows_removed": duplicate_rows,
        "skipped_rebalance_dates": skipped_dates,
        "non_overlapping_holding_periods": True,
        "periods": len(returns),
    }


def _portfolio_turnover(
    previous_weights: dict[str, float],
    current_weights: dict[str, float],
) -> float:
    """Calculate one-way turnover from old and new portfolio weights."""

    symbols = set(previous_weights) | set(current_weights)
    gross_turnover = sum(
        abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
        for symbol in symbols
    )
    return float(gross_turnover / 2)


def _annualized_return(returns: pd.Series) -> float:
    """Annualize 20-day holding-period returns."""

    compounded = (1 + returns).prod()
    years = len(returns) * 20 / 252
    return float(compounded ** (1 / years) - 1) if years > 0 else 0.0


def _sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""

    std = returns.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float(returns.mean() / std * math.sqrt(252 / 20))


def _sortino(returns: pd.Series) -> float:
    """Calculate annualized Sortino ratio."""

    downside = returns[returns < 0].std()
    if downside == 0 or pd.isna(downside):
        return 0.0
    return float(returns.mean() / downside * math.sqrt(252 / 20))


def _max_drawdown(returns: pd.Series) -> float:
    """Calculate max drawdown from periodic returns."""

    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())
