"""Rebalance-event simulator with explicit next-session fills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.config import BacktestConfig
from app.backtesting.costs.india import calculate_india_cash_cost
from app.backtesting.data.yahoo import add_tradability_fields
from app.backtesting.metrics.performance import calculate_performance_metrics
from app.backtesting.signals.base import SignalFunction, assert_signal_contract


@dataclass(slots=True)
class BacktestResult:
    """Complete auditable strategy output."""

    metrics: dict[str, Any]
    periods: pd.DataFrame
    trades: pd.DataFrame
    scored_panel: pd.DataFrame
    skipped_orders: pd.DataFrame


def run_strategy_backtest(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    signal: SignalFunction,
    config: BacktestConfig,
    *,
    trials_logged: int = 1,
) -> BacktestResult:
    """Run a long-only or market-neutral non-overlapping portfolio backtest.

    The signal is calculated with data through close T. Selection is stored at T,
    and all entry orders are filled at the adjusted open of the next market session.
    """

    config.validate()
    data = add_tradability_fields(panel, lookback=config.strategy.liquidity_lookback)
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    data["score"] = signal(data)
    assert_signal_contract(data, data["score"])
    daily_return = data.groupby("symbol")["adj_close"].pct_change()
    data["volatility"] = daily_return.groupby(data["symbol"]).transform(
        lambda values: values.rolling(config.strategy.volatility_lookback).std()
    )
    metadata = universe[["symbol", "sector"]].drop_duplicates("symbol")
    data = data.merge(metadata, on="symbol", how="left")
    data["sector"] = data["sector"].fillna("Unknown")
    symbol_frames = {
        str(symbol): group.sort_values("date").reset_index(drop=True)
        for symbol, group in data.groupby("symbol", sort=False)
    }

    benchmark = data[data["symbol"] == config.data.benchmark].sort_values("date")
    market_dates = pd.DatetimeIndex(benchmark["date"].drop_duplicates())
    rebalance_dates = _rebalance_dates(market_dates, config.strategy.rebalance_frequency)
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    previous_symbols: set[str] = set()
    for index, signal_date in enumerate(rebalance_dates[:-1]):
        next_signal_date = rebalance_dates[index + 1]
        entry_date = _next_date(market_dates, signal_date)
        exit_date = _next_date(market_dates, next_signal_date)
        if entry_date is None or exit_date is None:
            continue
        cross_section = data[
            (data["date"] == signal_date) & (data["symbol"] != config.data.benchmark)
        ].copy()
        eligible = cross_section[
            cross_section["score"].notna()
            & cross_section["is_tradable"]
            & cross_section["avg_traded_value_20d"].ge(config.data.min_avg_traded_value_20d)
        ]
        weights = _build_weights(eligible, config)
        if not weights:
            skipped.append({"signal_date": signal_date, "reason": "no_eligible_stocks"})
            continue
        gross_exposure = sum(abs(value) for value in weights.values())
        net_exposure = sum(weights.values())
        if gross_exposure > 1 + 1e-9:
            raise AssertionError(f"Gross portfolio exposure exceeds 100%: {gross_exposure:.6f}")
        if not config.strategy.long_short and net_exposure > 1 + 1e-9:
            raise AssertionError(f"Long portfolio exposure exceeds 100%: {net_exposure:.6f}")

        contributions: list[float] = []
        gross_contributions: list[float] = []
        executed_symbols: set[str] = set()
        for symbol, weight in weights.items():
            outcome = _simulate_trade(
                symbol_frames[symbol],
                symbol=symbol,
                signal_date=signal_date,
                entry_date=entry_date,
                planned_exit_date=exit_date,
                weight=weight,
                config=config,
            )
            if outcome.get("skipped"):
                skipped.append(outcome)
                continue
            trades.append(outcome)
            contributions.append(float(outcome["net_contribution"]))
            gross_contributions.append(float(outcome["gross_contribution"]))
            executed_symbols.add(symbol)
        if not contributions:
            continue
        benchmark_return = _benchmark_period_return(benchmark, entry_date, exit_date)
        turnover = len(executed_symbols.symmetric_difference(previous_symbols)) / max(
            1, 2 * len(executed_symbols | previous_symbols)
        )
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "gross_return": float(sum(gross_contributions)),
                "net_return": float(sum(contributions)),
                "benchmark_return": benchmark_return,
                "positions": len(executed_symbols),
                "turnover": float(turnover),
                "gross_exposure": float(sum(abs(weights[symbol]) for symbol in executed_symbols)),
                "net_exposure": float(sum(weights[symbol] for symbol in executed_symbols)),
            }
        )
        previous_symbols = executed_symbols

    period_frame = pd.DataFrame(rows)
    trade_frame = pd.DataFrame(trades)
    metrics = calculate_performance_metrics(period_frame, trade_frame, trials=trials_logged)
    return BacktestResult(
        metrics=metrics,
        periods=period_frame,
        trades=trade_frame,
        scored_panel=data,
        skipped_orders=pd.DataFrame(skipped),
    )


def _build_weights(frame: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    strategy = config.strategy
    longs = frame.nlargest(strategy.top_n, "score").copy()
    shorts = (
        frame.nsmallest(strategy.bottom_n, "score").copy()
        if strategy.long_short
        else frame.iloc[0:0]
    )
    long_budget = 0.5 if strategy.long_short else 1.0
    short_budget = 0.5 if strategy.long_short else 0.0
    weights = _allocate_side(
        longs,
        long_budget,
        1.0,
        strategy.weighting,
        strategy.max_position_weight,
        strategy.max_sector_weight,
    )
    weights.update(
        _allocate_side(
            shorts,
            short_budget,
            -1.0,
            strategy.weighting,
            strategy.max_position_weight,
            strategy.max_sector_weight,
        )
    )
    return weights


def _allocate_side(
    selected: pd.DataFrame,
    budget: float,
    direction: float,
    method: str,
    max_position: float,
    max_sector: float,
) -> dict[str, float]:
    if selected.empty or budget <= 0:
        return {}
    if method == "inverse_volatility":
        raw = 1 / selected["volatility"].replace(0, np.nan)
        raw = raw.fillna(raw.median()).fillna(1.0)
    else:
        raw = pd.Series(1.0, index=selected.index)
    raw = raw / raw.sum() * budget
    allocation: dict[str, float] = {}
    sector_used: dict[str, float] = {}
    for idx, row in selected.iterrows():
        remaining = budget - sum(abs(value) for value in allocation.values())
        if remaining <= 1e-12:
            break
        sector = str(row["sector"])
        sector_room = max(0.0, max_sector - sector_used.get(sector, 0.0))
        value = min(float(raw.loc[idx]), max_position, sector_room, remaining)
        if value <= 0:
            continue
        allocation[str(row["symbol"])] = direction * value
        sector_used[sector] = sector_used.get(sector, 0.0) + value
    return allocation


def _simulate_trade(
    symbol_frame: pd.DataFrame,
    *,
    symbol: str,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    planned_exit_date: pd.Timestamp,
    weight: float,
    config: BacktestConfig,
) -> dict[str, Any]:
    bars = symbol_frame[
        (symbol_frame["date"] >= entry_date) & (symbol_frame["date"] <= planned_exit_date)
    ]
    if bars.empty or bars.iloc[0]["date"] != entry_date:
        return {
            "symbol": symbol,
            "signal_date": signal_date,
            "skipped": True,
            "reason": "missing_entry_bar",
        }
    entry = bars.iloc[0]
    prior = symbol_frame[symbol_frame["date"] < entry_date].tail(1)
    gap = 0.0 if prior.empty else float(entry["adj_open"] / prior.iloc[0]["adj_close"] - 1)
    locked = (
        entry["adj_high"] == entry["adj_low"]
        and abs(gap) >= config.strategy.circuit_limit_pct * 0.95
    )
    if not bool(entry["is_tradable"]) or locked:
        return {
            "symbol": symbol,
            "signal_date": signal_date,
            "skipped": True,
            "reason": "untradable_entry",
        }
    direction = 1 if weight > 0 else -1
    entry_price = float(entry["adj_open"])
    exit_price = float(bars.iloc[-1]["adj_open"])
    exit_date = pd.Timestamp(bars.iloc[-1]["date"])
    exit_reason = "rebalance"
    peak = entry_price
    trough = entry_price
    for _, bar in bars.iloc[1:].iterrows():
        if direction > 0:
            fixed_stop = (
                entry_price * (1 - config.strategy.stop_loss_pct)
                if config.strategy.stop_loss_pct
                else -np.inf
            )
            trailing_stop = (
                peak * (1 - config.strategy.trailing_stop_pct)
                if config.strategy.trailing_stop_pct
                else -np.inf
            )
            stop = max(fixed_stop, trailing_stop)
            if float(bar["adj_low"]) <= stop:
                exit_price = min(float(bar["adj_open"]), stop)
                exit_date = pd.Timestamp(bar["date"])
                exit_reason = "stop_loss" if stop == fixed_stop else "trailing_stop"
                break
            peak = max(peak, float(bar["adj_high"]))
        else:
            fixed_stop = (
                entry_price * (1 + config.strategy.stop_loss_pct)
                if config.strategy.stop_loss_pct
                else np.inf
            )
            trailing_stop = (
                trough * (1 + config.strategy.trailing_stop_pct)
                if config.strategy.trailing_stop_pct
                else np.inf
            )
            stop = min(fixed_stop, trailing_stop)
            if float(bar["adj_high"]) >= stop:
                exit_price = max(float(bar["adj_open"]), stop)
                exit_date = pd.Timestamp(bar["date"])
                exit_reason = "stop_loss" if stop == fixed_stop else "trailing_stop"
                break
            trough = min(trough, float(bar["adj_low"]))
    gross_return = direction * (exit_price / entry_price - 1)
    volatility = float(entry.get("volatility", 0.0) or 0.0)
    traded_value = float(entry.get("traded_value", 0.0) or 0.0)
    capital = config.strategy.initial_capital_rupees
    entry_notional = abs(weight) * capital
    order_fraction = entry_notional / traded_value if traded_value > 0 else 0.0
    entry_side = "buy" if direction > 0 else "sell"
    exit_side = "sell" if direction > 0 else "buy"
    entry_cost = calculate_india_cash_cost(
        entry_notional,
        entry_side,
        config.costs,
        order_value_as_fraction_of_daily_volume=order_fraction,
        daily_volatility=volatility,
    ).total
    exit_cost = calculate_india_cash_cost(
        entry_notional * max(0.0, 1 + gross_return),
        exit_side,
        config.costs,
        order_value_as_fraction_of_daily_volume=order_fraction,
        daily_volatility=volatility,
    ).total
    cost_rupees = entry_cost + exit_cost
    cost_fraction = cost_rupees / capital
    net_return = gross_return - cost_fraction / abs(weight)
    return {
        "symbol": symbol,
        "signal_date": signal_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": "long" if direction > 0 else "short",
        "weight": abs(weight),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "net_return": net_return,
        "gross_contribution": abs(weight) * gross_return,
        "net_contribution": abs(weight) * gross_return - cost_fraction,
        "cost_rupees": cost_rupees,
        "cost_fraction": cost_fraction,
        "exit_reason": exit_reason,
        "holding_days": int((exit_date - entry_date).days),
        "skipped": False,
    }


def _benchmark_period_return(
    benchmark: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp
) -> float:
    entry = benchmark[benchmark["date"] == entry_date]
    exit_row = benchmark[benchmark["date"] == exit_date]
    if entry.empty or exit_row.empty:
        return 0.0
    return float(exit_row.iloc[0]["adj_open"] / entry.iloc[0]["adj_open"] - 1)


def _rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    series = pd.Series(dates).sort_values()
    periods = series.dt.to_period("W-FRI" if frequency == "weekly" else "M")
    return pd.DatetimeIndex(series.groupby(periods).max())


def _next_date(dates: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    location = dates.searchsorted(pd.Timestamp(date), side="right")
    return pd.Timestamp(dates[location]) if location < len(dates) else None
