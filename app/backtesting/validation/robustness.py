"""Time-ordered validation, baselines, sensitivity, and Monte Carlo checks."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.backtesting.config import TRIAL_LOG_PATH, BacktestConfig
from app.backtesting.costs.india import calculate_india_cash_cost
from app.backtesting.data.yahoo import add_tradability_fields
from app.backtesting.engine.portfolio import (
    BacktestResult,
    _allocate_side,
    _benchmark_period_return,
    _next_date,
    _rebalance_dates,
    _simulate_trade,
    run_strategy_backtest,
)
from app.backtesting.metrics.performance import calculate_performance_metrics
from app.backtesting.signals.base import SignalFunction


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One purged chronological validation fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_folds(dates: pd.Series, config: BacktestConfig) -> list[WalkForwardFold]:
    """Create expanding folds with a trading-session embargo."""

    sessions = pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()
    if config.validation.final_oos_start:
        sessions = sessions[sessions < pd.Timestamp(config.validation.final_oos_start)]
    first_test = sessions.min() + pd.DateOffset(years=config.validation.train_years)
    test_start = pd.Timestamp(sessions[sessions.searchsorted(first_test)])
    folds: list[WalkForwardFold] = []
    fold_id = 1
    while test_start < sessions.max():
        proposed_end = test_start + pd.DateOffset(months=config.validation.test_months)
        test_sessions = sessions[(sessions >= test_start) & (sessions < proposed_end)]
        if test_sessions.empty:
            break
        test_end = pd.Timestamp(test_sessions.max())
        test_position = int(sessions.get_loc(test_start))
        train_end_position = test_position - config.validation.embargo_days - 1
        if train_end_position < 0:
            break
        train_end = pd.Timestamp(sessions[train_end_position])
        embargo_start = pd.Timestamp(sessions[train_end_position + 1])
        embargo_end = pd.Timestamp(sessions[test_position - 1])
        folds.append(
            WalkForwardFold(
                fold=fold_id,
                train_start=pd.Timestamp(sessions.min()),
                train_end=train_end,
                embargo_start=embargo_start,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_id += 1
        next_position = sessions.searchsorted(test_end, side="right")
        if next_position >= len(sessions):
            break
        test_start = pd.Timestamp(sessions[next_position])
    return folds


def walk_forward_backtest(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    signal: SignalFunction,
    config: BacktestConfig,
) -> tuple[BacktestResult, pd.DataFrame]:
    """Stitch only unseen fold-period returns from chronological simulations."""

    benchmark_dates = panel.loc[panel["symbol"] == config.data.benchmark, "date"]
    folds = build_walk_forward_folds(benchmark_dates, config)
    period_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    scored = pd.DataFrame()
    skipped = pd.DataFrame()
    for fold in folds:
        available = panel[pd.to_datetime(panel["date"]) <= fold.test_end].copy()
        result = run_strategy_backtest(available, universe, signal, config)
        period = result.periods[
            (pd.to_datetime(result.periods["signal_date"]) >= fold.test_start)
            & (pd.to_datetime(result.periods["signal_date"]) <= fold.test_end)
        ].copy()
        trade = result.trades[
            (pd.to_datetime(result.trades["signal_date"]) >= fold.test_start)
            & (pd.to_datetime(result.trades["signal_date"]) <= fold.test_end)
        ].copy()
        period["fold"] = fold.fold
        trade["fold"] = fold.fold
        period_parts.append(period)
        trade_parts.append(trade)
        audit_rows.append(
            {
                **asdict(fold),
                "train_rows": int(
                    len(
                        panel[
                            pd.to_datetime(panel["date"]).between(fold.train_start, fold.train_end)
                        ]
                    )
                ),
                "test_rows": int(
                    len(
                        panel[pd.to_datetime(panel["date"]).between(fold.test_start, fold.test_end)]
                    )
                ),
                "embargo_sessions": config.validation.embargo_days,
                "no_overlap": fold.train_end
                < fold.embargo_start
                <= fold.embargo_end
                < fold.test_start,
            }
        )
        scored = result.scored_panel
        skipped = result.skipped_orders
    periods = pd.concat(period_parts, ignore_index=True) if period_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    from app.backtesting.metrics.performance import calculate_performance_metrics

    result = BacktestResult(
        metrics=calculate_performance_metrics(periods, trades, trials=count_logged_trials()),
        periods=periods,
        trades=trades,
        scored_panel=scored,
        skipped_orders=skipped,
    )
    audit = pd.DataFrame(audit_rows)
    if not audit.empty and not audit["no_overlap"].all():
        raise AssertionError("Walk-forward embargo violation")
    return result, audit


def parameter_sensitivity(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    signal: SignalFunction,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Vary top-N and stop loss by plus/minus the configured sensitivity."""

    pct = config.validation.sensitivity_pct
    top_values = sorted(
        {max(1, round(config.strategy.top_n * factor)) for factor in (1 - pct, 1, 1 + pct)}
    )
    base_stop = config.strategy.stop_loss_pct
    stop_values = (
        [None]
        if base_stop is None
        else sorted({base_stop * factor for factor in (1 - pct, 1, 1 + pct)})
    )
    rows: list[dict[str, Any]] = []
    for top_n in top_values:
        for stop in stop_values:
            candidate = copy.deepcopy(config)
            candidate.strategy.top_n = top_n
            candidate.strategy.stop_loss_pct = stop
            result, _ = walk_forward_backtest(panel, universe, signal, candidate)
            rows.append(
                {
                    "top_n": top_n,
                    "stop_loss_pct": stop,
                    "total_return": result.metrics.get("total_return"),
                    "sharpe": result.metrics.get("sharpe"),
                    "max_drawdown": result.metrics.get("max_drawdown"),
                    "trades": result.metrics.get("trades"),
                }
            )
            log_trial(candidate, experiment="sensitivity")
    return pd.DataFrame(rows)


def random_selection_baseline(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    config: BacktestConfig,
    *,
    samples: int | None = None,
    turnover_schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run random portfolios with matching rebalance timing, size, and turnover."""

    if config.strategy.long_short:
        raise NotImplementedError("Random baseline currently supports long-only strategies")
    sample_count = samples or config.validation.random_portfolios
    rng = np.random.default_rng(config.validation.random_seed)
    opportunities = _random_trade_opportunities(panel, universe, config)
    if opportunities.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for sample in range(sample_count):
        period_rows: list[dict[str, Any]] = []
        previous_symbols: set[str] = set()
        for signal_date, group in opportunities.groupby("signal_date", sort=True):
            target_size = min(len(group), config.strategy.top_n)
            target_turnover = _target_turnover(turnover_schedule, signal_date)
            replacements = min(
                target_size,
                round(target_turnover * target_size / max(1e-9, 1 - target_turnover)),
            )
            eligible_previous = list(previous_symbols.intersection(set(group["symbol"])))
            retain_count = min(len(eligible_previous), max(0, target_size - replacements))
            retained = (
                rng.choice(eligible_previous, size=retain_count, replace=False).tolist()
                if retain_count
                else []
            )
            remaining = group[~group["symbol"].isin(retained)]
            needed = target_size - len(retained)
            fresh_indices = (
                rng.choice(remaining.index.to_numpy(), size=needed, replace=False)
                if needed
                else np.array([], dtype=int)
            )
            selected = pd.concat(
                [group[group["symbol"].isin(retained)], group.loc[fresh_indices]],
                ignore_index=False,
            )
            selected["random_score"] = rng.random(len(selected))
            weights = _allocate_side(
                selected.sort_values("random_score", ascending=False),
                1.0,
                1.0,
                config.strategy.weighting,
                config.strategy.max_position_weight,
                config.strategy.max_sector_weight,
            )
            net_return = gross_return = 0.0
            selected_symbols: set[str] = set()
            for _, opportunity in selected.iterrows():
                symbol = str(opportunity["symbol"])
                if symbol not in weights:
                    continue
                weight = abs(weights[symbol])
                trade_return = float(opportunity["gross_trade_return"])
                capital = config.strategy.initial_capital_rupees
                entry_notional = weight * capital
                entry_cost = calculate_india_cash_cost(
                    entry_notional,
                    "buy",
                    config.costs,
                    order_value_as_fraction_of_daily_volume=(
                        entry_notional / float(opportunity["traded_value"])
                        if float(opportunity["traded_value"]) > 0
                        else 0.0
                    ),
                    daily_volatility=float(opportunity["volatility"]),
                ).total
                exit_cost = calculate_india_cash_cost(
                    entry_notional * max(0.0, 1 + trade_return),
                    "sell",
                    config.costs,
                    daily_volatility=float(opportunity["volatility"]),
                ).total
                gross_return += weight * trade_return
                net_return += weight * trade_return - (entry_cost + exit_cost) / capital
                selected_symbols.add(symbol)
            turnover = len(selected_symbols.symmetric_difference(previous_symbols)) / max(
                1, 2 * len(selected_symbols | previous_symbols)
            )
            first = group.iloc[0]
            period_rows.append(
                {
                    "exit_date": first["planned_exit_date"],
                    "net_return": net_return,
                    "gross_return": gross_return,
                    "benchmark_return": first["benchmark_return"],
                    "turnover": turnover,
                    "gross_exposure": sum(abs(value) for value in weights.values()),
                }
            )
            previous_symbols = selected_symbols
        metrics = calculate_performance_metrics(pd.DataFrame(period_rows), pd.DataFrame())
        rows.append(
            {
                "sample": sample + 1,
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "average_turnover": metrics.get("average_turnover"),
            }
        )
    return pd.DataFrame(rows)


def _target_turnover(schedule: pd.DataFrame | None, signal_date: pd.Timestamp) -> float:
    """Return the strategy's realized one-way turnover for a signal date."""

    if schedule is None or schedule.empty or "signal_date" not in schedule:
        return 0.5
    match = schedule[pd.to_datetime(schedule["signal_date"]) == pd.Timestamp(signal_date)]
    if match.empty:
        return 0.5
    return float(np.clip(match.iloc[0]["turnover"], 0.0, 0.5))


def _random_trade_opportunities(
    panel: pd.DataFrame,
    universe: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Precompute each eligible trade once so 1,000 random trials remain practical."""

    data = add_tradability_fields(panel, lookback=config.strategy.liquidity_lookback)
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    returns = data.groupby("symbol")["adj_close"].pct_change()
    data["volatility"] = returns.groupby(data["symbol"]).transform(
        lambda values: values.rolling(config.strategy.volatility_lookback).std()
    )
    data = data.merge(
        universe[["symbol", "sector"]].drop_duplicates("symbol"),
        on="symbol",
        how="left",
    )
    data["sector"] = data["sector"].fillna("Unknown")
    symbol_frames = {
        str(symbol): group.sort_values("date").reset_index(drop=True)
        for symbol, group in data.groupby("symbol", sort=False)
    }
    benchmark = symbol_frames[config.data.benchmark]
    market_dates = pd.DatetimeIndex(benchmark["date"].drop_duplicates())
    folds = build_walk_forward_folds(pd.Series(market_dates), config)
    allowed_dates = set()
    for fold in folds:
        allowed_dates.update(
            market_dates[(market_dates >= fold.test_start) & (market_dates <= fold.test_end)]
        )
    rebalance_dates = _rebalance_dates(market_dates, config.strategy.rebalance_frequency)
    zero_cost = copy.deepcopy(config)
    for field in (
        "brokerage_bps",
        "stt_buy_bps",
        "stt_sell_bps",
        "exchange_transaction_bps",
        "sebi_bps",
        "stamp_duty_buy_bps",
        "slippage_bps",
        "volume_impact_coefficient",
    ):
        setattr(zero_cost.costs, field, 0.0)
    rows: list[dict[str, Any]] = []
    for index, signal_date in enumerate(rebalance_dates[:-1]):
        if signal_date not in allowed_dates:
            continue
        entry_date = _next_date(market_dates, signal_date)
        exit_date = _next_date(market_dates, rebalance_dates[index + 1])
        if entry_date is None or exit_date is None:
            continue
        cross_section = data[
            (data["date"] == signal_date) & (data["symbol"] != config.data.benchmark)
        ]
        eligible = cross_section[
            cross_section["is_tradable"]
            & cross_section["avg_traded_value_20d"].ge(config.data.min_avg_traded_value_20d)
        ]
        benchmark_return = _benchmark_period_return(benchmark, entry_date, exit_date)
        for _, candidate in eligible.iterrows():
            symbol = str(candidate["symbol"])
            outcome = _simulate_trade(
                symbol_frames[symbol],
                symbol=symbol,
                signal_date=signal_date,
                entry_date=entry_date,
                planned_exit_date=exit_date,
                weight=0.01,
                config=zero_cost,
            )
            if outcome.get("skipped"):
                continue
            rows.append(
                {
                    "signal_date": signal_date,
                    "planned_exit_date": exit_date,
                    "symbol": symbol,
                    "sector": candidate["sector"],
                    "volatility": float(candidate.get("volatility", 0.0) or 0.0),
                    "traded_value": float(candidate.get("traded_value", 0.0) or 0.0),
                    "gross_trade_return": outcome["gross_return"],
                    "benchmark_return": benchmark_return,
                }
            )
    return pd.DataFrame(rows)


def monte_carlo(returns: pd.Series, *, samples: int, seed: int) -> pd.DataFrame:
    """Bootstrap period returns to estimate terminal-return and drawdown ranges."""

    clean = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    if clean.size == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for sample in range(samples):
        path = rng.choice(clean, size=len(clean), replace=True)
        equity = np.cumprod(1 + path)
        drawdown = equity / np.maximum.accumulate(equity) - 1
        rows.append(
            {
                "sample": sample + 1,
                "terminal_return": float(equity[-1] - 1),
                "max_drawdown": float(drawdown.min()),
            }
        )
    return pd.DataFrame(rows)


def log_trial(config: BacktestConfig, *, experiment: str) -> str:
    """Append a deterministic parameter-trial record and return its hash."""

    payload = {"experiment": experiment, "config": config.to_dict()}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    trial_hash = hashlib.sha256(encoded).hexdigest()
    TRIAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRIAL_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"trial_hash": trial_hash, **payload}, sort_keys=True) + "\n")
    return trial_hash


def count_logged_trials(path: Path = TRIAL_LOG_PATH) -> int:
    """Count unique parameter combinations tried."""

    if not path.exists():
        return 1
    hashes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            hashes.add(json.loads(line)["trial_hash"])
        except (json.JSONDecodeError, KeyError):
            continue
    return max(1, len(hashes))
