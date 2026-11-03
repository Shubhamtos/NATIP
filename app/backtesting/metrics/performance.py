"""Auditable portfolio and trade performance calculations."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    periods: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    trials: int = 1,
) -> dict[str, Any]:
    """Calculate portfolio, benchmark, risk, and multiple-testing metrics."""

    if periods.empty:
        return {}
    returns = pd.to_numeric(periods["net_return"], errors="coerce").dropna()
    gross = pd.to_numeric(periods["gross_return"], errors="coerce").dropna()
    benchmark = pd.to_numeric(periods["benchmark_return"], errors="coerce").reindex(returns.index)
    if (returns <= -1).any():
        raise ValueError(
            "A portfolio period return <= -100% indicates leverage or compounding error"
        )
    dates = pd.to_datetime(periods.loc[returns.index, "exit_date"])
    years = max((dates.max() - dates.min()).days / 365.25, 1 / 252)
    periods_per_year = _periods_per_year(dates)
    equity = (1 + returns).cumprod()
    gross_equity = (1 + gross).cumprod()
    benchmark_equity = (1 + benchmark.fillna(0)).cumprod()
    drawdown = equity / equity.cummax() - 1
    sharpe = _ratio(returns, periods_per_year)
    downside = returns.where(returns < 0).dropna()
    sortino = (
        returns.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year)
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    max_drawdown = float(drawdown.min())
    wins = (
        trades[trades["net_return"] > 0]["net_return"]
        if not trades.empty
        else pd.Series(dtype=float)
    )
    losses = (
        trades[trades["net_return"] < 0]["net_return"]
        if not trades.empty
        else pd.Series(dtype=float)
    )
    aligned = pd.concat([returns, benchmark], axis=1).dropna()
    beta = 0.0
    alpha = 0.0
    if len(aligned) > 2 and aligned.iloc[:, 1].var() > 0:
        beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
        alpha = float(
            (aligned.iloc[:, 0].mean() - beta * aligned.iloc[:, 1].mean()) * periods_per_year
        )
    return {
        "periods": int(len(returns)),
        "trades": int(len(trades)),
        "total_return": float(equity.iloc[-1] - 1),
        "gross_total_return": float(gross_equity.iloc[-1] - 1),
        "benchmark_total_return": float(benchmark_equity.iloc[-1] - 1),
        "cagr": cagr,
        "annualized_volatility": float(returns.std(ddof=1) * math.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "sortino": float(sortino),
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_periods": _max_drawdown_duration(drawdown),
        "win_rate": float((trades["net_return"] > 0).mean()) if not trades.empty else 0.0,
        "average_win": float(wins.mean()) if not wins.empty else 0.0,
        "average_loss": float(losses.mean()) if not losses.empty else 0.0,
        "profit_factor": (
            float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf")
        ),
        "expectancy": float(trades["net_return"].mean()) if not trades.empty else 0.0,
        "average_turnover": float(periods["turnover"].mean()),
        "average_holding_days": float(trades["holding_days"].mean()) if not trades.empty else 0.0,
        "average_exposure": float(periods["gross_exposure"].mean()),
        "alpha_annualized": alpha,
        "beta": beta,
        "net_cost_drag": float(gross_equity.iloc[-1] - equity.iloc[-1]),
        "rolling_12m_sharpe_last": _rolling_sharpe(returns, periods_per_year),
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            returns, sharpe=sharpe, trials=max(1, trials), periods_per_year=periods_per_year
        ),
        "parameter_trials_logged": int(max(1, trials)),
        "multiple_testing_warning": (
            "Multiple parameter trials inflate apparent Sharpe; inspect the deflated Sharpe "
            "probability."
            if trials > 1
            else "Single logged trial; unlogged manual trials remain a research risk."
        ),
    }


def equity_and_drawdown(periods: pd.DataFrame) -> pd.DataFrame:
    """Build aligned portfolio/benchmark equity and drawdown series."""

    result = periods[["exit_date", "net_return", "gross_return", "benchmark_return"]].copy()
    result["portfolio_equity"] = (1 + result["net_return"]).cumprod()
    result["gross_equity"] = (1 + result["gross_return"]).cumprod()
    result["benchmark_equity"] = (1 + result["benchmark_return"]).cumprod()
    result["drawdown"] = result["portfolio_equity"] / result["portfolio_equity"].cummax() - 1
    return result


def monthly_returns(periods: pd.DataFrame) -> pd.DataFrame:
    """Aggregate non-overlapping period returns into calendar months."""

    values = periods.copy()
    values["exit_date"] = pd.to_datetime(values["exit_date"])
    monthly = (
        values.set_index("exit_date")["net_return"]
        .resample("ME")
        .apply(lambda returns: (1 + returns).prod() - 1)
    )
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    return table.pivot(index="year", columns="month", values="return")


def stock_performance_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate executed trades into an auditable stock-level attribution table.

    Total net contribution is the primary ranking field because it incorporates
    the portfolio weight assigned to each trade. Trade-level returns and win
    rates remain visible to distinguish persistent results from isolated moves.
    """

    columns = [
        "rank",
        "symbol",
        "trades",
        "total_net_contribution",
        "total_gross_contribution",
        "average_net_return",
        "median_net_return",
        "win_rate",
        "profit_factor",
        "best_trade_return",
        "worst_trade_return",
        "extreme_trade_count",
        "data_review",
        "stop_loss_count",
        "stop_loss_rate",
        "average_holding_days",
        "first_entry_date",
        "last_exit_date",
        "active_folds",
        "sample_quality",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    required = {
        "symbol",
        "entry_date",
        "exit_date",
        "net_return",
        "gross_contribution",
        "net_contribution",
        "exit_reason",
        "holding_days",
    }
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"Trade log is missing stock-attribution columns: {missing}")

    executed = trades.copy()
    if "skipped" in executed.columns:
        skipped = executed["skipped"].fillna(False).astype(bool)
        executed = executed.loc[~skipped].copy()
    executed["net_return"] = pd.to_numeric(executed["net_return"], errors="coerce")
    executed["gross_contribution"] = pd.to_numeric(
        executed["gross_contribution"], errors="coerce"
    )
    executed["net_contribution"] = pd.to_numeric(
        executed["net_contribution"], errors="coerce"
    )
    executed["holding_days"] = pd.to_numeric(executed["holding_days"], errors="coerce")
    executed = executed.dropna(subset=["symbol", "net_return", "net_contribution"])
    if executed.empty:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for symbol, group in executed.groupby("symbol", sort=False):
        winners = group.loc[group["net_return"] > 0, "net_return"]
        losers = group.loc[group["net_return"] < 0, "net_return"]
        loss_total = float(losers.sum())
        profit_factor = (
            float(winners.sum() / abs(loss_total)) if loss_total < 0 else float("inf")
        )
        active_folds = int(group["fold"].nunique()) if "fold" in group.columns else 0
        trade_count = int(len(group))
        records.append(
            {
                "symbol": str(symbol),
                "trades": trade_count,
                "total_net_contribution": float(group["net_contribution"].sum()),
                "total_gross_contribution": float(group["gross_contribution"].sum()),
                "average_net_return": float(group["net_return"].mean()),
                "median_net_return": float(group["net_return"].median()),
                "win_rate": float((group["net_return"] > 0).mean()),
                "profit_factor": profit_factor,
                "best_trade_return": float(group["net_return"].max()),
                "worst_trade_return": float(group["net_return"].min()),
                "extreme_trade_count": int(group["net_return"].abs().gt(0.50).sum()),
                "data_review": (
                    "REVIEW_EXTREME_MOVE"
                    if group["net_return"].abs().gt(0.50).any()
                    else "PASS"
                ),
                "stop_loss_count": int((group["exit_reason"] == "stop_loss").sum()),
                "stop_loss_rate": float((group["exit_reason"] == "stop_loss").mean()),
                "average_holding_days": float(group["holding_days"].mean()),
                "first_entry_date": pd.to_datetime(group["entry_date"]).min().date(),
                "last_exit_date": pd.to_datetime(group["exit_date"]).max().date(),
                "active_folds": active_folds,
                "sample_quality": (
                    "SUPPORTED" if trade_count >= 5 and (active_folds >= 2 or active_folds == 0)
                    else "LOW_SAMPLE"
                ),
            }
        )

    summary = pd.DataFrame.from_records(records).sort_values(
        ["total_net_contribution", "trades", "symbol"],
        ascending=[False, False, True],
    )
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary.loc[:, columns].reset_index(drop=True)


def _periods_per_year(dates: pd.Series) -> float:
    if len(dates) < 2:
        return 12.0
    median_days = max(float(dates.sort_values().diff().dt.days.dropna().median()), 1.0)
    return 365.25 / median_days


def _ratio(returns: pd.Series, periods_per_year: float) -> float:
    std = returns.std(ddof=1)
    return float(returns.mean() / std * math.sqrt(periods_per_year)) if std > 0 else 0.0


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return int(longest)


def _rolling_sharpe(returns: pd.Series, periods_per_year: float) -> float | None:
    window = max(2, round(periods_per_year))
    if len(returns) < window:
        return None
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    value = (mean / std.replace(0, np.nan) * math.sqrt(periods_per_year)).iloc[-1]
    return None if pd.isna(value) else float(value)


def _deflated_sharpe_probability(
    returns: pd.Series,
    *,
    sharpe: float,
    trials: int,
    periods_per_year: float,
) -> float | None:
    if len(returns) < 3 or returns.std(ddof=1) == 0:
        return None
    normal = NormalDist()
    gamma = 0.5772156649
    trials = max(1, trials)
    if trials == 1:
        expected_max = 0.0
    else:
        expected_max = (1 - gamma) * normal.inv_cdf(1 - 1 / trials) + gamma * normal.inv_cdf(
            1 - 1 / (trials * math.e)
        )
    skew = float(returns.skew())
    kurtosis = float(returns.kurtosis() + 3)
    period_sharpe = sharpe / math.sqrt(periods_per_year)
    variance = (1 - skew * period_sharpe + ((kurtosis - 1) / 4) * period_sharpe**2) / (
        len(returns) - 1
    )
    if variance <= 0:
        return None
    statistic = (period_sharpe - expected_max / math.sqrt(max(len(returns), 1))) / math.sqrt(
        variance
    )
    return float(normal.cdf(statistic))
