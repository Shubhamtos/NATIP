"""Cross-sectional signal research with quantiles and information coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.backtesting.signals.base import SignalFunction, assert_signal_contract


@dataclass(slots=True)
class SignalResearchResult:
    """Tables produced by one signal research run."""

    scored_panel: pd.DataFrame
    quantile_returns: pd.DataFrame
    ic_periods: pd.DataFrame
    ic_summary: pd.DataFrame
    yearly: pd.DataFrame
    regimes: pd.DataFrame
    verdict: pd.DataFrame


def research_signal(
    panel: pd.DataFrame,
    signal: SignalFunction,
    *,
    horizons: list[int],
    quantiles: int = 5,
    rebalance_frequency: str = "monthly",
    benchmark_symbol: str = "^NSEI",
    min_mean_ic: float = 0.02,
    min_positive_ic_pct: float = 0.55,
) -> SignalResearchResult:
    """Rank a signal cross-sectionally and evaluate only future outcomes."""

    data = panel.sort_values(["symbol", "date"]).copy()
    data["score"] = signal(data)
    assert_signal_contract(data, data["score"])
    for horizon in horizons:
        data[f"forward_return_{horizon}d"] = data.groupby("symbol")["adj_open"].transform(
            lambda opening, periods=horizon: opening.shift(-(periods + 1)) / opening.shift(-1)
            - 1
        )
    benchmark = data[data["symbol"] == benchmark_symbol][["date", "adj_close"]].drop_duplicates(
        "date"
    )
    benchmark = benchmark.sort_values("date")
    benchmark["sma_200"] = benchmark["adj_close"].rolling(200).mean()
    benchmark["benchmark_return_20d"] = benchmark["adj_close"].pct_change(20)
    benchmark["regime"] = np.select(
        [
            benchmark["adj_close"] > benchmark["sma_200"] * 1.02,
            benchmark["adj_close"] < benchmark["sma_200"] * 0.98,
        ],
        ["bull", "bear"],
        default="sideways",
    )
    data = data[data["symbol"] != benchmark_symbol].merge(
        benchmark[["date", "regime"]], on="date", how="left"
    )
    dates = _rebalance_dates(data["date"], rebalance_frequency)
    sample = data[data["date"].isin(dates)].copy()
    sample["quantile"] = sample.groupby("date")["score"].transform(
        lambda values: _safe_quantile(values, quantiles)
    )

    quantile_rows: list[dict[str, object]] = []
    ic_rows: list[dict[str, object]] = []
    for horizon in horizons:
        forward = f"forward_return_{horizon}d"
        grouped = sample.dropna(subset=["score", forward, "quantile"])
        summary = grouped.groupby("quantile")[forward].agg(["mean", "median", "count"])
        for quantile, row in summary.iterrows():
            quantile_rows.append(
                {
                    "horizon": horizon,
                    "quantile": int(quantile),
                    "mean_forward_return": float(row["mean"]),
                    "median_forward_return": float(row["median"]),
                    "observations": int(row["count"]),
                }
            )
        top = summary.loc[quantiles, "mean"] if quantiles in summary.index else np.nan
        bottom = summary.loc[1, "mean"] if 1 in summary.index else np.nan
        quantile_rows.append(
            {
                "horizon": horizon,
                "quantile": "top_minus_bottom",
                "mean_forward_return": float(top - bottom),
                "median_forward_return": np.nan,
                "observations": int(len(grouped)),
            }
        )
        for date, group in grouped.groupby("date"):
            ic_rows.append(
                {
                    "date": date,
                    "horizon": horizon,
                    "ic": group["score"].corr(group[forward], method="spearman"),
                    "regime": (
                        group["regime"].mode().iloc[0]
                        if group["regime"].notna().any()
                        else "unknown"
                    ),
                    "stocks": len(group),
                }
            )

    ic_periods = pd.DataFrame(ic_rows).dropna(subset=["ic"])
    ic_summary = _summarize_ic(ic_periods, ["horizon"])
    yearly = _summarize_ic(
        ic_periods.assign(year=pd.to_datetime(ic_periods["date"]).dt.year), ["horizon", "year"]
    )
    regimes = _summarize_ic(ic_periods, ["horizon", "regime"])
    verdict = ic_summary.copy()
    verdict["pass_mean_ic"] = verdict["mean_ic"] >= min_mean_ic
    verdict["pass_positive_ic"] = verdict["positive_ic_pct"] >= min_positive_ic_pct
    verdict["verdict"] = np.where(
        verdict["pass_mean_ic"] & verdict["pass_positive_ic"], "PASS", "FAIL"
    )
    return SignalResearchResult(
        scored_panel=sample,
        quantile_returns=pd.DataFrame(quantile_rows),
        ic_periods=ic_periods,
        ic_summary=ic_summary,
        yearly=yearly,
        regimes=regimes,
        verdict=verdict,
    )


def _rebalance_dates(dates: pd.Series, frequency: str) -> pd.DatetimeIndex:
    unique = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values()
    periods = unique.dt.to_period("W-FRI" if frequency == "weekly" else "M")
    return pd.DatetimeIndex(unique.groupby(periods).max())


def _safe_quantile(values: pd.Series, quantiles: int) -> pd.Series:
    valid = values.dropna()
    result = pd.Series(np.nan, index=values.index)
    if valid.nunique() < quantiles:
        return result
    ranks = valid.rank(method="first")
    result.loc[valid.index] = pd.qcut(ranks, quantiles, labels=False) + 1
    return result


def _summarize_ic(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[*groups, "mean_ic", "ic_std", "ic_ir", "positive_ic_pct", "periods"]
        )
    output = frame.groupby(groups, dropna=False)["ic"].agg(["mean", "std", "count"])
    output["ic_ir"] = output["mean"] / output["std"].replace(0, np.nan)
    positive = frame.assign(positive=frame["ic"] > 0).groupby(groups)["positive"].mean()
    output["positive_ic_pct"] = positive
    return output.reset_index().rename(
        columns={"mean": "mean_ic", "std": "ic_std", "count": "periods"}
    )
