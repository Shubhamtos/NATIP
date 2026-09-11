"""Walk-forward research utilities for astro shadow experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class AstroExperimentConfig:
    """Research comparison configuration."""

    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    min_sample_size: int = 30


def run_astro_walk_forward_research(
    data: pd.DataFrame,
    *,
    experiment_signal_columns: Iterable[str],
    config: AstroExperimentConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Compare unchanged base strategy against base-plus-astro experiments.

    Expected columns:
    ``Date``, ``asset``, ``fold``, ``year``, ``regime``, ``future_return``,
    ``base_signal`` and one or more experiment signal columns.
    """

    cfg = config or AstroExperimentConfig()
    required = {"Date", "asset", "fold", "year", "regime", "future_return", "base_signal"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Astro research data missing columns: {sorted(missing)}")
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    experiments = ["base_signal", *list(experiment_signal_columns)]
    rows = [
        _evaluate_signal(frame, column, cfg) for column in experiments if column in frame.columns
    ]
    summary = pd.DataFrame(rows)
    summary["false_discovery_q"] = _benjamini_hochberg(summary["p_value_proxy"].tolist())
    by_year = _grouped_performance(frame, experiments, "year", cfg)
    by_regime = _grouped_performance(frame, experiments, "regime", cfg)
    by_fold = _grouped_performance(frame, experiments, "fold", cfg)
    outputs = {
        "summary": summary,
        "by_year": by_year,
        "by_regime": by_regime,
        "by_fold": by_fold,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            table.to_csv(output_dir / f"astro_{name}.csv", index=False)
    return outputs


def _evaluate_signal(
    frame: pd.DataFrame,
    signal_column: str,
    cfg: AstroExperimentConfig,
) -> dict[str, float | int | str]:
    selected = frame[frame[signal_column].astype(bool)].copy()
    net_cost = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
    selected["net_return"] = pd.to_numeric(selected["future_return"], errors="coerce") - net_cost
    returns = selected["net_return"].dropna()
    sample_size = int(len(returns))
    mean_return = float(returns.mean()) if sample_size else 0.0
    volatility = float(returns.std()) if sample_size > 1 else 0.0
    sharpe = mean_return / volatility if volatility > 0 else 0.0
    win_rate = float((returns > 0).mean()) if sample_size else 0.0
    return {
        "experiment": signal_column,
        "sample_size": sample_size,
        "meets_min_sample_size": sample_size >= cfg.min_sample_size,
        "mean_net_return": mean_return,
        "median_net_return": float(returns.median()) if sample_size else 0.0,
        "win_rate": win_rate,
        "sharpe_proxy": sharpe,
        "p_value_proxy": _normal_proxy_p_value(mean_return, volatility, sample_size),
    }


def _grouped_performance(
    frame: pd.DataFrame,
    experiments: list[str],
    group_column: str,
    cfg: AstroExperimentConfig,
) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(group_column):
        for experiment in experiments:
            if experiment not in group.columns:
                continue
            row = _evaluate_signal(group, experiment, cfg)
            row[group_column] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    total = len(p_values)
    q_values = [1.0] * total
    running = 1.0
    for rank, (index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p_value * total / rank)
        q_values[index] = min(running, 1.0)
    return q_values


def _normal_proxy_p_value(mean: float, std: float, n: int) -> float:
    if n < 2 or std <= 0:
        return 1.0
    z_score = abs(mean / (std / (n**0.5)))
    return max(0.0, min(1.0, 2 * (1 - _normal_cdf(z_score))))


def _normal_cdf(value: float) -> float:
    import math

    return 0.5 * (1 + math.erf(value / (2**0.5)))
