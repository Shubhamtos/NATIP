"""Command-line interface for NATIP signal research and strategy backtests."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.backtesting.config import BACKTEST_RUNS_DIR, BacktestConfig, load_config, save_config
from app.backtesting.data.yahoo import YahooDataStore, load_nifty_universe
from app.backtesting.reports.html import write_signal_report, write_strategy_report
from app.backtesting.signals.base import available_signals, get_signal
from app.backtesting.signals.research import research_signal
from app.backtesting.validation.robustness import (
    log_trial,
    monte_carlo,
    parameter_sensitivity,
    random_selection_baseline,
    walk_forward_backtest,
)

LOGGER = logging.getLogger("natip.backtest")


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser."""

    parser = argparse.ArgumentParser(prog="backtest", description="NATIP backtesting research")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    signal = subparsers.add_parser("signal", help="Research one signal before portfolio use")
    signal.add_argument("--name", required=True, choices=available_signals())
    signal.add_argument("--config", default="configs/backtesting/nifty100_momentum.json")
    signal.add_argument("--no-update", action="store_true")
    strategy = subparsers.add_parser("strategy", help="Run an after-cost portfolio backtest")
    strategy.add_argument("--config", required=True)
    strategy.add_argument("--no-update", action="store_true")
    strategy.add_argument("--robustness", action="store_true")
    strategy.add_argument("--random-samples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a signal or strategy research command."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    universe = load_nifty_universe(config.data.universe)
    symbols = [*universe["symbol"].tolist(), config.data.benchmark]
    store = YahooDataStore(config.data.cache_dir)
    panel, quality = store.load(
        symbols,
        start=config.data.start_date,
        end=config.data.end_date,
        update=not args.no_update and config.data.update_cache,
    )
    run_dir = _run_directory(config.run_name, args.command)
    run_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    save_config(config, run_dir / "config.json")
    _write_data_limitations(run_dir, config, quality)
    signal_function = get_signal(args.name if args.command == "signal" else config.strategy.signal)

    if args.command == "signal":
        result = research_signal(
            panel,
            signal_function,
            horizons=config.validation.forward_horizons,
            quantiles=config.validation.quantiles,
            rebalance_frequency=config.strategy.rebalance_frequency,
            benchmark_symbol=config.data.benchmark,
            min_mean_ic=config.validation.min_mean_ic,
            min_positive_ic_pct=config.validation.min_positive_ic_pct,
        )
        result.ic_summary.to_csv(run_dir / "ic_summary.csv", index=False)
        result.yearly.to_csv(run_dir / "ic_by_year.csv", index=False)
        result.regimes.to_csv(run_dir / "ic_by_regime.csv", index=False)
        result.verdict.to_csv(run_dir / "signal_verdict.csv", index=False)
        report = write_signal_report(result.quantile_returns, result.ic_periods, run_dir)
        print(json.dumps({"run_dir": str(run_dir), "report": str(report)}, indent=2))
        return 0

    trial_hash = log_trial(config, experiment="strategy")
    result, embargo = walk_forward_backtest(panel, universe, signal_function, config)
    embargo.to_csv(run_dir / "walk_forward_embargo_audit.csv", index=False)
    sensitivity = random = simulation = None
    if args.robustness:
        sensitivity = parameter_sensitivity(panel, universe, signal_function, config)
        random = random_selection_baseline(
            panel,
            universe,
            config,
            samples=args.random_samples or config.validation.random_portfolios,
            turnover_schedule=result.periods,
        )
        simulation = monte_carlo(
            result.periods["net_return"],
            samples=config.validation.monte_carlo_samples,
            seed=config.validation.random_seed,
        )
        stress = copy.deepcopy(config)
        for field in (
            "brokerage_bps",
            "stt_buy_bps",
            "stt_sell_bps",
            "exchange_transaction_bps",
            "sebi_bps",
            "stamp_duty_buy_bps",
            "slippage_bps",
        ):
            setattr(stress.costs, field, getattr(stress.costs, field) * 2)
        stress_result, _ = walk_forward_backtest(panel, universe, signal_function, stress)
        (run_dir / "cost_stress_metrics.json").write_text(
            json.dumps(stress_result.metrics, indent=2, default=str), encoding="utf-8"
        )
    report = write_strategy_report(
        result,
        run_dir,
        benchmark_name=config.data.benchmark,
        limitations=_limitations(config, quality),
        sensitivity=sensitivity,
        random_baseline=random,
        monte_carlo_results=simulation,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "report": str(report),
                "trial_hash": trial_hash,
                "metrics": result.metrics,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _run_directory(name: str, command: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return BACKTEST_RUNS_DIR / f"{timestamp}_{command}_{name}"


def _limitations(config: BacktestConfig, quality: pd.DataFrame) -> list[str]:
    failures = int((quality["status"] == "FAIL").sum()) if "status" in quality else 0
    return [
        "Yahoo Finance is not an exchange-grade point-in-time database; outages, revisions, "
        "and symbol changes can occur.",
        "The default universe uses current constituents and therefore has survivorship bias "
        "unless a historical constituent file is supplied.",
        "Adjusted OHLC is derived consistently from Yahoo's adjustment factor; volume remains "
        "raw and is never forward-filled.",
        "Fundamentals without publication dates use a conservative "
        f"{config.data.reporting_lag_days}-day estimate and are explicitly flagged.",
        f"{failures} requested symbols failed data validation in this run.",
        "Configured taxes and fees are modeling assumptions and must be verified before "
        "relying on the net-return estimate.",
        "Circuit-limit detection is approximate because Yahoo does not provide complete "
        "historical exchange order-book state.",
    ]


def _write_data_limitations(run_dir: Path, config: BacktestConfig, quality: pd.DataFrame) -> None:
    (run_dir / "known_limitations.txt").write_text(
        "\n".join(f"- {item}" for item in _limitations(config, quality)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
