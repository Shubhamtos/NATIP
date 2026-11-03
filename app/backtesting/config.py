"""Typed configuration for reproducible NATIP backtests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DATA_DIR = PROJECT_ROOT / "data" / "backtesting"
BACKTEST_CACHE_DIR = BACKTEST_DATA_DIR / "cache"
BACKTEST_RUNS_DIR = PROJECT_ROOT / "reports" / "backtesting"
TRIAL_LOG_PATH = BACKTEST_RUNS_DIR / "trial_log.jsonl"


@dataclass(slots=True)
class DataConfig:
    """Market-data and universe settings."""

    universe: str = "nifty100"
    start_date: str = "2015-01-01"
    end_date: str | None = None
    benchmark: str = "^NSEI"
    secondary_benchmark: str = "^CRSLDX"
    cache_dir: str = str(BACKTEST_CACHE_DIR)
    update_cache: bool = True
    min_history_days: int = 252
    min_avg_traded_value_20d: float = 10_000_000.0
    reporting_lag_days: int = 45


@dataclass(slots=True)
class CostConfig:
    """Configurable Indian cash-equity trading costs in basis points."""

    mode: Literal["delivery", "intraday"] = "delivery"
    brokerage_bps: float = 0.0
    brokerage_cap_rupees: float = 20.0
    stt_buy_bps: float = 10.0
    stt_sell_bps: float = 10.0
    exchange_transaction_bps: float = 0.297
    sebi_bps: float = 0.01
    gst_rate: float = 0.18
    stamp_duty_buy_bps: float = 1.5
    slippage_bps: float = 5.0
    volume_impact_coefficient: float = 0.0
    effective_date: str = "2026-01-01"
    verify_before_live_use: bool = True


@dataclass(slots=True)
class StrategyConfig:
    """Portfolio construction and execution settings."""

    signal: str = "momentum_6_1"
    initial_capital_rupees: float = 1_000_000.0
    rebalance_frequency: Literal["weekly", "monthly"] = "monthly"
    top_n: int = 10
    bottom_n: int = 10
    long_short: bool = False
    weighting: Literal["equal", "inverse_volatility"] = "equal"
    max_position_weight: float = 0.15
    max_sector_weight: float = 0.35
    stop_loss_pct: float | None = 0.12
    trailing_stop_pct: float | None = None
    fill_model: Literal["next_open"] = "next_open"
    liquidity_lookback: int = 20
    volatility_lookback: int = 20
    circuit_limit_pct: float = 0.20


@dataclass(slots=True)
class ValidationConfig:
    """Research, robustness, and walk-forward settings."""

    forward_horizons: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    prediction_horizon_days: int = 20
    quantiles: int = 5
    train_years: int = 5
    test_months: int = 12
    embargo_days: int = 20
    final_oos_start: str | None = None
    random_portfolios: int = 1000
    monte_carlo_samples: int = 2000
    sensitivity_pct: float = 0.20
    min_mean_ic: float = 0.02
    min_positive_ic_pct: float = 0.55
    random_seed: int = 42


@dataclass(slots=True)
class BacktestConfig:
    """Complete reproducible run configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    run_name: str = "nifty100_momentum"

    def validate(self) -> None:
        """Reject unsafe or internally inconsistent settings."""

        if self.strategy.top_n <= 0:
            raise ValueError("strategy.top_n must be positive")
        if self.strategy.initial_capital_rupees <= 0:
            raise ValueError("strategy.initial_capital_rupees must be positive")
        if not 0 < self.strategy.max_position_weight <= 1:
            raise ValueError("strategy.max_position_weight must be in (0, 1]")
        if not 0 < self.strategy.max_sector_weight <= 1:
            raise ValueError("strategy.max_sector_weight must be in (0, 1]")
        if self.validation.embargo_days < self.validation.prediction_horizon_days:
            raise ValueError("embargo_days must cover the prediction target horizon")
        if self.costs.gst_rate < 0:
            raise ValueError("costs.gst_rate cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot."""

        return asdict(self)


def load_config(path: str | Path) -> BacktestConfig:
    """Load and validate a JSON configuration file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = BacktestConfig(
        data=DataConfig(**payload.get("data", {})),
        costs=CostConfig(**payload.get("costs", {})),
        strategy=StrategyConfig(**payload.get("strategy", {})),
        validation=ValidationConfig(**payload.get("validation", {})),
        run_name=payload.get("run_name", "natip_backtest"),
    )
    config.validate()
    return config


def save_config(config: BacktestConfig, path: str | Path) -> None:
    """Persist the exact configuration used by a run."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
