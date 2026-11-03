"""Bias-aware signal research and portfolio backtesting for NATIP."""

from app.backtesting.config import BacktestConfig, load_config
from app.backtesting.engine.portfolio import BacktestResult, run_strategy_backtest
from app.backtesting.signals.base import SignalFunction, get_signal, register_signal

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "SignalFunction",
    "get_signal",
    "load_config",
    "register_signal",
    "run_strategy_backtest",
]
