"""Small public contract for plugging NATIP signals into the backtester."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

SignalFunction = Callable[[pd.DataFrame], pd.Series]
_SIGNALS: dict[str, SignalFunction] = {}


def register_signal(name: str, function: SignalFunction | None = None):  # type: ignore[no-untyped-def]
    """Register a signal function by name, directly or as a decorator."""

    def decorator(candidate: SignalFunction) -> SignalFunction:
        if name in _SIGNALS and _SIGNALS[name] is not candidate:
            raise ValueError(f"Signal already registered: {name}")
        _SIGNALS[name] = candidate
        return candidate

    return decorator(function) if function is not None else decorator


def get_signal(name: str) -> SignalFunction:
    """Return a registered signal, importing built-ins on first use."""

    if not _SIGNALS:
        from app.backtesting.signals import technical  # noqa: F401

    try:
        return _SIGNALS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown signal {name!r}. Available: {sorted(_SIGNALS)}") from exc


def available_signals() -> list[str]:
    """List registered signal names."""

    if not _SIGNALS:
        from app.backtesting.signals import technical  # noqa: F401
    return sorted(_SIGNALS)


def assert_signal_contract(frame: pd.DataFrame, score: pd.Series) -> None:
    """Validate that a signal preserves panel alignment and numeric output."""

    if not score.index.equals(frame.index):
        raise AssertionError("Signal output index must exactly match its input frame")
    if not pd.api.types.is_numeric_dtype(score):
        raise TypeError("Signal output must be numeric")
