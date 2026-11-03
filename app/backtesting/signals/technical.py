"""Backward-looking technical signals included with the backtesting module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtesting.signals.base import register_signal


@register_signal("momentum_6_1")
def momentum_6_1(frame: pd.DataFrame) -> pd.Series:
    """Six-month momentum excluding the most recent month."""

    grouped = frame.groupby("symbol")["adj_close"]
    return grouped.transform(lambda close: close.shift(21) / close.shift(126) - 1).rename("score")


@register_signal("rsi_14")
def rsi_14(frame: pd.DataFrame) -> pd.Series:
    """Fourteen-session RSI; higher values receive higher ranks."""

    def calculate(close: pd.Series) -> pd.Series:
        change = close.diff()
        gain = change.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = -change.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
        return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    return frame.groupby("symbol")["adj_close"].transform(calculate).rename("score")


@register_signal("atr_volatility")
def atr_volatility(frame: pd.DataFrame) -> pd.Series:
    """Negative normalized ATR so lower-volatility stocks rank higher."""

    previous_close = frame.groupby("symbol")["adj_close"].shift(1)
    true_range = pd.concat(
        [
            frame["adj_high"] - frame["adj_low"],
            (frame["adj_high"] - previous_close).abs(),
            (frame["adj_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.groupby(frame["symbol"]).transform(lambda values: values.rolling(14).mean())
    return (-(atr / frame["adj_close"])).rename("score")


@register_signal("vcp_score")
def vcp_score(frame: pd.DataFrame) -> pd.Series:
    """Use NATIP's deterministic VCP score when supplied by the caller."""

    if "vcp_score" not in frame.columns:
        raise ValueError(
            "vcp_score requires a point-in-time vcp_score column from NATIP's pattern scanner"
        )
    return pd.to_numeric(frame["vcp_score"], errors="coerce").rename("score")


@register_signal("vcp_breakout")
def vcp_breakout(frame: pd.DataFrame) -> pd.Series:
    """Rank only point-in-time VCP breakouts confirmed by volume."""

    required = {"vcp_score", "breakout_status", "volume_to_avg20"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"vcp_breakout requires columns: {sorted(missing)}")
    confirmed = (
        frame["breakout_status"].astype(str).str.lower().isin({"confirmed", "breakout"})
        & pd.to_numeric(frame["volume_to_avg20"], errors="coerce").ge(1.5)
    )
    return pd.to_numeric(frame["vcp_score"], errors="coerce").where(confirmed).rename("score")
