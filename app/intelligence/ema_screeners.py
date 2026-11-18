"""Backward-looking EMA distances and completed-close crossing screens."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

EMA_WINDOWS = (20, 50, 200)


def ema_observations(close: pd.Series, structural_segment: pd.Series) -> pd.DataFrame:
    """Reset EMA warm-up at missing bars/breaks; carry event dates, never prices."""
    groups = structural_segment + close.isna().cumsum()
    result = pd.DataFrame(index=close.index)
    previous = close.groupby(groups).shift(1)
    positions = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    dates = pd.Series(close.index, index=close.index)
    for window in EMA_WINDOWS:
        average = (
            close.groupby(groups)
            .transform(
                lambda values, w=window: values.ewm(span=w, adjust=False, min_periods=w).mean()
            )
            .where(close.notna())
        )
        prior_average = average.groupby(groups).shift(1)
        result[f"EMA{window}"] = average
        result[f"EMA{window} Distance %"] = close.div(average).sub(1).mul(100)
        for direction, crosses in (
            ("Up", previous.le(prior_average) & close.gt(average)),
            ("Down", previous.ge(prior_average) & close.lt(average)),
        ):
            result[f"EMA{window} {direction} Date"] = dates.where(crosses).groupby(groups).ffill()
            last_position = positions.where(crosses).groupby(groups).ffill()
            result[f"EMA{window} {direction} Age"] = positions - last_position
    return result


def near_ema200(
    stocks: pd.DataFrame, *, distance_pct: float = 3.0, side: str = "Both"
) -> pd.DataFrame:
    """Select currently tradable stocks within a symmetric percent distance of EMA200."""
    if not np.isfinite(distance_pct) or not 0 <= distance_pct <= 25:
        raise ValueError("EMA distance must be between 0% and 25%")
    if side not in {"Both", "Above", "Below"}:
        raise ValueError("EMA side must be Both, Above or Below")
    distance = stocks["EMA200 Distance %"]
    mask = stocks.Status.eq("PASS") & distance.abs().le(distance_pct + 1e-10)
    if side == "Above":
        mask &= distance.ge(0)
    elif side == "Below":
        mask &= distance.le(0)
    result = stocks.loc[mask].copy()
    result["Position"] = np.select(
        [result["EMA200 Distance %"].gt(1e-10), result["EMA200 Distance %"].lt(-1e-10)],
        ["Above EMA200", "Below EMA200"],
        default="At EMA200",
    )
    result = result.loc[result["EMA200 Distance %"].abs().sort_values(kind="stable").index]
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def ema_crossings(
    stocks: pd.DataFrame,
    *,
    windows: Sequence[int] = EMA_WINDOWS,
    direction: str = "Up",
    within_sessions: int = 1,
    match: str = "Any",
) -> pd.DataFrame:
    """One stock/direction row; recent crosses must still hold on the selected date."""
    selected = tuple(dict.fromkeys(windows))
    if not selected or any(w not in EMA_WINDOWS for w in selected):
        raise ValueError("Select at least one of EMA20, EMA50 and EMA200")
    if direction not in {"Up", "Down", "Both"} or match not in {"Any", "All"}:
        raise ValueError("Unsupported EMA direction or matching mode")
    if not isinstance(within_sessions, int) or not 1 <= within_sessions <= 20:
        raise ValueError("Crossing window must be 1 to 20 trading sessions")
    rows = []
    directions = ("Up", "Down") if direction == "Both" else (direction,)
    for current_direction in directions:
        flags = {}
        for window in selected:
            age = stocks[f"EMA{window} {current_direction} Age"]
            holds = (
                stocks.Close.gt(stocks[f"EMA{window}"])
                if current_direction == "Up"
                else stocks.Close.lt(stocks[f"EMA{window}"])
            )
            flags[window] = age.ge(0) & age.lt(within_sessions) & holds & stocks.Status.eq("PASS")
        hits = pd.DataFrame(flags, index=stocks.index)
        mask = hits.all(axis=1) if match == "All" else hits.any(axis=1)
        result = stocks.loc[mask].copy()
        result["Direction"] = "Crossed above" if current_direction == "Up" else "Crossed below"
        result["Crossed EMAs"] = (
            pd.Series(index=result.index, dtype="object")
            if result.empty
            else hits.loc[mask].apply(
                lambda row: ", ".join(f"EMA{w}" for w in selected if row[w]), axis=1
            )
        )
        ages = pd.DataFrame(
            {w: stocks[f"EMA{w} {current_direction} Age"].where(hits[w]) for w in selected}
        )
        result["Sessions Since Latest Cross"] = ages.loc[mask].min(axis=1)
        event_dates = pd.DataFrame(
            {w: stocks[f"EMA{w} {current_direction} Date"].where(hits[w]) for w in selected}
        )
        result["Latest Cross Date"] = event_dates.loc[mask].max(axis=1)
        for window in EMA_WINDOWS:
            result[f"EMA{window} Cross Date"] = (
                stocks.loc[mask, f"EMA{window} {current_direction} Date"].where(
                    hits[window].loc[mask]
                )
                if window in selected
                else pd.NaT
            )
        rows.append(result)
    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["Sessions Since Latest Cross", "Ticker"], kind="stable")
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)
