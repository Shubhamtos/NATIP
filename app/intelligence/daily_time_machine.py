"""Experimental daily block-bootstrap trade-plan research; no signal or order generation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

VERSION = "daily-block-bootstrap-v1.1"


@dataclass(frozen=True)
class DailyTradePlan:
    """Long-only hypothetical filled-entry plan, in adjusted-price units.

    Attributes:
        entry: Hypothetical filled entry before slippage, not an entry forecast.
        stop: Fixed stop below entry.
        target: Fixed target above entry.
        horizon: Number of future daily sessions.
        paths: Number of bootstrap scenarios, not independent historical samples.
        block_size: Consecutive historical return bars sampled per block.
        cost_bps: All-in charges per side, excluding slippage.
        slippage_bps: Adverse execution slippage per side.
        seed: Reproducible random seed.
    """

    entry: float
    stop: float
    target: float
    horizon: int = 20
    paths: int = 2000
    block_size: int = 5
    cost_bps: float = 10.0
    slippage_bps: float = 5.0
    seed: int = 19

    def validate(self) -> None:
        """Reject invalid levels, sizes and costs before simulation."""
        if not all(
            np.isfinite(value)
            for value in (self.entry, self.stop, self.target, self.cost_bps, self.slippage_bps)
        ):
            raise ValueError("Trade levels and costs must be finite.")
        if not 0 < self.stop < self.entry < self.target:
            raise ValueError("Long plan requires 0 < stop < entry < target.")
        if not 1 <= self.horizon <= 60 or not 100 <= self.paths <= 10000:
            raise ValueError("Use 1–60 sessions and 100–10,000 simulation paths.")
        if not 1 <= self.block_size <= 20:
            raise ValueError("Block size must be 1–20 sessions.")
        if not 0 <= self.cost_bps <= 500 or not 0 <= self.slippage_bps <= 500:
            raise ValueError("Costs and slippage must each be 0–500 bps per side.")


def clean_daily_history(history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Use only valid completed adjusted OHLCV bars at/before a supplied cutoff.

    The caller supplies a completed-session cutoff. No missing prices or volume
    are forward-filled, and returns across discarded rows are not manufactured.
    """
    frame = history.copy()
    if "Date" in frame:
        frame = frame.set_index("Date")
    frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None).normalize()
    cutoff = pd.to_datetime(as_of, utc=True).tz_localize(None).normalize()
    frame = frame.loc[frame.index <= cutoff].sort_index()
    if frame.index.duplicated().any():
        raise ValueError("Duplicate daily dates must be resolved before simulation.")
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame for column in required):
        raise ValueError("Adjusted daily Open/High/Low/Close/Volume are required.")
    inherited_segments = frame.get("_segment", pd.Series(0, index=frame.index))
    inherited_exclusions = [
        row for row in history.attrs.get("excluded_bars", []) if pd.Timestamp(row["Date"]) <= cutoff
    ]
    frame = frame[required].apply(pd.to_numeric, errors="coerce")
    invalid = (
        ~np.isfinite(frame).all(axis=1)
        | frame.le(0).any(axis=1)
        | frame.High.lt(frame[["Open", "Close"]].max(axis=1))
        | frame.Low.gt(frame[["Open", "Close"]].min(axis=1))
    )
    exclusions = list(inherited_exclusions)
    for date, row in frame.loc[invalid].iterrows():
        reasons = []
        if not np.isfinite(row).all():
            reasons.append("Missing/nonfinite OHLCV")
        if row.Volume <= 0:
            reasons.append("Zero/negative volume")
        if row[["Open", "High", "Low", "Close"]].le(0).any():
            reasons.append("Nonpositive price")
        if row.High < max(row.Open, row.Close) or row.Low > min(row.Open, row.Close):
            reasons.append("Invalid OHLC relationships")
        exclusions.append({"Date": date.date().isoformat(), "Reason": "; ".join(reasons)})
    # Segment boundaries prevent a discarded bar becoming a one-day return or
    # a supposedly contiguous bootstrap/replay window across a data gap.
    frame["_segment"] = inherited_segments + invalid.cumsum()
    frame = frame.loc[~invalid].copy()
    frame.attrs["excluded_bars"] = exclusions
    if len(frame) < 253:
        raise ValueError("At least 253 valid daily candles are required.")
    return frame


def _trade_outcome(bars: np.ndarray, plan: DailyTradePlan) -> dict:
    """Apply gap-aware, stop-first daily fills after a hypothetical entry."""
    outcome, exit_price, exit_day = "NEITHER", float(bars[-1, 3]), len(bars)
    for day, (open_, high, low, _close) in enumerate(bars, 1):
        if open_ <= plan.stop:
            outcome, exit_price, exit_day = "STOP", float(open_), day
            break
        if open_ >= plan.target:
            outcome, exit_price, exit_day = "TARGET", float(plan.target), day
            break
        if low <= plan.stop:
            outcome, exit_price, exit_day = "STOP", plan.stop, day
            break
        if high >= plan.target:
            outcome, exit_price, exit_day = "TARGET", plan.target, day
            break
    fee, slip = plan.cost_bps / 10000, plan.slippage_bps / 10000
    paid = plan.entry * (1 + slip) * (1 + fee)
    received = exit_price * (1 - slip) * (1 - fee)
    return {
        "Outcome": outcome,
        "Exit Session": exit_day,
        "Exit Price": exit_price,
        "Gross Return %": (exit_price / plan.entry - 1) * 100,
        "Net Return %": (received / paid - 1) * 100,
    }


def simulate_daily_plan(
    history: pd.DataFrame, plan: DailyTradePlan, *, as_of: pd.Timestamp
) -> dict:
    """Resample contiguous backward-looking daily blocks and stress a fixed trade plan.

    Returns:
        Outcome frequencies, net returns, price-path bands and limited sample paths.
        Frequencies are conditional bootstrap frequencies, not calibrated probabilities.
    """
    plan.validate()
    frame = clean_daily_history(history, as_of)
    ratios = (
        frame[["Open", "High", "Low", "Close"]]
        .div(frame.Close.shift(1), axis=0)
        .iloc[1:]
        .to_numpy()
    )
    valid_ratios = frame._segment.eq(frame._segment.shift(1)).iloc[1:].to_numpy()
    segments = frame._segment.iloc[1:].to_numpy()
    eligible_starts = np.array(
        [
            start
            for start in range(len(ratios) - plan.block_size + 1)
            if valid_ratios[start : start + plan.block_size].all()
            and segments[start] == segments[start + plan.block_size - 1]
        ],
        dtype=int,
    )
    if len(eligible_starts) < 20:
        raise ValueError(
            "Fewer than 20 intact historical blocks remain after cleaning. More valid history is required."
        )
    rng = np.random.default_rng(plan.seed)
    starts = rng.choice(
        eligible_starts,
        size=(plan.paths, int(np.ceil(plan.horizon / plan.block_size))),
    )
    closes, results = [], []
    for blocks in starts:
        sampled = np.concatenate([ratios[start : start + plan.block_size] for start in blocks])[
            : plan.horizon
        ]
        previous = plan.entry
        bars = []
        for day, row in enumerate(sampled):
            # Condition the first session on the user's filled opening entry.
            bar = row / row[0] * previous if day == 0 else row * previous
            bars.append(bar)
            previous = bar[3]
        bars = np.asarray(bars)
        closes.append(np.r_[plan.entry, bars[:, 3]])
        results.append(_trade_outcome(bars, plan))
    paths = np.asarray(closes)
    outcomes = pd.DataFrame(results)
    frequencies = outcomes.Outcome.value_counts(normalize=True).reindex(
        ["TARGET", "STOP", "NEITHER"], fill_value=0
    )
    bands = pd.DataFrame(
        np.quantile(paths, [0.1, 0.5, 0.9], axis=0).T, columns=["P10", "Median", "P90"]
    )
    bands.index.name = "Future session"
    return {
        "version": VERSION,
        "as_of": frame.index[-1].date().isoformat(),
        "history_rows": len(frame),
        "history_start": str(frame.index[0].date()),
        "excluded_bars": frame.attrs["excluded_bars"],
        "eligible_blocks": len(eligible_starts),
        "frequencies": frequencies.to_dict(),
        "outcomes": outcomes,
        "bands": bands,
        "sample_paths": pd.DataFrame(paths[:20].T),
        "net_mean_pct": float(outcomes["Net Return %"].mean()),
        "net_median_pct": float(outcomes["Net Return %"].median()),
        "reward_risk": (plan.target - plan.entry) / (plan.entry - plan.stop),
    }


def validate_daily_plan(
    history: pd.DataFrame, plan: DailyTradePlan, *, as_of: pd.Timestamp
) -> pd.DataFrame:
    """Non-overlapping chronological replay with predictions fit on past bars only.

    Uses the same percentage stop/target distances, entry at the next actual open,
    and 500 bootstrap scenarios per origin. This is unconditional plan validation,
    not evidence that Kronos or a BUY strategy has predictive skill.
    """
    from dataclasses import replace

    plan.validate()
    frame = clean_daily_history(history, as_of)
    rows = []
    for origin in range(252, len(frame) - plan.horizon, plan.horizon):
        if frame._segment.iloc[origin] != frame._segment.iloc[origin + plan.horizon]:
            continue
        entry = float(frame.Open.iloc[origin + 1])
        future_plan = replace(
            plan,
            entry=entry,
            stop=entry * plan.stop / plan.entry,
            target=entry * plan.target / plan.entry,
        )
        # Simulation uses a normalized hypothetical entry, not the unknown future open.
        past = frame.iloc[: origin + 1]
        simulation = simulate_daily_plan(past, replace(plan, paths=500), as_of=past.index[-1])
        actual = _trade_outcome(
            frame.iloc[origin + 1 : origin + 1 + plan.horizon][
                ["Open", "High", "Low", "Close"]
            ].to_numpy(),
            future_plan,
        )
        rows.append(
            {
                "Origin": past.index[-1],
                "Outcome End": frame.index[origin + plan.horizon],
                "Year": int(frame.index[origin + 1].year),
                "Regime": (
                    "Above SMA200"
                    if past.Close.iloc[-1] > past.Close.tail(200).mean()
                    else "Below SMA200"
                ),
                "Train Rows": len(past),
                "Predicted Target Frequency": simulation["frequencies"]["TARGET"],
                "Target Hit": int(actual["Outcome"] == "TARGET"),
                **actual,
            }
        )
    return pd.DataFrame(rows)
