# Kronos Daily Time Machine

Run a daily Kronos forecast, then use **Time Machine — Daily Trade-Plan Stress
Test** below the forecast. Enter a hypothetical long entry, stop and target;
the simulation horizon matches the completed forecast (5, 10 or 20 sessions).
Press **Run Daily Time Machine**. It never runs automatically or modifies
Kronos, frozen recommendations, position sizing or broker orders.

This is a separate daily implementation inspired by historical-resampling
research, not the supplied intraday Time Machine executed on daily bars.
It uses the same adjusted history already supplied to Kronos, at/before its
signal date. At least 253 valid daily candles are required. Duplicate dates
fail explicitly. Invalid OHLC, missing values and zero-volume
bars are excluded and reported with dates/reasons. Contiguous segments prevent
bootstrap blocks and replay holding windows from crossing excluded rows; no
return is calculated across those gaps. At least 253 valid bars and 20 intact
sampling blocks must remain. No prices are forward-filled. The caller's signal
date must be a completed session.

## Method

- Resample contiguous blocks of 3, 5 or 10 historical daily OHLC ratios.
- Normalize the first sampled open to the hypothetical filled entry. Later
  opens preserve sampled overnight gaps relative to the previous close.
- Follow a fixed stop and target for the selected number of sessions.
- If the market opens below the stop, exit at the worse open. A favourable
  gap above the target receives only the target price, conservatively.
- When both levels are touched intraday, count the stop first because daily
  OHLC cannot reveal the true sequence.
- Exit at the final close if neither level is reached. Apply selected charges
  and adverse slippage on both entry and exit. Defaults are editable examples,
  not verified market/broker-specific fee schedules.
- Save the seed, levels, costs, horizon and calculation version in the
  downloadable run configuration. Optional outcome replay is downloadable CSV.

The chart shows closing-price scenario bands, not confidence intervals.
Paths continue after a trade exits, whereas its P&L stops at exit. Bootstrap
frequencies are conditional on a filled entry; they do not estimate whether
that entry is obtainable or whether Kronos is directionally correct. More
paths reduce Monte Carlo sampling noise, not market uncertainty.

## Historical checks

The optional replay uses expanding historical windows and non-overlapping
future holding windows. Each simulation sees only data through that origin;
the actual comparison trade enters at the next observed open, with the same
percentage stop/target distances. Outputs include event Brier error, simulated
versus actual target-first frequency, net returns, year and stock-trend regime
breakdowns. The regime is stock price versus its trailing SMA200, not a market
regime forecast. Fewer than 30 trades triggers a small-sample warning.

This is unconditional trade-plan validation, not a walk-forward validation of
Kronos or any BUY strategy. Calibration has not been established. No default
plan or parameter is selected using those outcomes. No production ML artifacts
are retrained or altered.

## Limitations

Resampling assumes historical behaviour remains informative. Dependence
across block boundaries, unprecedented events, earnings gaps, exchange halts,
circuits, liquidity/market impact and precise intraday fills are not fully
modelled. Adjusted history can be revised for corporate actions. There is no
PIT guarantee for a currently downloaded historical snapshot. Risk limits can
be exceeded by gaps. All outputs are experimental research, not guaranteed
returns or instructions to trade.

Tests: `venv/bin/python -m pytest tests/test_daily_time_machine.py -q`
