# NATIP Market Strength

The Market Strength workspace is an end-of-day research surface built on NATIP's
existing adjusted-price caches. It brings market participation, relative-strength
rankings, setup scans, sector leadership, chart comparison, watchlists, alerts,
research notes and backtest evidence into one page.

The visible workspace navigation is: Overview, Sector Rotation, RS Rankings,
Breakouts, Scans, Charts, Watchlist, Backtests and Ask NATIP. These are direct
sections rather than controls hidden inside a generic dashboard.

## Data and ranking

- Supported local universes: Nifty 50, Nifty 100 and Nifty 250.
- The snapshot date is the latest completed session shared by enough of the
  selected universe and Nifty 50 benchmark.
- A stock requires at least 253 adjusted-price observations.
- NATIP RS is a 1-99 same-date percentile rank of an original composite:
  40% three-month return, 20% six-month return, 20% nine-month return and
  20% twelve-month return.
- RS history is recalculated on the benchmark's five- and twenty-session-prior
  dates. Future observations cannot enter the snapshot.
- Breadth uses the percentage of eligible stocks above SMA20, SMA50 and SMA200,
  advancing on the latest session, in Stage 2 and near a 52-week high.

The RS rating is a relative research rank. It is not a probability, target price,
or recommendation.

## Ready scans

The page includes transparent scans for RS leaders, Stage 2 and trend-template
leaders, stocks just entering Stage 2, RS-line new highs, stocks near 52-week
highs, near breakouts, confirmed price-volume breakouts, pocket pivots, volume
surges, relative strength in a weak market and laggards. NATIP's existing VCP and
DarvaX engines remain the single source for those pattern definitions and are
linked from the page rather than duplicated.

Breakouts have a dedicated workspace with setup type, pivot, distance to pivot,
volume ratio, RS rating, stage and the annotated price/volume chart. Touching a
pivot is not treated as a breakout.

## Persistent research

The watchlist and journal are stored locally in
`data/market_strength/research.sqlite3`. Watchlist alerts are deterministic and
cover RS leadership, near-breakout proximity, confirmed breakouts and rapid RS
deterioration. Journal entries are append-only and store the contemporaneous
market regime, sector, stage and RS context with the user's thesis and invalidation.

## Backtest evidence

The workspace reads completed NATIP backtest artifacts. It does not rerun or tune
a strategy from the dashboard. Historical results remain subject to survivorship
bias when current index constituents are used and must be interpreted after costs,
random baselines, sensitivity tests and data-quality limitations.

## Limitations

- Coverage is constrained to locally cached symbols; missing symbols appear in
  the data-quality table and are not guessed.
- Industry drill-down is not shown until NATIP has a reliable industry taxonomy.
  Sector values are enriched only from existing local reference data.
- This workspace is end-of-day, not intraday.
- Watchlist alerts are evaluated when the page runs; they are not push
  notifications.
- The implementation uses NATIP terminology, calculations and visual design. It
  does not copy third-party proprietary scores, code, data or branding.

