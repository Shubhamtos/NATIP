# Market Breadth

Open **Market Breadth** in NATIP's main navigation, or `/?page=market-breadth`.
This page is descriptive only: no models, portfolio rules or orders are changed.

## EMA Screeners

Two additional subtabs share the page's universe, as-of date and update controls:

- **Near EMA200**: stocks with absolute `100 * (adjusted_close / EMA200 - 1)` at
  or below the chosen distance (default 3%). Select above, below or both sides.
  Matches are ordered by absolute distance, closest first.
- **EMA Crossings**: select EMA20/50/200, upward/downward/both directions, last
  1/3/5/10/20 benchmark sessions, and any/all selected periods. Defaults are any
  selected EMA, upward crosses and the latest session. A price touch is not a cross.
  An upward cross requires previous close <= previous EMA and current close > current
  EMA; downward is the reverse. Recent crosses must still hold on the as-of date.
  All within one session means all selected periods crossed on that same session.
  For larger windows, each selected EMA may have a different crossing date, shown
  separately. Both directions may produce two explicitly labelled rows for one stock
  when it crossed different EMAs in opposite directions.

EMA uses the existing recursive pandas convention (`span`, `adjust=False`) with a
full-period minimum warm-up. EMA calculation and remembered crossing events restart
after missing/invalid/non-trading bars and known structural breaks. Only event dates
are carried forward, never missing prices. New listings without 200 uninterrupted
valid sessions are ineligible for EMA200, not labelled as failures. These screens
are not EMA/EMA crossovers, model features, probabilities or trade recommendations.
Each has sector/search filters and a downloadable matches table.

### Hosted visibility

The page displays UI build `EMA-1.2`. Both EMA subtabs remain visible even if
the hosted runtime has no local price cache; **Load prices and run screeners**
initializes the selected universe. Local caches are not committed to GitHub and
are not automatically shared with Streamlit Cloud. If an older overview without
the EMA tabs is still displayed after a GitHub push, reboot the hosted app once
from **Manage app**. File watching remains disabled to preserve the documented
bundled-runtime safeguards. Browser refresh is not a server reboot.

## Data and operation

- Universe filters: Nifty 50, 100, LargeMidcap 250 (labelled Nifty 250 in NATIP),
  500 and Smallcap 250. These are distinct saved/configured constituent snapshots.
- Start date for updates comes from `app.probability.config.START_DATE`.
- **Update breadth data** performs bounded incremental downloads using the existing
  YahooDataStore and keeps a per-ticker outcome report. Failures do not stop other stocks.
- Adjusted OHLC must be consistent. Raw OHLC is never paired with adjusted close.
- Reading compares clean probability, backtesting and Kronos adjusted caches.
  It selects the freshest available complete source per ticker; it does not splice
  differently adjusted price histories. The source of each series is in Data coverage.
- Only completed NSE sessions are used, bounded by the existing 16:00 IST cutoff.
  Actual Nifty 50 bars define the trading calendar, not an invented holiday calendar.
- Nothing is downloaded on routine filter reruns except a missing universe catalogue.
  Calculations are cached for five minutes. Update/Reload invalidates this cache.
- No credentials or new dependencies are required. Yahoo/NSE access may still be
  throttled or unavailable. Partial/stale coverage is disclosed, not manufactured.

## Definitions

All participation counts are equal-stock weighted. Each metric has its own denominator.
Newer listings can contribute to daily/short-window breadth without a year of history.
Missing, invalid, source-excluded and zero-volume observations are NaN, not negative votes.
Windows require consecutive valid Nifty sessions; missing prices and volume are not filled.

- Advances/declines: adjusted close versus the immediately previous benchmark session.
- Above SMA20/50/200: close strictly above its trailing arithmetic moving average,
  including the current completed close. Each denominator requires that complete window.
- Positive 20D return: close/close 20 sessions ago minus one, requiring 21 complete closes.
- New/equal 52W highs/lows: today's adjusted high/low at or beyond the **previous**
  252-session maximum/minimum (`shift(1)`), requiring a complete prior window.
- A-D ratio: advances divided by declines; undefined if declines are zero.
- Up-volume share: advancing share volume / (advancing + declining share volume).
  This is neither rupee turnover nor a money-flow estimate.
- A-D line: cumulative net advances, reset after a session with daily-return coverage
  below 60%. Low-coverage gaps are not treated as known zero net advances.
- Breadth50 change: percentage-point change over the stated number of actual sessions.
- VEDL: rolling windows restart at the known 2026-04-30 structural break. A total-return
  series including demerged-share value has not been reconstructed for this view.

References: [StockCharts A-D line](https://chartschool.stockcharts.com/table-of-contents/market-indicators/advance-decline-line)
and [percent above moving average](https://chartschool.stockcharts.com/table-of-contents/market-indicators/percent-above-moving-average).

## Limitations

Historical charts use the selected **current** constituent snapshot, not historical
index membership: survivorship/composition bias remains. Adjusted vendor history can
be revised. This view is not point-in-time model validation or a probability forecast.
Sector labels are the saved catalogue's industry/sector classification and may differ
from official sector-index membership. Coverage changes can move breadth independently
of individual price moves; inspect denominators and the quality report.

## Verification

`venv/bin/python -m pytest tests/test_market_breadth.py -q -p no:cacheprovider`

`venv/bin/python -m pytest tests/test_ema_screeners.py -q -p no:cacheprovider`

Tests use synthetic fixtures, not production output. They cover missing sessions,
new listings, no forward-fill, bad OHLC, zero volume, adjusted/raw selection, structural
breaks, future-data invariance, freshest-cache selection, partial download failures,
filters and chart rendering. Run the page against real cached prices to inspect the
actual as-of date and coverage before interpreting it.
