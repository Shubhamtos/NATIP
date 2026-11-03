# Kronos F&O screener

Open Kronos Forecast, then F&O screener. Refresh F&O universe to obtain the
official NSE market-lot list. Index derivatives are excluded. The locally cached
list shows its retrieval timestamp; a fallback reference has unverified freshness.

Select all stocks or a custom list, set the maximum stocks, daily context,
forecast horizon and sample paths, then press Scan F&O stocks with Kronos.
Forecasts run sequentially in a background worker. Stop takes effect after the
current download or inference finishes. Results survive ordinary page reruns.

The table separates UP, DOWN and SIDEWAYS forecasts. Ranking uses the absolute
projected move beyond the existing Kronos neutral band, not a success probability.
Inspect a completed forecast chart or download its CSV. The last snapshot is
saved under outputs/kronos_fno/latest.csv with latest_config.json. After a server
restart the table remains available, but in-memory chart objects do not.

Eligibility requires sufficient completed daily history, matching latest session,
no zero volume in the last 20 bars and the selected average traded-value minimum.
Bars after the completed-session cutoff are excluded. The cutoff skips weekends;
actual holiday availability comes from Yahoo. Future plotted sessions retain the
existing weekday-calendar limitation. Turnover uses adjusted close times volume
and is an approximate liquidity screen, not executable contract liquidity.

This is experimental daily research, not an intraday or options contract signal.
It does not check live F&O bans, spreads, expiry, lot sizes or broker permissions,
and does not place orders. Verify these separately. No frozen ML artifacts,
production recommendation rules or Kronos forecast formulas are changed.

Verification: actual cached pretrained Kronos-mini forecasts completed for
RELIANCE.NS and TCS.NS with fresh Yahoo history and a shared completed session.
Unit tests cover future-bar exclusion, mismatched dates, liquidity, zero volume,
isolated errors and cancellation. No full-universe accuracy claim is made.
