# US Market Strength

Open Market Strength and select **US stocks** under Market. Use Popular US
stocks (20 companies) or Choose US companies to search the Nasdaq Trader
directory. Select 2–200 companies, then click **Update latest data**.

Prices use the existing Yahoo adjusted OHLCV cache, preserving raw prices.
Downloads are incremental, failures are isolated, and each update writes
`data/market_strength/us_update_quality.csv`. Company profiles are cached for
30 days; unknown sectors remain Unclassified. The directory may omit some
security types and is not a point-in-time historical constituent list.

The benchmark is S&P 500 (`^GSPC`, price index, not total return). New York
session completion is bounded at 16:15 local time, with weekends skipped;
actual provider dates determine holiday availability. Future/in-progress
session rows are excluded from the US snapshot.

Existing RS, trend and pattern calculations are unchanged. RS percentiles
are relative to the selected eligible companies, not the entire US market.
At least 60% of selected companies (minimum two) must have 253 daily bars.
New listings may therefore be unavailable for long-window ranking.

USD values are displayed in dollars; traded value is in millions/day and the
workspace liquidity threshold is $10 million/day (NSE remains INR crore/day).
This is a separate currency-native filter, not an exchange-rate conversion.
The embedded research simulator retains its configurable percentage-cost
assumption; it does not implement a US-specific brokerage/tax schedule.

US workspace exports use `static/market-strength/data-us.json`, separate from
the NSE export. No production ML models, rules, or training artifacts change.
