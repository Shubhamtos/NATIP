# US Buying Agent

The Stock buying filter also provides **S&P 500**, **DJIA**, and **Nasdaq 100**. These select
the complete current constituent lists from the respective Wikipedia company
tables, validated before use (DJIA must contain 30 unique tickers). Lists are
cached daily under `data/market_strength/universes/`. A failed refresh uses a
validated saved list with a staleness notice; absent a valid list, the filter
reports an error rather than substituting popular stocks. Both filters retain
the existing S&P 500 research benchmark. Current membership must not be
treated as point-in-time historical membership for backtesting.

Open Buying Agent and select **Market > US stocks**. Choose Popular US stocks
or select **All US stocks** to access the full exchange directory. The All US
stocks view has ticker/company search, Stocks to scan (default 100), and Start
at company controls for explicit batches. Increase the count to include the
full list; large initial downloads can take considerable time. Choose US
companies remains available for a hand-picked selection of 2–200 companies.
Use **Update US data** when selecting stocks without local history.

The three existing task categories remain available in US mode:

- VCP Pattern Check: explicitly run the scanner, adjust the minimum score,
  inspect reasons, and download the output CSV.
- DarvaX Pattern Search: explicitly scan the latest daily candles, inspect
  matched patterns and quality notes, and download the output CSV.
- Fundamental + Technical Analysis: run cached technical analysis or request
  a current Yahoo financial snapshot for a selected company.

US price data shares the adjusted Yahoo cache with Market Strength and uses
S&P 500 (`^GSPC`) rather than Nifty. Only completed US daily sessions enter
the snapshot. A minimum of two sufficiently covered companies is required;
new listings without the existing long-window history may be ineligible.
Missing dates and prices are not filled to manufacture detector inputs.

Pattern formulas are reused from NATIP's existing scanner. Sector-relative
confirmation is unavailable in this adapter; Indian FII/DII, promoter and
NSE eligibility checks are not applied. Missing financial/institutional score
components remain unconfirmed, so US scores are not directly equivalent to
fully enriched Indian scores. Prices are USD; financial snapshots retain
their reported financialCurrency. Missing fields are shown as Unavailable.
Current Yahoo snapshots are not point-in-time historical fundamentals.

These pattern heuristics are research screens, not validated US BUY/SELL
signals. The frozen NSE ML models and production rules are not invoked or
modified. Each US scan has separate session-state keys from NSE scans;
results survive reruns, and changing the selection displays a warning until
the user explicitly runs the scanner again.

Tests: `venv/bin/python -m pytest tests/test_us_buying_agent.py tests/test_market_strength_us.py -q`
