# Commodity data coverage

Raw Material Impact has eight configured daily series. Six currently return
Yahoo data. The configured RSS3.SG and FCPO=F symbols return quote-not-found
errors; no replacement daily price is invented. Refresh now isolates failures,
reports each feed's status, and retains previously stored data. The tracker
uses actual timestamps to identify stale data instead of labelling every
nonempty history Fresh.

The Materials tab additionally provides the World Bank Pink Sheet monthly
reference catalogue, searchable by commodity, with source units, period,
monthly frequency, calendar-month comparisons, CSV export and a history chart.
The initial verified workbook contains 71 series through September 2026,
including Palm oil, Rubber RSS3 and Rubber TSR20. These are USD monthly
averages, not live exchange quotes or substitutes for the daily feed in
company impact calculations. Monthly values never enter the daily impact
store or get forward-filled into fabricated daily prices.

Use Refresh raw-material prices to refresh daily feeds and the monthly cache.
A failed monthly download leaves the previous cache intact. Raw cached
monthly observations are in data/raw_material/world_bank_monthly.parquet.
The source workbook can revise historical observations; this is display and
research reference data, not a point-in-time ML feature dataset.
