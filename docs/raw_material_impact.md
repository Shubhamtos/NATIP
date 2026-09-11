# Raw Material Impact

The Raw Material Impact module is an evidence-based decision-support layer for NATIP.
It tracks important commodity and currency moves, joins them with versioned
company-material exposure mappings, and estimates possible margin impact where
enough evidence exists.

The module is supporting evidence only. It cannot create BUY, SELL, CALL, PUT,
position-size, broker-order or risk-override decisions.

## Data Flow

Raw-material source adapters -> SQLite store -> RawMaterialImpactService ->
RawMaterialAgent -> API/Streamlit display.

Stored seed files:

- `data/raw_material/raw_material_master.csv`
- `data/raw_material/company_material_mappings.csv`

Local SQLite database:

- `data/raw_material/raw_material_impact.db`

## Tables

The SQLAlchemy store creates these MVP tables:

- `raw_material_master`
- `raw_material_price_history`
- `company_raw_material_mapping`
- `raw_material_impact_signal`

Each price row stores source, timestamp, retrieval timestamp, currency, unit,
status and quality score. Company mappings are versioned by symbol, raw material
and effective date.

## Data Sources

Current MVP source:

- Yahoo Finance fallback for EOD commodity/currency prices.

Tracked examples:

- Brent crude: `BZ=F`
- Copper: `HG=F`
- Aluminium proxy: `ALI=F`
- Natural gas: `NG=F`
- Silver: `SI=F`
- USD/INR: `INR=X`

Some proxies may be unavailable on Yahoo. Unavailable values remain displayed as
`Data unavailable`.

Production adapters should be added for MCX, PPAC, RBI/FBIL, World Bank Pink
Sheet, AGMARKNET, TradeStat and licensed providers when permitted.

## Formulas

INR-adjusted return:

```text
(1 + commodity_return) * (1 + usd_inr_return) - 1
```

Consumer estimated EBITDA-margin impact:

```text
-(material_spend_pct_revenue * inr_adjusted_price_change
  * (1 - hedge_ratio)
  * (1 - pass_through_ratio)
  * 10000)
```

Producer impact reverses the initial direction. Integrated companies estimate
upstream offset and downstream pressure separately in future versions.

## Guardrails

- Missing numeric values remain null.
- Missing values are never replaced with zero.
- Unverified template mappings show `Insufficient Data`.
- Historical probability is not shown unless enough validated shock events exist.
- Alert severity and evidence confidence are separate.
- The module does not alter final NATIP recommendations.

## Running

Start Streamlit:

```bash
venv/bin/streamlit run app/dashboard/streamlit_app.py --server.port 8501
```

Run API:

```bash
venv/bin/uvicorn app.api.main:app --reload
```

Run tests:

```bash
venv/bin/python -m pytest tests/test_raw_material_impact.py tests/test_api_main.py -q
```

## Remaining Production Work

- Add authorised official/licensed source adapters.
- Add full RBAC instead of the current local admin header guard.
- Add Alembic migration management if NATIP standardises migrations.
- Add historical event study and walk-forward validation.
- Add point-in-time evidence extraction from annual reports and filings.
- Add stock, sector and Nifty indexed comparison overlays using clean cache.
