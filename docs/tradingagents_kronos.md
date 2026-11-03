# TradingAgents forecast evidence review

NATIP's Kronos tab offers an optional **Check AI reasoning** button after a
forecast. The review does not modify the forecast or any production ML decisions.

The implementation uses actual bull, bear and conservative-risk node functions
from the user-supplied TradingAgents v0.7.1 archive, adapted to local context
imports under `app/vendor/tradingagents_research`. The Apache-2.0 license and a
modification notice are included there. The complete upstream LangGraph engine,
trader and portfolio manager are not installed or invoked.

## Workflow

1. Reuse the adjusted stock candles and truncate them at the Kronos signal date.
2. Calculate technical evidence using NATIP's existing technical summary. Include
   local adjusted Nifty/sector relative returns only when dates align exactly.
3. Execute the upstream bull node, bear node and conservative-risk node, in order.
4. Ask a NATIP research manager to assess the fixed forecast: `SUPPORTS`, `CONFLICTS`
   or `INSUFFICIENT_DATA`. These labels describe technical evidence agreement.

The existing NATIP Gemini transport handles four requests, with a 25-second
timeout per request, no automatic retry and no silent model fallback. Stages
are shown in Streamlit. A provider error stops remaining calls and preserves
completed roles in a `PARTIAL` report. The original forecast remains available.
The panel accepts NATIP's existing server/session Gemini configuration. Keys
are never included in prompts or saved reports; provider errors are redacted.

## Evidence and limitations

Every input has an evidence identifier, observation timestamp, source URL and
type (`OBSERVED`, `DERIVED`, `FORECAST`). Roles must cite supplied identifiers.
Missing/unknown references withhold confirmation. A deterministic guard rejects
inverted MACD-versus-signal comparisons. Reference validation alone does not
prove every natural-language claim correct.

This first integration supplies technical/price/volume evidence and aligned
local benchmark prices. Verified dated news, financials, valuation, exchange
restrictions, institutional flows and portfolio information are explicitly
marked missing. It does not call Yahoo's current financial snapshot historical
evidence, retrieve private account information or invent investor motives.
The review does not provide a new return probability or prove higher accuracy.
There are no broker or production-model calls in this research adapter.

## Persistence

Reports are saved as JSON under `outputs/kronos_research/<input_hash>.json`.
The hash includes the full forecast-path digest, supplied evidence, selected
Gemini model and adapter version. Complete matching reports survive reruns and
are reusable without another request; an explicit button press reruns research.
Reports can be downloaded from the panel. A changed stock, date, forecast,
evidence or model cannot accidentally reuse an unrelated conclusion.

## Verification

Run:

```bash
venv/bin/python -m pytest -q tests/test_kronos_research.py tests/test_gemini_reasoning.py
venv/bin/streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8501/?page=kronos-forecast`, generate a forecast, and click
**Check AI reasoning** below the chart. The complete review is displayed inline
in this exact order: **Bull case**, **Bear case**, **Risk review**, **Research
manager conclusion**, then **Dated evidence, source links and missing inputs**.
The manager status is printed literally as `SUPPORTS`, `CONFLICTS` or
`INSUFFICIENT_DATA`. The evidence table is available even before an AI request.
Forecast OHLC details and limitations remain available in collapsed sections.

Tests cover upstream node order, forecast immutability, historical cutoffs,
provider failure, key redaction, grouped invalid citations, numeric comparison
errors, research-only verdicts, cache identity, missing keys and Streamlit
button execution/rerun persistence. Offline provider responses in tests are
fixtures, never live findings.

A live four-stage Gemini run is required to verify connectivity in the user's
environment. Historical performance of the combined workflow remains untested;
any future accuracy study must compare identical dates and horizons with dated
evidence and separately report coverage of retained forecasts.
