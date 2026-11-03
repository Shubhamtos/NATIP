# NATIP Decision Framework

Fetch Analysis uses a deterministic decision pipeline. Optional AI providers explain the
completed decision once; they cannot change scores, actions, risk vetoes, prices, targets or
position size.

## Flow

1. Normalize Yahoo quote and adjusted OHLCV data.
2. Build a typed `AnalysisDecisionContext` for the quote timestamp.
3. Load the latest sector-rotation record and fundamental record available on or before that
   timestamp.
4. Classify the technical setup as breakout, pullback, value accumulation or no setup.
5. Apply instrument, surveillance, liquidity, reward-risk and technical-state gates.
6. Run deterministic Macro, Sector, Technical, Fundamentals, Valuation, Sentiment and Risk
   agents.
7. Build consensus. Hard Risk Agent vetoes override positive evidence.
8. Optionally request one Gemini or OpenAI explanation after consensus.
9. Persist the decision, context, sources and audit trace in local SQLite.

The dashboard uses `BUY`, `WATCH`, `REJECT`, `EXIT` and `MANUAL_REVIEW` from the decision
trace. Legacy `BUY/HOLD/SELL` fields remain only for compatibility with older code.

## Configuration

Settings are read from `.env` and `.env.local`; secrets must not be committed.

```text
NATIP_GEMINI_API_KEY=
NATIP_GEMINI_MODEL=gemini-3.5-flash-lite
OPENAI_API_KEY=
NATIP_OPENAI_MODEL=gpt-4.1-mini
NATIP_ANALYSIS_TOTAL_CAPITAL=
NATIP_ANALYSIS_RISK_PER_TRADE_PCT=1.0
NATIP_ANALYSIS_MAX_POSITION_PCT=5.0
NATIP_ANALYSIS_MAX_SECTOR_EXPOSURE_PCT=25.0
NATIP_ANALYSIS_MAX_NEW_BUYS_PER_DAY=3
NATIP_DECISION_LOG_PATH=data/decisions/decisions.sqlite3
```

Position sizing is research-only. NATIP does not place orders through this workflow.

## Data Boundaries

- Sector rotation: `data/sector_rotation/sector_rotation_history.csv`, latest row not after the
  analysis timestamp.
- Fundamentals: `data/fundamentals/point_in_time_fundamentals.parquet`, latest safely available
  row not after the analysis timestamp, supplemented by current Screener and Yahoo evidence.
- Eligibility: `config/reference/nse_securities.csv` and `config/reference/nse_fo_stocks.csv`.
- Macro proxies: Yahoo USD/INR, Brent, India VIX, Nifty 50 and US 10Y.
- Decision history: local SQLite at the configured decision-log path.

The live bid-ask spread, current F&O ban status and upcoming result date are not yet connected.
The dashboard identifies these as missing rather than inventing values. F&O eligibility is not
the same as current F&O-ban status.

## AI Failure Behavior

Gemini uses structured JSON output and legacy model names are migrated to
`gemini-3.5-flash-lite`. The request is bounded and fails closed to deterministic reasoning.
Provider errors never prevent charts or deterministic consensus from rendering.

