# Kronos Forecast

The Market selector supports **India (NSE)** and **US stocks**. US mode offers
an exchange directory from Nasdaq Trader (`nasdaqlisted.txt` and `otherlisted.txt`)
and a local popular-stock list. Test instruments and ETFs are excluded. Directory
data is cached daily, with a labelled local fallback on network failure. Yahoo
coverage is still required for any selected ticker.

US tickers use Yahoo notation such as `AAPL`, `MSFT`, `NVDA`, and `BRK-B`, without
an `.NS` suffix. Charts and price tables use USD; research evidence uses the New
York session close rather than NSE timestamps. India retains its Nifty filters,
INR pricing and NSE validation. US forecasts never use Nifty as their benchmark.

NATIP integrates the MIT-licensed Kronos candlestick model as a separate research tab. The
vendored source is from the user-supplied `Kronos-master` archive. Its original license is kept
at `app/vendor/kronos/LICENSE`.

The integration uses corporate-action-adjusted daily Yahoo OHLCV and a maximum of 512 historical
sessions. It forecasts the next 5, 10, or 20 business sessions with `Kronos-mini` and its matching
2k tokenizer. Model weights are downloaded from the public `NeoQuasar` Hugging Face repositories
on first use and cached under `data/models/kronos`.

Direction is derived from the final forecast close relative to the latest observed adjusted close.
The neutral band is volatility-aware and bounded between 1% and 3%. `UP`, `DOWN`, and `SIDEWAYS`
are model research labels, not calibrated probabilities or trading recommendations.

Known limitations:

- Kronos is not fine-tuned or independently validated for NATIP's NSE universe.
- A forecast path can be noisy or internally inconsistent.
- Future timestamps use business days and may include an NSE holiday.
- The model does not incorporate fundamentals, news, liquidity, transaction costs, or portfolio risk.
- No order is created and the forecast cannot override NATIP's deterministic risk controls.

## Historical direction audit

Run `venv/bin/python validate_kronos.py` from the project root. The default audit uses
RELIANCE, TCS, ICICIBANK, SUNPHARMA and LT, eight recent non-overlapping 10-session
outcome windows, 512 context candles, sample counts 1/2/3 and seeds 42/123.
It loads cached adjusted prices and fixed local pretrained weights; it does not train.
Each input is truncated at the signal date before calling the predictor. Outcomes
are measured at the tenth subsequent observed session. The live generic business-day
forecast timestamps are retained and calendar discrepancies are recorded explicitly.

Reports are saved under `reports/kronos/paths_1_2_3_context512_<hash>/`:

- `config.json`: fixed test design, dates and data fingerprints.
- `predictions.csv`: each prediction versus its actual outcome, with resumable checkpoints.
- `sample_path_comparison.csv`: three-class/sign accuracy, error and repeatability.
- `stock_comparison.csv`, `seed_comparison.csv`, `confusion_matrix.csv`: diagnostics.
- `date_block_bootstrap.csv`: 2,000 whole-date resamples and paired accuracy intervals.
- `validation_report.txt`: methodology, metrics and limitations.

The completed comparison appears in the Kronos tab's historical direction test expander.
Repeated seeds share the same realized outcomes and are not independent observations.
Eight date clusters are a small preliminary sample. Model pretraining overlap has not
been established, so historical inference isolation alone does not prove independent
out-of-sample performance. These tests do not measure tradable net profit.

To extend an audit to five paths, pass `--sample-counts 1 2 3 5` and optionally
`--reuse-from reports/kronos/<previous-audit-directory>`. Reuse verifies the model,
dates, seeds, horizon, tickers and price-data fingerprints before retaining earlier
predictions. Only new stock/date/seed/path combinations are evaluated. A separate
report directory preserves the earlier audit, and the tab displays the latest
completed comparison automatically.
