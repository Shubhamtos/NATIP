# NATIP Backtesting

The backtesting package is a research module. It does not modify NATIP's frozen
production models, recommendations, or broker interfaces.

## Structure

```text
app/backtesting/
├── config.py                 Typed JSON configuration
├── data/yahoo.py             Adjusted/raw OHLCV cache and PIT fundamentals merge
├── signals/base.py           Plug-in signal contract
├── signals/technical.py      Momentum, RSI, ATR volatility, VCP adapter
├── signals/research.py       Quantiles, forward returns, IC and regime stability
├── engine/portfolio.py       Next-open rebalance/trade simulator
├── costs/india.py            Configurable delivery/intraday cost model
├── metrics/performance.py    Return, risk, trade and multiple-testing metrics
├── validation/robustness.py  Walk-forward, sensitivity, random and Monte Carlo tests
├── reports/html.py           CSV, JSON and interactive HTML reports
└── cli.py                    `backtest signal` and `backtest strategy`
```

## Execution convention

A score calculated using the close of session T is not tradable on T. NATIP
stores the selection on T and fills at the adjusted open of the next available
session. Stops use the subsequent daily path. Portfolio weights are asserted to
remain at or below 100% gross exposure unless a future configuration explicitly
adds leverage support.

The engine is a rebalance-event simulator rather than a fully vectorized return
shortcut. That choice makes entry timing, skipped orders, stops, costs, and exit
reasons visible in the trade log while remaining practical for daily NSE data.

## Commands

```bash
backtest signal --name momentum_6_1 \
  --config configs/backtesting/nifty100_momentum.json

backtest strategy --config configs/backtesting/nifty100_momentum.json

backtest strategy --config configs/backtesting/nifty100_momentum.json \
  --robustness --random-samples 1000
```

During local development, `venv/bin/python backtest.py ...` runs the same CLI.
Add `--no-update` to use only the local cache.

Each run receives a timestamped folder under `reports/backtesting/`. It includes
the exact config, data-quality audit, trade log, skipped orders, period returns,
metrics, walk-forward embargo audit, charts, and known limitations. Robustness
runs add parameter sensitivity, random-selection, Monte Carlo, and doubled-cost
results. Every tried parameter configuration is logged in
`reports/backtesting/trial_log.jsonl`.

Order fees are calculated against `strategy.initial_capital_rupees`, so fixed
brokerage caps retain their rupee meaning before costs are converted back to a
portfolio-return fraction.

## Plug-in signal contract

Signals take one long-format `pandas.DataFrame` and return one numeric
`pandas.Series` with exactly the same index. A higher value means a higher rank.
The input includes `date`, `symbol`, adjusted/raw OHLCV, volume, liquidity, and
tradability fields. Register a function with `register_signal("name", function)`.

Fundamentals must contain a real `publication_date`. If unavailable, the helper
uses the configured reporting lag and sets `availability_estimated=True`.
Estimated rows must be disclosed and should not be treated as verified
point-in-time evidence.

## Data limitations

- Yahoo Finance is convenient but is not an exchange-grade point-in-time source.
- Current index constituents introduce survivorship bias. Supply historical
  constituent snapshots for publication-quality research.
- Yahoo adjustment factors are applied consistently to Open, High, Low, and
  Close. Raw Close is retained separately; volume is never forward-filled.
- Historical circuit-limit/order-book status is incomplete, so locked-market
  detection is approximate.
- Fees and taxes change. The JSON rates are configurable assumptions and should
  be verified for the broker, exchange, product, and effective date being tested.
- Capital-gains tax and depository participant charges are not included.
- A statistically interesting backtest is not a guarantee of future returns.
