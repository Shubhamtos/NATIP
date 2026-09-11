# NATIP

NSE Agentic Trading Intelligence Platform.

This repository contains a local-first NSE stock-analysis application. It uses
Yahoo Finance for market data, Streamlit for the desktop dashboard, and modular
agents/scanners for analysis workflows.

## Setup

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## FastAPI

```bash
uvicorn app.api.main:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /market/quote/RELIANCE`
- `GET /market/history/RELIANCE?days=30&interval=1d`
- `GET /market/status`
- `GET /evidence`

Bare symbols are normalized to Yahoo Finance NSE symbols with `.NS` by default.
For example, `RELIANCE` becomes `RELIANCE.NS`.

## Streamlit

```bash
streamlit run app/dashboard/streamlit_app.py
```

## Stock Probability %

The `Stock Probability %` tab ranks stocks by modeled probability of
outperforming Nifty 50 over the next 20 trading days. The production universe
is stored in:

```text
stocks_universe_2026_08.csv
```

It contains 150 stocks with `Ticker`, `Company`, `Sector`, `MarketCapCategory`,
and `Group` columns. The legacy 10-stock `stocks.csv` remains available for
small smoke tests.

To train the model:

```bash
python train_probability_model.py
```

To run the latest screener and save ranked probabilities:

```bash
python run_screener.py
```

Output is saved to:

```text
reports/probability/stock_probability_latest.csv
reports/probability/stock_probability_150_full_ranking.csv
reports/probability/stock_probability_150_top20.csv
reports/probability/stock_probability_150_data_quality.json
reports/probability/stock_probability_150_model_metrics.json
reports/probability/stock_probability_150_backtest.json
reports/probability/stock_probability_150_vs_10_comparison.json
```

The model uses only backward-looking features as of each date. Targets are
future 20-trading-day stock return minus future 20-trading-day Nifty return:

- `1`: Outperform if excess return is above +5%
- `0`: Neutral if excess return is between -5% and +5%
- `-1`: Underperform if excess return is below -5%

Splits are time-based:

- 2018-2022: training
- 2023-2024: validation
- 2025-latest: final out-of-sample test

To scale the universe later, replace `stocks_universe_2026_08.csv` with 250-500
NSE symbols in the same metadata format.
