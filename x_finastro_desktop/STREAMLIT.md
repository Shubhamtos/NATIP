# Streamlit Financial Astrology Dashboard

This repository now includes a Streamlit front end for the X financial-astrology scraper.

## Streamlit entry point

```
x_finastro_desktop/streamlit_app.py
```

## Run locally

```bash
cd x_finastro_desktop
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens in your browser.

## Modes

### 1. Hosted X API mode

Store the X bearer token as a Streamlit secret named:

```toml
X_BEARER_TOKEN = "..."
```

The app then lets you press **Start hosted scrape**. It paginates until X no longer returns a next page and then displays and exports the dataset.

### 2. Upload / dashboard mode

Upload CSV or XLSX from the existing Playwright desktop scraper. The Streamlit app re-runs the financial-astrology classifier and provides filters, metrics, tables, and downloads.

### 3. Full local browser scraper

The existing `app.py` remains available for the Playwright workflow. This is the preferred mode when X requires an interactive authenticated browser.

## Streamlit Community Cloud deployment

Because the repository is public, create a Streamlit Community Cloud app from:

- Repository: `Shubhamtos/NATIP`
- Branch: `main`
- Main file path: `x_finastro_desktop/streamlit_app.py`

Add `X_BEARER_TOKEN` under the app's **Secrets** if you want hosted scraping.

The bearer token must never be committed to GitHub.
