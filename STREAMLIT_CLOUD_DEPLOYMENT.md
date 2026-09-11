# NATIP Streamlit Cloud Deployment

Target URL:

```text
https://natip.streamlit.app
```

## Why this needs your account

The `natip.streamlit.app` subdomain can only be reserved from your Streamlit
Community Cloud account after the project is connected to a GitHub repository.
Codex can prepare the repo, but your Streamlit/GitHub account must create the
cloud app and claim the subdomain.

## Deploy Settings

- Platform: Streamlit Community Cloud
- Repository: your NATIP GitHub repository
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: `3.12`
- App URL/subdomain: `natip`

If `natip` is already taken, use a close variant such as:

```text
natip-ai
natip-nse
natip-research
```

## Secrets

In Streamlit Cloud, open app settings and add secrets using:

```toml
NATIP_ENVIRONMENT = "production"
NATIP_DEBUG = "false"
NATIP_GEMINI_API_KEY = "your_key_here"
NATIP_GEMINI_MODEL = "gemini-2.5-flash"
NATIP_ASTRO_ENABLED = "false"
NATIP_ASTRO_SHADOW_ONLY = "true"
NATIP_ASTRO_MAX_SCORE_ADJUSTMENT = "0"
```

Do not upload `.env`, passwords, broker tokens, Screener credentials, or local
SQLite/token files.

## Free-Tier Limits

The local project contains several gigabytes of historical cache. Do not push
that full cache to GitHub. The cloud app should run as a lightweight dashboard
using frozen model artifacts and small reference files. Heavy scans, retraining,
Screener scraping, and full cache rebuilds should remain local.

## Before Deploying

Run locally:

```bash
venv/bin/python -m pytest tests -q
venv/bin/streamlit run streamlit_app.py
```

Then push to GitHub and create the app at:

```text
https://share.streamlit.io
```
