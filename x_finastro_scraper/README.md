# X Financial Astrology Scraper

This project extracts financial-astrology-related posts from the X account **@anandrathi12** using the official X API.

## What it captures

For each matching post it stores:

- date/time
- full post text
- detected astrology terms
- detected market/asset terms
- bullish / bearish / neutral classification
- numeric levels mentioned in the post
- likes, reposts, replies and quotes
- post ID and direct X URL

The current financial-astrology dictionary includes terms such as:

`#finastro`, financial astrology, Mercury, Venus, Mars, Jupiter, Saturn,
Rahu, Ketu, Moon, full moon, new moon, eclipse, retrograde, nakshatra,
zodiac, transit and conjunction.

The market dictionary includes Nifty, Bank Nifty, Sensex, stocks, gold,
silver, crude, Bitcoin and common directional/technical terms.

## Security

**Never commit an X bearer token.**

For GitHub Actions, create a repository secret named:

`X_BEARER_TOKEN`

GitHub path:

**Repository → Settings → Secrets and variables → Actions → New repository secret**

## Run locally

```bash
cd x_finastro_scraper
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your bearer token, then run:

```bash
python src/main.py
```

Output is written to:

```
x_finastro_scraper/output/anandrathi12_financial_astrology.csv
x_finastro_scraper/output/anandrathi12_financial_astrology.xlsx
```

## Run in GitHub Actions

Open:

**Actions → X Financial Astrology Scraper → Run workflow**

The workflow uploads the generated CSV and Excel files as a GitHub Actions artifact.

## Important limitation

The amount of historical data available depends on the X API product/access level associated with your account. This project uses X's user-posts endpoint and paginates through everything the endpoint makes available.

## Interpretation

The directional label is a text classification of what the post appears to say. It is not investment advice and does not establish that astrological events cause market movements.
