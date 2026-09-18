# X Financial Astrology Scraper — Desktop App

A Python desktop application configured for **@anandrathi12** by default.

It opens X in a persistent Chromium session, continuously scrolls the account timeline, captures each unique post, checkpoints the dataset, and stops only after X repeatedly returns no additional posts and recovery attempts are exhausted.

## What the final output contains

The app creates:

- `anandrathi12_financial_astrology.xlsx`
  - **Financial_Astrology** sheet — only matched financial-astrology posts
  - **All_Scraped** sheet — all posts collected from the account
- `anandrathi12_financial_astrology.csv`
- `anandrathi12_all_scraped.csv`
- `anandrathi12_checkpoint.json`

For matched posts it adds:

- astrology factors/keywords
- market/asset terms
- bullish / bearish / neutral classification
- numbers or market levels found
- date/time
- likes, reposts, replies, bookmarks when visible
- tweet ID and direct URL

## Important meaning of “complete”

There is **no arbitrary tweet-count limit** in this app.

It keeps scrolling until the X timeline stops yielding new unique posts across repeated checks, confirms that the page is at the bottom, and performs multiple recovery reloads before declaring completion.

X itself may limit how much historical timeline it exposes to a logged-in web session. No scraper can guarantee access to posts that X does not return to that session.

## First-time setup

Requires Python 3.10+.

### macOS / Linux

```bash
cd x_finastro_desktop
python3 setup.py
python3 app.py
```

### Windows

```bat
cd x_finastro_desktop
python setup.py
python app.py
```

On first scrape, Chromium opens. If X asks you to log in, sign in in that browser window. The app detects the timeline automatically and continues. The saved browser profile remains **only on your computer**.

## Normal use after setup

macOS: double-click `run_mac.command` (you may need to allow execution once).

Windows: double-click `run_windows.bat`.

Then:

1. Keep `anandrathi12` as the username (or change it).
2. Choose the output folder.
3. Click **Start Scraping**.
4. If prompted by X, log in in the opened Chromium window.
5. Leave the app running. It checkpoints continually.
6. When the app reports **COMPLETE**, open the generated Excel file.

## Resume support

If the app or computer closes, start it again using the same output directory. It reloads the previous `*_all_scraped.csv`, deduplicates by tweet ID, and continues.

## Privacy and safety

The app does not ask for or save your X password. Authentication is handled inside the browser. The local Playwright browser profile is excluded from Git.

Use the software responsibly and in accordance with X's terms and applicable law. X can change its website markup at any time, which may require selector updates.
