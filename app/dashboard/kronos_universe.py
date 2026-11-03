"""Official, locally cached stock universes for the Kronos selector."""

from datetime import date, datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.dashboard.nse_symbols import all_nse_labels

UNIVERSE_URLS = {
    "Nifty 50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "Nifty 250": "https://nsearchives.nseindia.com/content/indices/ind_niftylargemidcap250list.csv",
    "Nifty 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "IPOs — last 2 years": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
}
UNIVERSE_OPTIONS = ("All NSE stocks", *UNIVERSE_URLS)
US_UNIVERSE_OPTIONS = ("All US stocks", "Popular US stocks")
US_POPULAR = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "LLY": "Eli Lilly",
    "AVGO": "Broadcom",
    "COST": "Costco",
    "NFLX": "Netflix",
    "AMD": "Advanced Micro Devices",
    "PLTR": "Palantir Technologies",
    "ORCL": "Oracle",
    "KO": "Coca-Cola",
}


def us_labels_from_frame(frame: pd.DataFrame) -> list[str]:
    """Normalize Nasdaq Trader directories, excluding ETFs and test instruments."""
    from app.intelligence.kronos_forecast import KronosForecastError, normalize_stock_symbol

    symbol_column = "Symbol" if "Symbol" in frame.columns else "ACT Symbol"
    if not {symbol_column, "Security Name", "Test Issue", "ETF"}.issubset(frame.columns):
        raise ValueError("US exchange directory is missing required fields.")
    selected = frame.loc[frame["Test Issue"].eq("N") & frame["ETF"].eq("N")]
    records = {}
    for _, row in selected.iterrows():
        raw = str(row[symbol_column]).strip().replace(".", "-")
        try:
            ticker = normalize_stock_symbol(raw, "US")
        except KronosForecastError:
            continue
        records[ticker] = f"{ticker} - {row['Security Name']}"
    return [records[ticker] for ticker in sorted(records)]


def load_us_universe(project_root: Path, universe: str) -> tuple[list[str], str]:
    """Load a cached US exchange stock directory with a small offline fallback."""
    fallback = [f"{s} - {name}" for s, name in sorted(US_POPULAR.items())]
    if universe == "Popular US stocks":
        return fallback, "Local popular-stock catalogue; not an index constituent list"
    if universe != "All US stocks":
        raise ValueError(f"Unsupported US universe: {universe}")
    cache = project_root / "data/kronos/universes/us_stocks.csv"
    today = datetime.now(ZoneInfo("America/New_York")).date()
    cached_date = (
        datetime.fromtimestamp(cache.stat().st_mtime, ZoneInfo("America/New_York")).date()
        if cache.exists()
        else None
    )
    warning = ""
    if cached_date != today:
        try:
            with requests.Session() as session:
                session.headers["User-Agent"] = "NATIP/1.0 (local equity research)"
                labels = set()
                for filename in ("nasdaqlisted.txt", "otherlisted.txt"):
                    response = session.get(
                        f"https://www.nasdaqtrader.com/dynamic/SymDir/{filename}", timeout=(5, 15)
                    )
                    response.raise_for_status()
                    labels.update(
                        us_labels_from_frame(pd.read_csv(StringIO(response.text), sep="|"))
                    )
            if not labels:
                raise ValueError("US exchange directory was empty.")
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            pd.DataFrame({"label": sorted(labels)}).to_csv(temporary, index=False)
            temporary.replace(cache)
            cached_date = today
        except (requests.RequestException, OSError, ValueError):
            if not cache.exists():
                return (
                    fallback,
                    "US directory unavailable; showing popular stocks. "
                    "Direct entry remains available.",
                )
            warning = " · Refresh unavailable; saved directory"
    return (
        pd.read_csv(cache)["label"].dropna().tolist(),
        f"Nasdaq Trader directory {cached_date}{warning}",
    )


def labels_from_frame(frame: pd.DataFrame, *, recent_ipos: bool, as_of: date) -> list[str]:
    """Select official constituents or equities listed in the past two years.

    Args:
        frame: NSE constituent or equity master table.
        recent_ipos: Whether to filter by the verified exchange listing date.
        as_of: Date used for the inclusive two-year listing window.

    Returns:
        Unique labels sorted by ticker.
    """
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if recent_ipos:
        required = {"SYMBOL", "NAME OF COMPANY", "DATE OF LISTING", "SERIES"}
        if not required.issubset(frame.columns):
            raise ValueError("NSE listing-date file is missing required columns.")
        listing_dates = pd.to_datetime(
            frame["DATE OF LISTING"].str.strip(), format="%d-%b-%Y", errors="coerce"
        )
        cutoff = pd.Timestamp(as_of) - pd.DateOffset(years=2)
        frame = frame.loc[
            listing_dates.between(cutoff, pd.Timestamp(as_of))
            & frame["SERIES"].str.strip().eq("EQ")
        ]
        symbol_column, name_column = "SYMBOL", "NAME OF COMPANY"
    else:
        symbol_column, name_column = "Symbol", "Company Name"
        if not {symbol_column, name_column}.issubset(frame.columns):
            raise ValueError("NSE constituent file is missing company/symbol columns.")
    records = {}
    for _, row in frame.dropna(subset=[symbol_column]).iterrows():
        symbol = str(row[symbol_column]).strip().upper()
        name = str(row[name_column]).strip()
        if symbol:
            records[symbol] = f"{symbol} - {name}"
    return [records[symbol] for symbol in sorted(records)]


def load_kronos_universe(project_root: Path, universe: str) -> tuple[list[str], str]:
    """Load selected NSE universe with a daily cache and stale-cache fallback.

    Args:
        project_root: NATIP project root.
        universe: One of the supported filter labels.

    Returns:
        Search labels and source/freshness information.
    """
    if universe == "All NSE stocks":
        return all_nse_labels(project_root), "Local NSE securities catalogue"
    if universe not in UNIVERSE_URLS:
        raise ValueError(f"Unsupported universe: {universe}")
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    cache = (
        project_root
        / "data/kronos/universes"
        / (
            "recent_listings.csv"
            if universe.startswith("IPOs")
            else universe.replace(" ", "_") + ".csv"
        )
    )
    cached_date = (
        datetime.fromtimestamp(cache.stat().st_mtime, ZoneInfo("Asia/Kolkata")).date()
        if cache.exists()
        else None
    )
    warning = ""
    if cached_date != today:
        try:
            with requests.Session() as session:
                session.headers["User-Agent"] = "NATIP/1.0 (local equity research)"
                session.mount(
                    "https://",
                    HTTPAdapter(
                        max_retries=Retry(
                            total=1, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
                        )
                    ),
                )
                response = session.get(UNIVERSE_URLS[universe], timeout=(5, 15))
                response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            labels_from_frame(frame, recent_ipos=universe.startswith("IPOs"), as_of=today)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(cache)
            cached_date = today
        except (requests.RequestException, OSError, ValueError) as exc:
            if not cache.exists():
                raise ValueError(
                    f"Unable to load {universe} from NSE. Try again or select All NSE stocks. {exc}"
                ) from exc
            warning = " · Using saved snapshot; NSE refresh unavailable"
    frame = pd.read_csv(cache)
    options = labels_from_frame(frame, recent_ipos=universe.startswith("IPOs"), as_of=today)
    return options, f"NSE snapshot {cached_date}{warning}"
