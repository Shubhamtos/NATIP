"""Validated, cached current US index constituent lists for research screens."""

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.intelligence.kronos_forecast import normalize_stock_symbol

INDEX_SOURCES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "DJIA": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
    "Nasdaq 100": "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
}
INDEX_CACHE_FILES = {"S&P 500": "sp500.csv", "DJIA": "djia.csv", "Nasdaq 100": "nasdaq100.csv"}


def parse_index_constituents(html: str, index: str) -> pd.DataFrame:
    """Extract and validate current constituent identities, not historical membership."""
    if index not in INDEX_SOURCES:
        raise ValueError(f"Unsupported US index: {index}")
    for table in pd.read_html(StringIO(html)):
        name_column = "Security" if "Security" in table else "Company"
        symbol_column = "Symbol" if "Symbol" in table else "Ticker"
        if symbol_column not in table or name_column not in table:
            continue
        selected = table[[symbol_column, name_column]].dropna().copy()
        selected.columns = ["ticker", "company"]
        selected["ticker"] = (
            selected.ticker.astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        selected["ticker"] = selected.ticker.map(lambda value: normalize_stock_symbol(value, "US"))
        selected = selected.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
        if index == "S&P 500":
            valid_size = 450 <= len(selected) <= 550
        elif index == "Nasdaq 100":
            valid_size = 100 <= len(selected) <= 105
        else:
            valid_size = len(selected) == 30
        if not valid_size:
            raise ValueError(f"Invalid {index} constituent count: {len(selected)}")
        return selected
    raise ValueError(f"No {index} constituent table found.")


def load_us_index_universe(root: Path, index: str) -> tuple[list[str], str]:
    """Use a daily validated cache; expose stale-cache fallback without substituting stocks."""
    if index not in INDEX_SOURCES:
        raise ValueError(f"Unsupported US index: {index}")
    path = root / "data/market_strength/universes" / INDEX_CACHE_FILES[index]
    now = datetime.now(timezone.utc)
    cached = None
    if path.exists():
        try:
            frame = pd.read_csv(path)
            # Revalidate cached identities/count before using a saved universe.
            cached = parse_index_constituents(
                frame[["ticker", "company"]]
                .rename(columns={"ticker": "Symbol", "company": "Company"})
                .to_html(index=False),
                index,
            )
            saved_at = pd.to_datetime(frame["retrieved_at"].iloc[0], utc=True)
            if saved_at.date() == now.date():
                return (
                    _labels(cached),
                    f"{index} constituents · retrieved {saved_at.date()} · {INDEX_SOURCES[index]}",
                )
        except (ValueError, KeyError, OSError):
            cached = None
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "NATIP/1.0 (local stock research; constituent lookup)"
            session.mount(
                "https://",
                HTTPAdapter(
                    max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[502, 503, 504])
                ),
            )
            response = session.get(INDEX_SOURCES[index], timeout=(5, 20))
            response.raise_for_status()
            frame = parse_index_constituents(response.text, index)
        frame["retrieved_at"] = now.isoformat()
        frame["source"] = INDEX_SOURCES[index]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        return (
            _labels(frame),
            f"{index} constituents · retrieved {now.date()} · {INDEX_SOURCES[index]}",
        )
    except (requests.RequestException, ValueError, OSError) as exc:
        if cached is not None:
            return (
                _labels(cached),
                f"{index} saved constituents from {saved_at.date()} · refresh unavailable · {INDEX_SOURCES[index]}",
            )
        raise ValueError(
            f"{index} constituent list unavailable. Retry later; no substitute universe was used."
        ) from exc


def _labels(frame: pd.DataFrame) -> list[str]:
    return [f"{row.ticker} - {row.company}" for row in frame.itertuples()]
