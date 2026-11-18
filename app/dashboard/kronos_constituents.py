"""Exact published constituent lists for Kronos's selected sector index."""

import json
from datetime import UTC, datetime
from io import StringIO

import pandas as pd
import requests
import streamlit as st

from app.intelligence.kronos_forecast import normalize_nse_symbol

FILES = {
    "Nifty Auto": "ind_niftyautolist.csv",
    "Nifty IT": "ind_niftyitlist.csv",
    "Nifty Metal": "ind_niftymetallist.csv",
    "Nifty FMCG": "ind_niftyfmcglist.csv",
    "Nifty Energy": "ind_niftyenergylist.csv",
    "Nifty Pharma": "ind_niftypharmalist.csv",
    "Nifty Realty": "ind_niftyrealtylist.csv",
    "Nifty Infrastructure": "ind_niftyinfralist.csv",
    "Nifty Chemicals": "ind_niftychemicalslist.csv",
    "Nifty Consumer Durables": "ind_niftyconsumerdurableslist.csv",
    "Nifty Bank": "ind_niftybanklist.csv",
    "Nifty 50": "ind_nifty50list.csv",
}


def normalize_constituents(frame, index_name):
    frame = frame.rename(columns=lambda c: str(c).strip().lstrip("\ufeff"))
    if not {"Symbol", "Company Name"}.issubset(frame):
        raise ValueError("Official constituent file has unexpected columns.")
    frame = frame.dropna(subset=["Symbol", "Company Name"]).copy()
    frame["symbol"] = frame.Symbol.map(lambda s: normalize_nse_symbol(str(s).strip()))
    frame["company"] = frame["Company Name"].astype(str)
    frame["sector"] = index_name
    frame["industry"] = frame["Industry"] if "Industry" in frame else None
    if frame.empty:
        raise ValueError("Official constituent list is empty.")
    return (
        frame[["symbol", "company", "sector", "industry"]]
        .drop_duplicates("symbol")
        .sort_values("symbol")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_constituents(root, index_name):
    """One bounded public request, then cache fallback; never use industry proxies."""
    if index_name not in FILES:
        raise ValueError(f"No official constituent source configured for {index_name}.")
    path = root / "data/kronos/sector_constituents" / FILES[index_name]
    meta = path.with_suffix(".json")
    url = "https://www.niftyindices.com/IndexConstituent/" + FILES[index_name]
    saved = json.loads(meta.read_text()) if meta.exists() else {}
    local_50 = root / "data/kronos/universes/Nifty_50.csv"
    if index_name == "Nifty 50" and not path.exists() and local_50.exists():
        return normalize_constituents(pd.read_csv(local_50), index_name), dict(
            source=str(local_50),
            retrieved_date="unverified",
            status="Existing saved Nifty 50 constituent snapshot",
        )
    today = datetime.now(UTC).date().isoformat()
    if path.exists() and saved.get("retrieved_date") == today:
        return normalize_constituents(pd.read_csv(path), index_name), saved
    try:
        response = requests.get(
            url, timeout=(3, 8), headers={"User-Agent": "NATIP/1.0 constituent research"}
        )
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        clean = normalize_constituents(raw, index_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        raw.to_csv(temporary, index=False)
        temporary.replace(path)
        saved = dict(source=url, retrieved_date=today, status="Published constituent snapshot")
        meta.write_text(json.dumps(saved))
        return clean, saved
    except (requests.RequestException, ValueError, OSError) as exc:
        if path.exists():
            return normalize_constituents(pd.read_csv(path), index_name), saved | {
                "status": "Saved constituent list; refresh unavailable"
            }
        raise ValueError(
            f"Cannot load actual {index_name} members. No broad-sector substitute used."
        ) from exc


def import_constituents(root, index_name, raw):
    """Validate a user-downloaded list without claiming independent source verification."""
    clean = normalize_constituents(raw, index_name)
    path = root / "data/kronos/sector_constituents" / FILES[index_name]
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source="User-imported constituent CSV",
                retrieved_date=datetime.now(UTC).date().isoformat(),
                status="User-imported list; source/membership not independently verified",
            )
        )
    )
    load_constituents.clear()
    return clean
